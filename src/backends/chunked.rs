//! A slow, fallible, block-oriented backend that stands in for a network or archive source.
//!
//! It is intentionally awkward for the naive "seek a shared handle and read" pattern: a file is a
//! list of disjoint blocks (no contiguous whole-file borrow is possible), a single read returns at
//! most `chunk_max` bytes, reads can be armed to fail, and the backend can refuse a fetch outright
//! because it is at capacity ([`ChunkedFs::set_connection_budget`]). Everything a correct descriptor
//! layer must tolerate shows up here.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::resource::{Filesystem, Resource};
use crate::types::{Access, Error, FileId, Result};

/// A deterministic, externally-armed fault trigger shared by a backend's resources. A test arms the
/// next read to fail; the read consumes the arming. This exercises the invariant that a failed
/// read must not move the caller's cursor — no timing races involved.
#[derive(Debug, Default)]
pub struct Faults {
    fail_next: AtomicBool,
    block_next: AtomicBool,
}

impl Faults {
    /// Next `read_at` returns [`Error::Backend`].
    pub fn arm_error(&self) {
        self.fail_next.store(true, Ordering::SeqCst);
    }
    /// Next `read_at` returns [`Error::WouldBlock`].
    pub fn arm_would_block(&self) {
        self.block_next.store(true, Ordering::SeqCst);
    }
    fn take(&self) -> Option<Fault> {
        if self.block_next.swap(false, Ordering::SeqCst) {
            Some(Fault::WouldBlock)
        } else if self.fail_next.swap(false, Ordering::SeqCst) {
            Some(Fault::Error)
        } else {
            None
        }
    }
}

enum Fault {
    WouldBlock,
    Error,
}

/// A per-backend cap on how many fetches may be in flight at once — the mock of a connection limit.
///
/// This is only expressible because the read position lives *above* the seam: the backend can
/// refuse service for pure capacity reasons and the caller's cursor is untouched, so the caller
/// retries from the same place. A backend whose own cursor lived inside the object being capped
/// could not shed load without losing its place in the file.
#[derive(Debug)]
struct ConnPool {
    budget: AtomicUsize,
    used: AtomicUsize,
    peak: AtomicUsize,
    denied: AtomicUsize,
}

impl Default for ConnPool {
    fn default() -> Self {
        Self::with_budget(usize::MAX)
    }
}

impl ConnPool {
    fn with_budget(budget: usize) -> Self {
        Self {
            budget: AtomicUsize::new(budget),
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            denied: AtomicUsize::new(0),
        }
    }

    /// Take one slot, or `None` if the budget is already spent. Never queues.
    fn acquire(&self) -> Option<Conn<'_>> {
        let budget = self.budget.load(Ordering::Relaxed);
        match self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |used| {
                if used < budget {
                    Some(used + 1)
                } else {
                    None
                }
            }) {
            Ok(used) => {
                self.peak.fetch_max(used + 1, Ordering::Relaxed);
                Some(Conn { pool: self })
            }
            Err(_) => {
                self.denied.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
}

/// One in-flight fetch. Released on drop, including when the fetch fails.
#[derive(Debug)]
struct Conn<'a> {
    pool: &'a ConnPool,
}

impl Drop for Conn<'_> {
    fn drop(&mut self) {
        self.pool.used.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Counters a caller may *report* on. Never used to assert wall-clock timing (see DESIGN.md).
#[derive(Debug, Default)]
pub struct ChunkedStats {
    reads: AtomicU64,
    bytes_returned: AtomicU64,
    errors: AtomicU64,
    would_blocks: AtomicU64,
    busy_ns: AtomicU64,
}

impl ChunkedStats {
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.reads.load(Ordering::Relaxed),
            self.bytes_returned.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
            self.would_blocks.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug)]
pub struct SlowConfig {
    /// Max bytes a single `read_at` will return, even if more is available. Forces callers to loop.
    pub chunk_max: usize,
    /// Optional per-read artificial latency. Defaults to `Duration::ZERO`. Reported, never asserted.
    pub latency: Duration,
}

impl Default for SlowConfig {
    fn default() -> Self {
        Self {
            chunk_max: 7,
            latency: Duration::ZERO,
        }
    }
}

#[derive(Debug)]
struct ChunkedResource {
    backend: &'static str,
    blocks: Vec<Arc<[u8]>>,
    size: u64,
    chunk_max: usize,
    latency: Duration,
    faults: Arc<Faults>,
    stats: Arc<ChunkedStats>,
    closed: Arc<AtomicBool>,
    pool: Arc<ConnPool>,
}

impl ChunkedResource {
    /// Byte at a logical offset, walking the disjoint block list. O(blocks); a real backend would
    /// index. This proves reads work without any contiguous borrow of the file.
    fn byte_at(&self, offset: usize) -> Option<u8> {
        let mut base = 0usize;
        for block in &self.blocks {
            let end = base + block.len();
            if offset < end {
                return Some(block[offset - base]);
            }
            base = end;
        }
        None
    }
}

impl Resource for ChunkedResource {
    fn size(&self) -> Option<u64> {
        Some(self.size)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::BackendShutdown(self.backend));
        }
        match self.faults.take() {
            Some(Fault::WouldBlock) => {
                self.stats.would_blocks.fetch_add(1, Ordering::Relaxed);
                return Err(Error::WouldBlock);
            }
            Some(Fault::Error) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                return Err(Error::backend(self.backend, "injected read failure"));
            }
            None => {}
        }

        if offset >= self.size {
            self.stats.reads.fetch_add(1, Ordering::Relaxed);
            return Ok(0); // EOF — known from the size, so it costs no round trip
        }
        // Capacity, not correctness: at budget the fetch is refused and the caller's cursor stays
        // exactly where it was. `Err` is what keeps it there (see `read_locked` in ofd.rs).
        let _conn = match self.pool.acquire() {
            Some(conn) => conn,
            None => return Err(Error::WouldBlock),
        };
        // The defining restriction: never more than `chunk_max` bytes in one call.
        let remaining = (self.size - offset) as usize;
        let want = usize::min(usize::min(buf.len(), remaining), self.chunk_max);

        let started = Instant::now();
        let start = offset as usize;
        let mut written = 0usize;
        while written < want {
            // byte_at already validated by `remaining`
            buf[written] = self.byte_at(start + written).unwrap_or(0);
            written += 1;
        }
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        self.stats.busy_ns.fetch_add(
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_returned
            .fetch_add(want as u64, Ordering::Relaxed);
        Ok(want)
    }

    // No `as_shared_bytes` override: this backend cannot hand back the whole file.
}

#[derive(Debug, Default)]
pub struct ChunkedFs {
    inner: Mutex<Inner>,
    next_ino: AtomicU64,
    faults: Arc<Faults>,
    stats: Arc<ChunkedStats>,
    config: Mutex<SlowConfig>,
    closed: Arc<AtomicBool>,
    pool: Arc<ConnPool>,
}

#[derive(Debug, Default)]
struct Inner {
    by_name: HashMap<String, FileId>,
    blocks: HashMap<FileId, Vec<Arc<[u8]>>>,
    sizes: HashMap<FileId, u64>,
}

impl ChunkedFs {
    pub const NAME: &'static str = "chunked";

    pub fn new(config: SlowConfig) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            next_ino: AtomicU64::new(0),
            faults: Arc::new(Faults::default()),
            stats: Arc::new(ChunkedStats::default()),
            config: Mutex::new(config),
            closed: Arc::new(AtomicBool::new(false)),
            pool: Arc::new(ConnPool::default()),
        }
    }

    /// Add a file as `chunk_size`-sized blocks — deliberately non-contiguous.
    pub fn add_chunked(&self, name: &str, bytes: impl Into<Vec<u8>>, chunk_size: usize) -> FileId {
        assert!(chunk_size > 0);
        let bytes: Vec<u8> = bytes.into();
        let size = bytes.len() as u64;
        let blocks: Vec<Arc<[u8]>> = bytes.chunks(chunk_size.max(1)).map(Arc::from).collect();
        let mut inner = self.lock();
        let id = match inner.by_name.get(name) {
            Some(existing) => *existing,
            None => {
                let ino = self.next_ino.fetch_add(1, Ordering::Relaxed);
                FileId::new(Self::NAME, ino)
            }
        };
        inner.by_name.insert(name.to_string(), id);
        inner.blocks.insert(id, blocks);
        inner.sizes.insert(id, size);
        id
    }

    pub fn faults(&self) -> Arc<Faults> {
        self.faults.clone()
    }

    pub fn stats(&self) -> Arc<ChunkedStats> {
        self.stats.clone()
    }

    /// Cap how many fetches this backend serves concurrently — the mock of a connection budget.
    /// Refusal at budget surfaces as [`Error::WouldBlock`], never as a queue and never as a failure.
    /// `usize::MAX` (the default) means unlimited; a request for `0` is clamped to `1`, since a
    /// backend that never serves anything is what [`Filesystem::shutdown`] is for.
    pub fn set_connection_budget(&self, budget: usize) {
        self.pool.budget.store(budget.max(1), Ordering::Relaxed);
    }

    pub fn live_connections(&self) -> usize {
        self.pool.used.load(Ordering::Relaxed)
    }

    /// Highest simultaneous connection count seen since the last reset.
    pub fn peak_connections(&self) -> usize {
        self.pool.peak.load(Ordering::Relaxed)
    }

    /// Fetches refused because the budget was already spent.
    pub fn connection_denials(&self) -> usize {
        self.pool.denied.load(Ordering::Relaxed)
    }

    /// Zero the peak and denial counters so a measurement starts from a known state.
    pub fn reset_connection_stats(&self) {
        self.pool.peak.store(0, Ordering::Relaxed);
        self.pool.denied.store(0, Ordering::Relaxed);
    }

    fn config(&self) -> SlowConfig {
        let g = self.config.lock().unwrap_or_else(|p| p.into_inner());
        SlowConfig {
            chunk_max: g.chunk_max,
            latency: g.latency,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Filesystem for ChunkedFs {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn lookup(&self, name: &str) -> Option<FileId> {
        self.lock().by_name.get(name).copied()
    }

    fn open(&self, id: FileId, _access: Access) -> Result<Arc<dyn Resource>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::BackendShutdown(Self::NAME));
        }
        let inner = self.lock();
        if id.backend != Self::NAME {
            return Err(Error::NoSuchFile(id));
        }
        let blocks = inner
            .blocks
            .get(&id)
            .cloned()
            .ok_or(Error::NoSuchFile(id))?;
        let size = inner.sizes.get(&id).copied().unwrap_or(0);
        let cfg = self.config();
        Ok(Arc::new(ChunkedResource {
            backend: Self::NAME,
            blocks,
            size,
            chunk_max: cfg.chunk_max,
            latency: cfg.latency,
            faults: self.faults.clone(),
            stats: self.stats.clone(),
            closed: self.closed.clone(),
            pool: self.pool.clone(),
        }))
    }

    fn shutdown(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }
}
