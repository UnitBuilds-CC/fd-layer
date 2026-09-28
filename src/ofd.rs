//! The open file description — the object the existing WASIX layer is missing.
//!
//! An OFD is what `open()` creates: it pairs one resource with **one shared cursor** and one
//! access mode, and its lifetime is its reference count. `fork`/`dup` share an `Arc<OpenFileDescription>`
//! (so the cursor advances together); two separate `open()`s create two separate OFDs (so their
//! cursors are independent). This is the POSIX three-object model with the middle object made
//! explicit, and it is what makes the resource lifetime structural: there is no manual "open
//! handle" counter to get out of sync — when the last descriptor referencing an OFD closes and no
//! thread holds a clone, `Arc` drops the OFD and the last clone of its resource drops the resource.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::resource::Resource;
use crate::types::{Access, Error, ReadOutcome, Result};

/// Shared cursor + resource for one open. Cloning the surrounding `Arc` (fork/dup) shares it.
#[derive(Debug)]
pub struct OpenFileDescription {
    access: Access,
    /// Readable through `&self`: pread-style reads take no lock at all, and sequential reads
    /// contend only on `position`, never on the resource.
    resource: Arc<dyn Resource>,
    /// The offset for sequential `read`. Guarded so that "read the cursor, fetch the bytes,
    /// advance the cursor" is one critical section — a failed fetch leaves the cursor untouched.
    position: Mutex<u64>,
    /// Test seam (see [`OpenFileDescription::arm_read_pause`]): two flags let a test freeze a read
    /// *inside* the cursor critical section — after the fetch, before the advance — so the "offset,
    /// fetch and advance are one atomic step" invariant is checked deterministically rather than by
    /// racing real threads and hoping to catch the window. Always false in normal use; the read path
    /// only pays one uncontended atomic load.
    read_pause_armed: AtomicBool,
    read_pause_hit: AtomicBool,
}

impl OpenFileDescription {
    pub(crate) fn new(access: Access, resource: Arc<dyn Resource>) -> Arc<Self> {
        Arc::new(Self {
            access,
            resource,
            position: Mutex::new(0),
            read_pause_armed: AtomicBool::new(false),
            read_pause_hit: AtomicBool::new(false),
        })
    }

    /// Total size if the backend reports it.
    pub fn size(&self) -> Option<u64> {
        self.resource.size()
    }

    /// Current sequential cursor.
    pub fn position(&self) -> u64 {
        *self.lock_position()
    }

    /// Move the sequential cursor. Provided because placement of a reader is a real need (and it
    /// demonstrates that seek and read share exactly one critical section). Returns the new
    /// position. Does not validate against size — reading past the end is defined to yield EOF.
    pub fn seek(&self, to: u64) -> u64 {
        let mut pos = self.lock_position();
        *pos = to;
        to
    }

    /// Whether two descriptors resolve to the *same* OFD (shared cursor) rather than two opens of
    /// the same file. Structural evidence for fork/dup semantics.
    pub fn is_same(a: &Arc<Self>, b: &Arc<Self>) -> bool {
        Arc::ptr_eq(a, b)
    }

    /// Zero-copy escape hatch: an owned view of the whole resource when the backend keeps it in
    /// memory. `None` for streaming/fallible backends — callers must handle that path.
    pub fn as_shared_bytes(&self) -> Option<Arc<[u8]>> {
        self.resource.as_shared_bytes()
    }

    /// Sequential read that blocks while another reader on the *same OFD* holds the cursor.
    pub fn read(&self, buf: &mut [u8]) -> Result<ReadOutcome> {
        if !self.access.read {
            return Err(Error::NotReadable);
        }
        if buf.is_empty() {
            return Ok(ReadOutcome::data(0));
        }
        let mut pos = self.lock_position();
        self.read_locked(&mut pos, buf)
    }

    /// Non-blocking variant. If another reader currently holds this OFD's cursor, returns
    /// [`Error::WouldBlock`] instead of queueing — the caller keeps its place and retries. This is
    /// *stricter* than the current WASIX behavior, which silently serializes readers onto one host
    /// handle and can interleave cursor updates; here a contended cursor is an explicit retry.
    pub fn try_read(&self, buf: &mut [u8]) -> Result<ReadOutcome> {
        if !self.access.read {
            return Err(Error::NotReadable);
        }
        if buf.is_empty() {
            return Ok(ReadOutcome::data(0));
        }
        let mut pos = self.try_lock_position()?;
        self.read_locked(&mut pos, buf)
    }

    /// **Zero-copy** sequential read. When the backend can borrow the whole resource (see
    /// [`Resource::as_shared_bytes`]) this advances the shared cursor and returns a borrowed
    /// [`SharedRead`] — *no bytes are copied into a caller buffer*. When the backend cannot borrow
    /// (a streaming/network/disk-file backend), it returns `Ok(None)` and the caller falls back to
    /// [`OpenFileDescription::read`]. It takes the *same* cursor lock as `read`, so sequential
    /// semantics and the failed-read-keeps-cursor invariant (I2) are identical on both paths.
    pub fn read_shared(&self, want: usize) -> Result<Option<SharedRead>> {
        if !self.access.read {
            return Err(Error::NotReadable);
        }
        let mut pos = self.lock_position();
        let start = *pos;
        match self.resource.as_shared_bytes() {
            Some(bytes) => {
                let avail = (bytes.len() as u64).saturating_sub(start);
                let len = usize::min(want, avail as usize);
                *pos = start.saturating_add(len as u64);
                Ok(Some(SharedRead {
                    bytes,
                    start,
                    len,
                    want,
                }))
            }
            None => Ok(None),
        }
    }

    /// Positioned read (POSIX `pread`): read at an explicit offset **without touching the shared
    /// cursor**. Takes no lock, so any number of `read_at` calls on one OFD run concurrently — this
    /// is the capability the old seek-per-read handle could not express.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if !self.access.read {
            return Err(Error::NotReadable);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        self.resource.read_at(offset, buf)
    }

    /// One critical section: fetch at the cursor, then advance it — and only on success.
    fn read_locked(&self, pos: &mut u64, buf: &mut [u8]) -> Result<ReadOutcome> {
        let start = *pos;
        match self.resource.read_at(start, buf) {
            Ok(0) => {
                // Backend reports end of file. Cursor stays (a subsequent read is still EOF).
                Ok(ReadOutcome::eof())
            }
            Ok(n) => {
                // Test seam: freeze here — the fetch has happened, the cursor has *not* advanced,
                // and the position lock is still held. See `arm_read_pause`.
                self.maybe_pause_for_test();
                *pos = start.saturating_add(n as u64);
                Ok(ReadOutcome::data(n))
            }
            Err(e) => Err(e), // cursor untouched — the atomicity invariant
        }
    }

    fn lock_position(&self) -> MutexGuard<'_, u64> {
        self.position
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn try_lock_position(&self) -> Result<MutexGuard<'_, u64>> {
        self.position.try_lock().map_err(|_| Error::WouldBlock)
    }

    // ---- deterministic test seam --------------------------------------------------
    //
    // These exist so a test can hold a read at the single most dangerous instant — inside the
    // cursor critical section, after `read_at` returned bytes but before the offset is committed —
    // and observe from another thread that (a) a competing `try_read` sees `WouldBlock` at the lock
    // and never reaches the backend, and (b) once released the cursor advances exactly once. That
    // turns the flagship race from "run threads and hope" into a deterministic check. Not part of
    // the semantic surface: `#[doc(hidden)]`, and the read path only pays an uncontended load.

    /// Arm the seam: the next sequential read through this OFD that fetches bytes will block,
    /// *holding the cursor lock*, just before it advances the offset.
    #[doc(hidden)]
    pub fn arm_read_pause(&self) {
        self.read_pause_hit.store(false, Ordering::Relaxed);
        self.read_pause_armed.store(true, Ordering::SeqCst);
    }

    /// Release a read parked by [`OpenFileDescription::arm_read_pause`] so it advances and returns.
    #[doc(hidden)]
    pub fn release_read_pause(&self) {
        self.read_pause_armed.store(false, Ordering::SeqCst);
    }

    /// Whether a paused read has reached the freeze point (test spin-gate).
    #[doc(hidden)]
    pub fn read_pause_reached(&self) -> bool {
        self.read_pause_hit.load(Ordering::Acquire)
    }

    fn maybe_pause_for_test(&self) {
        if !self.read_pause_armed.load(Ordering::SeqCst) {
            return;
        }
        // Publish "fetch complete, cursor not yet advanced" to a test spinning on `reached`.
        self.read_pause_hit.store(true, Ordering::Release);
        while self.read_pause_armed.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
    }
}

/// A borrowed result from [`OpenFileDescription::read_shared`]: an owned view of the whole resource
/// plus the range `[start, start + len)` that the sequential cursor consumed. [`SharedRead::slice`]
/// hands back the actual bytes with **no copy**. `len == 0` at a non-zero `want` means EOF at
/// `start`; `len == 0` with `want == 0` is a zero-length request (not EOF), matching the copy path.
#[derive(Debug)]
pub struct SharedRead {
    pub bytes: Arc<[u8]>,
    pub start: u64,
    pub len: usize,
    /// Bytes the caller asked for; kept so `end_of_file` can distinguish a genuine EOF from a
    /// zero-length request (per `types.rs`, `end_of_file` is meaningful only for a non-empty read).
    want: usize,
}

impl SharedRead {
    /// The consumed bytes, borrowed from the resource — the whole point of the fast path.
    pub fn slice(&self) -> &[u8] {
        let start = self.start as usize;
        &self.bytes[start..start + self.len]
    }

    /// Whether the read reached end of file: no bytes produced *at a non-zero-length request*.
    pub fn end_of_file(&self) -> bool {
        self.want > 0 && self.len == 0
    }
}
