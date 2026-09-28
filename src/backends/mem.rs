//! A fixed-collection, in-memory backend.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::resource::{Filesystem, Resource};
use crate::types::{Access, Error, FileId, Result};

/// Immutable, fully-resident file. `read_at` is a bounded slice copy; `as_shared_bytes` is a cheap
/// `Arc` bump. Reads cannot fail (beyond a past-end → EOF), which is exactly the assumption the
/// fallible backend exists to break. It observes the backend's shutdown flag so a whole-fs close is
/// visible to descriptors opened *before* the shutdown.
#[derive(Debug)]
struct MemResource {
    bytes: Arc<[u8]>,
    closed: Arc<AtomicBool>,
}

impl Resource for MemResource {
    fn size(&self) -> Option<u64> {
        Some(self.bytes.len() as u64)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::BackendShutdown(crate::backends::mem::MemFs::NAME));
        }
        let len = self.bytes.len() as u64;
        if offset >= len {
            return Ok(0); // past end → EOF, never a panic
        }
        let start = offset as usize;
        let n = usize::min(buf.len(), self.bytes.len() - start);
        buf[..n].copy_from_slice(&self.bytes[start..start + n]);
        Ok(n)
    }

    fn as_shared_bytes(&self) -> Option<Arc<[u8]>> {
        // Honor shutdown on the zero-copy path too: once closed, no borrow is handed out, so
        // `read_shared` reports `Ok(None)` and the caller's fallback `read` surfaces BackendShutdown.
        if self.closed.load(Ordering::Acquire) {
            return None;
        }
        Some(self.bytes.clone())
    }
}

/// A backend whose entire file set is known ahead of time (a fixed collection).
#[derive(Debug, Default)]
pub struct MemFs {
    inner: Mutex<Inner>,
    next_ino: AtomicU64,
    closed: Arc<AtomicBool>,
}

#[derive(Debug, Default)]
struct Inner {
    by_name: HashMap<String, FileId>,
    data: HashMap<FileId, Arc<[u8]>>,
}

impl MemFs {
    pub const NAME: &'static str = "mem";

    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a file, returning its stable [`FileId`]. `ino` is allocated here and
    /// never derived from the path — the identity stays valid even if a future rename moved it.
    pub fn add(&self, name: &str, bytes: impl Into<Vec<u8>>) -> FileId {
        let bytes: Arc<[u8]> = Arc::from(bytes.into());
        let mut inner = self.lock();
        let id = match inner.by_name.get(name) {
            Some(existing) => *existing,
            None => {
                let ino = self.next_ino.fetch_add(1, Ordering::Relaxed);
                FileId::new(Self::NAME, ino)
            }
        };
        inner.by_name.insert(name.to_string(), id);
        inner.data.insert(id, bytes);
        id
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Filesystem for MemFs {
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
        let bytes = inner
            .data
            .get(&id)
            .filter(|_| id.backend == Self::NAME)
            .cloned()
            .ok_or(Error::NoSuchFile(id))?;
        Ok(Arc::new(MemResource {
            bytes,
            closed: self.closed.clone(),
        }))
    }

    fn shutdown(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }
}
