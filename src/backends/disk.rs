//! A backend over **real files on disk** — the "proper file" check.
//!
//! The other two backends are synthetic (an in-memory buffer; a hand-built block mock). This one
//! opens an actual [`std::fs::File`] and reads it through the same `&self`-only [`Resource`]
//! interface, so the abstraction is exercised against genuine OS file I/O rather than canned data.
//!
//! Two open modes deliberately stress the two halves of the read contract:
//!
//! * [`Residency::Streaming`] keeps an `Arc<File>` and serves `read_at` with the OS *positional*
//!   read (`pread` on Unix, `seek_read` on Windows). It cannot borrow the whole file
//!   (`as_shared_bytes` is `None`), it takes no `&mut`, and because a positional read never touches
//!   shared seek state it is genuinely concurrent — exactly the shape the network backend has, but
//!   on a real file descriptor.
//! * [`Residency::Resident`] reads the file into an `Arc<[u8]>` once, at open, and *does* offer
//!   `as_shared_bytes`. This is what lets a zero-copy sequential read run against a real file with
//!   no `unsafe` and no memory-map dependency — the whole-file copy is paid once, not per read.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::resource::{Filesystem, Resource};
use crate::types::{Access, Error, FileId, Result};

/// How a disk resource exposes the file it stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Read the whole file into an `Arc<[u8]>` at open; `as_shared_bytes` is `Some` (zero-copy
    /// reads available), at the cost of an O(size) load once.
    Resident,
    /// Keep an `Arc<File>` and `read_at` via the OS positional read; `as_shared_bytes` is `None`.
    Streaming,
}

/// Positional read that never moves the file's own cursor. Safe to call from many threads on one
/// shared `File` because it does not mutate seek state.
fn positional_read(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::other(
            "positional read not implemented on this platform",
        ))
    }
}

/// Resident whole-file view (zero-copy capable).
#[derive(Debug)]
struct ResidentResource {
    bytes: Arc<[u8]>,
    backend: &'static str,
    closed: Arc<AtomicBool>,
}

/// Streaming real-fd view (positional reads; no whole-file borrow).
#[derive(Debug)]
struct StreamingResource {
    file: Arc<File>,
    size: u64,
    backend: &'static str,
    closed: Arc<AtomicBool>,
}

impl Resource for ResidentResource {
    fn size(&self) -> Option<u64> {
        Some(self.bytes.len() as u64)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::BackendShutdown(self.backend));
        }
        let len = self.bytes.len() as u64;
        if offset >= len {
            return Ok(0);
        }
        let start = offset as usize;
        let n = usize::min(buf.len(), self.bytes.len() - start);
        buf[..n].copy_from_slice(&self.bytes[start..start + n]);
        Ok(n)
    }

    fn as_shared_bytes(&self) -> Option<Arc<[u8]>> {
        // No borrow after shutdown — keeps the zero-copy path honest with the copy path's contract.
        if self.closed.load(Ordering::Acquire) {
            return None;
        }
        Some(self.bytes.clone())
    }
}

impl Resource for StreamingResource {
    fn size(&self) -> Option<u64> {
        Some(self.size)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::BackendShutdown(self.backend));
        }
        if buf.is_empty() || offset >= self.size {
            return Ok(0);
        }
        let n = positional_read(&self.file, buf, offset)
            .map_err(|e| Error::backend(self.backend, e.to_string()))?;
        Ok(n)
    }

    // No `as_shared_bytes`: this backend streams from a real file descriptor.
}

/// A backend rooted at a directory on the real filesystem. A file is addressed by its name
/// relative to that root.
#[derive(Debug)]
pub struct DiskFs {
    root: PathBuf,
    residency: Residency,
    next_ino: AtomicU64,
    closed: Arc<AtomicBool>,
    inner: Mutex<DiskInner>,
}

#[derive(Debug, Default)]
struct DiskInner {
    by_name: HashMap<String, FileId>,
    path_of: HashMap<FileId, PathBuf>,
}

impl DiskFs {
    pub const NAME: &'static str = "disk";

    /// Build a disk backend rooted at `root` (need not exist until a file is looked up).
    pub fn new(root: impl Into<PathBuf>, residency: Residency) -> Self {
        Self {
            root: root.into(),
            residency,
            next_ino: AtomicU64::new(0),
            closed: Arc::new(AtomicBool::new(false)),
            inner: Mutex::new(DiskInner::default()),
        }
    }

    fn path_for(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Resolve `name` to a stable [`FileId`] *if* a real file exists at `root/name`, minting and
    /// caching the id on first sight. Identity is allocated here, never derived from the path hash,
    /// so it stays valid even if the name later moved.
    fn resolve(&self, name: &str) -> Option<FileId> {
        let path = self.path_for(name);
        if !path.is_file() {
            return None;
        }
        let mut inner = self.lock();
        if let Some(existing) = inner.by_name.get(name) {
            return Some(*existing);
        }
        let ino = self.next_ino.fetch_add(1, Ordering::Relaxed);
        let id = FileId::new(Self::NAME, ino);
        inner.by_name.insert(name.to_string(), id);
        inner.path_of.insert(id, path);
        Some(id)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DiskInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Filesystem for DiskFs {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn lookup(&self, name: &str) -> Option<FileId> {
        self.resolve(name)
    }

    fn open(&self, id: FileId, _access: Access) -> Result<Arc<dyn Resource>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::BackendShutdown(Self::NAME));
        }
        let path = self.lock().path_of.get(&id).cloned();
        let path: PathBuf = path.ok_or(Error::NoSuchFile(id))?;
        let mut file = File::open(&path).map_err(|e| Error::backend(Self::NAME, e.to_string()))?;
        let size = file
            .metadata()
            .map_err(|e| Error::backend(Self::NAME, e.to_string()))?
            .len();
        match self.residency {
            Residency::Streaming => Ok(Arc::new(StreamingResource {
                file: Arc::new(file),
                size,
                backend: Self::NAME,
                closed: self.closed.clone(),
            })),
            Residency::Resident => {
                let mut v = Vec::with_capacity(size as usize);
                file.read_to_end(&mut v)
                    .map_err(|e| Error::backend(Self::NAME, e.to_string()))?;
                Ok(Arc::new(ResidentResource {
                    bytes: Arc::from(v),
                    backend: Self::NAME,
                    closed: self.closed.clone(),
                }))
            }
        }
    }

    fn shutdown(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }
}

/// Convenience for the (small) set of callers that want to build a disk backend over a temp dir.
pub fn rooted_at(path: impl AsRef<Path>, residency: Residency) -> DiskFs {
    DiskFs::new(path.as_ref().to_path_buf(), residency)
}
