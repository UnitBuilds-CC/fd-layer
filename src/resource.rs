//! The backend abstraction: a [`Resource`] you can read at an offset, and a [`Filesystem`] that
//! mints [`FileId`]s and opens them.

use std::sync::Arc;

use crate::types::{Access, FileId, Result};

/// An opened, readable resource. This is the *only* thing the descriptor layer asks a backend
/// for, and its shape is the whole point of the design:
///
/// * Every method takes `&self`. There is **no global mutable cursor** on the resource, so a
///   resource may be read from many positions concurrently, and a network/archive backend can
///   satisfy a read as a stateless range request rather than seeking a shared handle.
/// * Reads go **through a caller-supplied buffer** (`&mut [u8]`). A backend is never required to
///   hand out a borrowed slice of the whole file — only [`Resource::as_shared_bytes`] *may* offer
///   an owned view, and it defaults to `None`.
///
/// The trait is object-safe and requires no downcasting to use.
pub trait Resource: Send + Sync + std::fmt::Debug + 'static {
    /// Total size in bytes, when the backend can state it cheaply. `None` models a stream whose
    /// length is unknown until read; EOF is then detected purely by a read returning `0`.
    fn size(&self) -> Option<u64>;

    /// Read up to `buf.len()` bytes starting at `offset`.
    ///
    /// Returns the number of bytes written into `buf`. Contracts the whole layer relies on:
    /// * **`Ok(0)` means end of file.** A backend that transiently has no data yet must return
    ///   [`crate::Error::WouldBlock`], never `Ok(0)` — otherwise a caller cannot tell "empty now" from
    ///   "finished".
    /// * A read fully past the end returns `Ok(0)`, never an error and never a panic.
    /// * **`Err(..)` leaves no partial progress**: the offset the layer will advance from is
    ///   decided *after* this call returns `Ok`, so a failed read cannot corrupt the cursor.
    /// * Short reads (`Ok(n)` with `n < buf.len()`) are legal and common for range/network
    ///   backends; the caller decides whether to continue.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize>;

    /// Optional fast path: an owned, refcounted view of the whole resource when it already lives
    /// in memory. Mirrors `virtual_fs::VirtualFile::as_owned_buffer` — return `Some` only when the
    /// bytes are already resident and cloning is cheap (an `Arc` bump), never by copying the file.
    /// Backends that cannot (streams, archives, remote objects) leave the default `None`.
    fn as_shared_bytes(&self) -> Option<Arc<[u8]>> {
        None
    }
}

/// A file system backend: it owns file identity and opens resources. Kept intentionally small —
/// the descriptor layer does path resolution, caching, and offset management *above* this, so a
/// backend only has to answer "what is the id for this name" and "give me a resource for this id".
pub trait Filesystem: Send + Sync + std::fmt::Debug + 'static {
    /// Stable short name, used in [`FileId`] and diagnostics.
    fn name(&self) -> &'static str;

    /// Resolve a name to a backend-minted identity, if it exists. Identity derivation (path hash,
    /// inode number, archive index) is the backend's private concern.
    fn lookup(&self, name: &str) -> Option<FileId>;

    /// Open the resource behind `id` with the requested access. Each call returns an independent
    /// resource handle — the layer never assumes two opens share mutable state (and never assumes
    /// they don't; sharing is expressed by the backend returning resources backed by the same
    /// `Arc`).
    fn open(&self, id: FileId, access: Access) -> Result<Arc<dyn Resource>>;

    /// Close the whole file system: after this, open resources report [`crate::Error::BackendShutdown`]
    /// from reads and further opens fail. The descriptor layer keeps no refcount of its own, so
    /// this is a single hook per backend rather than a cross-layer teardown dance.
    fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}
