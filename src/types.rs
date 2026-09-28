//! Core value types shared across the layer: identifiers, access modes, errors.

use std::fmt;

/// A raw file descriptor: an index into a [`Process`](crate::process::Process) descriptor
/// table. Deliberately a thin `u32` (matching the WASIX `type Fd = u32` journal contract) so
/// that a slot number stays stable across fork/dup and can be pinned by a caller that needs a
/// specific value.
pub type RawFd = u32;

/// Highest descriptor value a default table will hand out. The table grows to this bound and
/// then reports [`Error::DescriptorTableFull`] rather than looping forever.
pub const MAX_FD: RawFd = 64 * 1024 - 1;

/// A backend-minted, stable identity for a file. The descriptor layer never derives this from a
/// path — it only carries it — so inode identity is entirely the backend's to define (path hash,
/// archive entry index, remote object key, inode number, …). `backend` disambiguates two
/// backends that might otherwise mint the same `ino`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId {
    pub backend: &'static str,
    pub ino: u64,
}

impl FileId {
    pub fn new(backend: &'static str, ino: u64) -> Self {
        Self { backend, ino }
    }
}

impl fmt::Display for FileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend, self.ino)
    }
}

/// Requested access when opening. The prototype models read/write only; `write` is accepted as
/// an *access* flag (so `Access::read_write()` opens successfully) but no backend actually
/// mutates data — see the DESIGN.md note on how writes evolve from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Access {
    pub read: bool,
    pub write: bool,
}

impl Access {
    pub const READ: Access = Access {
        read: true,
        write: false,
    };
    pub const WRITE: Access = Access {
        read: false,
        write: true,
    };
    pub const READ_WRITE: Access = Access {
        read: true,
        write: true,
    };

    pub fn read() -> Self {
        Self::READ
    }
    pub fn read_write() -> Self {
        Self::READ_WRITE
    }
}

/// The result of a successful read: how many bytes were placed in the caller's buffer, and
/// whether the read stopped because the end of the resource was reached.
///
/// `end_of_file` disambiguates the one thing a bare `usize` cannot: a zero-length read at the
/// end versus a zero-length read requested by the caller. In this layer `end_of_file` is set iff
/// the backend reported the offset at or beyond the resource's size for a non-empty request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOutcome {
    pub bytes_read: usize,
    pub end_of_file: bool,
}

impl ReadOutcome {
    pub fn data(bytes_read: usize) -> Self {
        Self {
            bytes_read,
            end_of_file: false,
        }
    }
    pub fn eof() -> Self {
        Self {
            bytes_read: 0,
            end_of_file: true,
        }
    }
}

/// Layer errors. Kept as one enum so callers can match on the semantic condition rather than a
/// backend-specific error type; the backend's own failure detail is carried in [`Error::Backend`]
/// as an opaque string (a real port would carry a typed source here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The file identity does not exist in the target backend.
    NoSuchFile(FileId),
    /// No descriptor is currently open at this slot.
    BadDescriptor(RawFd),
    /// The descriptor table reached [`MAX_FD`].
    DescriptorTableFull,
    /// `open_at`/`dup2`-style pinned placement refused: the requested slot is already occupied
    /// and the operation does not silently replace it.
    SlotAlreadyInUse(RawFd),
    /// The descriptor's open file description was not granted read access.
    NotReadable,
    /// No backend is registered under this name.
    UnknownBackend(String),
    /// The whole file system (backend) has been shut down; open descriptors now read as closed.
    BackendShutdown(&'static str),
    /// Non-blocking operation could not proceed right now (a contending reader holds the shared
    /// cursor, or the backend momentarily has no data). Corresponds to `EAGAIN`/`EWOULDBLOCK`.
    WouldBlock,
    /// A backend reported a failure reading. The resource is left unchanged.
    Backend {
        backend: &'static str,
        message: String,
    },
}

impl Error {
    pub fn backend(backend: &'static str, message: impl Into<String>) -> Self {
        Error::Backend {
            backend,
            message: message.into(),
        }
    }

    /// Whether this error is a transient "try again" rather than a hard failure.
    pub fn is_transient(&self) -> bool {
        matches!(self, Error::WouldBlock)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoSuchFile(id) => write!(f, "no such file: {id}"),
            Error::BadDescriptor(fd) => write!(f, "bad file descriptor: {fd}"),
            Error::DescriptorTableFull => write!(f, "descriptor table full"),
            Error::SlotAlreadyInUse(fd) => write!(f, "descriptor slot {fd} already in use"),
            Error::NotReadable => write!(f, "descriptor is not open for reading"),
            Error::UnknownBackend(b) => write!(f, "unknown backend: {b}"),
            Error::BackendShutdown(b) => write!(f, "backend '{b}' is shut down"),
            Error::WouldBlock => write!(f, "operation would block"),
            Error::Backend { backend, message } => {
                write!(f, "backend '{backend}' error: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;
