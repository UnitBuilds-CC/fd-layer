//! # fd-layer
//!
//! A standalone, stable-Rust prototype of an abstract **open-file-description (OFD)** layer.
//!
//! It models the three-object POSIX file-access stack that the existing WASIX file system collapses
//! onto two objects, and it is built so that the *shape* of each object maps onto a real integration
//! seam the current layer uses:
//!
//! ```text
//! Process (fd table)  →  RawFd slot  →  Arc<OpenFileDescription>  →  Arc<dyn Resource>  →  backend
//!        fork/dup shares the OFD           owns the shared cursor        &self, read_at
//! ```
//!
//! * [`process::Process`] owns a descriptor table mapping [`types::RawFd`] → `Arc<`[`ofd::OpenFileDescription`]`>`.
//! * [`ofd::OpenFileDescription`] is the object that owns **one shared cursor** and one access mode.
//!   Two `open()`s create two OFDs (independent cursors); `fork`/`dup` share one OFD (shared cursor).
//! * [`resource::Resource`] is the backend read interface: `&self`-only, read-at-offset, and never
//!   required to borrow the whole file — so a slow/fallible/range backend fits without pretending to
//!   be seekable.
//! * [`resource::Filesystem`] mints stable [`types::FileId`]s and opens resources; the layer never
//!   derives file identity from a path.
//!
//! ## Lifetime is structural, not counted
//!
//! There is no manual "open handle" reference counter anywhere. An OFD lives as long as an
//! `Arc<OpenFileDescription>` is reachable from a descriptor slot or an in-flight read; a resource
//! lives as long as its OFD. `close` simply clears a slot — dropping one `Arc`. This is the fix for
//! the duplicated/dropped-handle refcounts in the layer being replaced.
//!
//! ## No downcasting
//!
//! Callers use backends purely through the two object-safe traits; nothing in the public API
//! requires `Any`/`downcast_ref`.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod backends;
pub mod ofd;
pub mod process;
pub mod resource;
pub mod types;

pub use backends::chunked::SlowConfig;
pub use backends::disk::Residency;
pub use backends::{ChunkedFs, ChunkedStats, DiskFs, Faults, MemFs};
pub use ofd::{OpenFileDescription, SharedRead};
pub use process::Process;
pub use resource::{Filesystem, Resource};
pub use types::{Access, Error, FileId, RawFd, ReadOutcome, Result, MAX_FD};
