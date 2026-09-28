//! Three deliberately-different backends used to prove the abstraction holds.
//!
//! * [`mem::MemFs`] — everything resident in memory, reads are infallible, and it *can* hand out a
//!   whole-file `Arc<[u8]>` view. This is the "easy" backend the current layer assumes.
//! * [`chunked::ChunkedFs`] — a network/archive mock: the file is stored as disjoint blocks, a
//!   single `read_at` returns at most `chunk_max` bytes (forcing callers to loop for a full read),
//!   reads can be armed to fail transiently (`WouldBlock`) or hard (`Backend`), and it *cannot*
//!   provide a whole-file borrowed view (`as_shared_bytes` is `None`).
//! * [`disk::DiskFs`] — a **real file on disk**, so the abstraction is exercised against genuine OS
//!   I/O rather than canned data. Two modes: `Streaming` (positional `pread`, `as_shared_bytes`
//!   `None`, genuinely concurrent) and `Resident` (whole file loaded once, zero-copy capable).

pub mod chunked;
pub mod disk;
pub mod mem;

pub use chunked::{ChunkedFs, ChunkedStats, Faults};
pub use disk::{DiskFs, Residency};
pub use mem::MemFs;
