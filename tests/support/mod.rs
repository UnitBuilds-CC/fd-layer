//! Shared helpers for the integration tests (not a test binary itself).
//!
//! Compiled into every test binary; each binary uses only a subset of these helpers, so dead-code
//! warnings here are expected and suppressed.
#![allow(dead_code)]

use std::sync::Arc;

use fd_layer::{Access, ChunkedFs, MemFs, Process, SlowConfig};

/// A recognizable, non-trivially-sized corpus used across tests.
pub fn corpus() -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog, 0123456789, then again and again \
        until the buffer is clearly larger than one backend chunk."
        .to_vec()
}

/// A corpus whose bytes are all distinct (0..N). With unique byte values, "the multiset of read
/// bytes equals the file" is an *exact* positional statement — no duplicate/skip can hide behind
/// two equal values. Used by the flagship shared-cursor test.
pub fn distinct_corpus() -> Vec<u8> {
    (0u8..=200).collect()
}

/// A process with both backends registered and `corpus()` present in each as file `"data"`.
#[allow(clippy::type_complexity)]
pub fn wired_process() -> (Process, fd_layer::FileId, fd_layer::FileId, Arc<ChunkedFs>) {
    wire_with(corpus())
}

/// A process holding only a distinct-byte mem file, for exact multiset assertions.
pub fn wired_distinct() -> (Process, fd_layer::FileId, Arc<ChunkedFs>, fd_layer::FileId) {
    let data = distinct_corpus();
    let mem = Arc::new(MemFs::new());
    let mem_id = mem.add("data", data.clone());
    let slow = Arc::new(ChunkedFs::new(SlowConfig {
        chunk_max: 8,
        latency: std::time::Duration::ZERO,
    }));
    let slow_id = slow.add_chunked("data", data, 5);
    let proc = Process::new(1);
    proc.register(mem);
    proc.register(slow.clone());
    (proc, mem_id, slow, slow_id)
}

#[allow(clippy::type_complexity)]
fn wire_with(data: Vec<u8>) -> (Process, fd_layer::FileId, fd_layer::FileId, Arc<ChunkedFs>) {
    let mem = Arc::new(MemFs::new());
    let mem_id = mem.add("data", data.clone());

    let slow = Arc::new(ChunkedFs::new(SlowConfig {
        chunk_max: 8,
        latency: std::time::Duration::ZERO,
    }));
    let slow_id = slow.add_chunked("data", data.clone(), 5);

    let proc = Process::new(1);
    proc.register(mem);
    proc.register(slow.clone());
    (proc, mem_id, slow_id, slow)
}

/// Read a descriptor to EOF through the sequential cursor, looping over short reads. Returns the
/// bytes actually observed. Bounded to avoid an infinite loop if the cursor ever stalls.
pub fn drain(proc: &Process, fd: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 16];
    for _ in 0..10_000 {
        let outcome = proc.read(fd, &mut buf).expect("read ok");
        if outcome.end_of_file {
            break;
        }
        out.extend_from_slice(&buf[..outcome.bytes_read]);
    }
    out
}

/// Read a whole descriptor via positioned reads without disturbing the shared cursor.
pub fn drain_pread(proc: &Process, fd: u32, size: u64) -> Vec<u8> {
    let mut out = vec![0u8; size as usize];
    let mut off = 0u64;
    while off < size {
        let n = proc
            .read_at(fd, off, &mut out[off as usize..])
            .expect("pread ok");
        if n == 0 {
            break;
        }
        off += n as u64;
    }
    out.truncate(off as usize);
    out
}

pub const READ: Access = Access::READ;
