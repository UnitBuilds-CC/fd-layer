//! Backend-shape and lifetime edge cases — the properties that separate the two backends.

mod support;

use std::sync::Arc;

use support::{corpus, wired_process, READ};

use fd_layer::{ChunkedFs, Error, FileId, OpenFileDescription, Process, SlowConfig};

#[test]
fn in_memory_backend_offers_whole_file_view_chunked_does_not() {
    // `as_shared_bytes` is the *optional* fast path; the layer must work entirely without it.
    let (proc, mem, _slow, _fs) = wired_process();
    let fd = proc.open(mem, READ).unwrap();
    let ofd: Arc<OpenFileDescription> = proc.get(fd).unwrap();
    assert_eq!(ofd.as_shared_bytes().map(|b| b.to_vec()), Some(corpus()));
    assert_eq!(ofd.size(), Some(corpus().len() as u64));

    // The block/network backend cannot borrow the whole file — and the layer never asks it to.
    let slow = Arc::new(ChunkedFs::new(SlowConfig::default()));
    let id = slow.add_chunked("data", corpus(), 5);
    let p2 = Process::new(9);
    p2.register(slow);
    let fd = p2.open(id, READ).unwrap();
    assert!(p2.get(fd).unwrap().as_shared_bytes().is_none());
}

#[test]
fn chunked_read_never_returns_more_than_chunk_max() {
    // Covered, not sampled: several `chunk_max`/request-size pairs each drain the same file to a
    // real EOF. Without the completion assertions a backend that returned 0 bytes on the first call
    // would also pass the "never more than" half.
    let file = vec![7u8; 100];
    for (chunk_max, request) in [(1usize, 50usize), (3, 50), (7, 8), (100, 100), (200, 32)] {
        let slow = Arc::new(ChunkedFs::new(SlowConfig {
            chunk_max,
            latency: std::time::Duration::ZERO,
        }));
        let id = slow.add_chunked("d", file.clone(), 8);
        let proc = Process::new(3);
        proc.register(slow);
        let fd = proc.open(id, READ).unwrap();

        let mut drained = 0usize;
        let mut eof = false;
        for _ in 0..file.len() + 1 {
            let o = proc.read(fd, &mut vec![0u8; request]).unwrap();
            assert!(o.bytes_read <= chunk_max, "chunk_max {chunk_max} respected");
            drained += o.bytes_read;
            if o.end_of_file {
                eof = true;
                break;
            }
        }
        assert!(eof, "chunk_max {chunk_max} never reached EOF");
        assert_eq!(drained, file.len(), "chunk_max {chunk_max} lost bytes");
    }
}

#[test]
fn positioned_read_past_end_is_eof_on_both_backends() {
    let (proc, mem, slow, _fs) = wired_process();
    let n = corpus().len() as u64;
    let a = proc.open(mem, READ).unwrap();
    let b = proc.open(slow, READ).unwrap();
    assert_eq!(proc.read_at(a, n + 10, &mut [0u8; 4]).unwrap(), 0);
    assert_eq!(proc.read_at(b, n + 10, &mut [0u8; 4]).unwrap(), 0);
}

#[test]
fn fork_preserves_holes_and_slot_numbers() {
    // The table copy must be verbatim: same fd numbers, same holes. (Asserted nowhere upstream.)
    let (proc, mem, _slow, _fs) = wired_process();
    let a = proc.open(mem, READ).unwrap();
    let b = proc.open(mem, READ).unwrap();
    let c = proc.open(mem, READ).unwrap();
    proc.close(b).unwrap(); // leave a hole at slot b

    let child = proc.fork();
    assert!(child.is_open(a));
    assert!(!child.is_open(b), "hole is preserved");
    assert!(child.is_open(c));

    // Child allocations are independent of the parent's after fork.
    let cd = child.open(mem, READ).unwrap();
    assert_eq!(cd, b, "child reuses the hole");
    assert!(
        !proc.is_open(cd),
        "parent is unaffected by the child's open"
    );
}

#[test]
fn close_in_child_leaves_parent_descriptor_usable() {
    let (proc, mem, _slow, _fs) = wired_process();
    let fd = proc.open(mem, READ).unwrap();
    let child = proc.fork();
    assert!(child.close(fd).unwrap(), "child drops its reference");
    // parent still holds the only remaining reference and reads fine.
    assert_eq!(support::drain(&proc, fd), corpus());
}

#[test]
fn open_at_beyond_max_fd_is_rejected() {
    let (proc, mem, _slow, _fs) = wired_process();
    assert_eq!(
        proc.open_at(100_000, mem, READ),
        Err(Error::DescriptorTableFull)
    );
}

#[test]
fn unknown_backend_and_unminted_id_are_distinguished() {
    let (proc, _mem, _slow, _fs) = wired_process();
    // Registry routing is by name, and the layer resolves no paths, so an unregistered name simply
    // has no backend; `shutdown` is the operation that reports it as an error.
    assert!(proc.backend("nosuch").is_none());
    assert!(matches!(
        proc.shutdown("nosuch"),
        Err(Error::UnknownBackend(_))
    ));
    // A name the backend never advertised resolves to nothing *at the backend* — identity is minted
    // there, never derived here (I8).
    assert!(proc.backend("mem").unwrap().lookup("nope").is_none());
    // And an id no backend owns fails at `open`, carrying the caller's own id rather than a
    // synthesized stand-in.
    assert_eq!(
        proc.open(FileId::new("mem", 999), READ),
        Err(Error::NoSuchFile(FileId::new("mem", 999)))
    );
}

#[test]
fn stats_are_reported_never_asserted_on_timing() {
    // We exercise the fallible path a fixed number of times and read the counters. Timing fields
    // exist but are not asserted (CI-timing rule); correctness counters are.
    let slow = Arc::new(ChunkedFs::new(SlowConfig::default()));
    let id = slow.add_chunked("d", vec![1u8; 40], 6);
    let faults = slow.faults();
    let proc = Process::new(11);
    proc.register(slow.clone());
    let fd = proc.open(id, READ).unwrap();

    faults.arm_error();
    assert!(proc.read(fd, &mut [0u8; 4]).is_err());
    faults.arm_would_block();
    assert!(matches!(
        proc.read(fd, &mut [0u8; 4]),
        Err(Error::WouldBlock)
    ));

    let (reads, bytes, errors, would_blocks) = slow.stats().snapshot();
    assert_eq!(errors, 1);
    assert_eq!(would_blocks, 1);
    assert_eq!(
        reads, 0,
        "a faulted call is not a read: only fetches and past-end calls count"
    );
    assert_eq!(bytes, 0, "no bytes returned because both reads faulted");
}
