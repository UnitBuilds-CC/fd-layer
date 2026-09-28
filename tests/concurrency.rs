//! Concurrency correctness. The flagship test is the one the current WASIX layer fails: two
//! threads sharing one open file description must not duplicate or skip bytes.

mod support;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use support::{corpus, distinct_corpus, wired_distinct, wired_process, READ};

use fd_layer::{ChunkedFs, Error, Filesystem, Process, SlowConfig};

#[test]
fn shared_offset_fork_read_partitions_file_exactly() {
    // THE most valuable test. parent + forked child hold the SAME OFD at the same slot. Two OS
    // threads read through it concurrently. Because the cursor, the fetch, and the advance are one
    // critical section, every byte is returned exactly once (as a multiset), even though the order
    // across threads is nondeterministic. The current layer's split lock/offset race is what this
    // catches: a buggy implementation duplicates one byte and skips another. The corpus has all
    // distinct bytes, so multiset equality is an exact positional statement.
    let data = distinct_corpus();
    let (proc, mem, _slow, _sid) = wired_distinct();
    let fd = proc.open(mem, READ).unwrap();
    let child = Arc::new(proc.fork());
    let parent = Arc::new(proc);

    let mut handles = Vec::new();
    for who in [parent.clone(), child.clone()] {
        let n = data.len();
        handles.push(thread::spawn(move || {
            let mut mine = Vec::new();
            let mut buf = [0u8; 5];
            let mut spins = 0;
            loop {
                spins += 1;
                assert!(spins < n * 4 + 1000, "read loop failed to terminate");
                match who.read(fd, &mut buf) {
                    Ok(o) if o.end_of_file => break,
                    Ok(o) => mine.extend_from_slice(&buf[..o.bytes_read]),
                    Err(Error::WouldBlock) => continue,
                    Err(e) => panic!("unexpected read error: {e:?}"),
                }
            }
            mine
        }));
    }

    let a = handles.remove(0).join().unwrap();
    let b = handles.remove(0).join().unwrap();

    let mut combined: Vec<u8> = Vec::new();
    combined.extend_from_slice(&a);
    combined.extend_from_slice(&b);
    combined.sort_unstable();
    let mut expected = data.clone();
    expected.sort_unstable();
    assert_eq!(
        combined, expected,
        "no byte duplicated or dropped across the shared cursor"
    );
}

#[test]
fn independent_opens_drain_concurrently_without_touching_either_cursor() {
    // Two separate opens = two OFDs = two cursors (the fix vs the old per-inode host handle, which
    // put every reader of a file behind one write lock). The barrier releases both drains at the
    // same instant, so the two full reads genuinely overlap; each must return the complete file and
    // neither sequential cursor may move, since both drain with pread. What this does *not* assert
    // is that they made progress at the same wall-clock instant — the 1→8 thread cost is measured
    // in `examples/bench.rs` and reported, never asserted (no wall-clock gates in the suite).
    let data = corpus();
    let (proc, mem, _slow, _fs) = wired_process();
    let x = proc.open(mem, READ).unwrap();
    let y = proc.open(mem, READ).unwrap();
    let proc = Arc::new(proc);
    let size = data.len() as u64;
    let start = Arc::new(Barrier::new(2));

    let p1 = proc.clone();
    let b1 = start.clone();
    let h1 = thread::spawn(move || {
        b1.wait();
        drain_pread_owned(&p1, x, size)
    });
    let p2 = proc.clone();
    let b2 = start.clone();
    let h2 = thread::spawn(move || {
        b2.wait();
        drain_pread_owned(&p2, y, size)
    });

    assert_eq!(h1.join().unwrap(), data);
    assert_eq!(h2.join().unwrap(), data);
    // Crucially: independent sequential cursors never advanced (we used pread), proving reads on
    // distinct descriptors don't share a cursor.
    assert_eq!(proc.position(x).unwrap(), 0);
    assert_eq!(proc.position(y).unwrap(), 0);
}

#[test]
fn concurrent_pread_on_one_descriptor_never_moves_the_cursor() {
    // pread takes no cursor lock, so many threads can position-read one OFD at once; the shared
    // cursor must stay put and every read must be correct at its requested offset. The barrier is
    // what makes this a concurrency test rather than eight sequential ones: all eight threads are
    // released into their `read_at` together, and each checks its own 16-byte window afterwards.
    let data = corpus();
    let (proc, mem, _slow, _fs) = wired_process();
    let fd = proc.open(mem, READ).unwrap();
    let proc = Arc::new(proc);
    let size = data.len() as u64;
    let start = Arc::new(Barrier::new(8));

    let mut handles = Vec::new();
    for i in 0..8u64 {
        let p = proc.clone();
        let gate = start.clone();
        handles.push(thread::spawn(move || {
            gate.wait();
            let off = (i * 7) % size;
            let mut buf = vec![0u8; 16];
            let n = p.read_at(fd, off, &mut buf).unwrap();
            (off, buf[..n].to_vec())
        }));
    }
    for h in handles {
        let (off, got) = h.join().unwrap();
        assert_eq!(got, data[off as usize..][..got.len()]);
    }
    assert_eq!(
        proc.position(fd).unwrap(),
        0,
        "concurrent preads left the cursor untouched"
    );
}

#[test]
fn concurrent_try_read_on_shared_ofd_retries_then_completes() {
    // try_read surfaces WouldBlock instead of blocking; a set of threads sharing one OFD, each
    // retrying on WouldBlock, must still partition the (distinct-byte) file exactly.
    let data = distinct_corpus();
    let (proc, _mem, _slow, slow_id) = wired_distinct();
    let fd = proc.open(slow_id, READ).unwrap();
    let child = Arc::new(proc.fork());
    let parent = Arc::new(proc);

    let mut handles = Vec::new();
    for who in [parent, child] {
        let n = data.len();
        handles.push(thread::spawn(move || {
            let mut mine = Vec::new();
            let mut buf = [0u8; 5];
            let mut guard = 0;
            loop {
                guard += 1;
                assert!(guard < n * 8 + 1000, "try_read loop failed to terminate");
                match who.try_read(fd, &mut buf) {
                    Ok(o) if o.end_of_file => break,
                    Ok(o) => mine.extend_from_slice(&buf[..o.bytes_read]),
                    Err(Error::WouldBlock) => thread::yield_now(),
                    Err(e) => panic!("unexpected: {e:?}"),
                }
            }
            mine
        }));
    }
    let mut combined = handles.remove(0).join().unwrap();
    combined.extend(handles.remove(0).join().unwrap());
    combined.sort_unstable();
    let mut expected = data;
    expected.sort_unstable();
    assert_eq!(combined, expected);
}

#[test]
fn shutdown_does_not_wait_for_an_in_flight_reader() {
    // The name is the property: a read parked *inside* its critical section — holding the cursor lock
    // with bytes already fetched — must not make teardown block, and its bytes must not be lost. The
    // read-pause seam (used by I2's deterministic test below) fixes the "in flight" instant, so this
    // does not depend on how the scheduler places two threads.
    let slow = Arc::new(ChunkedFs::new(SlowConfig {
        chunk_max: 1024,
        latency: std::time::Duration::ZERO,
    }));
    let id = slow.add_chunked("big", corpus(), 16);
    let proc = Process::new(5);
    proc.register(slow.clone());
    let fd = proc.open(id, READ).unwrap();
    let ofd = proc.get(fd).unwrap();
    ofd.arm_read_pause();

    let reader = {
        let ofd = ofd.clone();
        thread::spawn(move || ofd.read(&mut [0u8; 4]))
    };
    // Bounded wait, so a broken seam reports a failure instead of hanging the suite.
    for _ in 0..5_000_000 {
        if ofd.read_pause_reached() {
            break;
        }
        thread::yield_now();
    }
    assert!(
        ofd.read_pause_reached(),
        "the armed read never parked mid-fetch"
    );

    // A reader holds the cursor lock and an `Arc<Resource>`: teardown must still return. Run it on its
    // own thread and wait *bounded*, so a shutdown that blocks behind the parked reader fails here
    // instead of hanging the suite.
    let fs = slow.clone();
    let done = Arc::new(AtomicBool::new(false));
    let signal = done.clone();
    let teardown = thread::spawn(move || {
        let result = fs.shutdown();
        signal.store(true, Ordering::SeqCst);
        result
    });
    for _ in 0..5_000_000 {
        if done.load(Ordering::SeqCst) {
            break;
        }
        thread::yield_now();
    }
    assert!(
        done.load(Ordering::SeqCst),
        "shutdown blocked behind the in-flight reader's cursor lock"
    );
    teardown.join().unwrap().unwrap();

    // Released, the in-flight read commits and delivers its bytes rather than being dropped.
    ofd.release_read_pause();
    let outcome = reader
        .join()
        .unwrap()
        .expect("the in-flight read completed");
    assert_eq!(
        outcome.bytes_read, 4,
        "in-flight bytes are delivered, not lost"
    );

    // The descriptor outlived the backend and says so.
    assert!(matches!(
        proc.read(fd, &mut [0u8; 4]),
        Err(Error::BackendShutdown("chunked"))
    ));
}

#[test]
fn cursor_lock_is_atomic_with_the_fetch_deterministically() {
    // Invariant I2 (DESIGN.md §2), as a deterministic version of the flagship property. Freeze a
    // sequential read at the one dangerous instant — inside the cursor critical section, after
    // `read_at` returned bytes but before the offset advances — and prove from a second thread
    // that (a) a competing `try_read` fails at the lock with `WouldBlock` and *never reaches the
    // backend* (fetch count stays 1), and (b) once released the cursor advances by exactly one
    // fetch's worth — no double-skip, no lost advance. This needs no thread-interleaving luck, so
    // it can't pass vacuously.
    let (proc, _mem, slow_id, fs) = wired_process();
    let fd = proc.open(slow_id, READ).unwrap();
    let ofd = proc.get(fd).unwrap();
    let stats = fs.stats();

    ofd.arm_read_pause();
    let reader = {
        let ofd = ofd.clone();
        thread::spawn(move || ofd.read(&mut [0u8; 64]))
    };

    // Wait (bounded, so a real failure reports instead of hanging) for the read to park at the
    // freeze point while it still holds the cursor lock.
    for _ in 0..5_000_000 {
        if ofd.read_pause_reached() {
            break;
        }
        thread::yield_now();
    }
    assert!(
        ofd.read_pause_reached(),
        "the armed read never parked at the freeze point"
    );

    // The fetch has happened (chunk_max = 8 in `wired_process`), so exactly one backend read is
    // recorded even though the cursor is still at 0 (not yet advanced).
    assert_eq!(
        stats.snapshot().0,
        1,
        "exactly one fetch completed pre-advance"
    );

    // A competing non-blocking read must be refused AT THE LOCK, not fetch again.
    assert_eq!(
        ofd.try_read(&mut [0u8; 4]),
        Err(Error::WouldBlock),
        "contended cursor surfaces as retry, not a second fetch"
    );
    assert_eq!(
        stats.snapshot().0,
        1,
        "the refused read never touched the backend"
    );

    // Release: the parked read commits its advance and returns.
    ofd.release_read_pause();
    let outcome = reader.join().unwrap().expect("read ok");
    assert_eq!(outcome.bytes_read, 8, "one chunk fetched");
    assert_eq!(
        proc.position(fd).unwrap(),
        8,
        "cursor advanced by exactly one fetch's worth"
    );
    assert_eq!(
        stats.snapshot().0,
        1,
        "still one backend fetch for one committed advance"
    );
}

fn drain_pread_owned(proc: &Process, fd: u32, size: u64) -> Vec<u8> {
    let mut out = vec![0u8; size as usize];
    let mut off = 0u64;
    while off < size {
        let n = proc.read_at(fd, off, &mut out[off as usize..]).unwrap();
        if n == 0 {
            break;
        }
        off += n as u64;
    }
    out.truncate(off as usize);
    out
}
