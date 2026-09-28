//! Capacity in the backend, correctness above it.
//!
//! These tests put a hard ceiling on how many fetches the network-shaped backend serves at once and
//! assert that the ceiling is respected *and* that the reads above it are still exactly right. The
//! point of the exercise: shedding load is only safe because the read position lives in the
//! descriptor layer, so a refused fetch leaves the caller's cursor where it was and the caller
//! retries from the same place. Nothing here asserts a duration — only counts and bytes.
//!
//! Where a test asserts that the budget *bit*, the backend is given a small per-fetch latency so
//! the threads really do collide; the latency is never itself asserted on.

mod support;

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use support::distinct_corpus;

use fd_layer::{Access, ChunkedFs, Error, Process, SlowConfig};

const READ: Access = Access::READ;

/// No latency: the fast case, and the control for the benchmark.
const IMMEDIATE: Duration = Duration::ZERO;
/// Enough per-fetch hold time that two threads at one budget collide almost certainly.
const HOLD: Duration = Duration::from_micros(200);

/// Generous iteration ceilings: a livelock must fail the test rather than hang the suite.
const SPIN_BOUND: usize = 200_000;

fn chunked(chunk_max: usize, latency: Duration) -> Arc<ChunkedFs> {
    Arc::new(ChunkedFs::new(SlowConfig { chunk_max, latency }))
}

/// Drain one descriptor through the sequential cursor, retrying capacity refusals.
fn drain_retrying(proc: &Process, fd: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 16];
    for _ in 0..SPIN_BOUND {
        match proc.read(fd, &mut buf) {
            Ok(o) if o.end_of_file => return out,
            Ok(o) => out.extend_from_slice(&buf[..o.bytes_read]),
            Err(Error::WouldBlock) => thread::yield_now(),
            Err(e) => panic!("unexpected read error: {e:?}"),
        }
    }
    panic!("sequential drain did not terminate within {SPIN_BOUND} iterations");
}

/// Drain one descriptor with positioned reads only, retrying capacity refusals.
fn pread_retrying(proc: &Process, fd: u32, size: u64) -> Vec<u8> {
    let mut out = vec![0u8; size as usize];
    let mut off = 0u64;
    for _ in 0..SPIN_BOUND {
        if off == size {
            return out;
        }
        match proc.read_at(fd, off, &mut out[off as usize..]) {
            Ok(0) => return out,
            Ok(n) => off += n as u64,
            Err(Error::WouldBlock) => thread::yield_now(),
            Err(e) => panic!("unexpected pread error: {e:?}"),
        }
    }
    panic!("positioned drain did not terminate within {SPIN_BOUND} iterations");
}

#[test]
fn budget_is_respected_while_independent_readers_all_fetch() {
    // Eight separate opens = eight cursors, so eight threads genuinely reach the backend at once.
    // The budget is the only thing limiting them now, and every reader must still see the whole
    // file, in order, byte for byte.
    let data = distinct_corpus();
    let slow = chunked(8, HOLD);
    let id = slow.add_chunked("data", data.clone(), 5);
    slow.set_connection_budget(2);
    let proc = Arc::new(Process::new(1));
    proc.register(slow.clone());

    let fds: Vec<u32> = (0..8).map(|_| proc.open(id, READ).unwrap()).collect();
    slow.reset_connection_stats();

    let mut handles = Vec::new();
    for fd in fds {
        let who = proc.clone();
        handles.push(thread::spawn(move || drain_retrying(&who, fd)));
    }
    let results: Vec<Vec<u8>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    for (i, got) in results.iter().enumerate() {
        assert_eq!(got, &data, "reader {i} saw a corrupted file under load");
    }
    assert!(
        slow.peak_connections() <= 2,
        "backend served {} fetches at once with a budget of 2",
        slow.peak_connections()
    );
    assert!(
        slow.connection_denials() > 0,
        "the budget never bit, so this test proved nothing"
    );
    assert_eq!(slow.live_connections(), 0, "a connection leaked");
}

#[test]
fn a_shared_cursor_is_a_tighter_bound_than_the_budget() {
    // Parent and forked children hold ONE open file description, so they serialize on the cursor
    // before they ever reach the backend: the budget is never the constraint here. If the peak ever
    // exceeded 1, the cursor would no longer be spanning the fetch — that is invariant I2, asserted
    // from the backend side.
    let data = distinct_corpus();
    let slow = chunked(8, IMMEDIATE);
    let id = slow.add_chunked("data", data.clone(), 5);
    slow.set_connection_budget(1);
    let proc = Arc::new(Process::new(1));
    proc.register(slow.clone());

    let fd = proc.open(id, READ).unwrap();
    let mut sharers: Vec<Arc<Process>> = vec![proc.clone()];
    for _ in 1..8 {
        sharers.push(Arc::new(proc.fork()));
    }
    slow.reset_connection_stats();

    let mut handles = Vec::new();
    for who in sharers {
        handles.push(thread::spawn(move || drain_retrying(&who, fd)));
    }
    let mut combined: Vec<u8> = Vec::new();
    for h in handles {
        combined.extend_from_slice(&h.join().unwrap());
    }
    let mut expected = data.clone();
    expected.sort_unstable();
    combined.sort_unstable();
    assert_eq!(
        combined, expected,
        "bytes duplicated or dropped through one shared cursor"
    );
    assert_eq!(
        slow.peak_connections(),
        1,
        "the cursor must serialize fetches on one OFD"
    );
    assert_eq!(slow.live_connections(), 0, "a connection leaked");
}

#[test]
fn positioned_and_sequential_readers_share_one_budget_without_deadlocking() {
    // `read_at` takes no cursor lock, so it can hold a connection while a sequential reader holds
    // the cursor and is waiting for a connection. Acquisition order is cursor -> connection and
    // never the reverse, which is what makes that harmless rather than deadlock-shaped: both readers
    // run to completion at a budget of 1.
    let data = distinct_corpus();
    let size = data.len() as u64;
    let slow = chunked(8, HOLD);
    let id = slow.add_chunked("data", data.clone(), 5);
    slow.set_connection_budget(1);
    let proc = Arc::new(Process::new(1));
    proc.register(slow.clone());

    let seq = proc.open(id, READ).unwrap();
    let pos = proc.open(id, READ).unwrap();
    slow.reset_connection_stats();

    let p1 = proc.clone();
    let sequencer = thread::spawn(move || drain_retrying(&p1, seq));
    let p2 = proc.clone();
    let positioner = thread::spawn(move || pread_retrying(&p2, pos, size));

    assert_eq!(
        sequencer.join().unwrap(),
        data,
        "sequential reader lost bytes"
    );
    assert_eq!(positioner.join().unwrap(), data, "pread reader lost bytes");
    assert!(
        slow.connection_denials() > 0,
        "one of the two readers never waited, so the ordering was not exercised"
    );
    assert!(
        slow.peak_connections() <= 1,
        "budget of 1 was exceeded: {}",
        slow.peak_connections()
    );
    assert_eq!(slow.live_connections(), 0, "a connection leaked");
}

#[test]
fn default_budget_is_unlimited_and_never_refuses() {
    // The control for the benchmark: the pool is on every fetch path by default, and at an
    // unlimited budget it never denies — so every pre-existing test and backend is unchanged.
    let data = distinct_corpus();
    let slow = chunked(8, IMMEDIATE);
    let id = slow.add_chunked("data", data.clone(), 5);
    let proc = Arc::new(Process::new(1));
    proc.register(slow.clone());

    let mut handles = Vec::new();
    for _ in 0..8 {
        let who = proc.clone();
        let fd = proc.open(id, READ).unwrap();
        handles.push(thread::spawn(move || drain_retrying(&who, fd)));
    }
    for h in handles {
        assert_eq!(h.join().unwrap(), data);
    }
    assert_eq!(slow.connection_denials(), 0, "unlimited budget refused");
    assert_eq!(slow.live_connections(), 0);
}
