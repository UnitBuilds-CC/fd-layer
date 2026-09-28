//! Behavior each mapping to a documented invariant. These assert *semantics*, not happy paths.

mod support;

use std::sync::Arc;

use support::{corpus, drain, drain_pread, wired_process, READ};

use fd_layer::{Access, Error, OpenFileDescription, Process};

// Each test rebinds the 4-tuple; the mem id is index 1, slow id index 2, chunked handle index 3.
macro_rules! wired {
    ($proc:ident, $mem:ident, $slow:ident, $fs:ident) => {
        let ($proc, $mem, $slow, $fs) = wired_process();
    };
}

#[test]
fn descriptor_reuse_tracks_lowest_hole_across_interleaved_ops() {
    // Guards the `next_free` accelerator in `Table`: open/dup must keep returning the *lowest* free
    // fd even after closes create holes below the frontier and `open_at` creates holes above it.
    let mem = Arc::new(fd_layer::MemFs::new());
    let id = mem.add("d", corpus());
    let proc = Process::new(1);
    proc.register(mem);

    let a = proc.open(id, READ).unwrap();
    let b = proc.open(id, READ).unwrap();
    let c = proc.open(id, READ).unwrap();
    assert_eq!((a, b, c), (0, 1, 2), "contiguous fill from 0");

    // Close the middle slot; the next open must land on the lowest hole (1), not append at 3.
    proc.close(b).unwrap();
    assert_eq!(
        proc.open(id, READ).unwrap(),
        1,
        "lowest hole reused, not append"
    );

    // A pinned high slot leaves holes below the frontier; open still prefers the lowest free.
    proc.open_at(6, id, READ).unwrap();
    assert_eq!(
        proc.open(id, READ).unwrap(),
        3,
        "open ignores the pinned high slot and takes the lowest free (3)"
    );

    // Close two low slots, then dup: dup shares the OFD but is also allocated at the lowest hole.
    proc.close(a).unwrap();
    assert_eq!(
        proc.dup(c).unwrap(),
        0,
        "dup allocates the lowest free slot (0) like open"
    );
    assert_eq!(
        proc.open(id, READ).unwrap(),
        4,
        "after 0 and 3 and 6 taken, next lowest free is 4"
    );

    // A pinned slot must not poison allocation: with 0..=4 and 6 live, 5 is still the lowest free,
    // and releasing the pinned 6 reopens exactly 6. Exhaustion itself is owned by
    // `descriptor_table_fills_and_reports_full`.
    assert_eq!(proc.open(id, READ).unwrap(), 5);
    proc.close(6).unwrap();
    assert!(!proc.is_open(6));
    assert_eq!(
        proc.open(id, READ).unwrap(),
        6,
        "releases reopen the exact freed slot"
    );
}

#[test]
fn independent_opens_have_independent_offsets() {
    wired!(proc, mem, _slow, _fs);
    let x = proc.open(mem, READ).unwrap();
    let y = proc.open(mem, READ).unwrap();

    proc.read(x, &mut [0u8; 10]).unwrap();
    assert_eq!(proc.position(x).unwrap(), 10);
    assert_eq!(proc.position(y).unwrap(), 0, "other descriptor unaffected");

    // y was never touched, so it reads the whole file from 0; x continues from its own cursor.
    assert_eq!(
        drain(&proc, y),
        corpus(),
        "independent full read on the second descriptor"
    );
    assert_eq!(
        drain(&proc, x),
        corpus()[10..],
        "first descriptor resumes at its own cursor"
    );
}

#[test]
fn fork_shares_one_cursor_and_partitions_the_file() {
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    let child = proc.fork();
    assert!(
        child.is_open(fd),
        "child inherits the descriptor at the same slot"
    );

    child.read(fd, &mut [0u8; 7]).unwrap();
    assert_eq!(
        proc.position(fd).unwrap(),
        7,
        "parent sees the child's advance (same OFD)"
    );

    // Rewind the shared cursor to 0 (visible to both), then alternate readers across the two
    // processes; the shared cursor must yield the corpus once, in order — the exact property the
    // current WASIX layer violates.
    proc.seek(fd, 0).unwrap();
    let mut acc = Vec::new();
    let mut buf = [0u8; 6];
    let mut flip = false;
    loop {
        let who = if flip { &child } else { &proc };
        flip = !flip;
        let o = who.read(fd, &mut buf).unwrap();
        if o.end_of_file {
            break;
        }
        acc.extend_from_slice(&buf[..o.bytes_read]);
    }
    assert_eq!(acc, corpus());
}

#[test]
fn dup_shares_cursor_but_open_does_not() {
    wired!(proc, mem, _slow, _fs);
    let a = proc.open(mem, READ).unwrap();
    let d = proc.dup(a).unwrap();
    let ga: Arc<OpenFileDescription> = proc.get(a).unwrap();
    let gd = proc.get(d).unwrap();
    let go = proc.get(proc.open(mem, READ).unwrap()).unwrap();
    assert!(OpenFileDescription::is_same(&ga, &gd), "dup shares the OFD");
    assert!(
        !OpenFileDescription::is_same(&ga, &go),
        "a fresh open is a distinct OFD"
    );

    proc.read(a, &mut [0u8; 4]).unwrap();
    assert_eq!(proc.position(d).unwrap(), 4, "dup shares the cursor");
}

#[test]
fn pread_does_not_move_the_shared_cursor() {
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    proc.read(fd, &mut [0u8; 3]).unwrap();
    let before = proc.position(fd).unwrap();

    assert_eq!(proc.read_at(fd, 20, &mut [0u8; 6]).unwrap(), 6);
    assert_eq!(
        proc.position(fd).unwrap(),
        before,
        "pread left the cursor put"
    );

    assert_eq!(drain_pread(&proc, fd, corpus().len() as u64), corpus());
    assert_eq!(
        proc.position(fd).unwrap(),
        before,
        "bulk pread left the cursor put"
    );
}

#[test]
fn close_leaves_the_other_descriptor_intact() {
    wired!(proc, mem, _slow, _fs);
    let keep = proc.open(mem, READ).unwrap();
    let temp = proc.open(mem, READ).unwrap();
    proc.close(temp).unwrap();
    assert!(!proc.is_open(temp));
    assert_eq!(drain(&proc, keep), corpus());
}

#[test]
fn resource_frees_only_at_last_reference() {
    // Structural lifetime: closing the last fd drops the resource; an open one keeps it alive.
    wired!(proc, mem, _slow, _fs);
    let a = proc.open(mem, READ).unwrap();
    let b = proc.dup(a).unwrap();
    assert!(proc.close(a).unwrap());
    assert!(
        proc.is_open(b),
        "dup keeps the OFD alive after the original closes"
    );
    assert_eq!(
        drain(&proc, b),
        corpus(),
        "the surviving descriptor reads fully"
    );
}

#[test]
fn reading_past_end_is_eof_not_an_error() {
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    proc.seek(fd, corpus().len() as u64 + 5).unwrap();
    let o = proc.read(fd, &mut [0u8; 8]).unwrap();
    assert!(o.end_of_file);
    assert_eq!(o.bytes_read, 0);
}

#[test]
fn zero_length_read_is_not_eof() {
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    let o = proc.read(fd, &mut []).unwrap();
    assert_eq!(o.bytes_read, 0);
    assert!(!o.end_of_file, "a 0-byte request is not end-of-file");
    assert_eq!(proc.position(fd).unwrap(), 0);
}

#[test]
fn chunked_backend_forces_short_reads_then_completes() {
    wired!(proc, _mem, slow, _fs);
    let fd = proc.open(slow, READ).unwrap();
    let o = proc.read(fd, &mut [0u8; 64]).unwrap();
    assert!(
        o.bytes_read > 0 && o.bytes_read <= 8,
        "chunked read returns at most chunk_max bytes, got {}",
        o.bytes_read
    );
    // cursor is now at o.bytes_read; the remainder still reads fully
    assert_eq!(drain(&proc, fd), corpus()[o.bytes_read..]);
}

#[test]
fn failed_read_leaves_the_cursor_untouched() {
    // The atomicity invariant, against the fallible backend.
    wired!(proc, _mem, slow, fs);
    let fd = proc.open(slow, READ).unwrap();
    proc.read(fd, &mut [0u8; 3]).unwrap();
    let before = proc.position(fd).unwrap();

    fs.faults().arm_would_block();
    assert_eq!(proc.read(fd, &mut [0u8; 4]), Err(Error::WouldBlock));
    assert_eq!(
        proc.position(fd).unwrap(),
        before,
        "transient error moved nothing"
    );

    fs.faults().arm_error();
    assert!(matches!(
        proc.read(fd, &mut [0u8; 4]),
        Err(Error::Backend { .. })
    ));
    assert_eq!(
        proc.position(fd).unwrap(),
        before,
        "hard error moved nothing"
    );

    // and the data is still all there once faults clear
    assert_eq!(drain(&proc, fd), corpus()[before as usize..]);
}

#[test]
fn not_readable_when_opened_write_only() {
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, Access::WRITE).unwrap();
    assert_eq!(proc.read(fd, &mut [0u8; 4]), Err(Error::NotReadable));
    assert_eq!(proc.read_at(fd, 0, &mut [0u8; 4]), Err(Error::NotReadable));
}

#[test]
fn bad_descriptor_errors_carry_the_slot() {
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    proc.close(fd).unwrap();
    assert_eq!(proc.read(fd, &mut [0u8; 1]), Err(Error::BadDescriptor(fd)));
    assert_eq!(proc.position(fd), Err(Error::BadDescriptor(fd)));
}

#[test]
fn open_at_pins_the_slot_and_refuses_live_ones() {
    wired!(proc, mem, _slow, _fs);
    proc.open_at(7, mem, READ).unwrap();
    assert!(proc.is_open(7));
    assert_eq!(proc.open_at(7, mem, READ), Err(Error::SlotAlreadyInUse(7)));
    assert_eq!(
        proc.open(mem, READ).unwrap(),
        0,
        "auto-alloc skips the pinned slot"
    );
}

#[test]
fn descriptor_table_fills_and_reports_full() {
    use fd_layer::MemFs;
    let mem = Arc::new(MemFs::new());
    let id = mem.add("d", vec![1, 2, 3]);
    let proc = Process::with_max_fd(2, 3); // slots 0..=3 => 4 descriptors max
    proc.register(mem);
    for expected in 0..=3u32 {
        assert_eq!(proc.open(id, READ).unwrap(), expected);
    }
    assert_eq!(proc.open(id, READ), Err(Error::DescriptorTableFull));
}

#[test]
fn shutdown_backend_fails_further_reads_and_opens() {
    wired!(proc, mem, slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    assert!(proc.read(fd, &mut [0u8; 4]).unwrap().bytes_read > 0);
    proc.shutdown("mem").unwrap();

    assert!(matches!(
        proc.read(fd, &mut [0u8; 4]),
        Err(Error::BackendShutdown("mem"))
    ));
    assert!(matches!(
        proc.open(mem, READ),
        Err(Error::BackendShutdown("mem"))
    ));
    // the independent backend still works
    assert!(proc.open(slow, READ).is_ok());
}

#[test]
fn shutdown_stops_the_zero_copy_borrow_too() {
    // The zero-copy path (`read_shared`) MUST NOT leak stale bytes after shutdown: once closed,
    // `as_shared_bytes` returns `None`, so `read_shared` reports `Ok(None)` and the caller's
    // documented fallback (`read`) surfaces `BackendShutdown`. This is what makes the layer's "next
    // read returns BackendShutdown" hold on both paths.
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, READ).unwrap();
    assert!(
        proc.read_shared(fd, 4).unwrap().is_some(),
        "MemFs borrows while live"
    );
    proc.shutdown("mem").unwrap();
    assert!(
        proc.read_shared(fd, 4).unwrap().is_none(),
        "no stale borrow handed out after shutdown"
    );
    assert!(matches!(
        proc.read(fd, &mut [0u8; 4]),
        Err(Error::BackendShutdown("mem"))
    ));
}

#[test]
fn access_check_precedes_empty_buffer_shortcut() {
    // A write-only descriptor must report NotReadable even for a 0-length request — the permission
    // check cannot be silently skipped by the empty-buffer early return.
    wired!(proc, mem, _slow, _fs);
    let fd = proc.open(mem, Access::WRITE).unwrap();
    assert_eq!(proc.read(fd, &mut []), Err(Error::NotReadable));
    assert_eq!(proc.try_read(fd, &mut []), Err(Error::NotReadable));
    assert_eq!(proc.read_at(fd, 0, &mut []), Err(Error::NotReadable));
}

#[test]
fn fork_of_max_pid_wraps_without_overflow_panicking() {
    // Guards the `pid.wrapping_add(1)` in `fork` against the earlier `.max(pid + 1)` overflow-panic
    // in debug builds at the u32 ceiling.
    let mem = Arc::new(fd_layer::MemFs::new());
    let id = mem.add("d", vec![1u8; 3]);
    let proc = Process::new(u32::MAX);
    proc.register(mem);
    let child = proc.fork();
    assert_eq!(child.pid(), 0, "pid wraps without panicking");
    assert!(child.open(id, READ).is_ok());
}
