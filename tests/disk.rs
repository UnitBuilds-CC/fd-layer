//! A **real file on disk** through the full descriptor stack.
//!
//! These tests write genuine bytes to a temp file and read them back through `DiskFs`, proving the
//! abstraction works against actual OS file I/O (not canned data), that the zero-copy path is
//! byte-identical to the copy path, and that positional reads on a real `File` are concurrent.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use fd_layer::{Access, DiskFs, Error, Filesystem, Process, Residency};

/// Deterministic, non-trivial content: a varying byte pattern so a wrong offset or a dropped/
/// duplicated byte is caught (not all-equal filler).
fn real_content(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 251) as u8 ^ ((i / 7) % 253) as u8)
        .collect()
}

/// A temp directory that removes itself on drop, so a failed assert can't litter the disk.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "fdlayer-disk-{}-{tag}-{}",
            std::process::id(),
            nanos()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
    fn path(&self) -> &PathBuf {
        &self.0
    }
    fn write(&self, name: &str, bytes: &[u8]) {
        fs::write(self.0.join(name), bytes).expect("write real file");
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---- streaming (real positional reads) ---------------------------------------

#[test]
fn disk_streaming_reads_real_file_byte_exact() {
    let tmp = TempDir::new("stream");
    let data = real_content(100_000);
    tmp.write("file.bin", &data);

    let fs = Arc::new(DiskFs::new(tmp.path().clone(), Residency::Streaming));
    let id = fs.lookup("file.bin").expect("file on disk is discoverable");
    let proc = Process::new(1);
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();

    // Drain through the shared-cursor sequential read.
    let mut got = Vec::new();
    let mut buf = vec![0u8; 4096];
    loop {
        let o = proc.read(fd, &mut buf).unwrap();
        got.extend_from_slice(&buf[..o.bytes_read]);
        if o.end_of_file {
            break;
        }
    }
    assert_eq!(
        got, data,
        "sequential read of a real file must be byte-exact"
    );
    assert_eq!(proc.position(fd).unwrap(), data.len() as u64);
}

#[test]
fn disk_streaming_cannot_borrow_so_read_shared_falls_back() {
    let tmp = TempDir::new("fallback");
    tmp.write("f", &real_content(1000));
    let fs = Arc::new(DiskFs::new(tmp.path().clone(), Residency::Streaming));
    let id = fs.lookup("f").unwrap();
    let proc = Process::new(1);
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();

    // The defining property of a streaming backend: no whole-file borrow, so the zero-copy path
    // reports None and the caller must use the copy `read`.
    assert!(proc.get(fd).unwrap().as_shared_bytes().is_none());
    assert!(
        proc.read_shared(fd, 16).unwrap().is_none(),
        "a streaming real-file backend must fall back, not fake a borrow"
    );
}

#[test]
fn disk_streaming_pread_is_concurrent_on_one_real_fd() {
    let tmp = TempDir::new("pread");
    let data = real_content(64 * 1024);
    tmp.write("f", &data);
    let fs = Arc::new(DiskFs::new(tmp.path().clone(), Residency::Streaming));
    let id = fs.lookup("f").unwrap();
    let proc = Arc::new(Process::new(1));
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();
    let before = proc.position(fd).unwrap();

    let head = data[..8192].to_vec();
    let tail_off = (data.len() - 8192) as u64;
    let tail = data[data.len() - 8192..].to_vec();

    // Two threads pread disjoint ranges of the SAME descriptor; a positional read must not move the
    // shared cursor nor corrupt either range.
    std::thread::scope(|s| {
        let p = proc.clone();
        let want = head.clone();
        s.spawn(move || {
            let mut b = vec![0u8; 8192];
            p.read_at(fd, 0, &mut b).unwrap();
            assert_eq!(b, want);
        });
        let p = proc.clone();
        let want = tail;
        s.spawn(move || {
            let mut b = vec![0u8; 8192];
            p.read_at(fd, tail_off, &mut b).unwrap();
            assert_eq!(b, want);
        });
    });
    assert_eq!(
        proc.position(fd).unwrap(),
        before,
        "pread on a real file must not move the shared cursor"
    );
}

// ---- resident (zero-copy) ----------------------------------------------------

#[test]
fn disk_resident_zero_copy_matches_copy_path_byte_for_byte() {
    let tmp = TempDir::new("resident");
    let data = real_content(50_000);
    tmp.write("f", &data);
    let fs = Arc::new(DiskFs::new(tmp.path().clone(), Residency::Resident));
    let id = fs.lookup("f").unwrap();
    let proc = Process::new(1);
    proc.register(fs);

    // Path A: zero-copy consumer (read_shared), assembling what the caller saw.
    let fd_zero = proc.open(id, Access::READ).unwrap();
    let mut via_shared = Vec::new();
    loop {
        let r = proc
            .read_shared(fd_zero, 1000)
            .unwrap()
            .expect("resident borrows");
        via_shared.extend_from_slice(r.slice());
        if r.end_of_file() {
            break;
        }
    }

    // Path B: classic copy `read` on an independent descriptor.
    let fd_copy = proc.open(id, Access::READ).unwrap();
    let mut via_copy = Vec::new();
    let mut buf = vec![0u8; 1000];
    loop {
        let o = proc.read(fd_copy, &mut buf).unwrap();
        via_copy.extend_from_slice(&buf[..o.bytes_read]);
        if o.end_of_file {
            break;
        }
    }

    assert_eq!(
        via_shared, data,
        "zero-copy read of a real file must be byte-exact"
    );
    assert_eq!(
        via_shared, via_copy,
        "the zero-copy and copy paths must produce identical bytes"
    );
    assert_eq!(
        proc.position(fd_zero).unwrap(),
        proc.position(fd_copy).unwrap(),
        "both paths must leave the shared cursor at the same place (EOF)"
    );
    assert_eq!(proc.position(fd_zero).unwrap(), data.len() as u64);
}

#[test]
fn read_shared_partial_and_eof_advance_cursor_like_read() {
    let tmp = TempDir::new("partial");
    let data = real_content(10);
    tmp.write("f", &data);
    let fs = Arc::new(DiskFs::new(tmp.path().clone(), Residency::Resident));
    let id = fs.lookup("f").unwrap();
    let proc = Process::new(1);
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();

    // Ask for far more than exists: len clamps to the remainder, cursor lands at EOF, not past it.
    let r = proc.read_shared(fd, 1000).unwrap().unwrap();
    assert_eq!(r.len, 10);
    assert_eq!(r.slice(), &data[..]);
    assert_eq!(proc.position(fd).unwrap(), 10);

    // Next read at EOF: produces zero bytes but must NOT move the cursor backwards or error.
    let r = proc.read_shared(fd, 1000).unwrap().unwrap();
    assert!(r.end_of_file());
    assert_eq!(r.slice(), &[][..]);
    assert_eq!(proc.position(fd).unwrap(), 10);
}

#[test]
fn read_shared_zero_length_request_is_not_eof() {
    // The zero-copy path must agree with the copy path (`zero_length_read_is_not_eof`): a
    // zero-length request mid-file yields no bytes but is NOT end-of-file.
    let tmp = TempDir::new("zero");
    let data = real_content(16);
    tmp.write("f", &data);
    let fs = Arc::new(DiskFs::new(tmp.path().clone(), Residency::Resident));
    let id = fs.lookup("f").unwrap();
    let proc = Process::new(1);
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();

    let r = proc.read_shared(fd, 0).unwrap().expect("resident borrows");
    assert_eq!(r.len, 0);
    assert!(!r.end_of_file(), "a want==0 read is not EOF");
    assert_eq!(proc.position(fd).unwrap(), 0, "cursor did not move");
}

// ---- lifecycle across a genuine shutdown -------------------------------------

#[test]
fn disk_shutdown_is_visible_to_already_open_real_descriptors() {
    for mode in [Residency::Streaming, Residency::Resident] {
        let tmp = TempDir::new("shutdown");
        tmp.write("f", &real_content(2048));
        let fs = Arc::new(DiskFs::new(tmp.path().clone(), mode));
        let id = fs.lookup("f").unwrap();
        let proc = Process::new(1);
        proc.register(fs.clone());
        let fd = proc.open(id, Access::READ).unwrap();
        assert!(proc.read(fd, &mut [0u8; 16]).unwrap().bytes_read > 0);

        // Shut the whole backend down; the open descriptor (real file) now reads as closed.
        fs.shutdown().unwrap();
        let err = proc.read(fd, &mut [0u8; 16]).unwrap_err();
        assert!(
            matches!(err, Error::BackendShutdown(_)),
            "{mode:?} read after shutdown: {err:?}"
        );
        // The zero-copy path must not leak stale bytes either: Resident stops handing out borrows
        // (Streaming never did), so read_shared reports None and the copy read is the shutdown signal.
        assert!(
            proc.read_shared(fd, 16).unwrap().is_none(),
            "{mode:?} must not borrow after shutdown"
        );
        // Further opens fail too.
        assert!(matches!(
            proc.open(id, Access::READ),
            Err(Error::BackendShutdown(_))
        ));
    }
}
