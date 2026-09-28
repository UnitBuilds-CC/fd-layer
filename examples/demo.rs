//! End-to-end demo: exercises the public API exactly as an integrator would, on real data, and
//! prints what it observes. Run with `cargo run --example demo`.
//!
//! Deliberately uses ONLY the public surface (`Process`, the traits, the two backends) — if this
//! compiles and prints correct results, the crate is usable as documented.

use std::sync::Arc;
use std::thread;

use fd_layer::{Access, ChunkedFs, Error, Filesystem, MemFs, Process, SlowConfig};

fn main() -> Result<(), Error> {
    let data: Vec<u8> = (0u32..=300).flat_map(|i| i.to_le_bytes()).collect();
    println!(
        "corpus: {} bytes (u32 LE sequence, all 4-byte groups distinct)",
        data.len()
    );

    // ---- wire one process with BOTH backends holding the same file --------------------------
    let mem = Arc::new(MemFs::new());
    let mem_id = mem.add("numbers.bin", data.clone());

    let slow = Arc::new(ChunkedFs::new(SlowConfig {
        chunk_max: 9, // forces ~35 round-trips to read a 1204-byte file
        latency: std::time::Duration::ZERO,
    }));
    let slow_id = slow.add_chunked("numbers.bin", data.clone(), 4);

    let proc = Process::new(101);
    proc.register(mem.clone());
    proc.register(slow.clone());

    // ---- open / read / EOF (fallible backend, must loop for short reads) --------------------
    let fd = proc.open(slow_id, Access::READ)?;
    println!("open  -> fd {fd} on backend '{}'", slow_id.backend);
    let mut full = Vec::new();
    let mut buf = [0u8; 64]; // ask for more than chunk_max each time
    let mut trips = 0;
    loop {
        trips += 1;
        let out = proc.read(fd, &mut buf)?;
        if out.end_of_file {
            break;
        }
        full.extend_from_slice(&buf[..out.bytes_read]);
    }
    assert_eq!(full, data, "chunked loop must reconstruct the file exactly");
    println!(
        "read  -> reassembled {} bytes in {trips} round-trips (chunk_max=9)",
        full.len()
    );
    println!("close -> {}", proc.close(fd)?);

    // ---- two opens => independent cursors ----------------------------------------------------
    let a = proc.open(mem_id, Access::READ)?;
    let b = proc.open(mem_id, Access::READ)?;
    proc.read(a, &mut [0u8; 40])?;
    println!(
        "open x2 -> a={a} cursor {} , b={b} cursor {} (independent)",
        proc.position(a)?,
        proc.position(b)?
    );
    let _ = proc.read_at(b, 500, &mut [0u8; 8])?; // pread leaves b's cursor put
    println!(
        "pread on b -> b cursor still {} (positioned read didn't move it)",
        proc.position(b)?
    );

    // ---- fork => shared OFD, concurrent readers partition the file ---------------------------
    let cfd = proc.open(slow_id, Access::READ)?;
    let child = proc.fork();
    let parent = Arc::new(proc);
    let child = Arc::new(child);
    println!(
        "fork  -> child pid {} inherits fd {cfd} sharing the parent's cursor",
        child.pid()
    );

    let mut halves = Vec::new();
    for who in [parent.clone(), child.clone()] {
        halves.push(thread::spawn(move || {
            let mut got = Vec::new();
            let mut b = [0u8; 7];
            loop {
                match who.read(cfd, &mut b) {
                    Ok(o) if o.end_of_file => break,
                    Ok(o) => got.extend_from_slice(&b[..o.bytes_read]),
                    Err(Error::WouldBlock) => continue,
                    Err(e) => panic!("read error: {e:?}"),
                }
            }
            got
        }));
    }
    let x = halves.remove(0).join().unwrap();
    let y = halves.remove(0).join().unwrap();
    let mut union = [x.as_slice(), y.as_slice()].concat();
    union.sort_unstable();
    let mut expected = data.clone();
    expected.sort_unstable();
    assert_eq!(
        union, expected,
        "shared-cursor readers must cover the file exactly once"
    );
    println!(
        "concurrent fork read -> thread A {} bytes + thread B {} bytes == whole file, no dup/skip",
        x.len(),
        y.len()
    );

    // ---- error handling: armed fault leaves the cursor untouched -----------------------------
    let fd2 = parent.open(slow_id, Access::READ)?;
    parent.read(fd2, &mut [0u8; 12])?;
    let before = parent.position(fd2)?;
    slow.faults().arm_error();
    let err = parent.read(fd2, &mut [0u8; 8]).unwrap_err();
    println!(
        "injected error -> {err} ; cursor stayed at {before} (now {})",
        parent.position(fd2)?
    );
    assert_eq!(parent.position(fd2)?, before);

    // ---- shutdown the whole file system ------------------------------------------------------
    parent.shutdown("chunked")?;
    let after = parent.read(fd2, &mut [0u8; 8]);
    println!(
        "shutdown('chunked') -> in-flight fd now reports {:?}",
        after.as_ref().err().map(|e| e.to_string())
    );
    assert!(matches!(after, Err(Error::BackendShutdown("chunked"))));

    // the fast path still available for an in-memory file
    let mfd = parent.open(mem_id, Access::READ)?;
    let bytes = parent
        .get(mfd)?
        .as_shared_bytes()
        .expect("mem offers a whole-file view");
    assert_eq!(&*bytes, &data[..]);
    println!(
        "as_shared_bytes -> mem backend handed a borrowed {}-byte view (zero loop)",
        bytes.len()
    );

    println!("\nALL DEMO ASSERTIONS PASSED");
    let _ = Filesystem::name(&*mem);
    let _ = Filesystem::name(&*slow);
    Ok(())
}
