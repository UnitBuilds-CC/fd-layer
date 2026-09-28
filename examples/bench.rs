//! Baseline benchmark harness — `cargo run --release --example bench`.
//!
//! This measures the *current* cost of the deliberately-naive paths the design leaves open, so the
//! "is it optimized?" question is answered with numbers instead of claims. It uses only `std`
//! (`Instant` + `thread::scope`) — no criterion, no extra dependencies — matching the crate's
//! zero-dep, stable-toolchain constraint.
//!
//! **These timings are reported, never asserted.** Nothing here is a CI gate; the correctness suite
//! (`cargo test`) is. Run under `--release` for meaningful numbers.

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fd_layer::{
    Access, ChunkedFs, DiskFs, Error, Filesystem, MemFs, Process, Residency, SlowConfig,
};

/// Number of timed repetitions per scenario (after warm-up).
const SAMPLES: usize = 5;

fn main() {
    if cfg!(debug_assertions) {
        println!("NOTE: built in debug — re-run with `--release` before reading these numbers.\n");
    }
    println!("fd-layer baseline — correctness is proven by `cargo test`, not here.");
    println!("This quantifies the baseline the DESIGN.md §6 gaps are ordered against.\n");

    mem_read_vs_zerocopy();
    real_file_copy_vs_zerocopy();
    chunked_block_scan();
    descriptor_scan();
    cursor_contention();
    connection_budget();
    remote_note();
}

// ---------------------------------------------------------------------------
// 1. MemFs: the sequential `read` copy vs the `as_shared_bytes` zero-copy path
// ---------------------------------------------------------------------------

/// The layer *always* copies bytes into the caller's buffer through `read_at`, even for an
/// in-memory file that could hand back a borrowed `Arc<[u8]>` for free. This measures the copy the
/// unused fast path would avoid.
fn mem_read_vs_zerocopy() {
    println!("[1] MemFs: sequential `read` (copy) vs `as_shared_bytes` (zero-copy)");
    const SIZE: usize = 8 << 20; // 8 MiB
    const ZC_ITER: u32 = 100_000; // batch the O(1) borrow so it clears the timer floor
    let fs = Arc::new(MemFs::new());
    let id = fs.add("big", vec![0u8; SIZE]);
    let proc = Process::new(1);
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();
    let ofd = proc.get(fd).unwrap();

    let copy = bench("copy whole file via read", || {
        proc.seek(fd, 0).unwrap();
        let mut buf = vec![0u8; 64 << 10];
        loop {
            let outcome = proc.read(fd, &mut buf).unwrap();
            black_box(&buf[..outcome.bytes_read]);
            if outcome.end_of_file {
                break;
            }
        }
    });
    let zc = bench("fetch Arc via as_shared_bytes", || {
        // A single Arc clone is below the timer floor (would read as 0.0 ns), so time ZC_ITER clones
        // and divide — black_box keeps each refcount bump from being elided.
        for _ in 0..ZC_ITER {
            black_box(ofd.as_shared_bytes().unwrap());
        }
    });
    let mib = |secs: f64| SIZE as f64 / (1 << 20) as f64 / secs;
    println!(
        "      {:<40} {:>10.1} ns/op  ~{:>8.0} MiB/s  (copies the whole 8 MiB)",
        "read (memcpy path)",
        copy.mean.as_nanos() as f64,
        mib(copy.mean_secs())
    );
    println!(
        "      {:<40} {:>10.1} ns/op  (O(1) Arc bump ×{ZC_ITER} averaged — moves no bytes, so no size scaling)",
        "as_shared_bytes (unused fast path)",
        zc.mean.as_nanos() as f64 / ZC_ITER as f64,
    );
    println!("      => the zero-copy path is available and unused; `read` pays a full copy.\n");
}

/// The same copy-vs-zero-copy contrast, but against a **real 8 MiB file on disk** — so the result is
/// a property of genuine OS file I/O, not a synthetic buffer. Streaming mode exercises the copy path
/// through the OS positional read (`pread`/`seek_read`); Resident mode loads once, then serves the
/// zero-copy `read_shared` path.
fn real_file_copy_vs_zerocopy() {
    println!("[1b] A PROPER file (8 MiB written to disk), via DiskFs");
    const SIZE: usize = 8 << 20;
    const CALL: usize = 64 << 10; // 64 KiB per read
    const CALLS: usize = SIZE / CALL;
    let dir = std::env::temp_dir().join(format!("fdlayer-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("big.bin"),
        (0..SIZE).map(|i| (i % 251) as u8).collect::<Vec<u8>>(),
    )
    .unwrap();

    // Streaming: `read` copies, each call a real OS positional read.
    let sfs = Arc::new(DiskFs::new(dir.clone(), Residency::Streaming));
    let sid = sfs.lookup("big.bin").unwrap();
    let sproc = Process::new(1);
    sproc.register(sfs);
    let sfd = sproc.open(sid, Access::READ).unwrap();
    let copy = bench("copy whole file via read (real pread)", || {
        sproc.seek(sfd, 0).unwrap();
        let mut buf = vec![0u8; CALL];
        loop {
            let o = sproc.read(sfd, &mut buf).unwrap();
            black_box(&buf[..o.bytes_read]);
            if o.end_of_file {
                break;
            }
        }
    });

    // Resident: one-time O(size) load at open (timed separately, excluded from the loop below),
    // then zero-copy `read_shared` — returns borrowed views, copying nothing.
    let rfs = Arc::new(DiskFs::new(dir.clone(), Residency::Resident));
    let rid = rfs.lookup("big.bin").unwrap();
    let rproc = Process::new(2);
    rproc.register(rfs);
    let open_start = Instant::now();
    let rfd = rproc.open(rid, Access::READ).unwrap(); // the whole-file load happens here
    let open_load = open_start.elapsed();
    let zc = bench("drain via read_shared (zero-copy)", || {
        rproc.seek(rfd, 0).unwrap();
        loop {
            let r = rproc
                .read_shared(rfd, CALL)
                .unwrap()
                .expect("resident borrows");
            black_box(r.slice());
            if r.end_of_file() {
                break;
            }
        }
    });

    let mib = |secs: f64| SIZE as f64 / (1 << 20) as f64 / secs;
    println!(
        "      copy via real pread   : {:>11.0} ns/pass (~{:>5.0} MiB/s, {:>5.0} ns/{k}KiB call)",
        copy.mean.as_nanos() as f64,
        mib(copy.mean_secs()),
        copy.mean.as_nanos() as f64 / CALLS as f64,
        k = CALL >> 10,
    );
    println!(
        "      zero-copy read_shared : {:>11.0} ns/pass, ~{:.0} ns/call of loop overhead",
        zc.mean.as_nanos() as f64,
        zc.mean.as_nanos() as f64 / CALLS as f64,
    );
    println!(
        "        (NO throughput quoted: read_shared moves zero bytes — reporting MiB/s would be"
    );
    println!(
        "         fabricated. The pass cost is the {CALLS}-iteration loop of returning views.)"
    );
    println!(
        "      resident one-time open load (paid once at open, excluded above): {open_load:?}"
    );
    println!("      => on a real file: copy path is syscall+memcpy bound; zero-copy is constant per\n         call (no syscall, no copy) — the win the layer could take and currently doesn't.\n");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 2. ChunkedFs: byte_at's O(blocks) walk made visible
// ---------------------------------------------------------------------------

/// `ChunkedResource::byte_at` walks the block list for *every* byte, so a read of `n` bytes over
/// `B` blocks is O(n·B). This is the mock's known-inefficient accessor; a real block backend would
/// index. Shown scaling with block count on a fixed file.
fn chunked_block_scan() {
    println!("[2] ChunkedFs: fill-loop cost vs block count (byte_at is O(blocks) per byte)");
    const SIZE: usize = 32 << 10; // 32 KiB fixed
    const CHUNK_MAX: usize = 1024;
    println!(
        "      {:<10} {:>8} {:>12} {:>10}",
        "block size", "blocks", "MB/s", "ns/byte"
    );
    for block in [4096usize, 1024, 256, 64, 16] {
        let cfs = Arc::new(ChunkedFs::new(SlowConfig {
            chunk_max: CHUNK_MAX,
            latency: Duration::ZERO,
        }));
        let id = cfs.add_chunked("d", vec![0u8; SIZE], block);
        let proc = Process::new(1);
        proc.register(cfs);
        let fd = proc.open(id, Access::READ).unwrap();
        let s = bench("drain chunked", || {
            drain_chunked(&proc, fd, CHUNK_MAX);
        });
        println!(
            "      {:<10} {:>8} {:>12.1} {:>10.1}",
            block,
            SIZE / block,
            (SIZE as f64) / 1e6 / s.mean_secs(),
            s.mean.as_nanos() as f64 / SIZE as f64,
        );
    }
    println!("      => ns/byte grows ~linearly with block count: O(n·B), not O(n).\n");
}

fn drain_chunked(proc: &Process, fd: u32, max: usize) -> usize {
    proc.seek(fd, 0).unwrap();
    let mut buf = vec![0u8; max];
    let mut total = 0;
    loop {
        let o = proc.read(fd, &mut buf).unwrap();
        total += o.bytes_read;
        if o.end_of_file {
            break;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// 3. fd table: the free-slot allocation cost (naive scan → measured after the fix)
// ---------------------------------------------------------------------------

/// `open` allocates the *lowest* free fd (POSIX). The old `first_free_locked` rescanned from slot 0
/// on every call, so a full table cost O(max_fd). The fix keeps a `next_free` lower bound plus an
/// `open_count` pigeonhole: allocation starts at the bound, and a full table is rejected in O(1)
/// without any scan. This measures both the residual low-hole path (still a short walk) and the
/// formerly-linear full-table rejection. `open_at` pins a slot directly and never scans.
///
/// A single post-fix `open` is below the OS timer quantum (a lone op reads as a clock artifact — see
/// `[1]`'s `0.0 ns`), so each timed sample runs `ITER` idempotent ops and we divide. Both loops are
/// state-preserving: a full-table `open` never mutates the table, and open-at-hole+`close` returns to
/// the same state — so batching measures the real per-op cost above the floor, not a single-shot read.
///
/// The `~51,300 ns` pre-fix number is the *recorded* baseline measured on the old from-slot-0 scan
/// (≈0.8 ns/slot across 65,535 slots), not something this run re-measures; the scan it describes no
/// longer exists in `src/`. Only the post-fix reject below is timed here.
fn descriptor_scan() {
    println!("[3] fd table: `open` free-slot allocation (scan fixed → next_free + open_count)");
    const MAXFD: u32 = 65_534;
    const ITER: u32 = 4_000;
    let fs = Arc::new(MemFs::new());
    let id = fs.add("x", vec![0u8; 16]);

    // Low-hole case: slot 1 free, next_free points at it → open+close per op (alloc + release).
    let empty = Process::new(2);
    empty.register(fs.clone());
    empty.open_at(0, id, Access::READ).unwrap(); // pin slot 0 so the timed open never lands there
    let t_empty = bench("open+close (hole at index 1)", || {
        for _ in 0..ITER {
            let fd = empty.open(id, Access::READ).unwrap();
            empty.close(fd).unwrap();
        }
    });

    // Full-table case: every 0..=max_fd is pinned via open_at (which never advances next_free), so
    // only the open_count pigeonhole can prove fullness — the old scan would walk all 65,535 slots.
    let full = Process::with_max_fd(3, MAXFD);
    full.register(fs);
    for slot in 0..=MAXFD {
        full.open_at(slot, id, Access::READ).unwrap();
    }
    let t_full = bench("open (table FULL → O(1) reject)", || {
        for _ in 0..ITER {
            black_box(full.open(id, Access::READ).unwrap_err());
        }
    });

    let ns_empty = t_empty.mean.as_nanos() as f64 / ITER as f64;
    let ns_full = t_full.mean.as_nanos() as f64 / ITER as f64;
    println!(
        "      open+close, low hole: {:>10.1} ns/op   (alloc at next_free + release, {ITER}×avg)",
        ns_empty
    );
    println!(
        "      open, full ({} slots): {:>10.1} ns/op   (O(1) pigeonhole reject, no scan, {ITER}×avg)",
        MAXFD + 1,
        ns_full,
    );
    println!(
        "      => full fd space now rejects in ~{:.0} ns/op — constant, no per-slot scan. The old\n         linear scan was ~51,300 ns/op; ~{:.0}× faster, and no longer scales with max_fd.\n",
        ns_full,
        51_300.0 / ns_full,
    );
}

// ---------------------------------------------------------------------------
// 4. Shared-cursor Mutex under contention (DESIGN.md §5, the rejected AtomicU64 cursor)
// ---------------------------------------------------------------------------

/// The sequential `read` holds `position`'s `Mutex` across the resource fetch (one critical
/// section, by design). That is the atomicity guarantee — and the serialization cost. Measured on
/// `position()` itself, which locks and releases the *same* `Mutex<u64>` with no I/O, so the
/// numbers isolate the lock's scaling. `pread`/`read_at` takes no lock at all.
fn cursor_contention() {
    println!("[4] `Mutex<position>` scaling under contention (DESIGN.md §5's mutex, measured)");
    let fs = Arc::new(MemFs::new());
    let id = fs.add("d", vec![0u8; 8]);
    let proc = Arc::new(Process::new(1));
    proc.register(fs);
    let fd = proc.open(id, Access::READ).unwrap();
    let ofd = proc.get(fd).unwrap();

    // Contention scaling is scheduler-sensitive, so run a longer cohort and keep the *best* of
    // several repetitions — this smooths thread-placement jitter that made short runs non-monotonic.
    const LOOPS: u64 = 1_000_000;
    const REPEATS: usize = 5;
    let cohort = |threads: usize| -> f64 {
        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..threads {
                let ofd = ofd.clone();
                s.spawn(move || {
                    for _ in 0..LOOPS {
                        black_box(ofd.position());
                    }
                });
            }
        });
        t.elapsed().as_nanos() as f64 / (LOOPS as f64 * threads as f64)
    };

    let mut base = 0.0f64;
    println!(
        "      {:>8} {:>14} {:>10}",
        "threads", "ns/lock-op", "scaling"
    );
    for threads in [1usize, 2, 4, 8] {
        let mut best = f64::MAX;
        for _ in 0..REPEATS {
            best = best.min(cohort(threads));
        }
        if threads == 1 {
            base = best;
        }
        println!("      {:>8} {:>14.1} {:>9.1}x", threads, best, best / base,);
    }
    println!(
        "      NOTE: an AtomicU64+retry variant is NOT implemented in the tree. These numbers"
    );
    println!("            quantify how the *current* Mutex scales; they are not a head-to-head.");
    println!("            `read_at` (pread) bypasses this lock entirely.\n");
}

// ---------------------------------------------------------------------------
// 5. Connection budget: what capping the backend's concurrency costs
// ---------------------------------------------------------------------------

/// A backend that refuses a fetch at capacity instead of queueing is only safe because the read
/// position lives above the seam — see `tests/connection_budget.rs`. This measures the other side of
/// that choice: what the cap costs when nobody has to wait, whether it really caps, and how much CPU
/// a spinning caller burns for a slot.
///
/// Two arms only. There is deliberately **no** multi-worker zero-latency arm: with a free fetch the
/// number is the per-backend bookkeeping (the shared stats counters and the descriptor lookup), not
/// the cap, and it reads misleadingly as throughput. The ceiling prediction is calibrated from a
/// *measured* round trip, because `thread::sleep` on Windows quantizes to the system timer and a
/// nominal figure would skew every ratio identically.
fn connection_budget() {
    println!("[5] Connection budget: uncontended price, then whether it caps");
    const FETCHES: usize = 40;
    /// A spawned Windows thread costs tens of µs, so the free-fetch arm needs enough fetches per
    /// worker that thread creation is a rounding error rather than the measurement.
    const FREE_FETCHES: usize = 50_000;
    const BUDGETS: [(&str, usize); 4] = [("unlimited", usize::MAX), ("8", 8), ("2", 2), ("1", 1)];
    const ROUND_TRIP: Duration = Duration::from_micros(1000);

    // (a) Uncontended: one worker, so a slot is always free. Same path with and without a bound, so
    //     the gap is the pool's own accounting — and the denial count must be zero either way.
    let free = budget_cohort(1, usize::MAX, Duration::ZERO, FREE_FETCHES);
    let capped = budget_cohort(1, 1, Duration::ZERO, FREE_FETCHES);
    println!(
        "      free fetch, 1 worker:  unlimited {:>7.1} ns/fetch   budget 1 {:>7.1} ns/fetch   ({:+.1}%), denials {}/{}",
        free.ns_per_fetch,
        capped.ns_per_fetch,
        (capped.ns_per_fetch - free.ns_per_fetch) / free.ns_per_fetch * 100.0,
        free.denials,
        capped.denials
    );

    // (b) Loaded: every fetch holds its connection for one measured round trip.
    let round_trip = budget_cohort(1, usize::MAX, ROUND_TRIP, FETCHES).ns_per_fetch;
    println!(
        "  -- one fetch, un-contended, measured at {:.0} µs --",
        round_trip / 1e3
    );
    println!(
        "      {:>7} {:>10} {:>12} {:>14} {:>6} {:>15}",
        "workers", "budget", "ns/fetch", "retries/fetch", "peak", "ceiling check"
    );
    for workers in [1usize, 8] {
        for (label, budget) in BUDGETS {
            let run = budget_cohort(workers, budget, ROUND_TRIP, FETCHES);
            let slots = if budget == usize::MAX {
                workers
            } else {
                budget.min(workers)
            };
            let predicted = run.fetches as f64 * round_trip / slots as f64;
            println!(
                "      {:>7} {:>10} {:>12.0} {:>14.0} {:>6} {:>14.2}x",
                workers,
                label,
                run.ns_per_fetch,
                run.retries_per_fetch,
                run.peak,
                run.wall_ns / predicted
            );
        }
    }
    println!(
        "      NOTE: `ns/fetch` is cohort wall time over *successful* fetches, so it rises as the"
    );
    println!(
        "            cap tightens — that is the price. `peak` is what it buys: never more than"
    );
    println!(
        "            `budget` fetches in flight, whatever the reader count. `ceiling check` is"
    );
    println!(
        "            measured wall / (fetches x round trip / slots); ~1.00x means the cap is the"
    );
    println!("            limit, not an artefact of placement. `retries/fetch` is the spin a");
    println!(
        "            `WouldBlock` caller pays: the honest cost of refusing instead of parking,"
    );
    println!("            and the figure to trade against a wait queue above this seam.");
    println!("            NOT MEASURED: a real TCP round trip (see [6]); host-fd counts.\n");
}

/// One timed cohort, reported as nanoseconds and counts only.
struct Run {
    wall_ns: f64,
    ns_per_fetch: f64,
    denials: usize,
    retries_per_fetch: f64,
    peak: usize,
    fetches: usize,
}

const BUDGET_REPEATS: usize = 3;

/// Repeat [`budget_run`] and keep the least-contended run — placement makes single runs
/// non-monotonic, the same reason section [4] takes a best-of.
fn budget_cohort(workers: usize, budget: usize, latency: Duration, per_worker: usize) -> Run {
    let mut best = budget_run(workers, budget, latency, per_worker);
    for _ in 1..BUDGET_REPEATS {
        let run = budget_run(workers, budget, latency, per_worker);
        if run.wall_ns < best.wall_ns {
            best = run;
        }
    }
    best
}

/// `workers` threads each complete `per_worker` positioned fetches against a backend capped at
/// `budget` simultaneous fetches, spinning only on refusal.
fn budget_run(workers: usize, budget: usize, latency: Duration, per_worker: usize) -> Run {
    let fs = Arc::new(ChunkedFs::new(SlowConfig {
        chunk_max: 64,
        latency,
    }));
    let id = fs.add_chunked("data", vec![7u8; 1 << 14], 4096);
    fs.set_connection_budget(budget);
    let proc = Arc::new(Process::new(1));
    proc.register(fs.clone());
    let fd = proc.open(id, Access::READ).unwrap();

    let mut warm = [0u8; 64];
    fetch_at(&proc, fd, &mut warm, 0);
    fs.reset_connection_stats();

    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..workers {
            let who = proc.clone();
            s.spawn(move || {
                let mut buf = [0u8; 64];
                for i in 0..per_worker {
                    fetch_at(&who, fd, &mut buf, (i % 16) as u64 * 64);
                }
            });
        }
    });
    let wall_ns = started.elapsed().as_nanos() as f64;
    let fetches = workers * per_worker;
    let denials = fs.connection_denials();
    Run {
        wall_ns,
        ns_per_fetch: wall_ns / fetches as f64,
        denials,
        retries_per_fetch: denials as f64 / fetches as f64,
        peak: fs.peak_connections(),
        fetches,
    }
}

/// One fetch, spinning only on capacity refusal. Anything else is a bug, not a result.
fn fetch_at(proc: &Process, fd: u32, buf: &mut [u8], offset: u64) {
    loop {
        match proc.read_at(fd, offset, buf) {
            Ok(n) if n > 0 => return,
            Ok(n) => panic!("backend returned a {n}-byte fetch"),
            Err(Error::WouldBlock) => std::hint::spin_loop(),
            Err(e) => panic!("unexpected fetch error: {e:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 6. The remote path — explicitly NOT measured here
// ---------------------------------------------------------------------------

fn remote_note() {
    println!("[6] Remote (HTTP-Range) backend: NOT MEASURED here");
    println!(
        "      An out-of-tree proof drives these same traits against a real nginx container over"
    );
    println!(
        "      HTTP Range (one fresh TCP connection per `read_at`: 1204 bytes in 20 requests = 20"
    );
    println!(
        "      20 connections. Quantifying that cost needs Docker up, so it is out of scope for"
    );
    println!(
        "      this std-only harness and is recorded as *not measured* rather than estimated.\n"
    );
}

// ---------------------------------------------------------------------------
// timing helpers
// ---------------------------------------------------------------------------

struct Sample {
    mean: Duration,
}

impl Sample {
    fn mean_secs(&self) -> f64 {
        self.mean.as_secs_f64()
    }
}

/// Warm up, then take `SAMPLES` timed runs and report the mean. `f` must do a fixed unit of work
/// and be re-runnable (idempotent via an internal seek/reset).
fn bench<F: FnMut()>(_label: &str, mut f: F) -> Sample {
    for _ in 0..20 {
        f();
    }
    let mut total = Duration::ZERO;
    for _ in 0..SAMPLES {
        let t = Instant::now();
        f();
        total += t.elapsed();
    }
    Sample {
        mean: total / SAMPLES as u32,
    }
}
