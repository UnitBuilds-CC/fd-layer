# DESIGN — `fd-layer`, an open-file-description layer

`fd-layer` is a standalone Rust prototype of the layer that maps a process's file descriptors onto
files served by different backends — in WASIX, the piece between a guest's `open` / `read` / `close`
/ `fork` calls and the filesystem underneath. It is deliberately small: stable Rust, no dependencies,
no `unsafe`, and 44 tests (7 against a real file) that run with `cargo test`. It is a design
demonstration, not a production filesystem.

## 1. The model: POSIX has three objects, not two

A **file descriptor** is a per-process index into the descriptor table; it holds no file state. An
`open()` creates a separate record — the **open file description** (OFD) — holding the cursor (current
offset) and access mode. The file's bytes are a third object, shared by all who open it.

Keeping the offset in the *middle* object makes POSIX semantics fall out for free: two `open()`s
create two OFDs, so their cursors are **independent**; `dup` and `fork` copy the *descriptor*, not the
OFD, so the copies **share one** cursor.

The current WASIX layer stores the offset on the descriptor and hangs one seekable host handle off the
inode. Two bugs follow:

- Readers share one seekable handle, so a read is "seek, then read" under a write lock: independent
  opens serialize, and a range-fetchable backend (network, archive) cannot express itself.
- With cursor and bytes in different objects, "read here, then advance" cannot be made atomic by any
  lock: a parent and child on one descriptor duplicate a byte and skip the next.

The fix is the missing object, not a cleverer lock: give the OFD its own cursor and make "fetch, then
advance" one critical section. The race is designed out, not managed.

## 2. Core objects and ownership

```
Process (fd table) → descriptor slot → Arc<OpenFileDescription> → Arc<dyn Resource> → backend
  owns the table       an index          THE shared cursor          read-at-offset
                                           + access mode             + optional bytes view
```

| Object | Defined in | Owns | Shared by |
|---|---|---|---|
| `Process` | `process.rs` | the descriptor table (slot → OFD, plus allocator state) and the backend registry | one process |
| `OpenFileDescription` | `ofd.rs` | one cursor, one access mode, one `Arc<dyn Resource>` | `fork` and `dup`, which clone the `Arc` |
| `Resource` (trait) | `resource.rs` | how to read bytes at an offset | the OFD that opened it |
| `Filesystem` (trait) | `resource.rs` | file identity, and opening resources | the process's backend registry |

Two properties do the heavy lifting. **Sharing needs no lookup**: because the cursor is an OFD field,
the two-`open()`s-versus-`fork`/`dup` rule of §1 falls out with no alias table to keep consistent.
**Lifetime is the reference count**: `close` releases the slot and the resource frees when the last
`Arc` drops, so use-after-close is not expressible.

## 3. Invariants, and what enforces them

Which invariants the types enforce, and which need runtime synchronization:

| # | Invariant | Enforced by | Where |
|---|---|---|---|
| I1 | A resource is reachable only while an OFD or reader holds it | type (`Arc`) | whole crate |
| I2 | Advancing the cursor is atomic with the read that earned it | runtime (`Mutex` spans fetch + advance) | `ofd.rs` |
| I3 | A failed read never moves the cursor | runtime (advance on the success arm only) | `ofd.rs` |
| I4 | The cursor is *not* a per-descriptor field | type (a descriptor is a bare index) | `process.rs` |
| I5 | A positional read never disturbs the shared cursor | type (`&self`, no lock taken) | `ofd.rs` |
| I6 | A backend works without lending out the whole file | type (the bytes view defaults to `None`) | `resource.rs` |
| I7 | Callers never downcast a backend | type (object-safe traits, complete interface) | `resource.rs` |
| I8 | File identity is minted by the backend, not derived | type + discipline (`FileId` is carried) | `types.rs` |
| I9 | Descriptor numbers survive `fork` verbatim | runtime (`fork` clones the table under one lock) | `process.rs` |

All are structural — the compiler rejects a violation — except I2, I3, and I9, which need a lock. I2 is
the point of the crate and cannot be structural: "fetch and advance together" is inherently runtime.
Two tests pin it deterministically. `shared_offset_fork_read_partitions_file_exactly` reads one shared
cursor from two threads and asserts the bytes return as a **multiset** (every distinct byte once), so a
duplicate cannot hide behind a matching length;
`cursor_lock_is_atomic_with_the_fetch_deterministically` parks a read *inside* the critical section
through a hidden seam and forces a competitor to arrive.

## 4. Semantics chosen

**`open`** uses POSIX's lowest-free-slot rule. A provably full table is rejected in constant time —
about 150 ns where a naive scan would be tens of microseconds (`examples/bench.rs`, arm [3]). An
optional `open_at` places a file at an exact slot or fails with `SlotAlreadyInUse`, because journal
restore renumbers a descriptor to a specific slot, which lowest-free allocation cannot reproduce.

**`read`** has four entry points, because "read" is not one operation:

- `read` — advances the shared cursor, waits behind contention. The ordinary sequential read.
- `try_read` — returns `WouldBlock` instead of queueing. Deliberately **stricter than POSIX**; the
  clean contention signal.
- `read_at` — a positional read (`pread`): no lock, no cursor movement, fully parallel on one
  descriptor.
- `read_shared` — the zero-copy sequential read: when the backend can lend its bytes it advances the
  cursor and returns a shared view (~38–75 ns/call vs 9,300–31,200 ns to copy a real 8 MiB file), and
  reports "unavailable" when it cannot.

A read returns a `ReadOutcome` with an explicit end-of-file flag, not just a count, set only when the
backend returned zero bytes for a *non-empty* request — so a zero-length read is never EOF.

**`close`** clears a slot; the resource lives until the last `Arc` drops (I1), so a child's inherited
descriptor keeps working after the parent closes its copy.

**`fork`** clones the descriptor table — slots and allocator state — under one lock, copying no
resource. Parent and child share one OFD and cursor, so the POSIX rule falls out of the object model
and holes and slot numbers survive verbatim (`fork_preserves_holes_and_slot_numbers`). `dup` and `seek`
exist only to make that sharing observable; `vfork`, `CLONE_FILES`, and `O_CLOEXEC` are out of scope.

## 5. How concurrent reads behave

- **Sequential reads through one shared OFD serialize on the cursor.** That serialization *is* I2 — what
  makes fetch-and-advance atomic. Readers queue, or get `WouldBlock` from `try_read`. Measured cost is
  about 6–8× per operation from 1 to 8 threads.
- **Positional reads (`read_at`) on one OFD run fully in parallel** — no lock, no cursor.
- **Reads through different OFDs never contend**, even on the same file.
- **`shutdown` is a per-backend hook, not a table teardown:** it flips one shared atomic flag every
  resource from that backend already holds, so a read starting after it fails with `BackendShutdown`
  while one already holding bytes commits them, and the zero-copy borrow stops.

No test asserts wall-clock time; the benchmark examples report it, never gated in CI.

## 6. The backend abstraction

The seam is one required method:

```
read_at(&self, offset, buf) -> Result<usize>
```

Reading at an explicit offset through `&self` deletes forced seekability: a range-fetchable source
answers `read_at(offset, ..)` directly instead of needing a global mutable cursor. The contract is
total: `Ok(0)` is EOF; a transient "no data yet" **must** be `WouldBlock`, never `Ok(0)`; `Err` means no
partial progress (the cursor advances only after success — I3); and short reads (`Ok(n)`,
`n < buf.len()`) are legal and common. The traits stay narrow (three methods on `Resource`, four on
`Filesystem`): path resolution, caching, and offset management belong *above* a backend. A backend may
offer `as_shared_bytes` for zero-copy, but it defaults to `None`, so a streaming backend is first-class.

**Three backends, not the brief's two,** stress the seam from opposite directions:

- **`mem`** — resident, infallible, lends the whole file. The fast path.
- **`chunked`** — answers at most `chunk_max` bytes per call (default 7), so a full read *must* loop;
  cannot lend; can be armed from outside to fail transiently or permanently (how I3 is tested without
  racing); and caps in-flight fetches, refusing past the cap.
- **`disk`** — a real file, read positionally or loaded once and lent zero-copy, with no `unsafe`.

## 7. Alternatives considered

- **One host handle per inode, readers serialized by a proper lock** — the current design with a better
  lock. It restores every symptom (opens serialize, the backend stays seekable, a second hidden cursor
  lives in the host) and gets resource economy right only by accident.
- **Offset on the descriptor plus a compare-and-swap** — the CAS either fails under a racing reader or
  succeeds over bytes from a different fetch. The race moves; it does not close.
- **An `AtomicU64` cursor** — cannot make read-then-advance atomic, and destroys `try_read`'s
  contention signal. The mutex is a correctness argument, not a speed claim.
- **A `Kind` enum or `downcast_ref`** — a type test in every syscall (the current layer mentions
  `Kind::` 265 times across 34 files), so zero-copy is a defaulted `Option` at the seam instead.
- **Async traits** — an executor, a waker story, and a `Pin`/lifetime tax in a guest that may have no
  threads to block, where `WouldBlock` already says "no data yet".

## 8. Where I would not trust this in production

None is a correctness bug; each is a cost or missing piece, with its fix.

1. **The cursor lock spans the backend fetch, with no deadline** — no `Condvar` or timed lock — so one
   wedged peer holds every sequential reader of that OFD indefinitely. Fix: a cancellation token on the
   *fetch*, under a watchdog. `cargo test` cannot see it; every in-tree backend returns promptly.
2. **One resource per open assumes a local backend:** N opens of one remote file are N sessions.
   `chunked` caps in-flight fetches instead (measured, `examples/bench.rs` arm [5]: connections peak at
   the budget, wall time within 0.92–1.05×, budget 1 costs 8.4×). Fix: a per-file shared *immutable*
   read resource — sharing removes round trips a cap only rations.
3. **No fetch policy for forced short reads:** at `chunk_max` 7, a 1 KiB read is 147 fetches — free
   in-process, 147 round trips remotely. Fix: a *bounded* per-open range cache; unbounded read-ahead is
   a page cache, with eviction and coherence questions.
4. **Descriptor numbers have no generation:** close-then-reopen can reuse a number under a caller
   holding the *number*, not the `Arc`. Fix: an epoch above the descriptor number.
5. **The backend error carries a string,** so "transient?" can only mean `WouldBlock`; a port cannot
   separate "no data yet" from "retry might succeed" without matching text. Fix: a typed error against
   the real errno set.

Types cannot enforce one thing: **`FileId` equality as the sharing key is a backend contract.** The
current layer hashes paths and its own source concedes hard links break under renames — such a backend
silently breaks sharing. Read-only scope, a flat namespace, and no pseudo-filesystems are stated
assumptions.

## 9. How the design grows

**Writes.** Add `write_at(&self, offset, buf)` to `Resource` and a write cursor to the OFD. Append /
read-modify-write atomicity (`O_APPEND` resolves offset-plus-write under one lock, as I2 does for reads)
and write coherence — where §8 item 2's shared read resource starts to matter — become load-bearing;
the ownership model does not move.

**Directories.** `read_dir` returns a position-carrying `DirStream`, so a directory is an OFD with a
position cursor and `telldir` / `seekdir` / `fork` / `dup` reuse the existing operations; path
resolution (`..`, symlinks, mounts) sits *above* the table, so identity stays the backend's to mint.

## 10. Restraint, and what AI changed

Three features were built, tested, and **removed** — a fetch-size hint, a stale-descriptor guard, a
typed retry-versus-permanent error — each already shown unnecessary by the simpler design. They ship as
re-appliable patches under `expansions/` (apply 03 → 02 → 01, `patch -p1 --fuzz=0`; 47 tests green when
restored), so the restraint is auditable.

The brief asks what AI materially influenced. Three architectural choices:

- **Accepted:** the explicit end-of-file flag on `ReadOutcome` over a length-only return, and
  barrier-choreographed contention tests over "spawn N threads and see" — each turns a semantic into
  something a test can hold.
- **Rejected after building:** the three removed features above — on scope, not merit.
- **Found wrong,** in an independent AI answer to this brief: a cursor that reserves the requested length
  under the lock and corrects the position afterward, so a racing reader on one OFD sees the position
  pass bytes nobody read — one byte duplicated, the next skipped. That is the hazard
  `shared_offset_fork_read_partitions_file_exactly` catches, and why it asserts multiset equality rather
  than a count. The same answer's "slow" backend absorbed its chunking internally, leaving the
  short-read contract unexercised — why `chunked` keeps short reads visible at the boundary.
