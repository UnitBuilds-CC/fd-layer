# fd-layer

A standalone, stable-Rust prototype of an abstract **open-file-description (OFD)** file-descriptor layer over
deliberately different backends. No dependencies, no `unsafe`, no downcasting in the public API. The argument is in
[`DESIGN.md`](DESIGN.md); this file only shows the shape.

POSIX file access uses three objects — a per-process **descriptor**, the **open file description** that owns the offset,
and the **resource** behind it. The existing WASIX layer keeps the first and the third and collapses the middle one, so
reads seek-then-read under a write lock, two independent `open()`s serialize, a non-seekable source cannot plug in, and a
forked reader can duplicate one byte while skipping the next. This prototype restores the middle object and nothing else.

```rust
use fd_layer::{Access, MemFs, Process};
use std::sync::Arc;

fn main() -> Result<(), fd_layer::Error> {
    let mem = Arc::new(MemFs::new());
    let id = mem.add("hello", b"hello world".to_vec());

    let proc = Process::new(1);
    proc.register(mem);

    let fd = proc.open(id, Access::READ)?;              // open  → a descriptor
    let mut buf = [0u8; 5];
    let got = proc.read(fd, &mut buf)?;                 // read  → up to n bytes + the EOF flag
    println!("{} bytes, eof={}", got.bytes_read, got.end_of_file);

    let child = proc.fork();                            // fork  → shares the OFD, so the cursor too
    println!("child sees the cursor at {}", child.position(fd)?);

    proc.close(fd)?;                                    // close → drops one Arc<OFD>
    proc.shutdown("mem")                                // shutdown the whole file system
}
```

```sh
cargo test                                  # 44 tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo run --example demo                    # the semantics, printed and asserted
cargo run --release --example bench         # baselines; timings reported, never asserted
```

Built and tested on `stable` 1.95 through 1.98. The `rust-version` field in `Cargo.toml` records the oldest toolchain
verified here, not a floor implied by the code.

## License

Dual-licensed under either of MIT or Apache-2.0, at your option. See [`LICENSE-MIT`](LICENSE-MIT) and
[`LICENSE-APACHE`](LICENSE-APACHE).
