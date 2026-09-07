# Use the native SDK from Rust

Rust is a frontend to the shared native SDK. It provides typed RMSNorm and ordinary FP8 blockscale GEMM descriptors, owned preparation handles and errors copied from the C API. It uses the same HIP/CK providers as the other native clients. It does not duplicate kernel selection, tuning or compilation in Rust.

The crate is local and not published. Build an AITER SDK first, then point Cargo to its installation:

```bash
export AITER_SDK_DIR=/path/to/installed/aiter-sdk
export LD_LIBRARY_PATH="$AITER_SDK_DIR/lib:$AITER_SDK_DIR/lib64:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --manifest-path bindings/rust/Cargo.toml --offline
cargo run --manifest-path bindings/rust/Cargo.toml --offline --example operators
```

The crate needs Rust 1.75 or newer and has no third-party Rust dependencies. The native SDK requires Linux and the declared ROCm environment. The GPU example qualifies its native gfx950 subset and checks both numerical results and invalid-buffer rejection. The ordinary Python installation does not need a Rust compiler.

Plan preparation and handle destruction have a typed Rust interface. Enqueue remains explicitly `unsafe`: the application owns raw HIP allocations, streams and asynchronous completion, and must keep them valid through queued work and any graph replay. A temporary Rust slice cannot prove those GPU lifetimes. The frontend therefore does not hide synchronization in a destructor or advertise an unproven safe asynchronous API. The example owns its test allocations and waits explicitly before reading or freeing them.

RMSNorm buffers are ordered input, weight, output. GEMM buffers are x, w, x_scale, w_scale, output; the buffer count is part of the plan's Rust type. Plans are neither cloneable nor transferable between threads in this frontend. The native C SDK retains executable code for graphs; Rust owns only the host preparation handle.

Future safe asynchronous bindings should integrate an actual allocation/stream owner and graph lifetime model before making stronger guarantees. Python/Torch remains the broad framework interface. Rust's role is a native application boundary, not a reason to rewrite specialized GPU kernels or introduce a second scheduler.

The acceptance runner builds and installs the candidate SDK, copies the Rust consumer outside the checkout, runs formatting, Clippy, ABI tests and thread-trait compile checks, then executes the GPU examples:

```bash
bash tests/integration/sdk/run-rust.sh
```

An explicit installed prefix can be supplied as its only argument. Both `lib` and `lib64` SDK installations are supported.
