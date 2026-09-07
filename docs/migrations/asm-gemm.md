# BF16 assembly GEMM correctness

`aiter.gemm_a16w16_asm` keeps its public Python signature. It accepts BF16
matrices, a BF16 or FP32 output, and an optional BF16 or FP32 bias vector.
The underlying assembly instructions multiply BF16 values; FP16 and mixed
inputs now fail explicitly instead of being interpreted as BF16 bits.

The former split-K path accumulated each partial result into the BF16 output.
On the checked gfx950 workload, sixteen partial sums caused about 22% of
output elements to exceed the existing comparison tolerance. The bridge now
provides FP32 temporary storage, accumulates there, adds bias in FP32, and
rounds once when writing BF16. Existing BF16-only single-pass kernels remain
available without bias. Each assembly manifest declares `supports_fp32`, which
both generation and offline tuning validate and the native dispatcher consumes.

An explicit `splitK` from 1 through 16 is honored exactly or rejected when no
kernel supports it. Previously, an explicit value was ignored unless a kernel
name was also supplied; requesting one split could therefore launch a split
kernel with an empty semaphore. The native bridge now checks the selected
kernel's semaphore requirement before launching it. Shapes, row strides,
devices, output layout, writable-buffer overlap, and address arithmetic are
also checked. Python additionally validates actual storage capacity.

The private native module is now `module_gemm_a16w16_asm_workspace`, with a
matching workspace-aware C entrypoint. An old cached binary cannot satisfy
that interface. The unused old recipe and pybind translation unit were
removed. Public Python imports remain unchanged.

This is a legacy convenience operation that allocates temporary storage. It
must be warmed before graph capture; tests cover changing-input replays and
separate streams. Semaphore allocations remain alive for the process lifetime
so creating other streams cannot evict a pointer retained by a captured graph.
The prepared runtime API has its own explicit resource model.

The extra FP32 buffer and final conversion have a cost. One local diagnostic
measured about 14.2 microseconds for the corrected BF16 path, compared with
about 9.8 microseconds for the inaccurate path. Those observations explain the
tradeoff; they are not a paired performance qualification. The original
numerical thresholds were preserved.
