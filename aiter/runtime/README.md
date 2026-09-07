# Prepare and execute GPU operations

A runtime prepares a fixed GPU implementation for an operation. Preparation checks its inputs, chooses a backend, compiles or loads the code and records its identity. Execution checks the supplied buffers and enqueues that prepared code on the chosen stream.

The caller allocates every input and output. This makes memory use predictable and lets several operations share intermediate buffers without copying them. Preparation does not change the caller's tensors.

```python
import torch
from aiter.runtime import Runtime

runtime = Runtime(device=0)
x = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device="cuda", dtype=x.dtype)
out = torch.empty_like(x)

plan = runtime.prepare_rmsnorm(x, weight, out, backend="hip")
plan.execute({"x": x, "weight": weight, "out": out})
print(plan.explain())
```

`explain()` identifies the operation, GPU, provider, exact executable digest, workspace requirement and pinned tuning manifest. A native plan also reports the library path. Returning from `execute()` means work was enqueued; the caller controls when to wait for completion.

## Connect quantization to matrix multiplication

The quantization operation produces the activation format expected by ordinary FP8 block scale GEMM. Both plans can use the same output and scale buffers.

```python
# gfx950 uses E4M3FN. gfx942 uses E4M3FNUZ for these prepared leaves.
m, n, k = 32, 256, 512
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
x_scale = torch.empty(m, k // 128, device="cuda", dtype=torch.float32)

# W and its scales would normally come from loading a quantized checkpoint.
w_fp8 = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
w_scale = torch.ones(n // 128, k // 128, device="cuda", dtype=torch.float32)
y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

quantize = runtime.prepare_quantize(x, x_fp8, x_scale, backend="triton")
gemm = runtime.prepare_gemm(x_fp8, w_fp8, x_scale, w_scale, y, backend="ck")

quantize.execute({"x": x, "out": x_fp8, "scales": x_scale})
gemm.execute({"x": x_fp8, "w": w_fp8, "x_scale": x_scale,
              "w_scale": w_scale, "out": y})
```

Calls on one stream execute in order. An explicit `stream=torch.cuda.Stream(...)` selects another stream; the caller establishes dependencies between streams. The plans can be captured together in a CUDA/HIP graph after preparation. Keep the buffers and plans alive through execution and every graph replay. Concurrent invocations may share a plan and read-only inputs, but need separate output buffers.

## Connect rotary embedding to attention

Rotary embedding and dense attention share the `[sequence,batch,heads,channels]` layout. The rotated query buffer can therefore feed attention directly. Apply a separate rotary plan to keys when the model requires it.

```python
q = torch.randn(33, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(33, 1, 2, 64, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
angles = torch.randn(33, 1, 1, 32, device="cuda", dtype=torch.float32)
q_rotated, k_rotated = torch.empty_like(q), torch.empty_like(k)
values = torch.empty_like(q)
lse = torch.empty(1, 4, 33, device="cuda", dtype=torch.float32)

rotate_q = runtime.prepare_rope(q, angles, q_rotated, backend="triton")
rotate_k = runtime.prepare_rope(k, angles, k_rotated, backend="triton")
attention = runtime.prepare_attention(
    q_rotated, k_rotated, v, values, lse, causal=True, backend="triton")

rotate_q.execute({"x": q, "freqs": angles, "out": q_rotated})
rotate_k.execute({"x": k, "freqs": angles, "out": k_rotated})
attention.execute({"q": q_rotated, "k": k_rotated, "v": v,
                   "out": values, "lse": lse})
```

Each pair of query heads shares one key/value head in this example. Attention computes `softmax(Q @ K.T / sqrt(head_dim)) @ V`; an explicit `scale=` replaces the default scale. `lse` contains the natural logarithm of each score row's exponential sum. The causal mask includes the query's own position. This operation does not create or update a KV cache.

## Use ordinary MXFP4 on gfx950

MXFP4 stores two E2M1 values in each byte and one E8M0 scale per group of 32 values. The even element uses the low nibble; the odd element uses the high nibble. These prepared operations use ordinary row-major packed storage. Shuffled weights and expert tensors have different interfaces.

```python
m, n, k = 16, 32, 256
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
x_packed = torch.empty(m, k // 2, device="cuda", dtype=torch.uint8)
w_packed = torch.empty(n, k // 2, device="cuda", dtype=torch.uint8)
x_scale = torch.empty(m, k // 32, device="cuda", dtype=torch.uint8)
w_scale = torch.empty(n, k // 32, device="cuda", dtype=torch.uint8)
y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

qx = runtime.prepare_mxfp4_quantize(x, x_packed, x_scale, backend="triton")
qw = runtime.prepare_mxfp4_quantize(w, w_packed, w_scale, backend="triton")
gemm = runtime.prepare_mxfp4_gemm(
    x_packed, w_packed, x_scale, w_scale, y, backend="triton")

qx.execute({"x": x, "out": x_packed, "scales": x_scale})
qw.execute({"x": w, "out": w_packed, "scales": w_scale})
gemm.execute({"x": x_packed, "w": w_packed, "x_scale": x_scale,
              "w_scale": w_scale, "out": y})
```

`K` must be divisible by 32. Quantization accepts finite FP16/BF16/FP32 input and preserves the specified exponent rounding and E2M1 packing. GEMM accumulates in FP32 and writes FP16 or BF16. Tests check packed bytes independently, including rounding ties, signed zero and extreme finite values. The vLLM bridge tests observe its dynamic-quantization call reaching the same AITER Triton leaf; they also account for vLLM's transposed scale argument.

## Choose and control an implementation

| Operation | Prepared providers | Tensor interface |
| --- | --- | --- |
| `RMSNorm` | HIP, Triton | `x[M,N]`, `weight[N]`, `out[M,N]` |
| `FP8BlockScaleGemm` | CK, Triton, Gluon | `x[M,K]`, `w[N,K]`, ordinary FP32 scales, `out[M,N]` |
| `BlockScaleQuantize` | Triton | `x[M,K]`, `out[M,K]`, `scales[M,ceil(K/128)]` |
| `MXFP4Quantize` | Triton on gfx950 | `x[M,K]`, packed `out[M,K/2]`, E8M0 `scales[M,K/32]` |
| `MXFP4Gemm` | Triton on gfx950 | packed `x[M,K/2]`, `w[N,K/2]`, row-group E8M0 scales, `out[M,N]` |
| `RotaryEmbedding` | Triton | `x[S,B,H,D]`, `freqs[S,1,1,D/2]`, matching output |
| `DenseAttention` | Triton | `q[Sq,B,Hq,D]`, `k/v[Sk,B,Hkv,D]`, matching query output, `lse[B,Hq,Sq]` |

These are explicit inference operations. `runtime.capabilities(description)` explains each provider's shape, dtype and target restrictions before compilation. This is code eligibility; preparation separately checks that a usable artifact can be loaded or built. CK currently uses the existing default tile with dimensions divisible by 16, 128 and 256 respectively; Triton and Gluon also handle tested masked GEMM tails. Gluon and prepared MXFP4 require gfx950. Other eligible prepared providers describe gfx942 and gfx950 support, with local GPU validation performed on gfx950. Rotary embedding supports full-head `neox` and `gptj` rotation for power-of-two head widths from 4 through 256. Dense attention supports FP16/BF16 head widths 16, 32, 64 and 128. Causal attention requires equal query and key sequence lengths.

Use `aiter.api` when metadata must exist before Torch or a GPU is initialized. `TensorSpec` describes shape, element strides, dtype and layout; each operation declares its named inputs and outputs through the `Operation` interface. The examples use ordinary contiguous storage; RMSNorm also accepts its declared padded input row stride. A preshuffled weight buffer needs its own operation and cannot be relabeled as ordinary storage.

`ExecutionPolicy(allow_compile=False)` prevents preparation from invoking a compiler. HIP and CK first use the native libraries bundled under `aiter/lib`, verifying their manifest, bytes, GPU target and ABI. Explicit `AITER_RMSNORM_LIBRARY` or `AITER_CK_BLOCKSCALE_LIBRARY` paths override that selection. Without a bundled SDK, HIP can use a prebuilt ABI-compatible library from `AITER_JIT_DIR`. A present but broken bundle fails instead of silently rebuilding. Triton and Gluon require compilation permission during preparation. After a plan is prepared, its execution never compiles, selects another provider, allocates GPU memory or synchronizes.

When an intact native bundle targets another GPU, compilation permission allows preparation to build the requested target from source. Local native build caches track declared sources, headers, compiler identity and build inputs; HIP also records the resulting binary digest in a receipt. A changed source or replaced binary invalidates reuse. These receipts support local cache correctness; a released artifact still needs the separate build and qualification evidence.

An optional `DispatchManifest` pins an approved provider and artifact for each exact operation and target. `Runtime(manifest=manifest)` checks that the recorded environment matches the observed dependencies and GPU profile. A missing selection, conflicting backend or different executable digest fails preparation. Updating tuning files cannot change a running plan.

## How the runtime is assembled

The runtime uses ports and adapters. Pure operation descriptions live in `aiter.api`; they do not import Torch, compiler code or providers. The immutable semantic registry references the actual descriptor classes, so domain discovery and their operation identifiers stay connected.

`runtime.composition.default_backends()` constructs a private adapter inventory for each runtime from `runtime.configuration`. `Runtime` works through the provider interface; it does not import concrete backend implementations. This is the one place that assembles the default prepared application. Legacy JIT and wheel building have their own focused composition points because they have different lifecycles.

Each Triton backend declares an immutable table of operation implementations. The same entry provides capability checks and preparation, preventing a supported operation from taking an unrelated fallback path. Domain code lives under `backends/triton/`: matrix multiplication, normalization, quantization, position transforms and attention have separate modules. HIP and CK share `backends/native/artifacts.py` for override/bundle/cache selection, target and ABI validation, and pinning exact executable bytes. The immutable file-pinning primitive is shared with the legacy JIT loader so replacement builds cannot reuse a resident library under a new identity. Their provider modules own the distinct native call signatures.

## Add another operator domain

A caller can pass `Runtime(backends=(MyBackend(),), policy=ExecutionPolicy(backend_order=("my-backend",)))` when the adapter declares `name = "my-backend"`. Each runtime keeps its own adapter inventory. Names are explicit, and a policy cannot enable an adapter that was not registered.

A descriptor implements `inputs()`, `outputs()`, `to_dict()` and `fingerprint()`. A backend implements `supports()` and `prepare()`. Its prepared launcher owns executable code and accepts borrowed buffers plus a stream. New numerical, layout, capture and lifetime tests establish that the boundary works before a legacy entry point adopts it.

`aiter.api.operator_domains()` maps the existing public surface into matrix multiplication, normalization, quantization, position transforms, attention, routing, MoE, collectives, activations, sequence state and support utilities. Paged or stateful attention, MoE and collectives continue through their existing APIs. Their entries list the state, workspace, routing and communicator requirements that must become explicit before they can offer the same prepared lifecycle.

Within an `OperationBackend`, register a new descriptor with an `OperationImplementation(descriptor_type, supports, prepare)` entry. The table is immutable after construction and rejects duplicate descriptor types. A new descriptor subclass needs its own registration. A plugin may instead implement the provider interface directly and pass its adapter to `Runtime`; no global registration mutation is required.
