# Native ragged attention check

The C++ consumer builds the production bridge from `csrc/cpp_itfs/pa`, then calls its generated FP16 attention kernel. It checks two sequences with different lengths, permuted pages and eight query heads sharing one KV head. Both caches use the actual `[block, KV head, token, dimension]` layout; every output value is compared with an independent CPU softmax calculation over the same FP16 inputs.

Use the Python environment that supplies AITER's build dependencies, a HIP compiler, and the system development packages for fmt and OpenSSL. Keep the build directory outside the source checkout:

```sh
HIP_VISIBLE_DEVICES=0 make -C tests/operators/hip/native/pa run \
  BUILD_DIR=/tmp/aiter-pa-check \
  PYTHON=/path/to/environment/bin/python \
  HIPCC='hipcc --offload-arch=gfx950'
```

The executable, bridge library and generated kernels stay under `BUILD_DIR`. The launcher selects the Python package through `AITER_PYTHON_ROOT`, which defaults to this checkout, and uses the selected interpreter's site-packages. It does not alter Python's search path inside the C++ program. Use a new build directory when changing the compiler, Python environment or target GPU.

The previous C++ test lived beside the production source. It passed float buffers to an FP16 interface, omitted the last-page lengths and had no numerical assertions. This replacement also exposed and corrected the bridge's stale generated-function argument order. It is a bounded native-interface regression, not a complete attention or framework qualification.
