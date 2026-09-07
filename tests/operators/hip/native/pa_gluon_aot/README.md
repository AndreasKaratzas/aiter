# Gluon attention C++ consumer

The C++ consumer links `csrc/cpp_itfs/pa_gluon_aot`, embeds Python for kernel preparation, and checks the native cold and cached execution paths against an independent CPU softmax reference over the exact stored BF16 inputs. Its Python counterpart is kept in this directory.

Use a ROCm Torch environment with pybind11, a HIP compiler, and the system development packages for Python, fmt and OpenSSL:

```sh
HIP_VISIBLE_DEVICES=0 make -C tests/operators/hip/native/pa_gluon_aot run \
  BUILD_DIR=/tmp/aiter-pa-gluon-check \
  PYTHON=/path/to/environment/bin/python \
  HIPCC='hipcc --offload-arch=gfx950'
```

`BUILD_DIR` must be outside the checkout. Both cold and cached executions check all 6,144 values, including the different causal endpoints of three query positions. The Makefile obtains Torch headers and libraries, Python embedding flags and Python site-packages from the chosen interpreter. `AITER_PYTHON_ROOT` selects the package to import and defaults to this source tree. The launcher supplies that environment explicitly; the C++ binary contains no compiled checkout path and performs no `sys.path` mutation. Use a fresh build directory after changing the target GPU or toolchain.
