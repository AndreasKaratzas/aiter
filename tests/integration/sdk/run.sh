#!/usr/bin/env bash
set -euo pipefail
source_root=${AITER_CI_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
native_build=$(mktemp -d "${TMPDIR:-/tmp}/aiter-native-ci.XXXXXXXX")
trap 'rm -rf "$native_build"' EXIT
cmake -S "$source_root" -B "$native_build" \
  -DCMAKE_BUILD_TYPE=Release -DAITER_BUILD_KERNELS=ON -DAITER_GPU_ARCH=gfx950 \
  -DCMAKE_HIP_COMPILER="${ROCM_PATH:-/opt/rocm}/lib/llvm/bin/clang++"
cmake --build "$native_build" --parallel "${MAX_JOBS:-2}"
ctest --test-dir "$native_build" --output-on-failure
"$native_build/aiter_native_tests" "$native_build/libaiter_rmsnorm_backend.so" "$native_build/libaiter_ck_backend.so"
python3 "$(dirname "${BASH_SOURCE[0]}")/test_library_reload.py" "$native_build/libaiter.so"
