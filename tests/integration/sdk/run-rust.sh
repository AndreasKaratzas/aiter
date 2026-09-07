#!/usr/bin/env bash
# Exercise the Rust frontend against an installed C SDK, outside the source tree.
set -euo pipefail
source_root=${AITER_CI_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
rust_work=$(mktemp -d "${TMPDIR:-/tmp}/aiter-rust-sdk.XXXXXXXX")
trap 'rm -rf "$rust_work"' EXIT

# An explicit prefix is useful for downstream SDK acceptance. CI passes no
# argument: its SDK is built from the exact candidate source being qualified.
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [installed-sdk-prefix]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  sdk_prefix=$(realpath "$1")
else
  sdk_prefix="$rust_work/sdk"
  cmake -S "$source_root" -B "$rust_work/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DAITER_BUILD_KERNELS=ON -DAITER_GPU_ARCH=gfx950 \
    -DCMAKE_HIP_COMPILER="${ROCM_PATH:-/opt/rocm}/lib/llvm/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX="$sdk_prefix"
  cmake --build "$rust_work/build" --parallel "${MAX_JOBS:-2}"
  cmake --install "$rust_work/build"
fi

# The consumer contains no AITER C++ source, editable installation or workspace
# dependency. The only native input is the installed SDK selected above.
mkdir "$rust_work/consumer"
cp "$source_root/bindings/rust/"{Cargo.toml,Cargo.lock,build.rs} "$rust_work/consumer/"
cp -R "$source_root/bindings/rust/"{src,examples} "$rust_work/consumer/"
export AITER_SDK_DIR="$sdk_prefix"
export CARGO_TARGET_DIR="$rust_work/target"
export LD_LIBRARY_PATH="$sdk_prefix/lib:$sdk_prefix/lib64:${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$rust_work/consumer"
cargo fmt --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo run --offline --locked --example operators
