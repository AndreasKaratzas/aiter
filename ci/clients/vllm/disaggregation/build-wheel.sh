#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Executed inside the selected immutable image, in the candidate workspace.
set -euo pipefail
git config --global --add safe.directory /workspace
shopt -s nullglob
rm -rf dist build aiter_meta ./*.egg-info
rm -f .aiter-build.log
python3 -c "from importlib.metadata import version; import flydsl; flydsl_version = version(\"flydsl\"); print(f\"Unmodified runtime dependency: flydsl={flydsl_version} at {flydsl.__file__}\")"
pip uninstall -y aiter amd-aiter || true
pip config set global.default-timeout 60
pip config set global.retries 10
pip install -r requirements/test/product.txt
pip install --upgrade -r requirements/test/legacy-overrides.txt
pip install --upgrade -r requirements/build/legacy.txt
flydsl_version="$(python3 -c "from importlib.metadata import version; print(version(\"flydsl\"))")"
echo "Packaging runtime dependency flydsl==${flydsl_version}"
python3 -m pip download \
  --no-deps \
  --only-binary=:all: \
  --dest dist \
  "flydsl==${flydsl_version}"
echo "Building AITER wheel for GPU_ARCHS=${GPU_ARCHS}, PREBUILD_KERNELS=${PREBUILD_KERNELS}, PREBUILD_MODULES=${PREBUILD_MODULES}, MAX_JOBS=${MAX_JOBS}"
python setup.py bdist_wheel 2>&1 | tee .aiter-build.log
ls -lh dist/*.whl
PYTHONPATH=/control python3 -m ci.clients.vllm.disaggregation.artifacts dist
