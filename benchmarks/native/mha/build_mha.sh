#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
exec "${PYTHON:-python3}" -m benchmarks.native.mha.build "$@"
