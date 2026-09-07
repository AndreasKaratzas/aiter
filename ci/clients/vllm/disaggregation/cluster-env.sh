#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Sourced on each compute node; wheel extraction happens on the login node.
overlay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aiter_overlay"
source "$(dirname "${BASH_SOURCE[0]}")/cluster.sh"
export GPU_ARCHS=gfx950
export VLLM_ROCM_USE_AITER=1
export PYTHONPATH="${overlay_root}${PYTHONPATH:+:${PYTHONPATH}}"
python3 - "${overlay_root}" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
import aiter
import flydsl

root = Path(sys.argv[1]).resolve()
for name, module in (("aiter", aiter), ("flydsl", flydsl)):
    loaded = Path(module.__file__).resolve()
    if root not in loaded.parents:
        raise SystemExit(f"{name} was not loaded from the admitted overlay {root}")
    print(f"[aiter-overlay] {name}: {loaded}")
print(f"[aiter-overlay] flydsl version: {version('flydsl')}")
PY
