#!/usr/bin/env bash
# Activate the optional manylinux compiler environment, then execute exact argv.
set -euo pipefail
if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
    source /opt/rh/gcc-toolset-13/enable 2>/dev/null || true
fi
export PATH="${AITER_BUILD_PYBIN:?Select the manylinux Python directory}:$PATH"
exec "$@"
