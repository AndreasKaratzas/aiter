# SPDX-License-Identifier: MIT
from . import run_ck


def main(argv=None, *, context=None):
    return run_ck(context, "example/ck_tile/50_sparse_attn/generate.py", argv)
