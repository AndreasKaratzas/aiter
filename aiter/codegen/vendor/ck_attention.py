# SPDX-License-Identifier: MIT
from . import run_ck


def main(argv=None, *, context=None):
    return run_ck(context, "example/ck_tile/01_fmha/generate.py", argv)
