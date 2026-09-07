# SPDX-License-Identifier: MIT
"""Execute pinned vendor tools with their declared resource root."""

import subprocess
import sys


def run_ck(context, script, argv):
    path = context.resource("ck", script)
    subprocess.run(
        [sys.executable, str(path), *(argv or ())],
        cwd=path.parent,
        env=context.child_environment(),
        check=True,
    )
