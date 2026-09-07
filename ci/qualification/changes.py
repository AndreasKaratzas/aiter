"""Collect both sides of renamed paths; uncertain diffs request full coverage."""

import argparse
import re
import subprocess
from pathlib import Path

from ci.qualification.catalog import relative_path


def changed_paths(root: Path, base: str) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{40}", base) or set(base) == {"0"}:
        return []
    exists = (
        subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", base + "^{commit}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not exists:
        return []
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            base,
            "HEAD",
            "--",
        ]
    )
    try:
        paths = [path for path in raw.decode("utf-8").split("\0") if path]
        for path in paths:
            relative_path(path)
        return sorted(set(paths))
    except (ValueError, UnicodeError):
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = changed_paths(args.source_root, args.base)
    args.output.write_text("".join(path + "\n" for path in paths))


if __name__ == "__main__":
    main()
