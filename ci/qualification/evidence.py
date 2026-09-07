"""Keep bounded model execution evidence separate from disposable model copies."""

import hashlib
from pathlib import Path

from ci.common.json import load_json, require, write_json


def inventory(directory: Path) -> dict:
    require(
        directory.is_dir() and not directory.is_symlink(),
        "missing model execution evidence",
    )
    files = []
    total = 0
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "model evidence cannot contain symlinks")
        if path.is_dir():
            continue
        require(path.is_file(), "model evidence must contain ordinary files")
        size = path.stat().st_size
        total += size
        require(
            size <= 32 * 1024 * 1024 and total <= 128 * 1024 * 1024,
            "model evidence exceeds the bounded evidence budget; keep model views in pytest cache",
        )
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    require(bool(files), "model execution produced no retained evidence")
    return {"schema_version": 1, "files": files}


def seal(attempt: Path) -> None:
    write_json(attempt / "e2e-evidence.json", inventory(attempt / "e2e"))


def verify(attempt: Path) -> None:
    require(
        load_json(attempt / "e2e-evidence.json") == inventory(attempt / "e2e"),
        "model execution evidence changed after the run",
    )
