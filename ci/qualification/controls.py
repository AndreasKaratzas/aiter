"""Copy reviewed execution controls without exposing a product checkout to Python."""

import shutil
from pathlib import Path

CONTROL_DIRECTORIES = ("tests", ".github", "docker", "ci", "benchmarks", "requirements")


def copy_controls(source: Path, target: Path) -> Path:
    """Create a new suite using the same declared controls for source and wheel tests."""
    target.mkdir()
    for relative in CONTROL_DIRECTORIES:
        if (source / relative).is_dir():
            shutil.copytree(
                source / relative,
                target / relative,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.bin"),
            )
    return target
