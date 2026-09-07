"""Compose an OCI build context from reviewed recipes and one verified candidate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ci.common.json import require
from ci.release.wheels import collect_source_identity, load_receipt, verify_wheel


def stage_context(source: Path, controls: Path, wheel: Path, output: Path) -> None:
    source, controls, wheel, output = (
        value.resolve() for value in (source, controls, wheel, output)
    )
    require(not output.exists(), "image context must be a new directory")
    control_identity = collect_source_identity(controls).to_dict()
    receipt = load_receipt(str(wheel) + ".receipt.json")
    verify_wheel(wheel, receipt, require_clean_source=True)
    require(
        collect_source_identity(source).to_dict() == receipt.source.to_dict(),
        "candidate source differs from the wheel receipt",
    )
    require(
        output != source
        and source not in output.parents
        and output != controls
        and controls not in output.parents,
        "image context must be outside both checkouts",
    )
    for root, relative in (
        (source, "include"),
        (controls, "ci"),
        (controls, "tests"),
        (controls, "docker"),
    ):
        directory = root / relative
        require(directory.is_dir(), "missing image context input: " + str(directory))
        require(
            all(path.resolve().is_relative_to(root) for path in directory.rglob("*")),
            "image context input escapes its checkout",
        )
    output.mkdir(parents=True)
    for root, relative in (
        (source, "include"),
        (controls, "ci"),
        (controls, "tests"),
        (controls, "docker"),
    ):
        shutil.copytree(
            root / relative,
            output / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    (output / "dist").mkdir()
    shutil.copy2(wheel, output / "dist" / wheel.name)
    require(
        collect_source_identity(source).to_dict() == receipt.source.to_dict(),
        "candidate changed while composing image context",
    )
    require(
        collect_source_identity(controls).to_dict() == control_identity,
        "controls changed while composing image context",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "controls", "wheel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    stage_context(args.source, args.controls, args.wheel, args.output)


if __name__ == "__main__":
    main()
