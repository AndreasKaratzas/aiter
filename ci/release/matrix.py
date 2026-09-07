"""Resolve changed capabilities and wheel artifacts into approved executor cells."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from ci.common.json import digest, require, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import load_registry, require_profile_environment
from ci.qualification.plan import plan_tests
from ci.release.wheels import collect_source_identity, load_receipt, verify_wheel


def executor(profile: str, architecture: str, lock: dict) -> dict:
    client = (
        "aiter"
        if profile
        in (
            "wheel",
            "product-fast",
            "product-nightly",
            "product-extended",
            "rust-sdk",
            "flydsl",
        )
        else profile
    )
    require_profile_environment(lock, client, supported=True)
    if profile == "flydsl":
        require(
            lock["packages"]["flydsl"] is not None,
            "FlyDSL executor must declare its dependency",
        )
    gpus = (
        "0"
        if profile in ("wheel", "rust-sdk", "flydsl") or architecture == "gfx942"
        else (
            "0,1,2,3,4,5,6,7"
            if profile in ("product-nightly", "product-extended")
            else "0,1"
        )
    )
    return {
        "profile": profile,
        "image": lock["image"],
        "environment_lock": json.dumps(lock, sort_keys=True, separators=(",", ":")),
        "environment_id": lock["id"],
        "environment_lock_digest": digest(lock),
        "architecture": architecture,
        "runner": (
            "linux-aiter-oci-mi300x-1"
            if architecture == "gfx942"
            else "linux-aiter-do-mi350x-8"
        ),
        "gpus": gpus,
    }


def change_matrix(
    catalog: dict,
    paths: list[str],
    source: dict,
    registry: dict[str, dict],
    *,
    product_profile: str = "product-fast",
) -> dict:
    entries = []
    require(
        product_profile in ("product-fast", "product-extended"),
        "unknown product cadence",
    )
    profiles = [product_profile, "pytorch", "vllm", "sglang", "rust-sdk"]
    if registry.get("rocm72-py312", {}).get("packages", {}).get("flydsl") is not None:
        profiles.append("flydsl")
    for profile in profiles:
        candidate = plan_tests(catalog, profile, paths, source)
        if not any(group["gpus"] for group in candidate["groups"].values()):
            continue
        if (
            profile in ("vllm", "sglang")
            and f"{profile}-bridge" not in candidate["groups"]
        ):
            continue
        key = profile if profile in ("vllm", "sglang") else "rocm72-py312"
        require(
            key in registry,
            f"configure approved support environment {key} for affected {profile}",
        )
        lock = registry[key]
        for arch in (
            ("gfx942", "gfx950") if profile == "product-fast" else ("gfx950",)
        ):
            plan = plan_tests(
                catalog,
                profile,
                paths,
                source,
                architecture=arch,
                environment_lock=lock,
            )
            if not any(group["gpus"] for group in plan["groups"].values()):
                continue
            entry = executor(profile, arch, lock)
            entry.update(
                label=f"{profile}-{arch}",
                source_revision=source["revision"],
                changed_paths=json.dumps(paths),
            )
            entries.append(entry)
    return {"include": entries}


def qualification_matrix(
    wheel_dir: Path, registry: dict[str, dict], *, channel: str = "stable"
) -> dict:
    require(channel in ("stable", "nightly"), "unknown qualification cadence")
    entries = []
    optional_cells = 0
    for wheel in sorted(wheel_dir.glob("*.whl")):
        receipt = load_receipt(str(wheel) + ".receipt.json")
        verify_wheel(wheel, receipt, require_clean_source=True)
        match = re.search(r"\+rocm(7\.[012])\.", receipt.wheel.version)
        require(
            match is not None or channel == "nightly",
            f"wheel has no supported ROCm tuple: {wheel.name}",
        )
        rocm = match.group(1) if match else "7.2"
        pythons = {tag.split("-")[0] for tag in receipt.wheel.tags}
        require(
            len(pythons) == 1 and pythons <= {"cp310", "cp312"},
            "unexpected wheel Python ABI",
        )
        python = "3.10" if pythons == {"cp310"} else "3.12"
        key = f"rocm{rocm.replace('.', '')}-py{python.replace('.', '')}"
        require(key in registry, f"configure approved support environment for {key}")
        lock = registry[key]
        require(
            lock["python"] == python and lock["rocm"] == rocm,
            "runtime lock differs from the wheel support tuple",
        )
        for arch in ("gfx942", "gfx950"):
            profiles = ["wheel", "product-nightly", "pytorch"]
            if arch == "gfx950" and lock["packages"]["flydsl"] is not None:
                profiles.append("flydsl")
                optional_cells += 1
            for profile in profiles:
                entry = executor(profile, arch, lock)
                entry.update(
                    wheel=wheel.name,
                    label=f"{key}-{arch}-{profile}",
                    cell=f"{wheel.name}|{profile}:py{python}:{arch}:rocm{rocm}",
                    source_revision=receipt.source.revision,
                )
                entries.append(entry)
        if python == "3.12" and rocm == "7.2":
            for profile in ("vllm", "sglang"):
                require(
                    profile in registry,
                    f"configure approved {profile} support environment",
                )
                client_lock = registry[profile]
                require(
                    client_lock["python"] == python and client_lock["rocm"] == rocm,
                    f"{profile} lock differs from the primary wheel tuple",
                )
                entry = executor(profile, "gfx950", client_lock)
                entry.update(
                    wheel=wheel.name,
                    label=profile,
                    cell=f"{wheel.name}|{profile}:py{python}:gfx950:rocm{rocm}",
                    source_revision=receipt.source.revision,
                )
                entries.append(entry)
    require(
        len(entries) == (38 if channel == "stable" else 14) + optional_cells,
        "stable requires six wheels and 38 base cells; nightly requires two wheels and 14 complete product/client base cells, plus declared FlyDSL cells",
    )
    require(
        len({entry["label"] for entry in entries}) == len(entries),
        "duplicate qualification cell",
    )
    return {"include": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel", choices=("stable", "nightly", "changes"), default="stable"
    )
    parser.add_argument(
        "--product-profile",
        choices=("product-fast", "product-extended"),
        default="product-fast",
    )
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--changed-paths", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(
        os.environ.get("AITER_SUPPORT_PROFILES") or "{}",
        allow_empty=args.channel == "changes",
    )
    if args.channel == "changes":
        matrix = change_matrix(
            load_catalog(),
            args.changed_paths.read_text().splitlines() if args.changed_paths else [],
            collect_source_identity(args.source_root).to_dict(),
            registry,
            product_profile=args.product_profile,
        )
    else:
        require(args.wheel_dir is not None, "wheel matrix requires --wheel-dir")
        matrix = qualification_matrix(args.wheel_dir, registry, channel=args.channel)
    write_json(args.output, matrix)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
            output.write("has_jobs=" + str(bool(matrix["include"])).lower() + "\n")


if __name__ == "__main__":
    main()
