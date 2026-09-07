"""Build immutable release records from verified wheels and complete test runs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import require_profile_environment
from ci.qualification.report import check_results
from ci.release.notes import read_reviewed_notes, validate_notes
from ci.release.wheels import load_receipt, verify_wheel


def release_manifest(
    wheel_dir: Path,
    runs: list[Path],
    *,
    channel: str,
    notes: str,
    required_cells: list[str],
    catalog: dict,
    notes_root: Path | None = None,
    required_environments: dict[str, str] | None = None,
) -> dict:
    require(
        channel in ("nightly-candidate", "nightly-tested", "stable"),
        "unknown release channel",
    )
    require(bool(notes.strip()), "release notes are required")
    require(
        bool(required_cells) and len(required_cells) == len(set(required_cells)),
        "declare distinct required support cells",
    )
    if channel != "nightly-candidate":
        require(
            isinstance(required_environments, dict)
            and set(required_environments) == set(required_cells),
            "each support cell requires its approved environment lock digest",
        )
    wheels = []
    versions = set()
    source = None
    for path in sorted(wheel_dir.glob("*.whl")):
        receipt = load_receipt(path.with_name(path.name + ".receipt.json"))
        verify_wheel(path, receipt, require_clean_source=True)
        if source is None:
            source = receipt.source.revision
        require(
            source == receipt.source.revision,
            "release contains multiple source revisions",
        )
        wheels.append(receipt.artifact.to_dict())
        if channel == "stable":
            versions.add(receipt.wheel.version.split("+", 1)[0])
    require(bool(wheels), "release contains no wheels")
    reviewed_notes = None
    if channel == "stable":
        require(len(versions) == 1, "stable wheels disagree on their release version")
        version = next(iter(versions))
        reviewed_notes = validate_notes(notes, version)
        curated = read_reviewed_notes(notes_root or Path.cwd(), version, source)
        require(
            notes == curated,
            "stable notes differ from the reviewed file at the released source revision",
        )
    bindings = {wheel["filename"]: wheel for wheel in wheels}
    for cell in required_cells:
        require(
            isinstance(cell, str) and "|" in cell,
            "support cells must name the wheel artifact",
        )
        require(
            cell.split("|", 1)[0] in bindings,
            "support cell refers to an unpublished wheel",
        )
    reports, observed, tested = [], set(), set()
    flydsl_required = {}
    for directory in runs:
        plan = load_json(directory / "plan.json")
        require(
            plan["source"]["revision"] == source,
            "test source differs from wheel source",
        )
        require(
            len(plan["artifacts"]) == 1,
            "each installed-wheel run must qualify exactly one wheel",
        )
        require(
            plan["selection"] == "complete-profile",
            "release requires complete-profile qualification",
        )
        require(
            set(plan["groups"]) | set(plan["not_applicable"])
            == set(catalog["profiles"][plan["profile"]]["groups"]),
            "release profile is missing declared groups",
        )
        lock = plan.get("environment_lock")
        require(
            isinstance(lock, dict),
            "qualified artifacts require an approved environment lock",
        )
        require_profile_environment(
            lock, catalog["profiles"][plan["profile"]]["client"], supported=True
        )
        require(
            plan.get("environment_lock_digest") == digest(lock),
            "qualification environment lock digest mismatch",
        )
        require(
            isinstance(plan.get("executor_image"), str),
            "qualification did not record the executor image",
        )
        report = check_results(plan, catalog, directory)
        require(report["status"] == "PASS", f"test evidence failed: {directory}")
        for artifact in plan["artifacts"]:
            require(
                bindings.get(artifact["filename"]) == artifact,
                "tested wheel differs from published wheel",
            )
            tested.add(artifact["filename"])
        for result_file in directory.glob("*/attempt-*/result.json"):
            result = load_json(result_file)
            environment = result["environment"]
            if (
                not environment.get("devices")
                or environment.get("qualification_kind") == "native-sdk-source"
            ):
                continue
            require(
                environment.get("import_mode") == "wheel",
                "test did not verify installed-wheel imports",
            )
            python = ".".join(environment["python"].split(".")[:2])
            for device in environment["devices"]:
                rocm = ".".join(str(environment["hip"]).split(".")[:2])
                cell = f"{artifact['filename']}|{plan['profile']}:py{python}:{device['architecture']}:rocm{rocm}"
                if cell in required_cells:
                    require(
                        required_environments[cell] == plan["environment_lock_digest"],
                        "observed cell ran in an environment other than its approved requirement",
                    )
                observed.add(cell)
        if (
            plan["profile"] == "product-nightly"
            and lock["packages"]["flydsl"] is not None
        ):
            flydsl_required.setdefault(artifact["filename"], set()).add(
                plan["environment_lock_digest"]
            )
        reports.append(
            {
                "plan_digest": plan["plan_digest"],
                "report_digest": report["report_digest"],
                "profile": plan["profile"],
                "environment_id": lock["id"],
                "environment_lock_digest": plan["environment_lock_digest"],
                "executor_image": plan["executor_image"],
                "artifact": artifact,
                "architecture": plan["architecture"],
                "not_applicable": plan["not_applicable"],
            }
        )
    if channel != "nightly-candidate":
        require(
            tested == set(bindings),
            "some published wheels have no passing installed-wheel tests",
        )
        require(
            set(required_cells) <= observed,
            "missing required support cells: "
            + ", ".join(sorted(set(required_cells) - observed)),
        )
    if channel != "nightly-candidate":
        required_profiles = {"product-nightly", "pytorch", "vllm", "sglang"}
        require(
            required_profiles <= {record["profile"] for record in reports},
            "qualified channels require full product and every supported client profile",
        )
        for filename in bindings:
            declared_profiles = {
                cell.split("|", 1)[1].split(":", 1)[0]
                for cell in required_cells
                if cell.startswith(filename + "|")
            }
            require(
                {"product-nightly", "pytorch"} <= declared_profiles,
                "each qualified wheel requires full product and PyTorch support cells",
            )
            if filename in flydsl_required:
                require(
                    "flydsl" in declared_profiles,
                    "declared FlyDSL support requires its artifact-bound qualification cell",
                )
                require(
                    flydsl_required[filename]
                    <= {
                        required_environments[cell]
                        for cell in required_cells
                        if cell.startswith(filename + "|flydsl:") and ":gfx950:" in cell
                    },
                    "FlyDSL qualification must use the same approved product dependency lock",
                )
        require(
            {"vllm", "sglang"}
            <= {cell.split("|", 1)[1].split(":", 1)[0] for cell in required_cells},
            "qualified client support must be declared before qualification",
        )
    manifest = {
        "schema_version": 1,
        "channel": channel,
        "source_revision": source,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": wheels,
        "required_cells": sorted(required_cells),
        "required_environments": required_environments,
        "observed_cells": sorted(observed),
        "qualification": reports,
        "release_notes": notes,
        "reviewed_notes": reviewed_notes,
    }
    manifest["release_digest"] = digest(manifest)
    return manifest


def revalidate_release(original: dict, reconstructed: dict) -> dict:
    """Keep the original immutable timestamp when retrying publication."""
    require(
        original.get("release_digest")
        == digest({k: v for k, v in original.items() if k != "release_digest"}),
        "release record digest mismatch",
    )
    omit = {"created_utc", "release_digest"}
    require(
        {k: v for k, v in original.items() if k not in omit}
        == {k: v for k, v in reconstructed.items() if k not in omit},
        "release evidence changed since qualification",
    )
    return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, default=[])
    parser.add_argument(
        "--channel",
        required=True,
        choices=("nightly-candidate", "nightly-tested", "stable"),
    )
    parser.add_argument("--required-cell", action="append", required=True)
    parser.add_argument(
        "--requirements", type=Path, help="frozen ci.release.matrix requirements"
    )
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = release_manifest(
            args.wheel_dir,
            args.run,
            channel=args.channel,
            notes=args.notes.read_text(),
            required_cells=args.required_cell,
            catalog=load_catalog(),
            required_environments=(
                {
                    item["cell"]: item["environment_lock_digest"]
                    for item in load_json(args.requirements)["include"]
                }
                if args.requirements
                else None
            ),
        )
        if args.output.exists():
            require(
                load_json(args.output) == manifest,
                "release record already exists; refusing replacement",
            )
        else:
            write_json(args.output, manifest)
        print(manifest["release_digest"])
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"release: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
