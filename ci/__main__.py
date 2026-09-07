"""Plan, run and verify AITER tests with the same commands locally and in CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ci.common.json import load_json, require, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.plan import plan_tests
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from ci.release.wheels import collect_source_identity, inspect_wheel

APPLICATIONS = {
    "architecture": "ci.architecture.__main__",
    "pipelines": "ci.pipelines.__main__",
    "channels": "ci.release.channels",
    "metrics": "ci.release.metrics",
}


def main(argv=None):
    selected = sys.argv[1:] if argv is None else argv
    if selected and selected[0] in APPLICATIONS:
        from importlib import import_module

        return import_module(APPLICATIONS[selected[0]]).main(selected[1:])
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Applications: architecture, pipelines, channels, metrics. "
        "Use 'python -m ci APPLICATION --help' for their commands.",
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--controls-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show profiles and executable test groups")
    commands.add_parser("validate", help="validate catalog references and test paths")
    coverage = commands.add_parser(
        "coverage", help="show test selectors, profile membership and unselected files"
    )
    coverage.add_argument("--client")
    coverage.add_argument("--format", choices=("markdown", "json"), default="markdown")
    coverage.add_argument("--output", type=Path)
    plan = commands.add_parser(
        "plan", help="select exact test groups for a client/profile"
    )
    plan.add_argument("--profile", default="product-fast")
    plan.add_argument(
        "--architecture", choices=("gfx942", "gfx950", "gfx1250"), default="gfx950"
    )
    plan.add_argument("--changed-paths", type=Path)
    plan.add_argument("--wheel", type=Path, action="append", default=[])
    plan.add_argument("--environment-lock", type=Path)
    plan.add_argument("--executor-image")
    plan.add_argument("--output", required=True, type=Path)
    run = commands.add_parser(
        "run", help="execute a plan and preserve logs and attempts"
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--gpus", default="")
    run.add_argument("--group", action="append")
    run.add_argument("--wheel-dir", type=Path)
    check = commands.add_parser(
        "check", help="require complete passing execution evidence"
    )
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--results", type=Path, required=True)
    check.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        catalog = (
            load_catalog(args.catalog)
            if args.catalog is not None
            else load_catalog(root=args.controls_root)
        )
        if args.command == "list":
            for name, profile in catalog["profiles"].items():
                print(f"{name}: {profile['description']}")
                print("  " + ", ".join(profile["groups"]))
            return 0
        if args.command == "coverage":
            import json

            from ci.qualification.coverage import render_inventory, selection_inventory

            inventory = selection_inventory(
                catalog, args.controls_root, client=args.client
            )
            rendered = (
                json.dumps(inventory, indent=2) + "\n"
                if args.format == "json"
                else render_inventory(inventory)
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered)
            else:
                print(rendered, end="")
            return 0
        if args.command == "validate":
            from ci.ownership.policy import load_policy, validate_routing

            validate_routing(
                args.source_root,
                load_policy(args.controls_root / "ci/ownership/owners.json"),
            )
            for group in catalog["groups"].values():
                for target in group["targets"]:
                    require(
                        (args.controls_root / target.split("::", 1)[0]).exists(),
                        f"test target does not exist: {target}",
                    )
                if "model_manifest" in group:
                    manifest = args.controls_root / group["model_manifest"]
                    require(
                        manifest.is_file()
                        and not manifest.is_symlink()
                        and manifest.resolve().is_relative_to(
                            args.controls_root.resolve()
                        ),
                        "reviewed model manifest is missing or escapes controls: "
                        + group["model_manifest"],
                    )
            for path in catalog["retained_workflows"]:
                require(
                    (args.controls_root / path).is_file(),
                    f"retained workflow is missing: {path}",
                )
            print(
                f"Validated {len(catalog['groups'])} groups and {len(catalog['profiles'])} profiles."
            )
        elif args.command == "plan":
            paths = (
                args.changed_paths.read_text().splitlines()
                if args.changed_paths
                else []
            )
            artifacts = [inspect_wheel(path)[0].to_dict() for path in args.wheel]
            output = plan_tests(
                catalog,
                args.profile,
                paths,
                collect_source_identity(args.source_root).to_dict(),
                artifacts=artifacts,
                architecture=args.architecture,
                environment_lock=(
                    load_json(args.environment_lock) if args.environment_lock else None
                ),
                executor_image=args.executor_image,
                control_source=collect_source_identity(args.controls_root).to_dict(),
            )
            write_json(args.output, output)
            print(
                f"{args.profile}: {len(output['groups'])} groups, {output['plan_digest']}"
            )
        elif args.command == "run":
            results = run_plan(
                load_json(args.plan),
                catalog,
                args.source_root,
                args.output_dir,
                python=args.python,
                gpus=args.gpus,
                groups=args.group,
                wheel_dir=args.wheel_dir,
                controls_root=args.controls_root,
            )
            return 0 if all(result["status"] == "PASS" for result in results) else 1
        else:
            result = check_results(load_json(args.plan), catalog, args.results)
            if args.output:
                write_json(args.output, result)
            print(
                f"{result['profile']}: {result['status']} ({result['groups']} groups, {result['attempts']} attempts)"
            )
            for problem in result["problems"]:
                print("  " + problem)
            return 0 if result["status"] == "PASS" else 1
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"ci: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
