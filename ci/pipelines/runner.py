"""Run the same declared delivery pipelines locally and from GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ci.common.json import load_json, parse_json, require
from ci.pipelines.images import compose_images
from ci.pipelines.nightly import WORKLOAD_PROFILES
from ci.pipelines.profile import execute_profile


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Platform applications share bootstrap and retained transports, while
    # keeping their own small typed command interfaces and area definitions.
    if argv and argv[0] == "sglang-downstream":
        from ci.clients.sglang.downstream import run_main

        return run_main(argv[1:])
    if argv and argv[0] == "product-legacy":
        from ci.pipelines.product import main as product_main

        return product_main(argv[1:])
    if argv and argv[0] == "vllm-disaggregation":
        from ci.clients.vllm.disaggregation.__main__ import main as disaggregation_main

        return disaggregation_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("profile", "images"):
        entry = sub.add_parser(command)
        for option in ("source", "controls", "output"):
            entry.add_argument("--" + option, type=Path, required=True)
        lock = entry.add_mutually_exclusive_group(required=True)
        lock.add_argument("--lock-file", type=Path)
        lock.add_argument("--lock-json")
    profile = sub.choices["profile"]
    profile.add_argument("--profile", required=True)
    profile.add_argument("--architecture", required=True)
    profile.add_argument("--gpus", required=True)
    profile.add_argument("--mode", choices=("source", "wheel"), default="source")
    profile.add_argument("--wheel-dir", type=Path)
    profile.add_argument("--wheel-name", default="")
    profile.add_argument("--image", required=True)
    profile.add_argument("--source-revision", default="")
    profile.add_argument("--base-revision", default="")
    images = sub.choices["images"]
    images.add_argument("--wheel", type=Path, required=True)
    images.add_argument(
        "--role", choices=("runtime", "development", "wheelhouse"), required=True
    )
    consumers = images.add_mutually_exclusive_group(required=True)
    consumers.add_argument("--consumers-file", type=Path)
    consumers.add_argument("--consumers-json")
    for command in ("workflows", "scripts"):
        navigation = sub.add_parser(command)
        navigation.add_argument("--check", action="store_true")
        navigation.add_argument("--write-index", action="store_true")
    canary = sub.add_parser("canary")
    canary.add_argument("--client", choices=("vllm", "sglang"), required=True)
    canary.add_argument("--case-json", required=True)
    canary.add_argument("--output", type=Path, required=True)
    canary.add_argument("--controls", type=Path)
    canary.add_argument("--wheel-dir", type=Path)
    canary.add_argument("--image")
    canary.add_argument("--container", default="ci_sglang")
    observe = sub.add_parser("canary-environments")
    observe.add_argument("--controls", type=Path, required=True)
    observe.add_argument("--output", type=Path, required=True)
    for command in ("vllm-nightly", "vllm-benchmark"):
        nightly = sub.add_parser(command)
        for option in ("source", "controls", "output"):
            nightly.add_argument("--" + option, type=Path, required=True)
        artifact = nightly.add_mutually_exclusive_group(required=True)
        artifact.add_argument("--wheel", type=Path)
        artifact.add_argument("--wheel-dir", type=Path)
        executor = nightly.add_mutually_exclusive_group(required=True)
        executor.add_argument("--image")
        executor.add_argument("--local", action="store_true")
        nightly.add_argument("--gpus", required=True)
    nightly = sub.choices["vllm-nightly"]
    nightly.add_argument("--variant")
    nightly.add_argument(
        "--workload-profile",
        choices=WORKLOAD_PROFILES,
        default="vllm-nightly",
    )
    nightly.add_argument(
        "--through", choices=("imports", "workloads"), default="workloads"
    )
    benchmark = sub.choices["vllm-benchmark"]
    benchmark.add_argument(
        "--benchmark-profile",
        choices=("baseline", "smoke", "throughput", "topology", "extended"),
        default="baseline",
    )
    benchmark.add_argument(
        "--benchmark-cases",
        help="Comma-separated exact case IDs within the selected profile",
    )
    args = parser.parse_args(argv)
    if args.command == "canary-environments":
        from ci.pipelines.canaries import resolve_environments

        matrix = resolve_environments(
            controls=args.controls,
            images=parse_json(os.environ["CANARY_IMAGES"]),
            output=args.output,
        )
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
                stream.write("matrix=" + json.dumps(matrix) + "\n")
        else:
            print(json.dumps(matrix))
        return
    if args.command in ("vllm-nightly", "vllm-benchmark"):
        wheel = args.wheel
        if args.wheel_dir is not None:
            wheels = sorted(args.wheel_dir.glob("*.whl"))
            require(
                len(wheels) == 1,
                "nightly qualification requires exactly one candidate wheel",
            )
            wheel = wheels[0]
        options = {
            "source": args.source,
            "controls": args.controls,
            "wheel": wheel,
            "output": args.output,
            "image": args.image,
            "gpus": args.gpus,
        }
        if args.command == "vllm-nightly":
            from ci.pipelines.nightly import run

            run(
                **options,
                variant=args.variant,
                through=args.through,
                workload_profile=args.workload_profile,
            )
        else:
            from ci.pipelines.benchmarks import run

            run(
                **options,
                benchmark_profile=args.benchmark_profile,
                benchmark_cases=args.benchmark_cases.split(",")
                if args.benchmark_cases
                else None,
            )
        return
    if args.command == "workflows":
        from ci.pipelines.workflows import workflow_index

        workflow_index(check=args.check, write=args.write_index)
        return
    if args.command == "scripts":
        from ci.pipelines.scripts import script_index

        script_index(check=args.check, write=args.write_index)
        return
    if args.command == "canary":
        from ci.pipelines.canaries import sglang_model, vllm_latency

        case = parse_json(args.case_json)
        if args.client == "vllm":
            require(
                args.controls is not None
                and args.wheel_dir is not None
                and args.image is not None,
                "vLLM canary needs controls, wheel directory and image",
            )
            vllm_latency(
                controls=args.controls,
                wheel_dir=args.wheel_dir,
                image=args.image,
                case=case,
                output=args.output,
            )
        else:
            sglang_model(container=args.container, case=case, output=args.output)
        return
    lock = load_json(args.lock_file) if args.lock_file else parse_json(args.lock_json)
    if args.command == "profile":
        require(
            lock["image"] == args.image,
            "selected executor differs from the declared environment",
        )
        require(
            not args.wheel_name
            or (
                args.wheel_dir is not None
                and Path(args.wheel_name).name == args.wheel_name
            ),
            "wheel needs one filename and its artifact directory",
        )
        execute_profile(
            source=args.source,
            controls=args.controls,
            output=args.output,
            profile=args.profile,
            architecture=args.architecture,
            gpus=args.gpus,
            lock=lock,
            mode=args.mode,
            wheel=args.wheel_dir / args.wheel_name if args.wheel_name else None,
            source_revision=args.source_revision,
            base_revision=args.base_revision,
        )
    else:
        compose_images(
            source=args.source,
            controls=args.controls,
            output=args.output,
            wheel=args.wheel,
            role=args.role,
            lock=lock,
            consumers=(
                load_json(args.consumers_file)
                if args.consumers_file
                else parse_json(args.consumers_json)
            ),
        )


if __name__ == "__main__":
    main()
