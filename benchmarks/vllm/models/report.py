"""Reconstruct recorded batch measurements without importing model runtimes."""

import hashlib
from dataclasses import asdict
from pathlib import Path

from benchmarks.common.measurements import summarize_batches
from benchmarks.vllm.models.config import Workload, engine_options
from ci.common.checkpoints import load_manifest, verify_snapshot
from ci.common.json import load_json, require


def validate_probe(workload, observations, workers):
    require(
        len(workers) == workload.tensor_parallel
        and {worker["rank"] for worker in workers}
        == set(range(workload.tensor_parallel)),
        "Missing or duplicate benchmark worker",
    )
    require(
        len(observations) == workload.tensor_parallel
        and {item["rank"] for item in observations}
        == set(range(workload.tensor_parallel)),
        "Missing rank observation",
    )
    require(
        len({worker["device"]["uuid"] for worker in workers})
        == workload.tensor_parallel
        and all(worker["device"]["uuid"] for worker in workers)
        and len({worker["pid"] for worker in workers}) == workload.tensor_parallel,
        "Workers did not use distinct GPUs",
    )
    for worker in workers:
        require(
            worker["world_size"] == workload.tensor_parallel
            and worker["instrumentation_active"] is False
            and worker["hooks_restored"] is True
            and worker["post_probe_observation_unchanged"] is True,
            "Wrong topology or live measurement hooks",
        )
        require(
            worker["parameter_dtypes"] == ["torch." + workload.dtype],
            "Observed model dtype differs",
        )
        require(
            worker["device"]["architecture"].split(":")[0] == "gfx950",
            "Worker executed an unselected architecture",
        )
    for observation in observations:
        for counters in (
            observation["operations"],
            observation["kernels"],
            observation["graph_replays"],
            *(
                capture[category]
                for capture in observation["graph_captures"].values()
                for category in ("operations", "kernels")
            ),
        ):
            require(
                type(counters) is dict
                and all(
                    type(name) is str and type(count) is int and count >= 0
                    for name, count in counters.items()
                ),
                "Invalid observation counters",
            )
        require(
            all(
                type(capture["complete"]) is bool
                for capture in observation["graph_captures"].values()
            ),
            "Invalid graph capture completion",
        )
        direct = any(
            "unified_attention" in name and count > 0
            for name, count in observation["kernels"].items()
        )
        replays = [
            name
            for name, count in observation["graph_replays"].items()
            if count > 0
            and observation["graph_captures"].get(name, {}).get("complete")
            and any(
                "unified_attention" in kernel and launches > 0
                for kernel, launches in observation["graph_captures"][name][
                    "kernels"
                ].items()
            )
        ]
        require(direct or replays, "No actual AITER attention on a rank")
        require(
            workload.execution != "graph" or replays,
            "Graph mode had no replay of an observed AITER capture",
        )
        require(
            workload.execution != "eager" or not observation["graph_replays"],
            "Eager measurement unexpectedly replayed graphs",
        )


def retained_files(output, repeats):
    names = [
        "request.json",
        "prompts.json",
        "engine-options.json",
        "untimed-probe.json",
        "model-before.json",
        "model-after.json",
    ]
    names += [f"sample-{index:04d}.json" for index in range(repeats)]
    return {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names
    }


def check_measurement(output):
    output = Path(output)
    report = load_json(output / "report.json")
    require(report["status"] == "PASS", "model benchmark did not pass")
    request = load_json(output / "request.json")
    require(
        type(request.get("schema_version")) is int
        and request["schema_version"] == 2
        and report.get("schema_version") == 2,
        "Current model measurements require evidence schema 2",
    )
    require(report["request"] == request, "benchmark request changed")
    workload = Workload(**request["workload"])
    workload.validate()
    require(request["workload"] == asdict(workload), "incomplete measurement workload")
    probe = load_json(output / "untimed-probe.json")
    require(
        set(probe) == {"workers", "worker", "observations", "outputs"},
        "Missing current per-rank probe evidence",
    )
    validate_probe(workload, probe["observations"], probe["workers"])
    require(
        report["workers"] == probe["workers"]
        and report["untimed_observations"] == probe["observations"],
        "Benchmark worker/probe evidence changed",
    )
    from benchmarks.vllm.models.runner import prompt_tokens

    require(
        load_json(output / "prompts.json") == prompt_tokens(workload),
        "Prompts differ from workload seed and shape",
    )
    manifest = Path(request["manifest"])
    require(
        hashlib.sha256(manifest.read_bytes()).hexdigest() == request["manifest_sha256"],
        "Model manifest changed",
    )
    models = load_manifest(manifest)
    require(
        request["model"] in models
        and request["model_declaration"] == models[request["model"]],
        "Selected model declaration differs",
    )
    receipt = load_json(output / "model-before.json")
    require(
        receipt
        == verify_snapshot(
            receipt["snapshot"], request["model_declaration"], strict=True
        ),
        "Model receipt differs from selected bytes",
    )
    require(
        load_json(output / "engine-options.json") == engine_options(workload, receipt),
        "Engine options differ from declared workload",
    )
    cache_root = Path(request["environment"]["AITER_JIT_DIR"]).resolve()
    for worker in probe["workers"]:
        require(
            worker["modules"].get("aiter", {}).get("path") == worker["aiter"]
            and worker["modules"].get("vllm", {}).get("path") == worker["vllm"],
            "Missing observed package origins",
        )
        for name, entry in worker["modules"].items():
            namespace = name.split(".")[0]
            require(
                namespace in {"aiter", "vllm"}, "Unexpected observed module namespace"
            )
            path = Path(entry["path"]).resolve()
            root = Path(worker[namespace]).resolve().parent
            require(
                path.is_relative_to(root)
                or (
                    namespace == "aiter"
                    and path.is_relative_to(cache_root)
                    and path.suffix == ".so"
                ),
                "Imported module escaped selected package/cache",
            )
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
                "Imported module bytes changed",
            )
    samples = [
        load_json(output / f"sample-{index:04d}.json")
        for index in range(workload.repeats)
    ]
    require(samples == report["samples"], "raw timing samples differ from report")
    for index, sample in enumerate(samples):
        require(
            sample["iteration"] == index and sample["outputs"] == probe["outputs"],
            "measured output differs from fixed-input probe",
        )
        require(
            len(sample["outputs"]) == workload.batch_size
            and all(
                len(item["token_ids"]) == workload.output_tokens
                for item in sample["outputs"]
            ),
            "measured output count differs from workload",
        )
        require(
            sample["output_token_counts"]
            == [len(item["token_ids"]) for item in sample["outputs"]],
            "throughput used an unobserved token count",
        )
    require(
        summarize_batches(samples) == report["summary"],
        "benchmark statistics do not reconstruct",
    )
    require(
        report["retained_files"] == retained_files(output, workload.repeats),
        "benchmark evidence bytes changed",
    )
    require(
        load_json(output / "model-before.json")
        == load_json(output / "model-after.json"),
        "model view changed",
    )
    require(
        probe["worker"] == report["worker"]
        and report["worker"]["instrumentation_active"] is False,
        "timed worker identity or instrumentation changed",
    )
    return report
