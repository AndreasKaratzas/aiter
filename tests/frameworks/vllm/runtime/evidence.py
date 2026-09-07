# SPDX-License-Identifier: MIT
"""Validate generation evidence independently of requested backend flags."""

import hashlib
from pathlib import Path

from .protocol import require


def validate_origins(workers, expected_root, expected_vllm, cache_root):
    """Re-read each worker's imported bytes within the selected package/cache roots."""
    checked = {}
    expected_vllm = Path(expected_vllm).resolve()
    cache_root = Path(cache_root).resolve() if cache_root else None
    for worker in workers:
        environment = worker["environment"]
        require(
            environment["vllm"] == str(expected_vllm), "Worker imported another vLLM"
        )
        modules = environment["observed_modules"]
        require(
            "aiter" in modules and "vllm" in modules, "Missing worker import evidence"
        )
        for name, entry in modules.items():
            namespace = name.split(".")[0]
            require(namespace in {"aiter", "vllm"}, "Unexpected observed namespace")
            path = Path(entry["path"]).resolve()
            root = (
                expected_root / "aiter"
                if namespace == "aiter"
                else expected_vllm.parent
            )
            owned_binary = (
                namespace == "aiter"
                and cache_root is not None
                and path.is_relative_to(cache_root)
                and path.name.endswith(".so")
            )
            require(
                path.is_relative_to(root) or owned_binary,
                f"Worker imported an unselected module: {name}: {path}",
            )
            if path not in checked:
                checked[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            require(checked[path] == entry["sha256"], f"Imported bytes changed: {name}")
        require(
            modules["vllm"]["path"] == str(expected_vllm), "Inconsistent vLLM origin"
        )
        require(
            modules["aiter"]["path"] == str(expected_root / "aiter/__init__.py"),
            "Inconsistent AITER origin",
        )


def observed(observation, category, fragment):
    direct = sum(
        count for name, count in observation[category].items() if fragment in name
    )
    replayed = 0
    for graph, count in observation["graph_replays"].items():
        capture = observation["graph_captures"].get(graph, {})
        if count > 0 and capture.get("complete"):
            replayed += count * sum(
                value for name, value in capture[category].items() if fragment in name
            )
    return direct + replayed


def validate_workers(result, request, expected_root):
    size = request["settings"]["tensor_parallel"]
    identities = result["workers"]
    require(
        len(identities) == size
        and {item["rank"] for item in identities} == set(range(size)),
        "Missing or duplicate tensor-parallel worker rank",
    )
    require(
        len({item["pid"] for item in identities}) == size,
        "Worker identities do not identify distinct processes",
    )
    if size > 1:
        require(
            all(item["device_uuid"] for item in identities)
            and len({item["device_uuid"] for item in identities}) == size,
            "Tensor-parallel workers did not execute on distinct GPUs",
        )
    for item in identities:
        require(item["world_size"] == size, "Wrong observed tensor-parallel world size")
        require(
            item["environment"]["aiter"] == str(expected_root / "aiter/__init__.py"),
            "A worker imported another AITER package",
        )
    require(
        [batch["name"] for batch in result["batches"]]
        == [batch["name"] for batch in request["batches"]],
        "Batch order changed",
    )
    for batch, inputs in zip(result["batches"], request["batches"], strict=True):
        require(len(batch["outputs"]) == len(inputs["prompts"]), "Wrong output count")
        require(
            len(batch["workers"]) == size
            and {worker["rank"] for worker in batch["workers"]} == set(range(size)),
            "Batch observations omit a tensor-parallel worker",
        )
        for worker in batch["workers"]:
            require(
                observed(worker, "operations", "rms_norm") > 0,
                "Generation bypassed AITER normalization on a worker",
            )
            validate_attention(
                worker, request["settings"].get("attention_backend", "unified")
            )
            if request["settings"].get("moe"):
                validate_experts(worker, request["settings"].get("moe_backend", "auto"))


def validate_attention(worker, backend):
    evidence = {
        "unified": observed(worker, "kernels", "unified_attention"),
        "flash": observed(worker, "operations", "flash_attn_varlen_func"),
        "mla": observed(worker, "operations", "mla_decode_fwd")
        + observed(worker, "kernels", "mla"),
    }
    require(
        backend in evidence and evidence[backend] > 0,
        f"Generation bypassed AITER attention: selected {backend}",
    )


def tokens(batch):
    return [output["token_ids"] for output in batch["outputs"]]


def replayed_aiter_graphs(worker):
    return {
        name: count
        for name, count in worker["graph_replays"].items()
        if count > 0
        and worker["graph_captures"].get(name, {}).get("complete")
        and any(
            "unified_attention" in kernel and launches > 0
            for kernel, launches in worker["graph_captures"][name]["kernels"].items()
        )
    }


def validate_experts(worker, backend):
    calls = (
        observed(worker, "kernels", "_moe_gemm_a16w4")
        if backend == "aiter_triton_mxfp4_bf16"
        else observed(worker, "operations", "fused_moe")
    )
    require(
        calls > 0, f"Generation bypassed selected AITER expert execution: {backend}"
    )
