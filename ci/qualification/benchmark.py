"""Reconstruct benchmark decisions from retained raw paired observations."""

from pathlib import Path

from ci.common.json import load_json, require


def benchmark_cases(path: Path) -> list[dict]:
    from benchmarks.common.comparison import compare_pairs

    report = load_json(path)
    require(
        report["schema_version"] == 1 and report["benchmark"] == "rmsnorm",
        "unexpected benchmark protocol",
    )
    expected = {
        (rows, columns, dtype)
        for rows, columns in ((1, 4096), (256, 4096), (1024, 8192))
        for dtype in ("float16", "bfloat16")
    }
    observed = set()
    cases = []
    for result in report["results"]:
        identity = (*result["shape"], result["dtype"])
        require(
            identity in expected and identity not in observed,
            "missing or duplicated benchmark workload",
        )
        observed.add(identity)
        require(
            result["protocol"] == "interleaved-21pairs-256calls-events-v1",
            "benchmark protocol changed",
        )
        require(
            result["correctness"]["passed"] is True,
            "benchmark did not pass independent correctness",
        )
        samples = result["samples_ns"]
        require(
            len(samples["baseline"])
            == len(samples["candidate"])
            == len(result["orders"])
            == 21,
            "incomplete paired benchmark observations",
        )
        require(
            all(
                sorted(order) == ["baseline", "candidate"] for order in result["orders"]
            ),
            "invalid benchmark pair ordering",
        )
        actual = compare_pairs(samples["baseline"], samples["candidate"])
        require(
            actual == result["comparison"],
            "benchmark decision differs from retained measurements",
        )
        cases.append(
            {
                "id": f"rmsnorm:{identity[0]}x{identity[1]}:{identity[2]}",
                "outcome": "passed" if actual["passed"] else "failed",
            }
        )
    require(observed == expected, "benchmark omitted required workload")
    require(
        report["passed"] == all(case["outcome"] == "passed" for case in cases),
        "benchmark summary differs from raw decisions",
    )
    return cases
