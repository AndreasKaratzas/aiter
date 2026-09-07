"""Follow only the submitted job's logs and retain an explicit accuracy verdict."""

from __future__ import annotations

import re
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

from ci.common.json import require, write_json


def job_ids(submit_log: Path) -> list[str]:
    text = submit_log.read_text(errors="replace") if submit_log.exists() else ""
    return sorted(set(re.findall(r"Submitted batch job\s+(\d+)", text)))


@contextmanager
def follow(submit_log: Path, log_root: Path):
    done = threading.Event()

    def stream():
        offsets = {}
        while not done.wait(1):
            paths = [submit_log]
            for job in job_ids(submit_log):
                paths.append(log_root / f"vllm-disagg-pd-{job}.log")
                paths.extend((log_root / job).glob("*.log"))
            for path in paths:
                try:
                    with path.open("rb") as source:
                        source.seek(offsets.get(path, 0))
                        data = source.read(65536)
                        offsets[path] = source.tell()
                    if data:
                        print(
                            f"[{path.name}] {data.decode(errors='replace')}", flush=True
                        )
                except OSError:
                    # Files are created asynchronously by Slurm. Final admission
                    # below requires the actual result, regardless of live output.
                    continue

    worker = threading.Thread(target=stream, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join(timeout=5)


def verdict(text: str) -> dict:
    matches = re.findall(
        r"\[vllm_disagg\] (PASS|FAIL): exact_match=([^\s]+).*?threshold=([^\s]+)", text
    )
    require(bool(matches), "no exact-match verdict from the upstream workload")
    status, score, threshold = matches[-1]
    score, threshold = float(score), float(threshold)
    require(
        0 <= score <= 1 and 0 <= threshold <= 1, "invalid accuracy score or threshold"
    )
    require(
        (status == "PASS") == (score >= threshold),
        "accuracy verdict disagrees with its score",
    )
    return {"status": status, "exact_match": score, "threshold": threshold}


def collect(submit_log: Path, log_root: Path, output: Path) -> dict:
    jobs = job_ids(submit_log)
    require(len(jobs) == 1, "expected exactly one submitted Slurm job")
    job = jobs[0]
    output.mkdir(parents=True, exist_ok=True)
    job_log = log_root / f"vllm-disagg-pd-{job}.log"
    require(job_log.is_file(), f"no result log for submitted Slurm job {job}")
    shutil.copyfile(job_log, output / "job.log")
    for path in (log_root / job).glob("*.log"):
        if path.is_file():
            destination = output / "roles"
            destination.mkdir(exist_ok=True)
            shutil.copyfile(path, destination / path.name)
    result = {"job_id": job, **verdict(job_log.read_text(errors="replace"))}
    write_json(output / "accuracy.json", result)
    require(result["status"] == "PASS", "upstream model accuracy failed")
    return result
