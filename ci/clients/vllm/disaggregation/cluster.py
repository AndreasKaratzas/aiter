"""Slurm-specific setup and cleanup behind the shared retained process runner."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ci.common.json import require, write_json
from ci.pipelines.process import Process

from . import logs


def exclusions(pool: str, present: list[str], required: int) -> list[str]:
    allowed = pool.split(",")
    require(required in (2, 4), "unsupported disaggregation topology")
    require(
        len(set(allowed)) == len(allowed) and len(allowed) >= required,
        "node pool is duplicate or too small",
    )
    require(
        all(re.fullmatch(r"[A-Za-z0-9.-]+", node) for node in [*allowed, *present]),
        "invalid Slurm node name",
    )
    require(set(allowed) <= set(present), "configured node is absent from Slurm")
    return sorted(set(present) - set(allowed))


def restrict_nodes(job_script: Path, excluded: list[str]) -> None:
    if not excluded:
        return
    text = job_script.read_text()
    require(
        "#SBATCH --exclude=" not in text, "upstream already declares a node exclusion"
    )
    require(
        text.count("#SBATCH --exclusive\n") == 1,
        "unrecognized upstream Slurm allocation",
    )
    job_script.write_text(
        text.replace(
            "#SBATCH --exclusive\n",
            "#SBATCH --exclusive\n#SBATCH --exclude=" + ",".join(excluded) + "\n",
        )
    )


def run(source: Path, case: dict, config: dict, env: dict, output: Path) -> dict:
    require(
        Path(env["DISAGG_SCRIPTS_STAGE"]).is_absolute(),
        "cluster logs require an absolute shared path",
    )
    log_root = Path(env["DISAGG_SCRIPTS_STAGE"]).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    present = (
        Process(output / "allocation", executable="sinfo")
        .command(["-N", "-h", "-o", "%N"], timeout=60)
        .splitlines()
    )
    excluded = exclusions(env["SPUR_NODE_POOL"], present, case["nodes"])
    restrict_nodes(source / ".buildkite/amd-disagg/run_xPyD_disagg.slurm", excluded)
    child_env = {
        **env,
        **case["environment"],
        "BUILDKITE_COMMIT": f"{config['source']['revision']}-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}-{case['case_slug']}",
        "VLLM_ROUTER_IMAGE": config["router_image"],
        "LOG_ROOT": str(log_root),
        "CLUSTER_ENV_IN": "/vllm-workspace/.buildkite/amd-disagg/aiter_cluster_env.sh",
    }
    execution = Process(output / "submission", executable="env")
    submit_log = execution.evidence / "001-bash.log"
    try:
        with logs.follow(submit_log, log_root):
            execution.command(case["argv"], cwd=source, env=child_env, timeout=34200)
    except BaseException as error:
        for job in logs.job_ids(submit_log):
            try:
                Process(output / ("cancel-" + job), executable="scancel").command(
                    [job], timeout=60
                )
            except (OSError, ValueError) as cleanup:
                if hasattr(error, "add_note"):
                    error.add_note(f"Could not cancel submitted job {job}: {cleanup}")
        raise
    finally:
        # Keep evidence and diagnostics even if the wrapper, cancellation or
        # accuracy gate fails. The workflow's independent collect phase can
        # then report the submitted job without inferring it from stale state.
        jobs = logs.job_ids(submit_log)
        write_json(
            output / "submitted.json",
            {"job_ids": jobs, "case": case, "excluded_nodes": excluded},
        )
        if submit_log.exists():
            shutil.copyfile(submit_log, output / "submit.log")
        if len(jobs) == 1 and shutil.which("sacct"):
            try:
                Process(output / "accounting", executable="sacct").command(
                    ["-j", jobs[0], "--format=JobID,State,ExitCode,Elapsed,NodeList"],
                    timeout=60,
                    allowed_returncodes=(0, 1),
                )
            except (OSError, ValueError) as diagnostic:
                # Accounting is supporting evidence; it must not replace the
                # submission exception or the workload's accuracy verdict.
                write_json(
                    output / "diagnostics_error.json", {"sacct": str(diagnostic)}
                )
    return logs.collect(submit_log, log_root, output / "results")
