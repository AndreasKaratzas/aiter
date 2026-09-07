"""Measure observed delivery work without inventing missing queue or adoption data."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.qualification.report import check_results
from ci.release.artifacts import ArtifactFile, hash_file
from ci.release.channels import (
    _candidate,
    _sealed,
    _verify,
    read_state,
    timestamp,
    verified_images,
)

PHASES = ("queue", "build", "test", "rework")


def _identity(source_revision: str, profile: str, run_id: str) -> None:
    require(
        isinstance(source_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", source_revision),
        "source must be a full Git revision",
    )
    require(isinstance(profile, str) and bool(profile.strip()), "profile is required")
    require(
        isinstance(run_id, str) and bool(run_id.strip()),
        "trusted executor run/job identity is required",
    )


def _save(directory: Path, kind: str, payload: dict, evidence: dict[str, Path]) -> dict:
    """Keep exact observation inputs beside their content-addressed record."""
    bindings = {}
    for name, path in evidence.items():
        require(
            Path(name).name == name and name not in (".", "..", "record.json"),
            "invalid evidence name",
        )
        sha, size = hash_file(Path(path))
        bindings[name] = {"sha256": sha, "size_bytes": size}
    record = _sealed(
        {
            "schema_version": 1,
            "kind": kind,
            "observation": payload,
            "evidence": bindings,
        },
        "observation_digest",
    )
    target = Path(directory) / record["observation_digest"].split(":", 1)[1]
    if target.exists():
        require(
            load_observation(target) == record,
            "observation directory differs from its identity",
        )
        return record
    target.mkdir(parents=True)
    for name, path in evidence.items():
        shutil.copyfile(path, target / name)
    write_json(target / "record.json", record)
    require(
        load_observation(target) == record, "observation input changed while copying"
    )
    return record


def load_observation(directory: Path) -> dict:
    record = load_json(Path(directory) / "record.json")
    require(
        set(record)
        == {"schema_version", "kind", "observation", "evidence", "observation_digest"},
        "invalid observation record",
    )
    _verify(record, "observation_digest")
    require(
        record["schema_version"] == 1
        and record["kind"]
        in ("interval", "workflow-job", "change", "adoption", "reuse"),
        "unknown observation schema or kind",
    )
    require(bool(record["evidence"]), "observations need retained evidence")
    for name, expected in record["evidence"].items():
        require(
            Path(name).name == name and name not in (".", "..", "record.json"),
            "invalid observation evidence path",
        )
        sha, size = hash_file(Path(directory) / name)
        require(
            expected == {"sha256": sha, "size_bytes": size},
            "observation evidence changed",
        )
    data = record["observation"]
    if record["kind"] == "interval":
        reconstructed = _interval_payload(
            load_json(Path(directory) / "execution.json"),
            phase=data["phase"],
            source_revision=data["source_revision"],
            profile=data["profile"],
            run_id=data["run_id"],
            artifact=data["artifact"],
        )
        require(
            reconstructed == data,
            "interval differs from its retained execution evidence",
        )
    elif record["kind"] == "workflow-job":
        observed = load_json(Path(directory) / "workflow-jobs.json")
        require(
            _workflow_job_payload(observed, data["job_id"]) == data,
            "workflow timing differs from retained API evidence",
        )
    elif record["kind"] == "change":
        require(
            load_json(Path(directory) / "change-event.json") == data,
            "change differs from its controller event",
        )
    return record


def record_interval(
    directory: Path,
    execution: Path,
    *,
    phase: str,
    source_revision: str,
    profile: str,
    run_id: str,
    artifact: dict | None = None,
) -> dict:
    """Read executor times; queue/rework records must explicitly identify that phase.

    A queue record contains queued_utc and started_utc. Build/test records are
    actual executor records. Rework is a separately observed interval, never
    inferred merely from the presence of a failed test or retry.
    """
    payload = _interval_payload(
        load_json(execution),
        phase=phase,
        source_revision=source_revision,
        profile=profile,
        run_id=run_id,
        artifact=artifact,
    )
    return _save(directory, "interval", payload, {"execution.json": execution})


def _interval_payload(
    data: dict,
    *,
    phase: str,
    source_revision: str,
    profile: str,
    run_id: str,
    artifact: dict | None,
) -> dict:
    _identity(source_revision, profile, run_id)
    require(phase in PHASES, "unknown delivery phase")
    require(
        all(
            data.get(key) == value
            for key, value in {
                "source_revision": source_revision,
                "profile": profile,
                "run_id": run_id,
            }.items()
        ),
        "execution does not identify this source, profile and executor",
    )
    if artifact is not None:
        ArtifactFile.from_dict(artifact)
        require(
            artifact in data.get("artifacts", []),
            "execution does not bind this artifact",
        )
    if phase == "queue":
        require(
            data.get("phase") == phase and data.get("run_id") == run_id,
            "queue observation must identify its executor",
        )
        start, finish = data["queued_utc"], data["started_utc"]
    else:
        start, finish = data["started_utc"], data["finished_utc"]
        if phase == "rework":
            require(
                data.get("phase") == phase and data.get("run_id") == run_id,
                "rework requires an explicit observed interval",
            )
        else:
            require(
                isinstance(data.get("command"), list) and bool(data["command"]),
                "build/test needs an executor command record",
            )
            require(
                type(data.get("returncode")) is int
                and type(data.get("timed_out")) is bool,
                "execution outcome is missing",
            )
    elapsed = (timestamp(finish) - timestamp(start)).total_seconds()
    require(elapsed >= 0, "execution finished before it started")
    measured = data.get("duration_ms")
    if measured is not None:
        require(
            type(measured) in (int, float) and measured >= 0,
            "invalid measured execution duration",
        )
        require(
            abs(measured / 1000 - elapsed) <= max(2.0, elapsed * 0.01),
            "wall and monotonic execution clocks disagree",
        )
    return {
        "phase": phase,
        "source_revision": source_revision,
        "profile": profile,
        "run_id": run_id,
        "artifact": artifact,
        "started_utc": start,
        "finished_utc": finish,
        "elapsed_seconds": elapsed,
        "measured_seconds": measured / 1000 if measured is not None else None,
        "returncode": data.get("returncode"),
        "timed_out": data.get("timed_out"),
    }


def record_change(directory: Path, event: Path) -> dict:
    """Retain a source event from the reviewed integration/merge controller."""
    data = load_json(event)
    require(
        set(data)
        == {"source_revision", "profile", "run_id", "changed_utc", "reference"},
        "invalid change event",
    )
    _identity(data["source_revision"], data["profile"], data["run_id"])
    timestamp(data["changed_utc"])
    require(
        isinstance(data["reference"], str) and bool(data["reference"]),
        "change event needs an evidence reference",
    )
    return _save(directory, "change", data, {"change-event.json": event})


def _workflow_job_payload(data: dict, job_id: int) -> dict:
    """Measure a completed build job, keeping workflow and artifact sources distinct."""
    require(
        set(data)
        == {"source_revision", "run_id", "workflow_run", "jobs", "jobs_endpoint"},
        "invalid workflow timing envelope",
    )
    _identity(data["source_revision"], "delivery", data["run_id"])
    workflow = data["workflow_run"]
    require(
        type(workflow["id"]) is int and type(workflow["run_attempt"]) is int,
        "workflow run/attempt identifiers are missing",
    )
    expected_run = f"{workflow['id']}:{workflow['run_attempt']}"
    require(data["run_id"] == expected_run, "workflow attempt differs from controller")
    require(
        isinstance(data["jobs_endpoint"], str)
        and data["jobs_endpoint"].endswith(
            f"/actions/runs/{workflow['id']}/attempts/{workflow['run_attempt']}/jobs"
        ),
        "jobs were not collected from the specified attempt",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", workflow["head_sha"]),
        "workflow control revision is invalid",
    )
    require(isinstance(data["jobs"], list), "workflow jobs must be a list")
    matches = [job for job in data["jobs"] if job.get("id") == job_id]
    require(len(matches) == 1, "workflow job is missing or duplicated")
    job = matches[0]
    require(
        job["run_id"] == workflow["id"] and job["head_sha"] == workflow["head_sha"],
        "job belongs to another workflow run or control revision",
    )
    require(
        job.get("run_attempt", workflow["run_attempt"]) == workflow["run_attempt"],
        "job belongs to another workflow attempt",
    )
    require(
        re.search(r"(?:^| / )build_whl_package(?: \(|$)", job["name"]),
        "job is not the declared wheel builder",
    )
    require(job["status"] == "completed", "build job has not completed")
    start, finish = timestamp(job["started_at"]), timestamp(job["completed_at"])
    require(finish >= start, "build job finished before it started")
    return {
        "phase": "build",
        "source_revision": data["source_revision"],
        "profile": "delivery",
        "run_id": expected_run + ":job:" + str(job_id),
        "artifact": None,
        "started_utc": start.isoformat(),
        "finished_utc": finish.isoformat(),
        "elapsed_seconds": (finish - start).total_seconds(),
        "measured_seconds": None,
        "job_id": job_id,
        "job_name": job["name"],
        "conclusion": job["conclusion"],
        "workflow_head_sha": workflow["head_sha"],
        "reference": job["html_url"],
        "measurement_scope": "Whole wheel-builder job, including setup, packaging and artifact upload; not compiler-only time or GPU queue time.",
    }


def record_workflow_jobs(
    directory: Path, evidence: Path, *, source_revision: str, run_id: str
) -> list[dict]:
    data = load_json(evidence)
    require(
        data.get("source_revision") == source_revision and data.get("run_id") == run_id,
        "workflow timing belongs to another delivery",
    )
    result = []
    for job in data["jobs"]:
        if (
            re.search(r"(?:^| / )build_whl_package(?: \(|$)", job.get("name", ""))
            and job.get("status") == "completed"
        ):
            payload = _workflow_job_payload(data, job["id"])
            result.append(
                _save(
                    directory, "workflow-job", payload, {"workflow-jobs.json": evidence}
                )
            )
    require(bool(result), "workflow evidence contains no completed wheel-builder jobs")
    return result


def _qualification(channels: Path, qualification_digest: str) -> tuple[dict, dict]:
    state = read_state(channels)
    event = next(
        (e for e in state["events"] if e["event_digest"] == qualification_digest), None
    )
    require(
        event is not None
        and event["kind"] == "result"
        and event["payload"]["status"] == "PASS",
        "observation needs a passing exact-artifact qualification",
    )
    return event, _candidate(state, event["payload"]["candidate_digest"])


def record_adoption(
    directory: Path,
    *,
    channels: Path,
    qualification_digest: str,
    consumer_checkout: Path,
    merged_ref: str,
    pin_path: str,
    merge_event: Path,
    image_evidence: dict,
    catalog: dict,
) -> dict:
    """Count adoption only after the exact pin is merged and its consumer image passes.

    The consumer keeps an AITER artifact lock JSON containing an ``artifact``
    object. The merge controller provides its observed merge time/reference.
    Local Git proves the recorded commit is in the accepted branch and contains
    that exact artifact lock. This function never changes the consumer checkout.
    """
    qualification, candidate = _qualification(channels, qualification_digest)
    merge = load_json(merge_event)
    require(
        set(merge)
        == {"consumer", "merged_commit", "merged_utc", "reference", "run_id"},
        "invalid merge observation",
    )
    _identity(merge["merged_commit"], merge["consumer"], merge["run_id"])
    require(
        bool(merge["reference"]), "merge observation needs its controller reference"
    )
    require(
        not merged_ref.startswith("-") and bool(merged_ref),
        "invalid accepted consumer ref",
    )
    require(
        not pin_path.startswith("/")
        and all(p not in ("", ".", "..") for p in pin_path.split("/")),
        "invalid consumer lock path",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(consumer_checkout),
            "merge-base",
            "--is-ancestor",
            merge["merged_commit"],
            merged_ref,
        ],
        check=True,
        capture_output=True,
    )
    text = subprocess.check_output(
        [
            "git",
            "-C",
            str(consumer_checkout),
            "show",
            f"{merge['merged_commit']}:{pin_path}",
        ],
        text=True,
    )
    pin = json.loads(text)
    require(
        isinstance(pin, dict) and pin.get("artifact") == candidate["artifact"],
        "merged consumer lock does not pin the qualified wheel",
    )
    images = verified_images([image_evidence], candidate, catalog)
    image = images[0]
    consumer_profiles = {check["profile"] for check in image["checks"]}
    require(
        merge["consumer"] + "-image" in consumer_profiles,
        "image lacks the adopting consumer's complete checks",
    )
    consumer_finishes = []
    for run in image_evidence["runs"]:
        plan = load_json(Path(run) / "plan.json")
        if plan["profile"] != merge["consumer"] + "-image":
            continue
        for result_path in Path(run).glob("*/attempt-*/result.json"):
            result = load_json(result_path)
            framework = (
                result["environment"].get("frameworks", {}).get(merge["consumer"])
            )
            if framework is not None:
                require(
                    framework.get("revision") == merge["merged_commit"]
                    and framework.get("dirty") is False,
                    "consumer image did not execute the merged clean revision",
                )
                consumer_finishes.append(timestamp(result["finished_utc"]))
    require(
        bool(consumer_finishes),
        "consumer image has no observed merged revision and completion time",
    )
    merged = timestamp(merge["merged_utc"])
    qualified = timestamp(qualification["created_utc"])
    require(merged >= qualified, "adoption predates the artifact qualification")
    accepted = max(merged, *consumer_finishes)
    # The retained image record binds its independently reconstructed checks.
    payload = {
        **merge,
        "source_revision": candidate["source_revision"],
        "profile": candidate["scope"]["profile"],
        "scope": candidate["scope"],
        "qualification_digest": qualification_digest,
        "qualified_utc": qualification["created_utc"],
        "artifact": candidate["artifact"],
        "image_id": image["image_id"],
        "image_record_digest": image["image_record_digest"],
        "qualified_to_merged_seconds": (merged - qualified).total_seconds(),
        "accepted_utc": accepted.isoformat(),
        "qualified_to_accepted_seconds": (accepted - qualified).total_seconds(),
        "accepted_ref": merged_ref,
        "pin_path": pin_path,
        "pin_digest": digest(pin),
    }
    return _save(
        directory,
        "adoption",
        payload,
        {
            "merge-event.json": merge_event,
            "image-record.json": Path(image_evidence["record"]),
        },
    )


def record_reuse(
    directory: Path,
    event: Path,
    *,
    channels: Path,
    qualification_digest: str,
    wheel: Path | None = None,
) -> dict:
    """Track eligible consumer builds and exact byte reuse, including non-reuse."""
    qualification, candidate = _qualification(channels, qualification_digest)
    data = load_json(event)
    require(
        set(data)
        == {
            "consumer",
            "run_id",
            "observed_utc",
            "environment_lock_digest",
            "artifact",
            "eligible",
            "reason",
            "reference",
        },
        "invalid reuse observation",
    )
    _identity(candidate["source_revision"], data["consumer"], data["run_id"])
    require(
        type(data["eligible"]) is bool
        and bool(data["reason"])
        and bool(data["reference"]),
        "reuse needs eligibility reasoning and executor evidence",
    )
    require(
        timestamp(data["observed_utc"]) >= timestamp(qualification["created_utc"]),
        "reuse predates qualification",
    )
    same_environment = (
        data["environment_lock_digest"] == candidate["scope"]["environment_lock_digest"]
    )
    require(
        not data["eligible"] or same_environment,
        "a new environment is not eligible exact-artifact reuse",
    )
    if data["artifact"] is not None:
        ArtifactFile.from_dict(data["artifact"])
        require(
            wheel is not None, "reuse observation needs the actual consumed wheel bytes"
        )
        sha, size = hash_file(Path(wheel))
        require(
            data["artifact"]
            == {"filename": Path(wheel).name, "sha256": sha, "size_bytes": size},
            "consumed wheel differs from reuse observation",
        )
    payload = {
        **data,
        "source_revision": candidate["source_revision"],
        "profile": candidate["scope"]["profile"],
        "qualification_digest": qualification_digest,
        "qualified_artifact": candidate["artifact"],
        "exact_artifact_reused": data["eligible"]
        and data["artifact"] == candidate["artifact"],
    }
    return _save(directory, "reuse", payload, {"reuse-event.json": event})


def _union_seconds(intervals: list[tuple[datetime, datetime]]) -> float:
    total, end = 0.0, None
    for start, finish in sorted(intervals):
        if end is None or start >= end:
            total += (finish - start).total_seconds()
            end = finish
        elif finish > end:
            total += (finish - end).total_seconds()
            end = finish
    return total


def summarize(
    observations: Path,
    *,
    channels: Path | None = None,
    runs: list[Path] | None = None,
    catalog: dict | None = None,
) -> dict:
    """Report measured samples and coverage; missing phases are null, never zero."""
    records = [
        load_observation(path.parent)
        for path in sorted(Path(observations).glob("*/record.json"))
    ]
    require(
        len({r["observation_digest"] for r in records}) == len(records),
        "duplicate observation",
    )
    groups = {}
    for record in records:
        data = record["observation"]
        key = data["source_revision"] + "|" + data["profile"]
        group = groups.setdefault(
            key,
            {
                "source_revision": data["source_revision"],
                "profile": data["profile"],
                "phases": {},
                "adoptions": [],
                "reuse": {
                    "eligible_builds": 0,
                    "exact_reuses": 0,
                    "ineligible_builds": 0,
                },
                "qualifications": [],
                "change_events": [],
            },
        )
        if record["kind"] in ("interval", "workflow-job"):
            phase = group["phases"].setdefault(data["phase"], [])
            require(
                not any(item["run_id"] == data["run_id"] for item in phase),
                "duplicate phase for the same executor job",
            )
            phase.append(data)
        elif record["kind"] == "change":
            group["change_events"].append(data)
        elif record["kind"] == "adoption":
            require(
                not any(
                    item["consumer"] == data["consumer"]
                    and item["merged_commit"] == data["merged_commit"]
                    for item in group["adoptions"]
                ),
                "duplicate accepted consumer upgrade",
            )
            group["adoptions"].append(data)
        else:
            seen = group.setdefault("reuse_run_ids", [])
            require(
                data["run_id"] not in seen, "duplicate downstream build observation"
            )
            seen.append(data["run_id"])
            group["reuse"]["eligible_builds"] += int(data["eligible"])
            group["reuse"]["exact_reuses"] += int(data["exact_artifact_reused"])
            group["reuse"]["ineligible_builds"] += int(not data["eligible"])
    if channels is not None:
        state = read_state(channels)
        for event in state["events"]:
            if event["kind"] != "result" or event["payload"]["status"] != "PASS":
                continue
            candidate = _candidate(state, event["payload"]["candidate_digest"])
            key = candidate["source_revision"] + "|" + candidate["scope"]["profile"]
            if key not in groups:
                continue
            group = groups[key]
            changes = group["change_events"]
            require(
                len({c["changed_utc"] for c in changes}) <= 1,
                "ambiguous source-change observation",
            )
            if changes:
                seconds = (
                    timestamp(event["created_utc"])
                    - timestamp(changes[0]["changed_utc"])
                ).total_seconds()
                require(seconds >= 0, "qualification predates its change")
            else:
                seconds = None
            group["qualifications"].append(
                {
                    "qualification_digest": event["event_digest"],
                    "scope": candidate["scope"],
                    "artifact": candidate["artifact"],
                    "change_to_qualified_seconds": seconds,
                }
            )
    for group in groups.values():
        for phase in PHASES:
            samples = group["phases"].get(phase, [])
            group["phases"][phase] = {
                "observed_jobs": len(samples),
                "worker_seconds": (
                    sum(s["elapsed_seconds"] for s in samples) if samples else None
                ),
                "wall_seconds": (
                    _union_seconds(
                        [
                            (timestamp(s["started_utc"]), timestamp(s["finished_utc"]))
                            for s in samples
                        ]
                    )
                    if samples
                    else None
                ),
            }
        reuse = group["reuse"]
        reuse["exact_reuse_fraction"] = (
            reuse["exact_reuses"] / reuse["eligible_builds"]
            if reuse["eligible_builds"]
            else None
        )
        group.pop("change_events")
        group.pop("reuse_run_ids", None)
    execution = []
    for run in runs or []:
        require(catalog is not None, "run timing requires the matching catalog")
        plan = load_json(Path(run) / "plan.json")
        report = check_results(plan, catalog, Path(run))
        elapsed, attempts = 0.0, 0
        for path in Path(run).glob("*/attempt-*/result.json"):
            result = load_json(path)
            _verify(result, "result_digest")
            require(
                result["plan_digest"] == plan["plan_digest"],
                "run timing has a stale result",
            )
            duration = result.get("duration_ms")
            require(
                type(duration) in (int, float) and duration >= 0, "invalid run duration"
            )
            elapsed += duration / 1000
            attempts += 1
        execution.append(
            {
                "plan_digest": plan["plan_digest"],
                "profile": plan["profile"],
                "report_digest": report["report_digest"],
                "status": report["status"],
                "observed_attempts": attempts,
                "test_worker_seconds": elapsed if attempts else None,
            }
        )
    return _sealed(
        {
            "schema_version": 1,
            "interpretation": "Observed samples only. Parallel worker time is not elapsed lead time. Missing observations are null. These local content hashes do not authenticate the external event producer.",
            "delivery": groups,
            "test_runs": execution,
        },
        "metrics_digest",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    interval = commands.add_parser(
        "interval", help="retain an observed queue/build/test/rework interval"
    )
    interval.add_argument("--execution", type=Path, required=True)
    interval.add_argument("--phase", choices=PHASES, required=True)
    interval.add_argument("--source-revision", required=True)
    interval.add_argument("--profile", required=True)
    interval.add_argument("--run-id", required=True)
    interval.add_argument("--artifact", type=Path)
    change = commands.add_parser(
        "change", help="retain a reviewed controller's source-change event"
    )
    change.add_argument("--event", type=Path, required=True)
    adoption = commands.add_parser(
        "adoption", help="verify a merged exact pin and its passing consumer image"
    )
    adoption.add_argument("--consumer-checkout", type=Path, required=True)
    adoption.add_argument("--merged-ref", required=True)
    adoption.add_argument("--pin-path", required=True)
    adoption.add_argument("--merge-event", type=Path, required=True)
    adoption.add_argument("--image-evidence", type=Path, required=True)
    reuse = commands.add_parser(
        "reuse", help="record eligible builds and verify consumed wheel bytes"
    )
    reuse.add_argument("--event", type=Path, required=True)
    reuse.add_argument("--wheel", type=Path)
    for command in (interval, change, adoption, reuse):
        command.add_argument("--output-dir", type=Path, required=True)
    for command in (adoption, reuse):
        command.add_argument("--channels", type=Path, required=True)
        command.add_argument("--qualification", required=True)
    summary = commands.add_parser(
        "summarize", help="report measured work; leave missing observations unknown"
    )
    summary.add_argument("--observations", type=Path, required=True)
    summary.add_argument("--channels", type=Path)
    summary.add_argument("--run", type=Path, action="append", default=[])
    summary.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        from ci.qualification.catalog import load_catalog

        if args.command == "interval":
            output = record_interval(
                args.output_dir,
                args.execution,
                phase=args.phase,
                source_revision=args.source_revision,
                profile=args.profile,
                run_id=args.run_id,
                artifact=load_json(args.artifact) if args.artifact else None,
            )
        elif args.command == "change":
            output = record_change(args.output_dir, args.event)
        elif args.command == "adoption":
            output = record_adoption(
                args.output_dir,
                channels=args.channels,
                qualification_digest=args.qualification,
                consumer_checkout=args.consumer_checkout,
                merged_ref=args.merged_ref,
                pin_path=args.pin_path,
                merge_event=args.merge_event,
                image_evidence=load_json(args.image_evidence),
                catalog=load_catalog(args.catalog),
            )
        elif args.command == "reuse":
            output = record_reuse(
                args.output_dir,
                args.event,
                channels=args.channels,
                qualification_digest=args.qualification,
                wheel=args.wheel,
            )
        else:
            output = summarize(
                args.observations,
                channels=args.channels,
                runs=args.run,
                catalog=load_catalog(args.catalog),
            )
            if args.output:
                write_json(args.output, output)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"metrics: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
