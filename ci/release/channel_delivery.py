"""Reconstruct a delivery's local channel state from retained workflow artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import require_profile_environment
from ci.qualification.plan import plan_tests, validate_plan
from ci.release.channels import (
    channel_index,
    evaluate_result,
    read_state,
    record_candidate,
    record_failure,
    record_result,
    verified_images,
)
from ci.release.metrics import record_interval, record_workflow_jobs, summarize
from ci.release.wheels import load_receipt, verify_wheel


def _one(paths, label: str) -> Path | None:
    paths = list(paths)
    require(len(paths) <= 1, f"multiple {label} artifacts")
    return paths[0] if paths else None


def _jobs(job_results: dict) -> tuple[dict, list[str]]:
    require(isinstance(job_results, dict), "job results must be an object")
    observed = {}
    for name, value in job_results.items():
        result = value.get("result") if isinstance(value, dict) else value
        require(
            result in ("success", "failure", "cancelled", "skipped"),
            f"unknown job outcome for {name}",
        )
        observed[name] = result
    required = {"build", "qualification-plan", "qualify", "release-record", "images"}
    required.add("publish" if "publish" in observed else "promote")
    if "prepare" in observed:
        required.add("prepare")
    problems = [
        f"Delivery job {name}: {observed.get(name, 'missing')}"
        for name in sorted(required)
        if observed.get(name) != "success"
    ]
    return observed, problems


def _entry(entry: dict, source_revision: str, catalog: dict) -> dict:
    require(isinstance(entry, dict), "matrix entry must be an object")
    require(
        re.fullmatch(r"[a-zA-Z0-9_.-]+", entry["label"]),
        "invalid matrix artifact label",
    )
    require(
        entry["source_revision"] == source_revision,
        "matrix belongs to another source revision",
    )
    require(entry["profile"] in catalog["profiles"], "matrix profile is unknown")
    require(
        entry["architecture"] in ("gfx942", "gfx950", "gfx1250"),
        "matrix architecture is unknown",
    )
    require(
        Path(entry["wheel"]).name == entry["wheel"], "matrix wheel is not a filename"
    )
    lock = (
        json.loads(entry["environment_lock"])
        if isinstance(entry["environment_lock"], str)
        else entry["environment_lock"]
    )
    require_profile_environment(
        lock, catalog["profiles"][entry["profile"]]["client"], supported=True
    )
    require(
        entry["environment_lock_digest"] == digest(lock), "matrix lock digest differs"
    )
    return lock


def _failure(
    store: Path,
    *,
    run_id: str,
    source_revision: str,
    owner: str,
    delta: str,
    problems: list[str],
    scope=None,
    environment_lock=None,
):
    state = read_state(store)
    prior = next(
        (
            event
            for event in state["events"]
            if event["kind"] == "failure" and event["payload"]["run_id"] == run_id
        ),
        None,
    )
    if prior is not None:
        require(
            prior["payload"]["source_revision"] == source_revision
            and prior["payload"]["scope"] == scope
            and prior["payload"]["problems"] == problems
            and prior["payload"]["owner"] == owner
            and prior["payload"]["delta"] == delta
            and prior["payload"]["environment_lock"] == environment_lock,
            "delivery retry changed a recorded failure",
        )
        return prior
    return record_failure(
        store,
        source_revision=source_revision,
        run_id=run_id,
        owner=owner,
        delta=delta,
        problems=problems,
        scope=scope,
        environment_lock=environment_lock,
        expected_digest=state["state_digest"],
    )["event"]


def _wheel(artifacts: Path, filename: str, source_revision: str):
    matches = [
        path
        for path in artifacts.rglob(filename)
        if path.parent.name.startswith("aiter-whl-packages-py")
    ]
    path = _one(matches, filename)
    require(path is not None, f"missing wheel artifact: {filename}")
    receipt = load_receipt(str(path) + ".receipt.json")
    verify_wheel(
        path,
        receipt,
        expected_source_revision=source_revision,
        require_clean_source=True,
    )
    return path, receipt


def _image_evidence(
    artifacts: Path, source_revision: str, catalog: dict, entries: list[dict]
) -> tuple[list[dict], list[str]]:
    images, problems = [], []
    approved = {entry["environment_lock_digest"] for entry in entries}
    for folder in sorted(artifacts.glob("image-candidate-*")):
        if not (folder / "record.json").is_file():
            problems.append(f"Image evidence is incomplete: {folder.name}")
            continue
        try:
            record = load_json(folder / "record.json")
            require(
                record["source_revision"] == source_revision,
                "image belongs to another source",
            )
            expected_plans = {check["plan_digest"] for check in record["checks"]}
            runs = []
            for path in folder.rglob("plan.json"):
                if (
                    path.parent.name == "run" or path.parent.name.endswith("-run")
                ) and load_json(path).get("plan_digest") in expected_plans:
                    runs.append(path.parent)
            require(
                len(runs) == len(expected_plans),
                "image execution evidence is missing or duplicated",
            )
            runs.sort(
                key=lambda directory: load_json(directory / "plan.json")["profile"]
                != "image"
            )
            plan = load_json(runs[0] / "plan.json")
            candidate = {
                "source_revision": source_revision,
                "artifact": record["wheel"],
                "scope": {
                    "profile": "image",
                    "architecture": plan["architecture"],
                    "environment_lock_digest": plan["environment_lock_digest"],
                },
                "plan": plan,
            }
            for run in runs:
                observed = load_json(run / "plan.json")
                require(
                    observed["environment_lock_digest"] in approved,
                    "image uses an undeclared environment",
                )
            evidence = {
                "record": str(folder / "record.json"),
                "archive": str(folder / "image.tar"),
                "runs": [str(run) for run in runs],
            }
            verified_images([evidence], candidate, catalog)
            images.append(
                {
                    "evidence": evidence,
                    "record": record,
                    "plans": [load_json(run / "plan.json") for run in runs],
                }
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            problems.append(f"Image {folder.name}: {error}")
    # Every wheel's published role must remain available as verified evidence.
    for filename in sorted({entry["wheel"] for entry in entries}):
        matches = [
            item for item in images if item["record"]["wheel"]["filename"] == filename
        ]
        targets = {item["record"]["role"] for item in matches}
        if not {"runtime", "development", "wheelhouse"} <= targets:
            problems.append(
                f"Missing runtime/development/wheelhouse image evidence for {filename}"
            )
        required_clients = {
            entry["profile"]
            for entry in entries
            if entry["wheel"] == filename and entry["profile"] in ("vllm", "sglang")
        }
        for item in matches:
            observed_profiles = {check["profile"] for check in item["record"]["checks"]}
            for client in required_clients:
                if client + "-image" not in observed_profiles:
                    problems.append(
                        f"Missing {client} inheritance evidence for {filename} {item['record']['role']}"
                    )
    return images, problems


def finalize(
    artifacts: Path,
    store: Path,
    output: Path,
    *,
    source_revision: str,
    run_id: str,
    owner: str,
    delta: str,
    job_results: dict,
    catalog: dict,
    matrix: Path | None = None,
    jobs_evidence: Path | None = None,
) -> dict:
    """Record failures successfully; publishing remains a separate conditional write."""
    require(
        isinstance(source_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", source_revision),
        "controller requires the exact source revision",
    )
    require(
        all(
            isinstance(value, str) and value.strip() for value in (run_id, owner, delta)
        ),
        "controller needs run identity, owner and delta",
    )
    artifacts, store, output = Path(artifacts), Path(store), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observed_jobs, blockers = _jobs(job_results)
    write_json(output / "job-results.json", observed_jobs)
    matrix_path = matrix or _one(
        artifacts.glob("release-qualification-matrix/release-matrix.json"),
        "qualification matrix",
    )
    entries, outcomes, metric_problems = [], [], []
    if matrix_path is not None and matrix_path.is_file():
        try:
            payload = load_json(matrix_path)
            require(
                set(payload) == {"include"}
                and isinstance(payload["include"], list)
                and bool(payload["include"]),
                "qualification matrix is empty or invalid",
            )
            entries = payload["include"]
            for entry in entries:
                _entry(entry, source_revision, catalog)
            require(
                len({entry["label"] for entry in entries}) == len(entries),
                "qualification labels are duplicated",
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            entries = []
            blockers.append(f"Cannot use qualification matrix: {error}")
    else:
        blockers.append("Build or planning produced no qualification matrix.")
    require(
        not entries
        or not any(
            event["kind"] == "failure" and event["payload"]["run_id"] == run_id
            for event in read_state(store)["events"]
        ),
        "this workflow attempt already failed before planning; use a new attempt",
    )
    if not entries:
        event = _failure(
            store,
            run_id=run_id,
            source_revision=source_revision,
            owner=owner,
            delta=delta,
            problems=blockers,
        )
        outcomes.append(
            {"label": None, "status": "ERROR", "event_digest": event["event_digest"]}
        )
    images, image_problems = (
        _image_evidence(artifacts, source_revision, catalog, entries)
        if entries
        else ([], [])
    )
    blockers.extend(image_problems)
    attempt_input_digest = digest(
        {
            "source_revision": source_revision,
            "run_id": run_id,
            "owner": owner,
            "delta": delta,
            "jobs": observed_jobs,
            "matrix": entries,
            "images": [item["record"] for item in images],
            "blockers": blockers,
        }
    )
    for entry in entries:
        label = entry["label"]
        lock = _entry(entry, source_revision, catalog)
        scope = {
            "profile": entry["profile"],
            "architecture": entry["architecture"],
            "environment_lock_digest": digest(lock),
        }
        cell_run_id = run_id + ":" + label
        try:
            recorded_failure = next(
                (
                    event
                    for event in read_state(store)["events"]
                    if event["kind"] == "failure"
                    and event["payload"]["run_id"] == cell_run_id
                ),
                None,
            )
            if recorded_failure is not None:
                payload = recorded_failure["payload"]
                require(
                    payload["source_revision"] == source_revision
                    and payload["scope"] == scope
                    and payload["owner"] == owner
                    and payload["delta"] == delta
                    and payload["environment_lock"] == lock,
                    "retry changed a failed attempt's identity",
                )
                outcomes.append(
                    {
                        "label": label,
                        "status": "ERROR",
                        "event_digest": recorded_failure["event_digest"],
                    }
                )
                continue
            wheel, receipt = _wheel(artifacts, entry["wheel"], source_revision)
            artifact = receipt.artifact.to_dict()
            run = artifacts / ("support-evidence-" + label) / "run"
            planned = plan_tests(
                catalog,
                entry["profile"],
                [],
                receipt.source.to_dict(),
                artifacts=[artifact],
                architecture=entry["architecture"],
                environment_lock=lock,
            )
            if (run / "plan.json").is_file():
                actual = load_json(run / "plan.json")
                validate_plan(actual, catalog)
                require(
                    actual["source"] == receipt.source.to_dict()
                    and actual["artifacts"] == [artifact],
                    "execution source/artifact differs from wheel receipt",
                )
                require(
                    actual["profile"] == entry["profile"]
                    and actual["architecture"] == entry["architecture"]
                    and actual["environment_lock_digest"] == digest(lock),
                    "execution differs from the declared matrix cell",
                )
                require(
                    actual["selection"] == "complete-profile",
                    "execution profile was narrowed",
                )
                planned = actual
            release = {
                "schema_version": 1,
                "channel": "nightly-candidate",
                "source_revision": source_revision,
                "artifacts": [artifact],
            }
            release["release_digest"] = digest(release)
            state = read_state(store)
            previous = next(
                (
                    e
                    for e in state["events"]
                    if e["kind"] == "candidate"
                    and e["payload"].get("run_id") == cell_run_id
                ),
                None,
            )
            if previous is None:
                candidate = record_candidate(
                    store,
                    release,
                    planned,
                    wheel.parent,
                    catalog=catalog,
                    owner=owner,
                    delta=delta,
                    expected_digest=state["state_digest"],
                    run_id=cell_run_id,
                    attempt_input_digest=attempt_input_digest,
                )["event"]["payload"]
            else:
                candidate = previous["payload"]
                require(
                    candidate["plan"] == planned
                    and candidate["artifact"] == artifact
                    and candidate["owner"] == owner
                    and candidate["delta"] == delta
                    and candidate.get("attempt_input_digest") == attempt_input_digest,
                    "retry changed the candidate's sealed identity",
                )
            state = read_state(store)
            previous_result = next(
                (
                    e
                    for e in state["events"]
                    if e["kind"] == "result"
                    and e["payload"]["candidate_digest"]
                    == candidate["candidate_digest"]
                ),
                None,
            )
            relevant_profile = (
                planned["client"] + "-image"
                if planned["client"] in ("vllm", "sglang")
                else "image"
            )
            relevant_images = [
                item["evidence"]
                for item in images
                if item["record"]["wheel"] == artifact
                and any(
                    p["profile"] == relevant_profile
                    and p["architecture"] == entry["architecture"]
                    and p["environment_lock_digest"] == digest(lock)
                    for p in item["plans"]
                )
            ]
            if previous_result is not None:
                current = evaluate_result(
                    candidate,
                    run,
                    catalog=catalog,
                    images=relevant_images,
                    delivery_blockers=blockers,
                )
                require(
                    current == previous_result["payload"],
                    "retry changed the sealed qualification evidence; use a new workflow attempt",
                )
                result = previous_result
            else:
                result = record_result(
                    store,
                    candidate["candidate_digest"],
                    run,
                    catalog=catalog,
                    expected_digest=state["state_digest"],
                    images=relevant_images,
                    delivery_blockers=blockers,
                )["event"]
            outcomes.append(
                {
                    "label": label,
                    "status": result["payload"]["status"],
                    "event_digest": result["event_digest"],
                    "candidate_digest": candidate["candidate_digest"],
                }
            )
            for path in sorted(run.glob("*/attempt-*/*.execution.json")):
                try:
                    execution = load_json(path)
                    record_interval(
                        output / "observations",
                        path,
                        phase="test",
                        source_revision=source_revision,
                        profile=entry["profile"],
                        run_id=execution["run_id"],
                        artifact=(
                            artifact
                            if artifact in execution.get("artifacts", [])
                            else None
                        ),
                    )
                except (ValueError, TypeError, KeyError, OSError) as error:
                    metric_problems.append(f"{label}/{path.name}: {error}")
        except (ValueError, TypeError, KeyError, OSError) as error:
            event = _failure(
                store,
                run_id=cell_run_id,
                source_revision=source_revision,
                owner=owner,
                delta=delta,
                problems=[*blockers, str(error)],
                scope=scope,
                environment_lock=lock,
            )
            outcomes.append(
                {
                    "label": label,
                    "status": "ERROR",
                    "event_digest": event["event_digest"],
                }
            )
    if jobs_evidence is not None:
        try:
            record_workflow_jobs(
                output / "observations",
                jobs_evidence,
                source_revision=source_revision,
                run_id=run_id,
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            metric_problems.append(f"Workflow job timings: {error}")
    metrics = summarize(output / "observations", channels=store)
    write_json(output / "metrics.json", metrics)
    state = read_state(store)
    # A state-addressed published view must be reproducible on an unchanged retry.
    write_json(
        output / "index.json",
        channel_index(store, as_of=state["events"][-1]["created_utc"]),
    )
    result = {
        "source_revision": source_revision,
        "run_id": run_id,
        "state_digest": read_state(store)["state_digest"],
        "status": (
            "PASS"
            if not blockers
            and outcomes
            and all(item["status"] == "PASS" for item in outcomes)
            else "FAIL"
        ),
        "outcomes": outcomes,
        "delivery_blockers": blockers,
        "metric_problems": metric_problems,
    }
    result["finalization_digest"] = digest(result)
    write_json(output / "finalization.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("finalize")
    for name in ("artifacts", "store", "output"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--matrix", type=Path)
    command.add_argument("--catalog", type=Path)
    command.add_argument("--jobs-evidence", type=Path)
    for name in ("source-revision", "run-id", "owner", "delta", "job-results"):
        command.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(
            args.artifacts,
            args.store,
            args.output,
            source_revision=args.source_revision,
            run_id=args.run_id,
            owner=args.owner,
            delta=args.delta,
            job_results=json.loads(args.job_results),
            catalog=load_catalog(args.catalog),
            matrix=args.matrix,
            jobs_evidence=args.jobs_evidence,
        )
        print(json.dumps(result, sort_keys=True))
        # Successfully recording a failed night is itself a successful collector run.
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"channel delivery: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
