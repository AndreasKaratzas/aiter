"""Keep local candidate history, qualified profile pointers and exact rollback."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ci.common.json import digest, load_json, require
from ci.qualification.environments import validate_lock
from ci.qualification.plan import validate_plan
from ci.qualification.report import check_results
from ci.release.wheels import load_receipt, verify_wheel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: str) -> datetime:
    require(isinstance(value, str), "timestamp must be an ISO UTC string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() is not None, "timestamp requires an explicit timezone")
    require(parsed.utcoffset().total_seconds() == 0, "timestamp must use UTC")
    return parsed


def _sealed(value: dict, field: str) -> dict:
    return {**value, field: digest(value)}


def _verify(value: dict, field: str) -> None:
    require(isinstance(value, dict), f"missing {field} object")
    require(
        value.get(field) == digest({k: v for k, v in value.items() if k != field}),
        f"{field} mismatch",
    )


def _text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} is required")


def _empty() -> dict:
    return _sealed({"schema_version": 1, "events": []}, "state_digest")


def read_state(store: Path) -> dict:
    """Validate a complete event chain; hashes identify content, not its author."""
    path = Path(store) / "state.json"
    state = load_json(path) if path.exists() else _empty()
    require(
        set(state) == {"schema_version", "events", "state_digest"},
        "invalid channel state",
    )
    require(
        state["schema_version"] == 1 and isinstance(state["events"], list),
        "invalid channel schema",
    )
    _verify(state, "state_digest")
    previous, last_time, candidates, qualified = None, None, {}, {}
    completed = set()
    for number, event in enumerate(state["events"], 1):
        require(
            set(event)
            == {
                "sequence",
                "previous",
                "created_utc",
                "kind",
                "payload",
                "event_digest",
            },
            "invalid channel event",
        )
        _verify(event, "event_digest")
        require(
            event["sequence"] == number and event["previous"] == previous,
            "broken channel history",
        )
        moment = timestamp(event["created_utc"])
        require(
            last_time is None or moment >= last_time, "channel time moved backwards"
        )
        payload = event["payload"]
        if event["kind"] == "candidate":
            require(
                payload["candidate_digest"] not in candidates, "duplicate candidate"
            )
            _verify(payload, "candidate_digest")
            plan = payload["plan"]
            _verify(plan, "plan_digest")
            require(
                plan["selection"] == "complete-profile",
                "stored candidate narrows qualification",
            )
            require(
                plan["artifacts"] == [payload["artifact"]]
                and plan["source"]["revision"] == payload["source_revision"],
                "stored candidate identity differs from plan",
            )
            require(
                payload["scope"]
                == {
                    "profile": plan["profile"],
                    "architecture": plan["architecture"],
                    "environment_lock_digest": plan["environment_lock_digest"],
                },
                "stored candidate scope differs from plan",
            )
            require(
                payload["scope_id"] == digest(payload["scope"]),
                "stored candidate scope digest differs",
            )
            require(
                plan["environment_lock"]["status"] == "supported"
                and plan["environment_lock_digest"] == digest(plan["environment_lock"]),
                "stored candidate environment is not supported",
            )
            candidates[payload["candidate_digest"]] = payload
        elif event["kind"] == "result":
            require(
                payload["candidate_digest"] in candidates, "unknown result candidate"
            )
            require(
                payload["candidate_digest"] not in completed,
                "duplicate candidate result",
            )
            completed.add(payload["candidate_digest"])
            require(
                payload["status"] in ("PASS", "FAIL", "ERROR"),
                "invalid qualification status",
            )
            _verify(payload["report"], "report_digest")
            blockers = payload.get("delivery_blockers", [])
            require(
                isinstance(blockers, list)
                and all(isinstance(item, str) and item.strip() for item in blockers),
                "invalid stored delivery blockers",
            )
            require(
                not blockers or payload["status"] == "FAIL",
                "blocked candidate cannot pass",
            )
            require(
                payload["report"]["status"] in ("PASS", "FAIL", "ERROR"),
                "invalid stored test report status",
            )
            require(
                payload["report"]["status"] == payload["status"]
                or (
                    payload["status"] == "FAIL"
                    and bool(payload.get("delivery_blockers"))
                ),
                "stored report status differs",
            )
            if payload["status"] == "PASS":
                require(
                    payload["report"].get("plan_digest")
                    == candidates[payload["candidate_digest"]]["plan"]["plan_digest"]
                    and not payload["report"].get("problems"),
                    "stored passing report differs from candidate",
                )
                qualified[event["event_digest"]] = candidates[
                    payload["candidate_digest"]
                ]
        elif event["kind"] == "failure":
            _verify(payload, "failure_digest")
            require(
                re.fullmatch(r"[0-9a-f]{40}", payload["source_revision"]),
                "failure source is invalid",
            )
            require(
                isinstance(payload["problems"], list)
                and bool(payload["problems"])
                and all(isinstance(p, str) and p for p in payload["problems"]),
                "failure details are missing",
            )
            if payload["scope"] is not None:
                _failure_scope(payload["scope"], payload["environment_lock"])
                require(
                    payload["scope"]["environment_lock_digest"]
                    == digest(payload["environment_lock"]),
                    "failure scope differs from environment",
                )
            else:
                require(
                    payload["environment_lock"] is None,
                    "unscoped failure has an environment",
                )
            _text(payload["owner"], "failure owner")
            _text(payload["run_id"], "failure executor")
            _text(payload["delta"], "failure delta")
        elif event["kind"] == "rollback":
            require(
                payload["qualification_digest"] in qualified,
                "rollback is not qualified",
            )
            require(
                payload["scope"] == qualified[payload["qualification_digest"]]["scope"],
                "rollback scope changed",
            )
            _text(payload["owner"], "rollback owner")
            _text(payload["reason"], "rollback reason")
        else:
            raise ValueError("unknown channel event kind")
        previous, last_time = event["event_digest"], moment
    return state


@contextmanager
def _locked(store: Path):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    with (store / ".lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield store


def _append(store: Path, kind: str, payload: dict, expected_digest: str | None) -> dict:
    with _locked(store) as directory:
        state = read_state(directory)
        require(
            expected_digest == state["state_digest"]
            or (expected_digest is None and not state["events"]),
            "channel changed; reread state before updating",
        )
        created = utc_now()
        if state["events"]:
            require(
                timestamp(created) >= timestamp(state["events"][-1]["created_utc"]),
                "clock moved backwards",
            )
        event = _sealed(
            {
                "sequence": len(state["events"]) + 1,
                "previous": (
                    state["events"][-1]["event_digest"] if state["events"] else None
                ),
                "created_utc": created,
                "kind": kind,
                "payload": payload,
            },
            "event_digest",
        )
        updated = _sealed(
            {"schema_version": 1, "events": [*state["events"], event]}, "state_digest"
        )
        fd, temporary = tempfile.mkstemp(prefix=".state-", dir=directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(updated, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "state.json")
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"state_digest": updated["state_digest"], "event": event}


def record_candidate(
    store: Path,
    release: dict,
    plan: dict,
    wheel_dir: Path,
    *,
    catalog: dict,
    owner: str,
    delta: str,
    expected_digest: str | None = None,
    run_id: str | None = None,
    attempt_input_digest: str | None = None,
) -> dict:
    """Register verified bytes and their complete, locked qualification plan."""
    _text(owner, "investigating owner")
    _text(delta, "delta from the previous qualified build")
    if run_id is not None:
        _text(run_id, "candidate executor")
    if attempt_input_digest is not None:
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", attempt_input_digest),
            "invalid attempt input digest",
        )
    _verify(release, "release_digest")
    validate_plan(plan, catalog)
    require(
        plan["selection"] == "complete-profile", "channel requires the complete profile"
    )
    require(len(plan["artifacts"]) == 1, "channel candidate must bind one wheel")
    require(
        any(
            group["gpus"] > 0 and group["environment"].get("AITER_CI_PROBE") != "native"
            for group in plan["groups"].values()
        ),
        "channel requires installed-wheel GPU qualification",
    )
    require(
        plan["source"]["revision"] == release["source_revision"],
        "candidate source differs from release",
    )
    lock = plan.get("environment_lock")
    require(
        isinstance(lock, dict) and lock.get("status") == "supported",
        "channel requires a supported environment lock",
    )
    require(
        plan.get("environment_lock_digest") == digest(lock),
        "environment lock digest mismatch",
    )
    artifact = plan["artifacts"][0]
    require(artifact in release["artifacts"], "candidate wheel is absent from release")
    wheel = Path(wheel_dir) / artifact["filename"]
    receipt = load_receipt(str(wheel) + ".receipt.json")
    verify_wheel(
        wheel,
        receipt,
        expected_source_revision=release["source_revision"],
        require_clean_source=True,
    )
    require(
        receipt.artifact.to_dict() == artifact,
        "candidate wheel differs from verified bytes",
    )
    scope = {
        "profile": plan["profile"],
        "architecture": plan["architecture"],
        "environment_lock_digest": plan["environment_lock_digest"],
    }
    payload = _sealed(
        {
            "scope": scope,
            "scope_id": digest(scope),
            "plan": plan,
            "release_digest": release["release_digest"],
            "source_revision": release["source_revision"],
            "artifact": artifact,
            "owner": owner,
            "delta": delta,
            "run_id": run_id,
            "attempt_input_digest": attempt_input_digest,
        },
        "candidate_digest",
    )
    require(
        not any(
            e["kind"] == "candidate"
            and e["payload"]["candidate_digest"] == payload["candidate_digest"]
            for e in read_state(store)["events"]
        ),
        "candidate already recorded",
    )
    return _append(store, "candidate", payload, expected_digest)


def _candidate(state: dict, candidate_digest: str) -> dict:
    for event in state["events"]:
        if (
            event["kind"] == "candidate"
            and event["payload"]["candidate_digest"] == candidate_digest
        ):
            return event["payload"]
    raise ValueError("unknown candidate")


def verified_images(evidence: list[dict], candidate: dict, catalog: dict) -> list[dict]:
    """Recheck image bytes and every referenced execution before retaining them."""
    from ci.release.images import verify_archive

    records = []
    for item in evidence:
        record = load_json(item["record"])
        verify_archive(Path(item["archive"]), record)
        require(
            record["wheel"] == candidate["artifact"], "image contains another wheel"
        )
        require(
            record["source_revision"] == candidate["source_revision"],
            "image source differs",
        )
        checks, matches_scope = [], False
        for directory in item["runs"]:
            directory = Path(directory)
            plan = load_json(directory / "plan.json")
            require(
                plan["artifacts"] == [candidate["artifact"]],
                "image test artifact differs",
            )
            require(
                plan["architecture"] == candidate["scope"]["architecture"],
                "image architecture differs",
            )
            require(
                plan.get("environment_lock", {}).get("status") == "supported",
                "image environment is not supported",
            )
            require(
                plan["selection"] == "complete-profile",
                "image test profile is narrowed",
            )
            report = check_results(plan, catalog, directory)
            require(report["status"] == "PASS", "image test did not pass")
            client = candidate["plan"]["client"]
            relevant_profile = (
                client + "-image" if client in ("vllm", "sglang") else "image"
            )
            matches_scope |= (
                plan["profile"] == relevant_profile
                and plan["environment_lock_digest"]
                == candidate["scope"]["environment_lock_digest"]
            )
            checks.append(
                {
                    "profile": plan["profile"],
                    "plan_digest": plan["plan_digest"],
                    "report_digest": report["report_digest"],
                    **(
                        {
                            "environment_lock_digest": plan["environment_lock_digest"],
                            "executor_image": plan["executor_image"],
                        }
                        if any(
                            "environment_lock_digest" in check
                            for check in record["checks"]
                        )
                        else {}
                    ),
                }
            )
        require(
            matches_scope,
            "image has no complete check for the candidate environment and consumer",
        )
        require(
            sorted(checks, key=digest) == sorted(record["checks"], key=digest),
            "image evidence does not match all declared checks",
        )
        require(
            record["image_id"] not in {r["image_id"] for r in records},
            "duplicate image",
        )
        records.append(record)
    return records


def record_result(
    store: Path,
    candidate_digest: str,
    run: Path,
    *,
    catalog: dict,
    expected_digest: str,
    images: list[dict] | None = None,
    delivery_blockers: list[str] | None = None,
) -> dict:
    """Retain failures and missing jobs; only independently passing evidence advances."""
    state = read_state(store)
    candidate = _candidate(state, candidate_digest)
    require(
        not any(
            e["kind"] == "result"
            and e["payload"]["candidate_digest"] == candidate_digest
            for e in state["events"]
        ),
        "candidate already has a result; create a new candidate for rework",
    )
    payload = evaluate_result(
        candidate,
        run,
        catalog=catalog,
        images=images,
        delivery_blockers=delivery_blockers,
    )
    return _append(store, "result", payload, expected_digest)


def evaluate_result(
    candidate: dict,
    run: Path,
    *,
    catalog: dict,
    images: list[dict] | None = None,
    delivery_blockers: list[str] | None = None,
) -> dict:
    """Reconstruct one result without modifying history, including on a retry."""
    image_records = []
    blockers = delivery_blockers or []
    require(
        isinstance(blockers, list)
        and all(isinstance(item, str) and bool(item.strip()) for item in blockers),
        "invalid delivery blockers",
    )
    try:
        require(
            load_json(Path(run) / "plan.json") == candidate["plan"],
            "execution uses another candidate plan",
        )
        report = check_results(candidate["plan"], catalog, Path(run))
        if report["status"] == "PASS":
            image_records = verified_images(images or [], candidate, catalog)
    except (ValueError, TypeError, KeyError, OSError) as error:
        report = _sealed({"status": "ERROR", "problems": [str(error)]}, "report_digest")
    payload = {
        "candidate_digest": candidate["candidate_digest"],
        "status": "FAIL" if blockers else report["status"],
        "report": report,
        "images": image_records,
        "delivery_blockers": blockers,
    }
    return payload


def _failure_scope(scope: dict, environment_lock: dict) -> None:
    require(
        isinstance(scope, dict)
        and set(scope) == {"profile", "architecture", "environment_lock_digest"},
        "invalid failure scope",
    )
    _text(scope["profile"], "failure profile")
    require(
        scope["architecture"] in ("gfx942", "gfx950", "gfx1250"),
        "unknown failure architecture",
    )
    validate_lock(environment_lock)
    require(
        environment_lock["status"] == "supported",
        "failure scope must use a supported environment",
    )
    require(
        scope["environment_lock_digest"] == digest(environment_lock),
        "failure environment differs from scope",
    )


def record_failure(
    store: Path,
    *,
    source_revision: str,
    run_id: str,
    owner: str,
    delta: str,
    problems: list[str],
    expected_digest: str | None,
    scope: dict | None = None,
    environment_lock: dict | None = None,
) -> dict:
    """Expose work that failed before verified candidate bytes or a plan existed."""
    require(
        isinstance(source_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", source_revision),
        "failure source requires a full revision",
    )
    for value, label in (
        (run_id, "failure executor"),
        (owner, "failure owner"),
        (delta, "failure delta"),
    ):
        _text(value, label)
    require(
        isinstance(problems, list)
        and bool(problems)
        and all(isinstance(p, str) and p.strip() for p in problems),
        "failure details are required",
    )
    if scope is not None:
        _failure_scope(scope, environment_lock)
    else:
        require(
            environment_lock is None, "unscoped failure cannot declare an environment"
        )
    payload = _sealed(
        {
            "source_revision": source_revision,
            "run_id": run_id,
            "owner": owner,
            "delta": delta,
            "problems": problems,
            "scope": scope,
            "environment_lock": environment_lock,
        },
        "failure_digest",
    )
    return _append(store, "failure", payload, expected_digest)


def rollback(
    store: Path,
    qualification_digest: str,
    *,
    owner: str,
    reason: str,
    expected_digest: str,
) -> dict:
    """Restore a previously qualified exact identity; never rebuild or rename bytes."""
    _text(owner, "rollback owner")
    _text(reason, "rollback reason")
    state = read_state(store)
    event = next(
        (e for e in state["events"] if e["event_digest"] == qualification_digest), None
    )
    require(
        event is not None
        and event["kind"] == "result"
        and event["payload"]["status"] == "PASS",
        "rollback requires a previously passing qualification",
    )
    candidate = _candidate(state, event["payload"]["candidate_digest"])
    return _append(
        store,
        "rollback",
        {
            "scope": candidate["scope"],
            "qualification_digest": qualification_digest,
            "owner": owner,
            "reason": reason,
        },
        expected_digest,
    )


def channel_index(store: Path, *, as_of: str | None = None) -> dict:
    """Render newest candidate, retained last good, failures and honest age per profile."""
    state = read_state(store)
    observed = timestamp(as_of or utc_now())
    scopes, candidates, results, unscoped_failures = {}, {}, {}, []
    for event in state["events"]:
        require(
            observed >= timestamp(event["created_utc"]), "index time precedes history"
        )
        payload = event["payload"]
        if event["kind"] == "candidate":
            candidates[payload["candidate_digest"]] = event
            scope = scopes.setdefault(
                payload["scope_id"],
                {
                    "scope": payload["scope"],
                    "newest_candidate": None,
                    "last_qualified": None,
                },
            )
            scope["newest_candidate"] = {
                "candidate_digest": payload["candidate_digest"],
                "created_utc": event["created_utc"],
                "source_revision": payload["source_revision"],
                "artifact": payload["artifact"],
                "plan_digest": payload["plan"]["plan_digest"],
                "environment_lock": payload["plan"]["environment_lock"],
                "owner": payload["owner"],
                "delta": payload["delta"],
                "status": "PENDING",
                "problems": ["Qualification has not completed."],
            }
        elif event["kind"] == "result":
            candidate = candidates[payload["candidate_digest"]]["payload"]
            scope = scopes[candidate["scope_id"]]
            newest = (
                scope["newest_candidate"]["candidate_digest"]
                == payload["candidate_digest"]
            )
            if newest:
                scope["newest_candidate"].update(
                    status=payload["status"],
                    problems=payload["report"].get("problems", [])
                    + payload.get("delivery_blockers", []),
                )
            if payload["status"] == "PASS":
                qualified = {
                    "qualification_digest": event["event_digest"],
                    "candidate_digest": candidate["candidate_digest"],
                    "qualified_utc": event["created_utc"],
                    "source_revision": candidate["source_revision"],
                    "artifact": candidate["artifact"],
                    "images": payload["images"],
                    "plan_digest": candidate["plan"]["plan_digest"],
                    "report_digest": payload["report"]["report_digest"],
                }
                results[event["event_digest"]] = qualified
                # A slow older run cannot silently undo a newer candidate decision.
                barrier = scope.get("rollback", {}).get("sequence", 0)
                after_decision = (
                    candidates[payload["candidate_digest"]]["sequence"] > barrier
                )
                if newest and after_decision:
                    scope["last_qualified"] = qualified
                    scope.pop("rollback", None)
        elif event["kind"] == "failure":
            failed = {
                "candidate_digest": payload["failure_digest"],
                "created_utc": event["created_utc"],
                "source_revision": payload["source_revision"],
                "artifact": None,
                "plan_digest": None,
                "environment_lock": payload["environment_lock"],
                "owner": payload["owner"],
                "delta": payload["delta"],
                "status": "ERROR",
                "problems": payload["problems"],
                "run_id": payload["run_id"],
            }
            if payload["scope"] is None:
                unscoped_failures.append(failed)
            else:
                scope = scopes.setdefault(
                    digest(payload["scope"]),
                    {
                        "scope": payload["scope"],
                        "newest_candidate": None,
                        "last_qualified": None,
                    },
                )
                scope["newest_candidate"] = failed
        else:
            scope = scopes[digest(payload["scope"])]
            scope["last_qualified"] = results[payload["qualification_digest"]]
            scope["rollback"] = {
                "sequence": event["sequence"],
                "created_utc": event["created_utc"],
                "owner": payload["owner"],
                "reason": payload["reason"],
            }
    for scope in scopes.values():
        qualified = scope["last_qualified"]
        scope["last_qualified_age_seconds"] = (
            (observed - timestamp(qualified["qualified_utc"])).total_seconds()
            if qualified
            else None
        )
    return {
        "schema_version": 1,
        "state_digest": state["state_digest"],
        "as_of_utc": observed.isoformat(),
        "profiles": scopes,
        "unscoped_failures": unscoped_failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser(
        "candidate", help="record verified candidate bytes and their complete plan"
    )
    candidate.add_argument("--release", type=Path, required=True)
    candidate.add_argument("--plan", type=Path, required=True)
    candidate.add_argument("--wheel-dir", type=Path, required=True)
    candidate.add_argument("--owner", required=True)
    candidate.add_argument("--delta", required=True)
    result = commands.add_parser(
        "result", help="reconstruct qualification, retaining any failure"
    )
    result.add_argument("--candidate", required=True)
    result.add_argument("--run", type=Path, required=True)
    result.add_argument(
        "--images", type=Path, help="JSON list of record/archive/runs evidence objects"
    )
    restore = commands.add_parser(
        "rollback", help="restore a previously qualified exact identity locally"
    )
    restore.add_argument("--qualification", required=True)
    restore.add_argument("--owner", required=True)
    restore.add_argument("--reason", required=True)
    index = commands.add_parser(
        "index", help="show newest candidates and last qualified artifacts"
    )
    for command in (candidate, result, restore, index):
        command.add_argument("--store", type=Path, required=True)
        command.add_argument("--output", type=Path)
    candidate.add_argument("--expected-digest")
    for command in (result, restore):
        command.add_argument("--expected-digest", required=True)
    args = parser.parse_args(argv)
    try:
        from ci.common.json import write_json
        from ci.qualification.catalog import load_catalog

        if args.command == "candidate":
            output = record_candidate(
                args.store,
                load_json(args.release),
                load_json(args.plan),
                args.wheel_dir,
                catalog=load_catalog(args.catalog),
                owner=args.owner,
                delta=args.delta,
                expected_digest=args.expected_digest,
            )
        elif args.command == "result":
            output = record_result(
                args.store,
                args.candidate,
                args.run,
                catalog=load_catalog(args.catalog),
                expected_digest=args.expected_digest,
                images=load_json(args.images) if args.images else None,
            )
        elif args.command == "rollback":
            output = rollback(
                args.store,
                args.qualification,
                owner=args.owner,
                reason=args.reason,
                expected_digest=args.expected_digest,
            )
        else:
            output = channel_index(args.store)
        if args.output:
            write_json(args.output, output)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"channels: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
