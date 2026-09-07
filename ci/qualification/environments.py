"""Approved runtime requirements that a test result must actually satisfy."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from ci.common.json import digest, require

IMAGE = re.compile(r"[A-Za-z0-9_./:-]+@sha256:[0-9a-f]{64}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")


def validate_lock(lock: dict) -> dict:
    require(
        isinstance(lock, dict)
        and set(lock) - {"candidate_substitution"}
        == {
            "schema_version",
            "id",
            "status",
            "image",
            "python",
            "rocm",
            "packages",
            "torch_revision",
            "frameworks",
        },
        "environment lock has unknown or missing fields",
    )
    require(
        type(lock["schema_version"]) is int and lock["schema_version"] == 1,
        "unsupported environment lock schema",
    )
    require(
        isinstance(lock["id"], str) and re.fullmatch(r"[a-z][a-z0-9-]*", lock["id"]),
        "invalid environment id",
    )
    require(
        lock["status"] in ("supported", "canary", "development"),
        "invalid environment status",
    )
    require(
        (lock["status"] == "development" and lock["image"] is None)
        or (isinstance(lock["image"], str) and IMAGE.fullmatch(lock["image"])),
        "environment image must be digest-pinned",
    )
    require(
        isinstance(lock["python"], str)
        and re.fullmatch(r"3\.(?:10|12)", lock["python"]),
        "unsupported environment Python tuple",
    )
    require(
        isinstance(lock["rocm"], str) and re.fullmatch(r"7\.[012]", lock["rocm"]),
        "unsupported environment ROCm tuple",
    )
    require(
        isinstance(lock["torch_revision"], str)
        and SHA.fullmatch(lock["torch_revision"]),
        "environment needs the exact Torch source revision",
    )
    packages = lock["packages"]
    require(
        isinstance(packages, dict)
        and set(packages)
        in ({"torch", "triton", "flydsl"}, {"torch", "pytorch-triton-rocm", "flydsl"}),
        "declare Torch, Triton and FlyDSL versions explicitly",
    )
    for name, version in packages.items():
        require(
            (version is None and name == "flydsl")
            or (
                isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version)
            ),
            "package versions must be exact; only unused FlyDSL may be absent",
        )
    require(
        isinstance(lock["frameworks"], dict)
        and all(
            isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in lock["frameworks"]
        ),
        "unknown framework requirement",
    )
    for name, framework in lock["frameworks"].items():
        require(
            isinstance(framework, dict) and set(framework) == {"version", "revision"},
            f"invalid {name} requirement",
        )
        require(
            isinstance(framework["version"], str)
            and re.fullmatch(r"[A-Za-z0-9_.+!-]+", framework["version"]),
            f"{name} needs an exact version",
        )
        require(
            isinstance(framework["revision"], str)
            and SHA.fullmatch(framework["revision"]),
            f"{name} needs an exact source revision",
        )
    if "candidate_substitution" in lock:
        from ci.qualification.substitution import validate_policy

        require(
            lock["status"] == "development",
            "supported/canary image locks cannot relax candidate dependency pins",
        )
        policy = validate_policy(lock["candidate_substitution"])
        require(
            policy["rule"]["consumer"] in lock["frameworks"],
            "candidate substitution consumer is absent from environment",
        )
    return lock


def load_registry(payload: str, *, allow_empty: bool = False) -> dict[str, dict]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate environment registry key")
            result[key] = value
        return result

    registry = json.loads(payload, object_pairs_hook=unique)
    require(
        isinstance(registry, dict) and (bool(registry) or allow_empty),
        "configure AITER_SUPPORT_PROFILES with approved environment locks",
    )
    for key, lock in registry.items():
        require(
            isinstance(key, str) and re.fullmatch(r"[a-z][a-z0-9-]*", key),
            "invalid environment registry key",
        )
        validate_lock(lock)
    require(
        len({lock["id"] for lock in registry.values()}) == len(registry),
        "environment ids must be unique",
    )
    return registry


def require_profile_environment(
    lock: dict, client: str, *, supported: bool = False
) -> None:
    validate_lock(lock)
    if supported:
        require(
            lock["status"] == "supported",
            "qualification requires an approved supported environment; canaries remain advisory",
        )
    if client not in ("aiter", "pytorch"):
        require(
            re.fullmatch(r"[a-z][a-z0-9_]*", client),
            "client needs an explicit supported module/distribution identity; hyphenated IDs are not normalized",
        )
        require(
            client in lock["frameworks"],
            f"environment does not declare {client} identity",
        )


def dependency_problems(distribution) -> list[str]:
    """Check declared default dependencies; extras need their own client profile."""
    from packaging.requirements import Requirement

    problems = []
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            actual = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"missing dependency: {requirement}")
            continue
        if requirement.specifier and actual not in requirement.specifier:
            problems.append(f"dependency {requirement} has installed version {actual}")
        if requirement.url:
            dependency = importlib.metadata.distribution(requirement.name)
            direct = json.loads(dependency.read_text("direct_url.json") or "{}")
            if direct.get("url") != requirement.url:
                problems.append(
                    f"dependency {requirement.name} has a different direct source"
                )
    return sorted(problems)


def checkout_observation(checkout: Path) -> dict:
    return {
        "checkout": str(checkout),
        "revision": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"]
            )
        ),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    "--no-textconv",
                    "HEAD",
                ]
            )
        ).hexdigest(),
    }


def framework_observation(name: str) -> dict:
    distribution = importlib.metadata.distribution(name)
    direct_text = distribution.read_text("direct_url.json")
    direct = json.loads(direct_text) if direct_text else {}
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    result = {
        "name": name,
        "origin": str(origin),
        "dependency_problems": dependency_problems(distribution),
        "version": distribution.version,
        "revision": None,
        "dirty": False,
        "direct_url": direct_text,
    }
    revision = direct.get("vcs_info", {}).get("commit_id")
    if isinstance(revision, str) and SHA.fullmatch(revision):
        result["revision"] = revision
    url = urlparse(direct.get("url", ""))
    if url.scheme == "file" and direct.get("dir_info", {}).get("editable"):
        checkout = Path(unquote(url.path)).resolve()
        require(
            origin.is_relative_to(checkout),
            f"imported {name} is outside its declared editable checkout",
        )
        result.update(checkout_observation(checkout))
    else:
        require(
            origin == Path(distribution.locate_file(name + "/__init__.py")).resolve(),
            f"imported {name} differs from its installed distribution",
        )
    if not direct and result["revision"] is None:
        # Legacy editable installs may expose only an egg-info directory. Infer
        # no source from a version suffix: require this actual imported file to
        # be tracked in the observed Git checkout before recording its revision.
        found = subprocess.run(
            ["git", "-C", str(origin.parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if found.returncode == 0:
            checkout = Path(found.stdout.strip()).resolve()
            if origin.is_relative_to(checkout):
                tracked = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "ls-files",
                        "--error-unmatch",
                        "--",
                        str(origin.relative_to(checkout)),
                    ],
                    capture_output=True,
                    check=False,
                )
                if tracked.returncode == 0:
                    result.update(checkout_observation(checkout))
    if url.scheme == "https" and url.netloc == "github.com":
        archive = re.search(r"/archive/([0-9a-f]{40})(?:\.tar\.gz|\.zip)$", url.path)
        if archive:
            result["revision"] = archive.group(1)
    if result["revision"] is None:
        from ci.clients.vllm.nightly import wheel_revision

        if name == "vllm":
            # Only an immutable official ROCm artifact URL, never a short version suffix.
            hashes = direct.get("archive_info", {}).get("hashes", {})
            if isinstance(hashes.get("sha256"), str) and re.fullmatch(
                r"[0-9a-f]{64}", hashes["sha256"]
            ):
                result["revision"] = wheel_revision(direct.get("url", ""))
    return result


def observe_requirements(lock: dict | None, client: str, torch) -> dict:
    result = {
        "packages": {},
        "torch_revision": getattr(torch.version, "git_version", None),
        "frameworks": {},
    }
    for module_name, providers in (
        ("torch", ("torch",)),
        ("triton", ("triton", "pytorch-triton-rocm")),
        ("flydsl", ("flydsl",)),
    ):
        installed = []
        for provider in providers:
            try:
                installed.append((provider, importlib.metadata.distribution(provider)))
            except importlib.metadata.PackageNotFoundError:
                pass
        if not installed:
            result["packages"][module_name] = None
            continue
        module = (
            torch if module_name == "torch" else importlib.import_module(module_name)
        )
        origin = Path(module.__file__).resolve()
        matching = [
            (name, distribution)
            for name, distribution in installed
            if origin
            == Path(distribution.locate_file(module_name + "/__init__.py")).resolve()
            and any(
                str(item) == module_name + "/__init__.py"
                for item in distribution.files or []
            )
        ]
        require(
            len(matching) == 1,
            f"imported {module_name} has no unique distribution owner",
        )
        provider, distribution = matching[0]
        result["packages"][provider] = distribution.version
    frameworks = set(lock["frameworks"]) if lock else set()
    if client not in ("aiter", "pytorch"):
        frameworks.add(client)
    result["frameworks"] = {
        name: framework_observation(name) for name in sorted(frameworks)
    }
    if lock and "candidate_substitution" in lock:
        from ci.qualification.substitution import observe

        result["candidate_substitution"] = observe(lock["candidate_substitution"])
    return result


def verify_native_observation(lock: dict, observed: dict) -> None:
    """Check the native worker tuple without requiring a Torch installation."""
    validate_lock(lock)
    require(
        ".".join(observed["python"].split(".")[:2]) == lock["python"],
        "native Python differs from approved environment",
    )
    version = observed["hip_runtime_version"]
    require(
        type(version) is int and version > 0, "native HIP runtime version is invalid"
    )
    major, remainder = divmod(version, 10000000)
    minor = remainder // 100000
    require(
        f"{major}.{minor}" == lock["rocm"],
        "native ROCm differs from approved environment",
    )
    require(
        observed.get("environment_lock_digest") == digest(lock),
        "native executor did not bind the approved environment lock",
    )


def verify_observation(lock: dict, observed: dict) -> None:
    validate_lock(lock)
    require(
        ".".join(observed["python"].split(".")[:2]) == lock["python"],
        "Python differs from approved environment",
    )
    require(
        ".".join(str(observed["hip"]).split(".")[:2]) == lock["rocm"],
        "ROCm differs from approved environment",
    )
    require(
        observed.get("packages") == lock["packages"],
        "Torch/DSL package versions differ from approved environment",
    )
    require(
        observed.get("torch_revision") == lock["torch_revision"],
        "Torch source revision differs from approved environment",
    )
    if "candidate_substitution" in lock:
        from ci.qualification.substitution import validate_observation

        actual_policy = observed.get("candidate_substitution")
        validate_observation(lock["candidate_substitution"], actual_policy)
        require(
            actual_policy["candidate_origin"] == observed.get("aiter_origin"),
            "candidate substitution origin differs from the qualified import",
        )
    else:
        require(
            "candidate_substitution" not in observed,
            "undeclared candidate substitution",
        )
    for name, expected in lock["frameworks"].items():
        actual = observed.get("frameworks", {}).get(name)
        require(isinstance(actual, dict), f"missing observed {name} identity")
        require(
            actual.get("version") == expected["version"]
            and actual.get("revision") == expected["revision"],
            f"{name} version/source revision differs from approved environment",
        )
        if lock["status"] == "supported":
            require(
                actual.get("dirty") is False, f"supported {name} source is modified"
            )
            require(
                actual.get("dependency_problems") == [],
                f"supported {name} has unmet or unverified declared dependencies",
            )
    require(
        observed.get("environment_lock_digest") == digest(lock),
        "executor did not bind the approved environment lock",
    )


def main():
    """Resolve a rolling canary once; never rewrite an approved baseline."""
    import argparse
    import platform

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--client", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = observe_requirements(None, args.client, torch)
    framework = actual["frameworks"][args.client]
    lock = validate_lock(
        {
            "schema_version": 1,
            "id": args.client + "-rolling-canary",
            "status": "canary",
            "image": args.image,
            "python": ".".join(platform.python_version().split(".")[:2]),
            "rocm": ".".join(str(torch.version.hip).split(".")[:2]),
            "packages": actual["packages"],
            "torch_revision": actual["torch_revision"],
            "frameworks": {
                args.client: {
                    "version": framework["version"],
                    "revision": framework["revision"],
                }
            },
        }
    )
    args.output.write_text(json.dumps(lock, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
