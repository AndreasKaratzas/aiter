"""Connect tested wheel bytes, OCI image content and consumer execution records."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import load_registry, require_profile_environment
from ci.qualification.report import check_results
from ci.release.artifacts import ArtifactFile, hash_file
from ci.release.notes import validate_notes
from ci.release.recipes import load_recipes
from ci.release.wheels import load_receipt, verify_wheel

IMAGE = re.compile(r"[A-Za-z0-9_./:-]+@sha256:[0-9a-f]{64}\Z")


def image_matrix(wheels: Path, release: dict, images: dict[str, dict]) -> dict:
    require(
        release.get("release_digest")
        == digest(
            {key: value for key, value in release.items() if key != "release_digest"}
        ),
        "release record digest mismatch",
    )
    require(
        release["channel"] in ("nightly-tested", "stable"),
        "images require qualified wheel records",
    )
    if release["channel"] == "stable":
        reviewed = release.get("reviewed_notes")
        require(
            isinstance(reviewed, dict),
            "stable images require reviewed release guidance",
        )
        require(
            validate_notes(release.get("release_notes", ""), reviewed.get("version"))
            == reviewed,
            "stable image notes differ from reviewed guidance",
        )
    recipes = load_recipes()
    entries = []
    for artifact in release["artifacts"]:
        path = wheels / artifact["filename"]
        receipt = load_receipt(str(path) + ".receipt.json")
        verify_wheel(
            path,
            receipt,
            expected_source_revision=release["source_revision"],
            require_clean_source=True,
        )
        require(
            receipt.artifact.to_dict() == artifact,
            "image wheel differs from qualified wheel",
        )
        python = (
            "310"
            if any(tag.startswith("cp310-") for tag in receipt.wheel.tags)
            else (
                "312"
                if any(tag.startswith("cp312-") for tag in receipt.wheel.tags)
                else None
            )
        )
        require(
            python is not None, "image wheel requires an explicit supported Python ABI"
        )
        rocm_match = re.search(r"\+rocm(7\.[012])\.", receipt.wheel.version)
        rocm = rocm_match.group(1) if rocm_match else "7.2"
        key = f"rocm{rocm.replace('.', '')}-py{python}"
        lock = images.get(key)
        require(
            isinstance(lock, dict), f"configure approved image environment for {key}"
        )
        require_profile_environment(lock, "aiter", supported=True)
        base = lock["image"]
        consumers = []
        if python == "312" and rocm == "7.2":
            for client in recipes["inheritance_clients"]:
                client_lock = images.get(client)
                require(
                    isinstance(client_lock, dict),
                    f"configure approved primary {client} environment",
                )
                require_profile_environment(client_lock, client, supported=True)
                require(
                    client_lock["python"] == lock["python"]
                    and client_lock["rocm"] == lock["rocm"],
                    "consumer base differs from primary wheel runtime tuple",
                )
                consumers.append(
                    {
                        "client": client,
                        "base_image": client_lock["image"],
                        "environment_lock": client_lock,
                        "profile": recipes["consumers"][client]["profile"],
                    }
                )
        for target in ("runtime", "development", "wheelhouse"):
            entries.append(
                {
                    "wheel": artifact["filename"],
                    "wheel_sha256": artifact["sha256"],
                    "source_revision": release["source_revision"],
                    "base_image": base,
                    "environment_lock": json.dumps(lock, sort_keys=True),
                    "consumers": json.dumps(consumers, sort_keys=True),
                    "target": target,
                    "label": f"{target}-{key}",
                }
            )
    require(
        bool(entries) and len({item["label"] for item in entries}) == len(entries),
        "ambiguous image support matrix",
    )
    return {"include": entries}


def image_record(
    wheel: Path,
    run: Path,
    inspection: dict,
    base_image: str,
    archive: Path,
    *,
    catalog: dict,
    consumers: list[dict] | None = None,
    execution_inspection: dict | None = None,
    role: str = "runtime",
) -> dict:
    require(
        IMAGE.fullmatch(base_image) is not None,
        "base image must use an immutable digest",
    )
    receipt = load_receipt(str(wheel) + ".receipt.json")
    verify_wheel(wheel, receipt, require_clean_source=True)
    image_id = inspection.get("Id", "")
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is not None,
        "invalid built image identity",
    )
    require(
        bool(inspection.get("RootFS", {}).get("Layers")), "image has no content layers"
    )
    require(role in ("runtime", "development", "wheelhouse"), "unknown image role")
    execution_id = (execution_inspection or inspection)["Id"]
    checks, inherited = [], []
    requirements = [
        {
            "run": run,
            "profile": "image",
            "client": "aiter",
            "base_image": base_image,
            "image_id": execution_id,
        }
    ]
    for consumer in consumers or []:
        require(
            set(consumer) == {"client", "run", "base_image", "inspection"},
            "invalid consumer image input",
        )
        client = consumer["client"]
        require(
            isinstance(client, str) and re.fullmatch(r"[a-z][a-z0-9-]*", client),
            "invalid image consumer",
        )
        require(
            client not in {item["client"] for item in inherited},
            "duplicate image consumer",
        )
        require(
            IMAGE.fullmatch(consumer["base_image"]),
            "consumer base must use an immutable digest",
        )
        profile = client + "-image"
        require(
            profile in catalog["profiles"]
            and catalog["profiles"][profile]["client"] == client,
            "unknown consumer image profile",
        )
        identity = consumer["inspection"].get("Id", "")
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", identity),
            "invalid consumer image identity",
        )
        inherited.append(
            {
                "client": client,
                "base_image": consumer["base_image"],
                "image_id": identity,
            }
        )
        requirements.append(
            {
                "run": consumer["run"],
                "profile": profile,
                "client": client,
                "base_image": consumer["base_image"],
                "image_id": identity,
            }
        )
    for requirement in requirements:
        directory = Path(requirement["run"])
        plan = load_json(directory / "plan.json")
        require(
            plan["artifacts"] == [receipt.artifact.to_dict()],
            "image tests did not bind the installed wheel",
        )
        require(
            plan["source"]["revision"] == receipt.source.revision,
            "image tests use another source revision",
        )
        require(
            plan["selection"] == "complete-profile", "image requires a complete profile"
        )
        require(
            plan["profile"] == requirement["profile"],
            "unexpected image qualification profile",
        )
        lock = plan.get("environment_lock")
        require(
            isinstance(lock, dict), "image check requires its approved environment lock"
        )
        require_profile_environment(lock, requirement["client"], supported=True)
        require(
            lock["image"] == requirement["base_image"],
            "image check environment differs from approved base",
        )
        require(
            plan.get("executor_image") == requirement["image_id"],
            "image check did not execute the inspected image",
        )
        result = check_results(plan, catalog, directory)
        require(
            result["status"] == "PASS", "image test evidence is incomplete or failed"
        )
        checks.append(
            {
                "profile": plan["profile"],
                "environment_lock_digest": plan["environment_lock_digest"],
                "executor_image": plan["executor_image"],
                "plan_digest": plan["plan_digest"],
                "report_digest": result["report_digest"],
            }
        )
    archive_sha, archive_size = hash_file(archive)
    record = {
        "schema_version": 2,
        "role": role,
        "execution_image_id": execution_id,
        "image_id": image_id,
        "base_image": base_image,
        "layers": inspection["RootFS"]["Layers"],
        "source_revision": receipt.source.revision,
        "wheel": receipt.artifact.to_dict(),
        "wheel_receipt_digest": receipt.receipt_digest,
        "archive": {
            "filename": archive.name,
            "sha256": archive_sha,
            "size_bytes": archive_size,
        },
        "checks": checks,
        "consumers": inherited,
    }
    actual_id, actual_layers = archive_identity(archive)
    require(
        actual_id == image_id and actual_layers == record["layers"],
        "exported image differs from the tested image",
    )
    if role == "wheelhouse":
        verify_wheelhouse(archive, record["wheel"])
    record["image_record_digest"] = digest(record)
    verify_archive(archive, record)
    return record


def archive_identity(archive: Path) -> tuple[str, list[str]]:
    """Read a docker-save archive without extracting any path or executing code."""
    import json
    import tarfile

    from ci.release.artifacts import validate_relative_path
    from ci.release.wheels import _unique_object

    with tarfile.open(archive) as source:
        members = {}
        for member in source.getmembers():
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            if name in ("", ".") and member.isdir():
                continue
            validate_relative_path(name.rstrip("/"), "image archive member")
            require(name not in members, "duplicate image archive member")
            require(
                member.isfile() or member.isdir(),
                "image archive links are not accepted",
            )
            members[name] = member

        def read_json_member(name):
            require(
                name in members
                and members[name].isfile()
                and members[name].size <= 4 * 1024 * 1024,
                "missing or oversized image metadata",
            )
            with source.extractfile(members[name]) as data:
                content = data.read()
            return content, json.loads(content, object_pairs_hook=_unique_object)

        _, manifests = read_json_member("manifest.json")
        require(
            isinstance(manifests, list) and len(manifests) == 1,
            "expected one image in the archive",
        )
        config_path = manifests[0]["Config"]
        validate_relative_path(config_path, "image config")
        config_bytes, config = read_json_member(config_path)
        layer_paths = manifests[0].get("Layers")
        rootfs = config.get("rootfs")
        require(
            isinstance(rootfs, dict) and rootfs.get("type") == "layers",
            "unsupported image rootfs type",
        )
        diff_ids = rootfs.get("diff_ids")
        require(
            isinstance(layer_paths, list)
            and isinstance(diff_ids, list)
            and len(layer_paths) == len(diff_ids)
            and bool(diff_ids)
            and all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in diff_ids
            ),
            "image layer count or digest format is invalid",
        )
        import hashlib

        for layer, expected in zip(layer_paths, diff_ids):
            validate_relative_path(layer, "image layer")
            require(layer in members and members[layer].isfile(), "missing image layer")
            actual = hashlib.sha256()
            with source.extractfile(members[layer]) as data:
                prefix = data.read(512)
                require(
                    not prefix.startswith(
                        (b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd")
                    ),
                    "compressed image layers are unsupported; use docker save with uncompressed layer tar entries",
                )
                actual.update(prefix)
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    actual.update(block)
            require(
                "sha256:" + actual.hexdigest() == expected,
                "image layer bytes do not match ordered rootfs diff IDs",
            )

        return "sha256:" + hashlib.sha256(config_bytes).hexdigest(), diff_ids


def verify_archive(archive: Path, record: dict) -> None:
    expected = {
        "schema_version",
        "role",
        "execution_image_id",
        "image_id",
        "base_image",
        "layers",
        "source_revision",
        "wheel",
        "wheel_receipt_digest",
        "archive",
        "checks",
        "image_record_digest",
    }
    require(isinstance(record, dict), "image record must be an object")
    expected.add("consumer_base" if record.get("schema_version") == 1 else "consumers")
    require(
        set(record) == expected,
        "image record has unknown or missing fields",
    )
    require(
        type(record["schema_version"]) is int and record["schema_version"] in (1, 2),
        "unsupported image record schema",
    )
    ArtifactFile.from_dict(record["wheel"])
    require(
        re.fullmatch(r"[0-9a-f]{40}", record["source_revision"]) is not None,
        "invalid image source revision",
    )
    require(
        IMAGE.fullmatch(record["base_image"]) is not None, "image base is not pinned"
    )
    require(
        isinstance(record["checks"], list) and bool(record["checks"]),
        "missing image checks",
    )
    for check in record["checks"]:
        require(
            set(check)
            == {
                "profile",
                "plan_digest",
                "report_digest",
                "environment_lock_digest",
                "executor_image",
            }
            and isinstance(check["profile"], str)
            and (
                check["profile"] == "image"
                or re.fullmatch(r"[a-z][a-z0-9-]*-image", check["profile"])
            ),
            "invalid image execution record",
        )
        require(
            all(
                re.fullmatch(r"sha256:[0-9a-f]{64}", check[key])
                for key in (
                    "plan_digest",
                    "report_digest",
                    "environment_lock_digest",
                    "executor_image",
                )
            ),
            "invalid image execution digest",
        )
    require(
        record.get("image_record_digest")
        == digest(
            {
                key: value
                for key, value in record.items()
                if key != "image_record_digest"
            }
        ),
        "image record digest mismatch",
    )
    sha, size = hash_file(archive)
    require(
        record["archive"]
        == {"filename": archive.name, "sha256": sha, "size_bytes": size},
        "image archive differs from qualified content",
    )
    require(
        {item["profile"] for item in record["checks"]} >= {"image"},
        "missing image execution record",
    )
    actual_id, actual_layers = archive_identity(archive)
    require(
        actual_id == record["image_id"] and actual_layers == record["layers"],
        "archive contains another image",
    )
    require(
        record["role"] in ("runtime", "development", "wheelhouse"), "invalid image role"
    )
    require(
        next(item for item in record["checks"] if item["profile"] == "image")[
            "executor_image"
        ]
        == record["execution_image_id"],
        "image execution identity differs from recorded check",
    )
    if record["role"] == "wheelhouse":
        verify_wheelhouse(archive, record["wheel"])
    else:
        require(
            record["execution_image_id"] == record["image_id"],
            "runtime execution used a different image",
        )
    if record["schema_version"] == 1:
        if record.get("consumer_base"):
            require(
                any(item["profile"] == "vllm-image" for item in record["checks"]),
                "missing required inheritance check",
            )
    else:
        consumers = record["consumers"]
        require(isinstance(consumers, list), "consumer records must be a list")
        names = []
        for consumer in consumers:
            require(
                isinstance(consumer, dict)
                and set(consumer) == {"client", "base_image", "image_id"},
                "invalid consumer image record",
            )
            client = consumer["client"]
            require(
                isinstance(client, str)
                and re.fullmatch(r"[a-z][a-z0-9-]*", client)
                and client not in names,
                "duplicate or invalid image consumer",
            )
            require(
                IMAGE.fullmatch(consumer["base_image"]), "consumer base must be pinned"
            )
            matches = [
                check
                for check in record["checks"]
                if check["profile"] == client + "-image"
            ]
            require(
                len(matches) == 1
                and matches[0]["executor_image"] == consumer["image_id"],
                "missing or mismatched inheritance check",
            )
            names.append(client)
        require(
            len({check["profile"] for check in record["checks"]})
            == len(record["checks"]),
            "duplicate image qualification profile",
        )
        require(
            {check["profile"] for check in record["checks"]}
            == {"image", *(name + "-image" for name in names)},
            "image checks differ from declared consumers",
        )


def verify_wheelhouse(archive: Path, artifact: dict) -> None:
    """A scratch wheelhouse has one layer containing exactly its declared wheel."""
    import hashlib
    import tarfile

    with tarfile.open(archive) as outer:
        manifests = json.load(outer.extractfile("manifest.json"))
        require(
            len(manifests) == 1 and len(manifests[0]["Layers"]) == 1,
            "wheelhouse must have one immutable export layer",
        )
        with tarfile.open(
            fileobj=outer.extractfile(manifests[0]["Layers"][0]), mode="r|"
        ) as layer:
            wheels = []
            for member in layer:
                if member.name.endswith(".whl"):
                    require(
                        member.isfile()
                        and member.name.lstrip("./")
                        == "opt/aiter-wheel/" + artifact["filename"],
                        "unexpected wheelhouse artifact",
                    )
                    stream = layer.extractfile(member)
                    value = hashlib.sha256()
                    while block := stream.read(1024 * 1024):
                        value.update(block)
                    require(
                        value.hexdigest() == artifact["sha256"]
                        and member.size == artifact["size_bytes"],
                        "wheelhouse bytes differ from qualified wheel",
                    )
                    wheels.append(member.name)
            require(
                len(wheels) == 1, "wheelhouse must contain exactly one qualified wheel"
            )


def verify_publications(directory: Path, release: dict) -> list[dict]:
    """Require every advertised wheel/OCI role before a channel can advance."""
    artifacts = {item["filename"]: item for item in release["artifacts"]}
    observed = set()
    publications = []
    for path in sorted(directory.glob("image-publication-*/publication.json")):
        publication = load_json(path)
        record = load_json(path.with_name("record.json"))
        require(
            record["image_record_digest"]
            == digest({k: v for k, v in record.items() if k != "image_record_digest"}),
            "image record digest mismatch",
        )
        require(
            publication["image_record_digest"] == record["image_record_digest"]
            and publication["image_id"] == record["image_id"],
            "published image differs from tested image",
        )
        require(
            publication["wheel"]
            == record["wheel"]
            == artifacts.get(record["wheel"]["filename"]),
            "published image has an unqualified wheel",
        )
        require(
            record["source_revision"] == release["source_revision"],
            "image source differs from release",
        )
        require(
            publication["references"]
            and all(IMAGE.fullmatch(value) for value in publication["references"]),
            "publication lacks immutable OCI references",
        )
        cell = (record["wheel"]["filename"], record["role"])
        require(cell not in observed, "duplicate image publication role")
        observed.add(cell)
        publications.append(publication)
    required = {
        (name, role)
        for name in artifacts
        for role in ("runtime", "development", "wheelhouse")
    }
    require(observed == required, "missing or unexpected published image roles")
    return publications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--wheel-dir", type=Path, required=True)
    matrix.add_argument("--release", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    record = commands.add_parser("record")
    record.add_argument("--wheel", type=Path, required=True)
    record.add_argument("--run", type=Path, required=True)
    record.add_argument("--inspection", type=Path, required=True)
    record.add_argument("--base-image", required=True)
    record.add_argument("--archive", type=Path, required=True)
    record.add_argument(
        "--consumers",
        type=Path,
        help="JSON list of client/run/base_image/inspection inputs",
    )
    record.add_argument("--execution-inspection", type=Path)
    record.add_argument(
        "--role", choices=("runtime", "development", "wheelhouse"), default="runtime"
    )
    record.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--record", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "matrix":
        import json

        output = image_matrix(
            args.wheel_dir,
            load_json(args.release),
            load_registry(os.environ.get("AITER_SUPPORT_PROFILES", "{}")),
        )
        write_json(args.output, output)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                stream.write(
                    "matrix=" + json.dumps(output, separators=(",", ":")) + "\n"
                )
    elif args.command == "record":
        inspection = load_json(args.inspection)
        require(
            isinstance(inspection, list) and len(inspection) == 1,
            "expected one inspected image",
        )
        output = image_record(
            args.wheel,
            args.run,
            inspection[0],
            args.base_image,
            args.archive,
            catalog=load_catalog(),
            consumers=load_json(args.consumers) if args.consumers else None,
            execution_inspection=(
                load_json(args.execution_inspection)[0]
                if args.execution_inspection
                else None
            ),
            role=args.role,
        )
        write_json(args.output, output)
    else:
        verify_archive(args.archive, load_json(args.record))
        print("verified image archive and wheel qualification chain")


if __name__ == "__main__":
    main()
