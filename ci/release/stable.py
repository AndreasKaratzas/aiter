"""Reconstruct a stable candidate, then publish only a complete verified draft."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from ci.common.json import load_json, require, write_json
from ci.qualification.catalog import load_catalog
from ci.release.images import verify_publications
from ci.release.manifest import release_manifest, revalidate_release
from ci.release.notes import read_reviewed_notes


def record_candidate(
    wheels: Path, matrix: Path, runs: Path, source: Path, version: str
) -> dict:
    entries = load_json(matrix)["include"]
    require(bool(entries), "stable qualification matrix is empty")
    revisions = {entry["source_revision"] for entry in entries}
    require(len(revisions) == 1, "stable matrix mixes source revisions")
    notes = read_reviewed_notes(source, version, revisions.pop())
    return release_manifest(
        wheels,
        sorted(runs.glob("support-evidence-*/run")),
        channel="stable",
        notes=notes,
        required_cells=[entry["cell"] for entry in entries],
        required_environments={
            entry["cell"]: entry["environment_lock_digest"] for entry in entries
        },
        catalog=load_catalog(),
    )


def _gh(*arguments: str) -> str:
    return subprocess.check_output(["gh", *arguments], text=True)


def _view(tag: str) -> dict | None:
    # Listing has a successful empty result for absent tags; authentication errors
    # must never be interpreted as permission to create a new release.
    records = json.loads(
        _gh("release", "list", "--limit", "1000", "--json", "tagName,isDraft")
    )
    matches = [item for item in records if item["tagName"] == tag]
    require(len(matches) <= 1, "duplicate release tag")
    if not matches:
        return None
    return json.loads(_gh("release", "view", tag, "--json", "isDraft,assets"))


def upload_draft(
    tag: str, branch: str, notes: Path, assets: list[Path], directory: Path
) -> dict:
    """Resume identical assets; leave any failure unpublished and never clobber bytes."""
    names = {path.name for path in assets}
    require(len(names) == len(assets), "duplicate release asset filename")
    require(all(path.is_file() for path in assets), "missing release asset")
    existing = _view(tag)
    require(
        existing is None or existing["isDraft"] is True,
        "published releases are immutable",
    )
    require(
        existing is None or {asset["name"] for asset in existing["assets"]} <= names,
        "draft contains unexpected assets",
    )
    if existing is None:
        _gh(
            "release",
            "create",
            tag,
            "--draft",
            "--verify-tag",
            "--target",
            branch,
            "--title",
            "AITER " + tag,
            "--notes-file",
            str(notes),
        )
        existing = {"assets": []}
    else:
        _gh(
            "release",
            "edit",
            tag,
            "--title",
            "AITER " + tag,
            "--notes-file",
            str(notes),
        )
    known = {asset["name"] for asset in existing["assets"]}
    directory.mkdir(parents=True, exist_ok=False)
    for path in assets:
        if path.name in known:
            _gh(
                "release",
                "download",
                tag,
                "--pattern",
                path.name,
                "--dir",
                str(directory),
            )
            require(
                (directory / path.name).read_bytes() == path.read_bytes(),
                "existing draft asset differs: " + path.name,
            )
            (directory / path.name).unlink()
        else:
            _gh("release", "upload", tag, str(path))
    _gh("release", "download", tag, "--dir", str(directory))
    require(
        {path.name for path in directory.iterdir()} == names,
        "downloaded release asset inventory differs",
    )
    for path in assets:
        require(
            (directory / path.name).read_bytes() == path.read_bytes(),
            "downloaded release bytes differ: " + path.name,
        )
    result = _view(tag)
    require(
        result is not None and result["isDraft"] is True,
        "release was published during verification",
    )
    require(
        {asset["name"] for asset in result["assets"]} == names,
        "remote release asset inventory differs",
    )
    return result


def render_notes(curated: str, generated: str, assets: list[dict], version: str) -> str:
    pattern = re.compile(
        rf"{re.escape(version)}\+rocm(?P<rocm>7\.[012])\.manylinux\.2\.28-(?P<py>cp3(?:10|12))-(?P=py)-"
    )
    rows = {}
    for asset in assets:
        if not asset["name"].endswith(".whl"):
            continue
        match = pattern.search(asset["name"])
        require(match is not None, "unexpected wheel asset name")
        cell = match.group("rocm"), match.group("py")
        require(cell not in rows, "duplicate wheel support tuple")
        rows[cell] = asset
    require(
        set(rows)
        == {(rocm, py) for rocm in ("7.0", "7.1", "7.2") for py in ("cp310", "cp312")},
        "missing stable wheel support tuple",
    )
    lines = [
        curated,
        "",
        "## Wheels",
        "",
        "Prebuilt manylinux_2_28 wheels, `GPU_ARCHS=gfx942;gfx950`:",
        "",
        "| ROCm | Python | Wheel |",
        "|---|---|---|",
    ]
    for (rocm, python), asset in sorted(rows.items()):
        lines.append(
            f"| {rocm} | {python} | [{asset['name']}]({asset['browser_download_url']}) |"
        )
    lines.extend(
        [
            "",
            "<details>",
            "<summary>What's Changed</summary>",
            "",
            generated.strip(),
            "",
            "</details>",
            "",
        ]
    )
    return "\n".join(lines)


def publish(
    wheels: Path,
    matrix: Path,
    runs: Path,
    original: Path,
    images: Path,
    source: Path,
    *,
    tag: str,
    version: str,
    branch: str,
    previous_tag: str,
    repository: str,
) -> dict:
    require(
        re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?", tag) is not None
        and tag == "v" + version,
        "release tag/version mismatch",
    )
    record = revalidate_release(
        load_json(original), record_candidate(wheels, matrix, runs, source, version)
    )
    publications = verify_publications(images, record)
    generated = _gh(
        "api",
        "--method",
        "POST",
        f"repos/{repository}/releases/generate-notes",
        "-f",
        "tag_name=" + tag,
        "-f",
        "previous_tag_name=" + previous_tag,
        "-f",
        "target_commitish=" + branch,
        "--jq",
        ".body",
    )
    curated = record["release_notes"]
    with tempfile.TemporaryDirectory(prefix="aiter-stable-publication-") as temporary:
        root = Path(temporary)
        qualification = root / "qualification.json"
        write_json(qualification, record)
        image_record = root / "release-images.json"
        write_json(image_record, publications)
        notes = root / "release-notes.md"
        notes.write_text(curated + "\n\n## Generated change list\n\n" + generated)
        assets = (
            sorted(wheels.glob("*.whl"))
            + sorted(wheels.glob("*.receipt.json"))
            + [qualification, image_record]
        )
        verified = upload_draft(tag, branch, notes, assets, root / "verified")
        notes.write_text(render_notes(curated, generated, verified["assets"], version))
        # This is the only visibility-changing operation. Every asset has already
        # been downloaded and compared, including wheels, receipts and OCI records.
        _gh(
            "release",
            "edit",
            tag,
            "--title",
            "AITER " + tag,
            "--notes-file",
            str(notes),
            "--draft=false",
        )
    return {
        "release_tag": tag,
        "source_revision": record["source_revision"],
        "release_digest": record["release_digest"],
        "assets": [path.name for path in assets],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "publish"))
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument(
        "--version",
        default=os.environ.get("RELEASE_VERSION"),
        required="RELEASE_VERSION" not in os.environ,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--images", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        require(args.output is not None, "record requires --output")
        write_json(
            args.output,
            record_candidate(
                args.wheels, args.matrix, args.runs, args.source, args.version
            ),
        )
    else:
        require(
            args.original is not None and args.images is not None,
            "publish requires original release record and image records",
        )
        print(
            json.dumps(
                publish(
                    args.wheels,
                    args.matrix,
                    args.runs,
                    args.original,
                    args.images,
                    args.source,
                    tag=os.environ["RELEASE_TAG"],
                    version=args.version,
                    branch=os.environ["RELEASE_BRANCH"],
                    previous_tag=os.environ["PREVIOUS_TAG"],
                    repository=os.environ["GITHUB_REPOSITORY"],
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
