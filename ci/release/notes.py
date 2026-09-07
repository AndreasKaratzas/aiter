"""Validate curated release guidance from the exact released source revision."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from ci.common.json import require

VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?\Z")
SECTIONS = (
    "Overview",
    "Compatibility",
    "Upgrade",
    "Known issues",
    "Rollback",
    "Qualification",
)
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|FIXME|PLACEHOLDER|CHANGEME|REPLACE_ME|INSERT_HERE)\b"
    r"|\b(?:fill (?:this|in)|to be (?:completed|written|reviewed))\b|<[^>\n]+>",
    re.IGNORECASE,
)


def validate_notes(notes: str, version: str) -> dict:
    require(
        isinstance(version, str) and VERSION.fullmatch(version),
        "invalid release version",
    )
    require(
        isinstance(notes, str) and notes.strip(), "reviewed release notes are missing"
    )
    require(
        not PLACEHOLDER.search(notes),
        "release notes contain an unfilled template marker",
    )
    lines = notes.splitlines()
    require(
        lines[0] == f"# AITER {version}", "release notes title names another version"
    )
    sections = {}
    metadata = []
    current = None
    for line in lines[1:]:
        if line.startswith("## "):
            current = line[3:].strip()
            require(
                current in SECTIONS and current not in sections,
                "unknown or duplicate release note section",
            )
            sections[current] = []
        elif current is None:
            metadata.append(line)
        else:
            sections[current].append(line)
    require(
        set(sections) == set(SECTIONS),
        "release notes are missing required guidance sections",
    )
    headers = {}
    for line in metadata:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        require(
            separator
            and key in {"Reviewed by", "Review reference"}
            and key not in headers,
            "invalid release review metadata",
        )
        headers[key] = value.strip()
    require(
        set(headers) == {"Reviewed by", "Review reference"},
        "release notes need review attribution and a review reference",
    )
    reviewers = [value.strip() for value in headers["Reviewed by"].split(",")]
    require(
        bool(reviewers)
        and len(set(reviewers)) == len(reviewers)
        and all(
            re.fullmatch(r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", value)
            for value in reviewers
        ),
        "invalid release reviewer attribution",
    )
    require(
        re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[1-9][0-9]*(?:#pullrequestreview-[0-9]+)?",
            headers["Review reference"],
        ),
        "release review reference must name a GitHub pull request or review",
    )
    bodies = []
    for name, content in sections.items():
        body = "\n".join(content).strip()
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", body)
        require(
            len(body) >= 50
            and len(words) >= 8
            and len({word.lower() for word in words}) >= 6,
            f"release section {name} needs substantive guidance",
        )
        bodies.append(" ".join(body.lower().split()))
    require(
        len(set(bodies)) == len(bodies), "release sections repeat generic boilerplate"
    )
    return {
        "version": version,
        "path": f"releases/{version}.md",
        "sha256": hashlib.sha256(notes.encode()).hexdigest(),
        "reviewers": reviewers,
        "review_reference": headers["Review reference"],
    }


def read_reviewed_notes(root: Path, version: str, revision: str) -> str:
    require(VERSION.fullmatch(version), "invalid release version")
    require(
        re.fullmatch(r"[0-9a-f]{40}", revision),
        "release notes require an exact source revision",
    )
    result = subprocess.run(
        ["git", "show", f"{revision}:releases/{version}.md"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        result.returncode == 0, f"released source has no reviewed releases/{version}.md"
    )
    validate_notes(result.stdout, version)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    record = validate_notes(args.file.read_text(), args.version)
    require(
        args.file.name == f"{args.version}.md" and args.file.parent.name == "releases",
        "accepted notes belong in releases/<VERSION>.md",
    )
    print(
        f"Validated reviewed release guidance for {record['version']}; review enforcement remains a repository setting"
    )


if __name__ == "__main__":
    main()
