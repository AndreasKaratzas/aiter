"""Resolve effective code ownership and keep routing aligned with product domains."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ci.common.json import load_json, require
from ci.qualification.catalog import relative_path

OWNER = re.compile(r"@[A-Za-z0-9][A-Za-z0-9-]*(?:/[A-Za-z0-9_.-]+)?\Z")


@dataclass(frozen=True)
class Rule:
    pattern: str
    owners: tuple[str, ...]
    line: int

    def matches(self, path: str) -> bool:
        relative_path(path)
        pattern = self.pattern
        anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
        directory = pattern.endswith("/")
        pattern = pattern.strip("/")
        regex = "^" if anchored else r"(?:^|.*/)"
        index = 0
        while index < len(pattern):
            if pattern[index : index + 3] == "**/":
                regex += r"(?:.*/)?"
                index += 3
            elif pattern[index : index + 2] == "**":
                regex += ".*"
                index += 2
            elif pattern[index] == "*":
                regex += "[^/]*"
                index += 1
            elif pattern[index] == "?":
                regex += "[^/]"
                index += 1
            else:
                regex += re.escape(pattern[index])
                index += 1
        final_segment = pattern.rsplit("/", 1)[-1]
        if directory:
            regex += r"/.*$"
        elif not any(character in final_segment for character in "*?"):
            regex += r"(?:/.*)?$"
        else:
            regex += "$"
        return re.fullmatch(regex, path) is not None


def parse_codeowners(text: str) -> list[Rule]:
    require(
        len(text.encode()) < 3 * 1024 * 1024, "CODEOWNERS exceeds GitHub's size limit"
    )
    rules = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        pattern, *owners = fields
        require(
            not any(value in pattern for value in ("!", "[", "]", "\\")),
            f"unsupported ownership pattern at line {line_number}",
        )
        require(
            not any(part in (".", "..") for part in pattern.split("/")),
            "ownership path cannot contain dot segments",
        )
        require(
            all(OWNER.fullmatch(owner) for owner in owners),
            f"invalid owner at line {line_number}",
        )
        require(len(owners) == len(set(owners)), "duplicate owner")
        rules.append(Rule(pattern, tuple(owners), line_number))
    require(bool(rules), "CODEOWNERS has no rules")
    return rules


def effective_owner(rules: list[Rule], path: str) -> Rule | None:
    return next((rule for rule in reversed(rules) if rule.matches(path)), None)


def load_policy(path: Path | None = None) -> dict:
    policy = load_json(path or Path(__file__).with_name("owners.json"))
    require(
        set(policy) == {"schema_version", "local_steward", "domains", "rules"},
        "unknown ownership policy fields",
    )
    require(
        policy["schema_version"] == 1 and type(policy["schema_version"]) is int,
        "unsupported ownership schema",
    )
    require(
        OWNER.fullmatch(policy["local_steward"]) is not None, "invalid local steward"
    )
    require(
        isinstance(policy["domains"], dict) and bool(policy["domains"]),
        "missing ownership domains",
    )
    for name, domain in policy["domains"].items():
        require(re.fullmatch(r"[a-z][a-z-]*", name) is not None, "invalid domain")
        require(
            set(domain) == {"responsibility", "primary", "backup", "accepted"},
            "invalid ownership domain fields",
        )
        require(
            isinstance(domain["responsibility"], str)
            and bool(domain["responsibility"]),
            "missing domain responsibility",
        )
        require(
            type(domain["accepted"]) is bool, "ownership acceptance must be boolean"
        )
        for role in ("primary", "backup"):
            require(
                domain[role] is None
                or isinstance(domain[role], str)
                and OWNER.fullmatch(domain[role]),
                "invalid owner identity",
            )
        if domain["accepted"]:
            require(
                domain["primary"]
                and domain["backup"]
                and domain["primary"] != domain["backup"],
                "accepted domain requires distinct primary and backup",
            )
    require(
        isinstance(policy["rules"], list) and bool(policy["rules"]),
        "missing domain rules",
    )
    for rule in policy["rules"]:
        require(
            set(rule) == {"pattern", "domain"} and rule["domain"] in policy["domains"],
            "invalid domain rule",
        )
        parse_codeowners(rule["pattern"] + " " + policy["local_steward"])
    return policy


def render_codeowners(policy: dict) -> str:
    lines = [
        "# Local architecture review routing. No repository settings are changed by this file.",
        "# Domain owners and backups are recorded in ci/ownership/owners.json.",
        "# Unassigned domains route to the local design steward until owners accept them.",
        "# The last matching rule wins; multiple names on one line do not require both approvals.",
    ]
    for rule in policy["rules"]:
        domain = policy["domains"][rule["domain"]]
        owner = domain["primary"] if domain["accepted"] else policy["local_steward"]
        lines.append(f"{rule['pattern']} {owner} # {rule['domain']}")
    return "\n".join(lines) + "\n"


def validate_routing(root: Path, policy: dict) -> dict:
    path = root / ".github/CODEOWNERS"
    require(
        path.read_text() == render_codeowners(policy),
        "CODEOWNERS differs from domain policy; run python -m ci.ownership.policy render",
    )
    rules = parse_codeowners(path.read_text())
    files = (
        subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    missing = [name for name in files if name and effective_owner(rules, name) is None]
    require(
        not missing, "tracked files lack ownership routing: " + ", ".join(missing[:10])
    )
    return {
        "files": sum(bool(name) for name in files),
        "domains": len(policy["domains"]),
        "unassigned_domains": [
            name for name, domain in policy["domains"].items() if not domain["accepted"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate effective review routing")
    commands.add_parser("render", help="regenerate CODEOWNERS from domain policy")
    commands.add_parser("ready", help="require accepted domain owners and backups")
    show = commands.add_parser("show", help="show the last matching rule for a file")
    show.add_argument("path")
    args = parser.parse_args()
    policy = load_policy(args.source_root / "ci/ownership/owners.json")
    if args.command == "render":
        (args.source_root / ".github/CODEOWNERS").write_text(render_codeowners(policy))
    elif args.command == "show":
        rule = effective_owner(parse_codeowners(render_codeowners(policy)), args.path)
        require(rule is not None, "file has no owner")
        print(f"{args.path}: {' '.join(rule.owners)} (last match: {rule.pattern})")
    else:
        result = validate_routing(args.source_root, policy)
        if args.command == "ready":
            require(
                not result["unassigned_domains"],
                "assign accepted primary/backup owners for: "
                + ", ".join(result["unassigned_domains"]),
            )
        print(
            f"{result['files']} tracked files have review routing across {result['domains']} domains."
        )
        if result["unassigned_domains"]:
            print(
                "Domain staffing remains unassigned: "
                + ", ".join(result["unassigned_domains"])
            )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        import sys

        print(f"ownership: {error}", file=sys.stderr)
        raise SystemExit(2)
