"""Materialize hierarchical workflow sources as GitHub's flat entrypoints."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from ci.common.json import load_json, require

DOMAINS = {"reusable", "repository", "library", "frameworks", "release", "schedules"}
SOURCE_ROOT = ".github/workflow-sources"
EXTENSIONS = {".yaml", ".yml"}
LOCAL_CALL = re.compile(r"uses:\s*[\"\']?\./\.github/workflows/([^\s\"\']+)")


def contained(root: Path, relative: str) -> Path:
    """Reject redirects at every component, including output parents."""
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        require(not path.is_symlink(), f"workflow path is a symlink: {relative}")
    require(
        path.resolve().is_relative_to(root), f"workflow path escapes root: {relative}"
    )
    return path


def files(directory: Path) -> set[str]:
    result = set()
    if directory.exists():
        for path in directory.rglob("*"):
            require(not path.is_symlink(), f"workflow tree contains a symlink: {path}")
            if path.suffix in EXTENSIONS and path.is_file():
                result.add(path.relative_to(directory).as_posix())
    return result


def load_sources(root: Path) -> tuple[dict, dict[str, bytes]]:
    inventory = load_json(contained(root, SOURCE_ROOT + "/registry.json"))
    require(
        isinstance(inventory, dict)
        and set(inventory) == {"schema_version", "domains", "workflows"}
        and type(inventory["schema_version"]) is int
        and inventory["schema_version"] == 1,
        "invalid workflow inventory",
    )
    domains = inventory["domains"]
    require(
        isinstance(domains, dict) and set(domains) == DOMAINS, "unknown workflow domain"
    )
    require(
        all(isinstance(value, str) and value.strip() for value in domains.values()),
        "workflow domains need nonempty descriptions",
    )
    entries = inventory["workflows"]
    require(
        isinstance(entries, list) and entries,
        "workflow inventory must be a nonempty list",
    )
    sources, destinations = {}, set()
    for entry in entries:
        require(
            isinstance(entry, dict)
            and set(entry) == {"file", "source", "domain", "purpose", "application"},
            "invalid workflow inventory entry",
        )
        domain, source, filename = entry["domain"], entry["source"], entry["file"]
        require(
            isinstance(domain, str) and domain in domains,
            "unknown workflow entry domain",
        )
        require(
            isinstance(source, str)
            and re.fullmatch(r"[a-z0-9_./-]+\.(?:yaml|yml)", source),
            "invalid workflow source path",
        )
        parts = PurePosixPath(source)
        require(
            str(parts) == source
            and not parts.is_absolute()
            and ".." not in parts.parts
            and len(parts.parts) >= (3 if domain in {"frameworks", "schedules"} else 2)
            and parts.parts[0] == domain,
            "workflow source must have a canonical owner path",
        )
        require(
            isinstance(filename, str)
            and re.fullmatch(
                r"(?:reusable|repository|library|frameworks|release|schedule)-[a-z0-9-]+\.(?:yaml|yml)",
                filename,
            ),
            "invalid flat workflow filename",
        )
        require(
            filename.startswith({"schedules": "schedule"}.get(domain, domain) + "-"),
            "workflow filename does not match its domain",
        )
        require(
            source not in sources and filename not in destinations,
            "duplicate workflow source or entrypoint",
        )
        require(
            isinstance(entry["purpose"], str)
            and entry["purpose"].strip()
            and not any(char in entry["purpose"] for char in "|\r\n"),
            "invalid workflow purpose",
        )
        applications = entry["application"]
        require(
            isinstance(applications, list)
            and all(isinstance(module, str) for module in applications)
            and len(set(applications)) == len(applications),
            "workflow applications must be a list of unique module strings",
        )
        for module in applications:
            require(
                re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", module
                )
                and (module.startswith("ci.") or module == "docs.website"),
                "invalid workflow application module",
            )
            relative = module.replace(".", "/")
            require(
                contained(root, relative + ".py").is_file()
                or contained(root, relative + "/__init__.py").is_file(),
                "workflow application module is missing",
            )
        location = contained(root, SOURCE_ROOT + "/" + source)
        require(location.is_file(), f"workflow source is missing: {source}")
        body = location.read_bytes()
        require(
            body
            and body.endswith(b"\n")
            and b"\r" not in body
            and not body.startswith(b"\xef\xbb\xbf"),
            "workflow source must be nonempty UTF-8 with LF and a final newline",
        )
        body.decode("utf-8")
        require(
            not body.startswith(b"# Generated by ci.workflows"),
            "canonical workflow contains a generated entrypoint",
        )
        sources[source] = body
        destinations.add(filename)
    require(
        files(contained(root, SOURCE_ROOT)) == set(sources),
        "workflow inventory omits a source or retains a missing source",
    )
    actual = files(contained(root, ".github/workflows"))
    require(
        all("/" not in name for name in actual),
        "GitHub cannot discover nested workflow entrypoints",
    )
    require(
        actual <= destinations,
        "unregistered GitHub entrypoint; add its source or explicitly remove it",
    )
    for body in sources.values():
        for called in LOCAL_CALL.findall(body.decode("utf-8")):
            require(
                "/" not in called and called in destinations,
                "workflow calls a missing or nested entrypoint",
            )
    return inventory, sources


def entrypoint(source: str, body: bytes) -> bytes:
    banner = (
        "# Generated by ci.workflows; edit the canonical source, not this file.\n"
        f"# Source: {SOURCE_ROOT}/{source}\n"
        f"# Directory guide: {SOURCE_ROOT}/{PurePosixPath(source).parent}/README.md\n"
        f"# Source SHA256: {hashlib.sha256(body).hexdigest()}\n"
        "# Regenerate: python -m ci.workflows --write\n\n"
    )
    return banner.encode("utf-8") + body


def navigation(inventory: dict, *, github: bool) -> str:
    if github:
        return landing(inventory)
    lines = [
        "# Workflow source index",
        "",
        "Start with the [workflow guide](README.md) if you are new to AITER. This index maps every editable source to the flat file that GitHub executes. After editing a source, run `python -m ci.workflows --write` and `python -m ci.workflows --check`.",
        "",
        "Sources are ordinary GitHub Actions YAML. The generator copies their bytes after a source-path comment; it does not interpret or expand their jobs. [Pipeline controllers](../../ci/pipelines/README.md) implement execution, and [script helpers](../scripts/README.md) support specialized jobs.",
        "",
    ]
    for domain, description in inventory["domains"].items():
        lines += [
            "## " + domain.title(),
            "",
            description + f" Read the [{domain} guide]({domain}/README.md).",
            "",
        ]
        for entry in sorted(inventory["workflows"], key=lambda item: item["source"]):
            if entry["domain"] != domain:
                continue
            source = entry["source"]
            destination = "../workflows/" + entry["file"]
            lines += [
                f"### {source.removesuffix('.yaml').removesuffix('.yml')}",
                "",
                entry["purpose"] + ".",
                "",
                f"[Edit source]({source}) · [GitHub entrypoint]({destination})",
                "",
            ]
            if entry["application"]:
                lines += [
                    "Execution: "
                    + ", ".join("`" + module + "`" for module in entry["application"])
                    + ".",
                    "",
                ]
    return "\n".join(lines)


def landing(inventory: dict) -> str:
    """Keep GitHub's required flat directory a short, useful starting point."""
    lines = [
        "# Find the workflow you need",
        "",
        "**Edit the directories in [workflow-sources](../workflow-sources/README.md).** The YAML files here are generated copies. GitHub [does not support workflow subdirectories](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow), so the editable hierarchy lives immediately beside this directory.",
        "",
        "Start with [vLLM](../workflow-sources/frameworks/vllm/README.md), [SGLang](../workflow-sources/frameworks/sglang/README.md), or [scheduled runs](../workflow-sources/schedules/README.md). Each guide explains the files and links to the test definitions and execution code.",
        "",
        "| Directory | What belongs there |",
        "| --- | --- |",
    ]
    for domain, description in inventory["domains"].items():
        lines.append(
            f"| [{domain}/](../workflow-sources/{domain}/README.md) | {description} |"
        )
    lines += [
        "",
        "Small shared steps live in [actions/common](../actions/common/README.md); complete reusable jobs live in [workflow-sources/reusable](../workflow-sources/reusable/README.md). Framework integration is not a client/server relationship. Older material called it ‘client’ CI and called repository checks ‘host’ CI.",
        "",
        "After editing a source, run these commands from the repository root:",
        "",
        "```bash",
        "python -m ci.workflows --write",
        "python -m ci.workflows --check",
        "```",
        "",
        "Include the source and generated output in the same change. The [complete index](../workflow-sources/index.md) maps every source to its GitHub filename. Each generated file also names its source and directory guide at the top.",
        "",
    ]
    return "\n".join(lines)


def generate(
    *, root: Path | None = None, check: bool = False, write: bool = False
) -> str:
    require(not (check and write), "choose either check or write")
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    inventory, sources = load_sources(root)
    outputs = {
        ".github/workflows/" + entry["file"]: entrypoint(
            entry["source"], sources[entry["source"]]
        )
        for entry in inventory["workflows"]
    }
    rendered = navigation(inventory, github=True)
    outputs[".github/workflows/README.md"] = rendered.encode("utf-8")
    outputs[".github/workflow-sources/index.md"] = navigation(
        inventory, github=False
    ).encode("utf-8")
    # Admit every destination before any write. Invalid later entries cannot leave
    # a partially generated result or follow an existing output symlink.
    destinations = {name: contained(root, name) for name in outputs}
    stale = [
        name
        for name, path in destinations.items()
        if not path.is_file() or path.read_bytes() != outputs[name]
    ]
    if check:
        require(
            not stale,
            "workflow generation is stale; run python -m ci.workflows --write: "
            + ", ".join(stale),
        )
    if write:
        for name in stale:
            destinations[name].parent.mkdir(parents=True, exist_ok=True)
            destinations[name].write_bytes(outputs[name])
    if not check and not write:
        print(rendered)
    return rendered
