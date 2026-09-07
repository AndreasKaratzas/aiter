"""Validate script ownership and render navigation for retained workflow adapters."""

import re
from pathlib import Path, PurePosixPath

from ci.common.json import load_json, require


def script_index(*, root=None, check=False, write=False):
    root = Path(root or Path(__file__).resolve().parents[2])
    directory = root / ".github/scripts"
    inventory = load_json(root / "ci/pipelines/scripts.json")
    require(
        isinstance(inventory, dict)
        and set(inventory)
        == {"schema_version", "domains", "scripts", "external_references"}
        and type(inventory["schema_version"]) is int
        and inventory["schema_version"] == 1,
        "invalid script inventory",
    )
    domains = inventory["domains"]
    require(
        isinstance(domains, dict)
        and set(domains)
        == {"common", "library", "repository", "frameworks", "release"},
        "invalid script owners",
    )
    require(
        all(isinstance(v, str) and v.strip() for v in domains.values()),
        "script owners need descriptions",
    )
    require(isinstance(inventory["scripts"], list), "script entries must be a list")
    entries = {}
    for entry in inventory["scripts"]:
        require(
            isinstance(entry, dict) and set(entry) == {"file", "purpose"},
            "invalid script entry",
        )
        name, purpose = entry["file"], entry["purpose"]
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9_./-]+\.(?:sh|py)", name),
            "invalid script path",
        )
        path = PurePosixPath(name)
        require(
            str(path) == name
            and not path.is_absolute()
            and ".." not in path.parts
            and len(path.parts) > 1
            and path.parts[0] in domains,
            "script must have a canonical owner path",
        )
        require(name not in entries, "duplicate script entry")
        require(
            isinstance(purpose, str) and purpose.strip() and "|" not in purpose,
            "script needs a description",
        )
        location = directory / name
        require(
            location.is_file()
            and not location.is_symlink()
            and location.resolve().is_relative_to(directory.resolve()),
            "script is missing or escapes its owner",
        )
        entries[name] = purpose
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.suffix in {".sh", ".py"} and p.is_file()
    }
    require(
        set(entries) == actual,
        "script inventory omits an executable or retains a removed script",
    )
    external = {}
    require(
        isinstance(inventory["external_references"], list),
        "external script references must be a list",
    )
    for entry in inventory["external_references"]:
        require(
            isinstance(entry, dict) and set(entry) == {"workflow", "file", "reason"},
            "invalid external script reference",
        )
        require(
            all(isinstance(v, str) and v.strip() for v in entry.values()),
            "external script references need descriptions",
        )
        key = (entry["workflow"], entry["file"])
        require(
            key not in external
            and key[1] not in entries
            and (root / ".github/workflows" / key[0]).is_file(),
            "invalid or duplicate external script owner",
        )
        external[key] = entry["reason"]
    used_external = set()
    callers = {name: set() for name in entries}
    for workflow in (root / ".github/workflows").iterdir():
        if workflow.suffix not in {".yaml", ".yml"}:
            continue
        text = workflow.read_text()
        names = re.findall(r"\.github/scripts/([A-Za-z0-9_./-]+\.(?:py|sh))", text)
        names += re.findall(r"script:\s*(frameworks/[A-Za-z0-9_./-]+\.(?:py|sh))", text)
        for name in names:
            if name in entries:
                callers[name].add(workflow.name)
            else:
                key = (workflow.name, name)
                require(
                    key in external,
                    f"workflow references an unowned script: {workflow.name}: {name}",
                )
                used_external.add(key)
    require(used_external == set(external), "stale external script reference")
    lines = [
        "# Workflow script owners",
        "",
        "These adapters support retained specialized workflows. Qualified delivery orchestration lives in [ci/pipelines](../../ci/pipelines/README.md); numerical and model acceptance belongs in [tests](../../tests/README.md), and measurements belong in [benchmarks](../../benchmarks/README.md). The script directory contains no flat executable files. [Workflow sources](../workflow-sources/README.md) generate the flat GitHub callers listed below.",
        "",
        "Choose the owner that matches the caller: repository maintenance, library testing, framework integration or release work. `common` contains small shared adapters. Each directory has a short guide. Edit `ci/pipelines/scripts.json` when changing an adapter, then run `python -m ci.pipelines scripts --write-index`. Repository checks verify every maintained script, its owner, workflow references and this index. ATOM helpers listed separately below belong to its external checkout.",
        "",
    ]
    for domain, description in domains.items():
        lines.extend(
            [
                "## " + domain.title(),
                "",
                description + f" See the [{domain} guide]({domain}/README.md).",
                "",
                "| Adapter | Responsibility | Direct workflow callers |",
                "|---|---|---|",
            ]
        )
        for name, purpose in entries.items():
            if name.split("/")[0] != domain:
                continue
            links = (
                ", ".join(f"[{x}](../workflows/{x})" for x in sorted(callers[name]))
                or "Called by another adapter or used manually"
            )
            lines.append(f"| [{name}]({name}) | {purpose} | {links} |")
        lines.append("")
    lines.extend(["## External checkout adapters", ""])
    for (workflow, name), reason in sorted(external.items()):
        lines.append(f"- [{workflow}](../workflows/{workflow}): `{name}`. {reason}")
    rendered = "\n".join(lines) + "\n"
    destination = directory / "README.md"
    if check:
        require(
            destination.read_text() == rendered,
            "script navigation is stale; run scripts --write-index",
        )
    if write:
        destination.write_text(rendered)
    if not check and not write:
        print(rendered, end="")
    return rendered
