"""Validate and render the domain index of flat GitHub workflow entrypoints."""

from __future__ import annotations

import re
from pathlib import Path

from ci.common.json import load_json, require


def workflow_index(
    *, check: bool = False, write: bool = False, root: Path | None = None
) -> str:
    root = root or Path(__file__).resolve().parents[2]
    inventory = load_json(root / "ci/pipelines/workflows.json")
    require(
        isinstance(inventory, dict)
        and set(inventory) == {"schema_version", "domains", "workflows"}
        and type(inventory["schema_version"]) is int
        and inventory["schema_version"] == 1,
        "invalid workflow inventory",
    )
    actual = {
        p.name
        for p in (root / ".github/workflows").iterdir()
        if p.suffix in (".yml", ".yaml")
    }
    require(
        isinstance(inventory["domains"], dict)
        and set(inventory["domains"]) == {"host", "product", "client", "release"},
        "unknown workflow domain",
    )
    require(
        isinstance(inventory["workflows"], list), "workflow inventory must be a list"
    )
    require(
        all(
            isinstance(description, str) and description.strip()
            for description in inventory["domains"].values()
        ),
        "workflow domains need nonempty descriptions",
    )
    for entry in inventory["workflows"]:
        require(
            isinstance(entry, dict)
            and set(entry) == {"file", "domain", "purpose", "application"},
            "invalid workflow inventory entry",
        )
        require(
            isinstance(entry["domain"], str)
            and entry["domain"] in inventory["domains"],
            "unknown workflow entry domain",
        )
        require(isinstance(entry["file"], str), "invalid workflow filename")
        require(
            isinstance(entry["purpose"], str)
            and bool(entry["purpose"].strip())
            and "|" not in entry["purpose"],
            "invalid workflow purpose",
        )
        require(
            isinstance(entry["application"], list)
            and all(isinstance(module, str) for module in entry["application"])
            and len(set(entry["application"])) == len(entry["application"]),
            "workflow applications must be a list of unique module strings",
        )
    declared = []
    sections = [
        "# Find a workflow",
        "",
        "GitHub requires these entrypoint files directly in `.github/workflows/`. Qualified delivery and selected model jobs call the applications listed below. Specialized jobs still execute their documented workflow steps. This index groups both by purpose without generating their YAML.",
        "",
        "[GitHub documents the flat reusable-workflow requirement](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow).",
        "",
        "[Workflow script owners](../scripts/README.md) group shared adapters, operator jobs, framework launchers and release helpers into physical directories. Reusable pipeline applications own the qualified execution and evidence protocols; specialized legacy jobs retain their stated adapter scope.",
        "",
        "Edit `ci/pipelines/workflows.json` when adding or removing an entrypoint, then run `python -m ci.pipelines workflows --write-index`. Host CI checks the inventory, local workflow references and this generated index.",
        "",
    ]
    for domain, description in inventory["domains"].items():
        sections.extend(
            [
                "## " + domain.title(),
                "",
                description,
                "",
                "| Entrypoint | Purpose | Execution |",
                "|---|---|---|",
            ]
        )
        for entry in inventory["workflows"]:
            if entry["domain"] != domain:
                continue
            filename = entry["file"]
            require(
                filename.startswith(domain + "-")
                and "/" not in filename
                and filename in actual,
                "workflow filename does not match its domain or is missing",
            )
            require(filename not in declared, "duplicate workflow inventory entry")
            for module in entry["application"]:
                require(
                    isinstance(module, str)
                    and re.fullmatch(
                        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", module
                    )
                    and (module.startswith("ci.") or module == "docs.website"),
                    "invalid workflow application module",
                )
                location = root.joinpath(*module.split("."))
                require(
                    location.with_suffix(".py").is_file()
                    or (location / "__init__.py").is_file(),
                    "workflow application module is missing",
                )
            text = (root / ".github/workflows" / filename).read_text()
            for called in re.findall(
                r"uses:\s*[\"\']?\./\.github/workflows/([^\s\"\']+)", text
            ):
                require(
                    "/" not in called and called in actual,
                    "workflow calls a missing or nested entrypoint",
                )
            declared.append(filename)
            application = (
                ", ".join("`" + module + "`" for module in entry["application"])
                or "Workflow steps"
            )
            sections.append(
                f'| [{filename}]({filename}) | {entry["purpose"]} | {application} |'
            )
        sections.append("")
    require(
        set(declared) == actual,
        "workflow inventory omits or duplicates actual entrypoints",
    )
    rendered = "\n".join(sections)
    destination = root / ".github/workflows/README.md"
    if check:
        require(
            destination.read_text() == rendered,
            "workflow navigation is stale; run workflows --write-index",
        )
    if write:
        destination.write_text(rendered)
    if not check and not write:
        print(rendered)
    return rendered
