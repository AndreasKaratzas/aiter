"""Check Python dependency edges and executable placement without imports.

This is a static boundary check. Literal importlib/__import__ calls are checked;
computed dynamic imports are reported for review, not claimed to be resolved.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .policy import load_policy, matches


@dataclass(frozen=True)
class Dependency:
    line: int
    target: str
    kind: str = "import"


def module_name(path, root):
    # Keep package facades explicit: an exception for __init__ must not exempt
    # every implementation module in that package.
    return ".".join(path.relative_to(root).with_suffix("").parts)


def dependency_edges(path, root):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = path.relative_to(root).parent.parts
    importlib_names = {"importlib"}
    builtin_names = {"builtins"}
    import_functions = {"__import__"}
    edges, unresolved = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(Dependency(node.lineno, alias.name))
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
                if alias.name == "builtins":
                    builtin_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    edges.append(Dependency(node.lineno, "<invalid relative import>"))
                    continue
                parent = package[: len(package) - node.level + 1]
                module = ".".join((*parent, *filter(None, (node.module,))))
            else:
                module = node.module or ""
            for alias in node.names:
                target = module if alias.name == "*" else f"{module}.{alias.name}"
                edges.append(Dependency(node.lineno, target))
                if module == "importlib" and alias.name == "import_module":
                    import_functions.add(alias.asname or alias.name)
                if module == "builtins" and alias.name == "__import__":
                    import_functions.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        dynamic = (
            isinstance(function, ast.Name) and function.id in import_functions
        ) or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (
                (
                    function.attr == "import_module"
                    and function.value.id in importlib_names
                )
                or (
                    function.attr == "__import__" and function.value.id in builtin_names
                )
            )
        )
        if not dynamic:
            continue
        value = (
            node.args[0]
            if node.args
            else next((k.value for k in node.keywords if k.arg == "name"), None)
        )
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            target = value.value
            if target.startswith("."):
                # Relative importlib calls need the supplied package argument.
                package_arg = (
                    node.args[1]
                    if len(node.args) > 1
                    else next(
                        (k.value for k in node.keywords if k.arg == "package"), None
                    )
                )
                if isinstance(package_arg, ast.Constant) and isinstance(
                    package_arg.value, str
                ):
                    from importlib.util import resolve_name

                    try:
                        target = resolve_name(target, package_arg.value)
                    except (ImportError, ValueError):
                        target = "<invalid relative import>"
                else:
                    unresolved.append(node.lineno)
                    continue
            edges.append(Dependency(node.lineno, target, "dynamic-literal"))
        else:
            unresolved.append(node.lineno)
    return edges, sorted(set(unresolved))


def _allowed(target, rule, roots):
    if target.startswith("<"):
        return False
    if any(matches(target, prefix) for prefix in rule["deny"]):
        return False
    top = target.split(".", 1)[0]
    for permitted in rule["allow"]:
        if permitted == "*":
            return True
        if permitted == "@stdlib" and top in sys.stdlib_module_names:
            return True
        if permitted == "@external" and top not in roots:
            return True
        if not permitted.startswith("@") and matches(target, permitted):
            return True
    return False


def _source_paths(directory, root):
    """Reject links rather than silently omitting linked Python subpackages."""
    paths, violations = [], []

    def linked(path):
        if not path.is_symlink():
            return False
        violations.append(
            {
                "path": path.relative_to(root).as_posix(),
                "line": 0,
                "rule": "source-origin",
                "target": None,
                "reason": "Source packages must contain regular files and directories, not symbolic links.",
            }
        )
        return True

    if linked(directory):
        return paths, violations
    for current, directories, files in os.walk(directory, followlinks=False):
        base = Path(current)
        directories[:] = [
            name
            for name in directories
            if name != "__pycache__"
            and base / name != root / "aiter/jit/build"
            and not linked(base / name)
        ]
        for name in files:
            path = base / name
            if not linked(path) and path.suffix == ".py":
                paths.append(path)
    return sorted(paths), violations


def _layout_problems(root):
    problems = []
    for path in (root / "csrc").rglob("*.py"):
        problems.append(
            (path, "native-sources", "Python generators belong in aiter/codegen.")
        )
    for path in (root / ".github/workflows").rglob("*"):
        if (
            path.suffix in {".yml", ".yaml"}
            and path.parent != root / ".github/workflows"
        ):
            problems.append(
                (
                    path,
                    "workflow-entrypoints",
                    "GitHub workflow YAML must be directly inside .github/workflows.",
                )
            )
    for path in (root / "docker").glob("*"):
        if path.is_file() and (
            path.name == "Dockerfile"
            or path.name.startswith("Dockerfile.")
            or path.suffix == ".Dockerfile"
        ):
            problems.append(
                (
                    path,
                    "container-domains",
                    "Container recipes belong under docker/common or a framework directory.",
                )
            )
    retired = {
        "kernels": "Managed kernel resources belong beside their manager in aiter/kernels/data/.",
        "aiter/bert_padding.py": "Sequence padding belongs in aiter/ops/attention/padding.py.",
        "aiter/rotary_embedding.py": "Rotary helpers belong in aiter/ops/position/rotary.py.",
        "aiter/int4_utils.py": "Packed quantization helpers belong in aiter/ops/quantization/packing.py.",
        "aiter/test_mha_common.py": "Reusable attention references belong in aiter/testing/attention.py.",
        "tests/performance": "Measurements belong in benchmarks/.",
        "tests/native": "Native SDK consumers belong in tests/integration/sdk/.",
        "op_tests": "Operator checks belong in tests/operators/.",
        "tools/quality": "Repository qualification belongs in ci/.",
        "aiter_contracts": "Operation descriptions belong in aiter/api/.",
        "tests/support.py": "Shared harness paths belong in tests/common/paths.py.",
        "aiter/ci": "Automation belongs in the repository ci/ application, outside the installed SDK.",
        "gradlib": "BLAS bridges belong in csrc/blas/; offline tuning belongs in aiter/tuning/.",
        "hsa": "Precompiled kernel bytes and selection data belong in aiter/kernels/data/.",
        "aiter_logs": "Trace analysis belongs in benchmarks/traces/; generated reports belong outside the checkout.",
    }
    for name, reason in retired.items():
        if (root / name).exists():
            problems.append(
                (
                    root / name,
                    "repository-layout",
                    reason,
                )
            )
    requirement_files = list(root.glob("requirements*.txt"))
    for name in (
        "aiter",
        "benchmarks",
        "ci",
        "docs",
        "docker",
        "scripts",
        ".github",
        "tests",
    ):
        requirement_files.extend((root / name).rglob("requirements*.txt"))
    for path in requirement_files:
        problems.append(
            (
                path,
                "dependency-inputs",
                "Maintain named dependency inputs under requirements/ and reference them from their consumers.",
            )
        )
    for path in (root / "tests/frameworks").glob("test_*.py"):
        problems.append(
            (
                path,
                "framework-domains",
                "Place these cases in a named framework directory or common/.",
            )
        )
    return [
        {
            "path": str(path.relative_to(root)),
            "line": 0,
            "rule": rule,
            "target": None,
            "reason": reason,
        }
        for path, rule, reason in problems
    ]


def check_repository(root, policy=None, *, layout=True):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {root}")
    policy = policy or load_policy()
    violations, dynamic_imports, dependencies = [], [], []
    count = 0
    for name in policy["source_roots"]:
        if layout and not (root / name).is_dir():
            violations.append(
                {
                    "path": name,
                    "line": 0,
                    "rule": "source-roots",
                    "target": None,
                    "reason": "Declared source package is missing.",
                }
            )
        paths, origins = _source_paths(root / name, root)
        violations.extend(origins)
        for path in paths:
            count += 1
            relative = path.relative_to(root).as_posix()
            source = module_name(path, root)
            rules = [
                rule
                for rule in policy["rules"]
                if any(matches(source, prefix) for prefix in rule["sources"])
                and not any(matches(source, prefix) for prefix in rule["exclude"])
            ]
            try:
                edges, unresolved = dependency_edges(path, root)
            except (SyntaxError, UnicodeError) as error:
                violations.append(
                    {
                        "path": relative,
                        "line": getattr(error, "lineno", 0),
                        "rule": "python-syntax",
                        "target": None,
                        "reason": str(error),
                    }
                )
                continue
            dynamic_imports.extend(
                {"path": relative, "line": line} for line in unresolved
            )
            for edge in edges:
                dependencies.append({"source": source, **asdict(edge)})
                for rule in rules:
                    if not _allowed(edge.target, rule, policy["source_roots"]):
                        violations.append(
                            {
                                "path": relative,
                                "line": edge.line,
                                "rule": rule["name"],
                                "target": edge.target,
                                "reason": rule["reason"],
                            }
                        )
    if layout:
        violations.extend(_layout_problems(root))
    return {
        "schema_version": 1,
        "status": "FAIL" if violations else "PASS",
        "python_files": count,
        "dependency_edges": len(dependencies),
        "violations": violations,
        "unresolved_dynamic_imports": dynamic_imports,
    }
