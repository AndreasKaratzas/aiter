"""Check the repository dependency inputs without contacting a package index."""

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path

from ci.common.json import require


def resolve(path: Path, home: Path, active=()) -> list[str]:
    path, home = path.resolve(), home.resolve()
    require(
        path.is_relative_to(home) and path.is_file(),
        "requirement include escapes its home or is missing",
    )
    require(path not in active, "requirement include cycle")
    result = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        require(not value.endswith("\\"), "use one dependency per line")
        if value.startswith(("-r ", "-c ")):
            included = value[3:].strip()
            require(
                included and not Path(included).is_absolute(),
                "requirement include must be relative",
            )
            result.extend(resolve(path.parent / included, home, (*active, path)))
        else:
            result.append(value)
    require(bool(result), f"empty dependency input: {path}")
    return result


def check(root: Path) -> dict:
    root = root.resolve()
    home = root / "requirements"
    paths = sorted(home.rglob("*.txt"))
    require(bool(paths), "requirements home is empty")
    for path in paths:
        resolve(path, home)
    text = (root / "pyproject.toml").read_text()
    sections = re.findall(r"(?ms)^\[build-system\]\s*\n(.*?)(?=^\[|\Z)", text)
    require(len(sections) == 1, "expected one PEP 517 build-system section")
    assignments = list(re.finditer(r"(?m)^requires\s*=\s*(?=\[)", sections[0]))
    require(len(assignments) == 1, "expected a literal PEP 517 requirement list")
    tail = sections[0][assignments[0].end() :]
    lines = tail.splitlines(keepends=True)
    depth, end = 0, None
    for token in tokenize.generate_tokens(io.StringIO(tail).readline):
        if token.type == tokenize.OP:
            depth += token.string == "["
            depth -= token.string == "]"
            if depth == 0:
                end = sum(map(len, lines[: token.end[0] - 1])) + token.end[1]
                break
    require(end is not None, "unterminated PEP 517 requirement list")
    # This maintained TOML field is a literal string list, not executable Python.
    bootstrap = ast.literal_eval(tail[:end])
    require(
        isinstance(bootstrap, list)
        and all(isinstance(value, str) for value in bootstrap),
        "invalid build-system dependency list",
    )
    require(
        bootstrap == resolve(home / "build/frontend.txt", home),
        "PEP 517 metadata and frontend requirements differ",
    )
    for name in ("runtime/base.txt", "runtime/native.txt"):
        values = resolve(home / name, home)
        require(
            all(not value.startswith("-") for value in values),
            "runtime metadata cannot contain installer options",
        )
    require(
        "graft requirements" in (root / "MANIFEST.in").read_text().splitlines(),
        "source archives must include dependency inputs",
    )
    return {"files": len(paths), "metadata_requirements": bootstrap}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = check(args.root)
    print(
        f"Dependency inputs: {result['files']} files; metadata and include graph agree."
    )


if __name__ == "__main__":
    main()
