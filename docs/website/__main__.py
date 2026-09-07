# SPDX-License-Identifier: MIT
"""Build and inspect the AITER website from the repository root."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .checks import check_links


def build(output: Path):
    root = Path(__file__).resolve().parents[2]
    output = output.resolve()
    marker = output / ".aiter-website.json"
    if output.exists() and any(output.iterdir()) and not marker.is_file():
        raise ValueError(
            "Choose an empty output directory; this directory was not created by the website builder."
        )
    if marker.exists() and json.loads(marker.read_text()).get("source") != str(
        root / "docs"
    ):
        raise ValueError("The existing site belongs to a different source checkout.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="aiter-site-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "html"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "--keep-going",
                "-E",
                "-a",
                "-d",
                str(Path(temporary) / "doctrees"),
                "-b",
                "html",
                "docs",
                str(stage),
            ],
            cwd=root,
            check=True,
        )
        links = check_links(stage)
        (stage / ".aiter-website.json").write_text(
            json.dumps(
                {"schema_version": 1, "source": str(root / "docs"), "links": links},
                indent=2,
            )
            + "\n"
        )
        if output.exists():
            shutil.rmtree(output)
        shutil.move(stage, output)
    print(f"Built {links['pages']} pages at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser(
        "build", help="Fresh warning-fatal build and local link validation"
    )
    build_parser.add_argument("--output", type=Path, required=True)
    check = commands.add_parser(
        "check",
        help="Check links, real Chromium rendering and desktop/mobile interactions",
    )
    check.add_argument("--site", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output)
    else:
        from .browser import check_browser

        links = check_links(args.site)
        result = check_browser(args.site, args.output)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "links": links,
                    "rendered_page_viewports": len(result["pages"]),
                    "evidence": str(args.output),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
