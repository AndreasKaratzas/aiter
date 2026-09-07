# SPDX-License-Identifier: MIT
"""Validate the local website graph without fetching external URLs."""

from html.parser import HTMLParser
from os.path import normpath
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attribute in ("href", "src", "action"):
            if attrs.get(attribute):
                self.links.append(attrs[attribute])


def check_links(directory: Path):
    directory = directory.resolve()
    pages = {}
    for path in directory.rglob("*.html"):
        page = Page()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path] = page
    if directory / "index.html" not in pages:
        raise ValueError("The site has no index.html")
    failures, external, checked = [], set(), 0
    for path, page in pages.items():
        for uri in page.links:
            parsed = urlsplit(uri)
            if parsed.scheme or parsed.netloc:
                external.add(uri)
                continue
            if not parsed.path:
                target = path
            elif parsed.path.startswith("/"):
                target = directory / unquote(parsed.path.lstrip("/"))
            else:
                target = path.parent / unquote(parsed.path)
            requested = Path(normpath(target))
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if (
                not requested.is_relative_to(directory)
                or not target.is_relative_to(directory)
                or not target.is_file()
            ):
                failures.append(
                    f"{path.relative_to(directory)}: missing local target {uri}"
                )
            elif any(
                part.startswith(".") for part in requested.relative_to(directory).parts
            ):
                failures.append(
                    f"{path.relative_to(directory)}: hidden local target omitted by artifact uploads {uri}"
                )
            elif (
                parsed.fragment
                and target in pages
                and unquote(parsed.fragment) not in pages[target].ids
            ):
                failures.append(f"{path.relative_to(directory)}: missing anchor {uri}")
    if failures:
        raise ValueError("\n".join(failures))
    return {
        "pages": len(pages),
        "local_links_checked": checked,
        "external_links_not_fetched": sorted(external),
    }
