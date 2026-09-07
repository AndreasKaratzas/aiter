# SPDX-License-Identifier: MIT
"""Render canonical repository guides and local diagrams in the Sphinx site."""

import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote, unquote, urlsplit

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename
from pygments.util import ClassNotFound
from sphinx import addnodes
from sphinx.errors import ExtensionError
from sphinx.util.osutil import relative_uri


class Mermaid(Directive):
    """Emit escaped source for the pinned, locally served Mermaid renderer."""

    has_content = True
    option_spec: ClassVar = {"caption": directives.unchanged}

    def run(self):
        self.assert_has_content()
        caption = self.options.get("caption", "Diagram")
        source = html.escape("\n".join(self.content))
        body = (
            f'<figure class="architecture-diagram" data-diagram-type="{html.escape(self.content[0].split()[0], quote=True)}" aria-label="{html.escape(caption, quote=True)}">'
            '<div class="diagram-viewport"><pre class="mermaid">'
            f"{source}</pre></div>"
            '<figcaption class="diagram-caption">'
            f"<span>{html.escape(caption)}</span>"
            '<span class="diagram-actions"></span></figcaption></figure>'
        )
        return [nodes.raw("", body, format="html")]


def guides(app):
    return json.loads((Path(app.srcdir) / "website/guides.json").read_text())


def canonical(app, docname):
    root = Path(app.srcdir).parent
    return (
        root / guides(app)[docname]
        if docname in guides(app)
        else Path(app.env.doc2path(docname))
    )


def read_source(app, docname, source):
    if docname in guides(app):
        path = canonical(app, docname)
        app.env.note_dependency(str(path))
        source[0] = path.read_text(encoding="utf-8")


def public_source_path(relative):
    """Keep source URLs visible to artifact upload and Pages packaging.

    Escape literal tildes first so a real ``~dot-github`` directory cannot
    collide with the published name of ``.github``. Displayed paths and file
    contents still identify the original repository file.
    """
    parts = []
    for part in Path(relative).parts:
        escaped = part.replace("~", "~~")
        parts.append("~dot-" + escaped[1:] if part.startswith(".") else escaped)
    return "/".join(parts)


def resolve_links(app, doctree, docname):
    """Resolve Markdown relative links against their actual maintained source."""
    root = Path(app.srcdir).parent.resolve()
    source = canonical(app, docname).resolve()
    reverse = {(root / path).resolve(): doc for doc, path in guides(app).items()}
    for doc in app.env.found_docs:
        reverse[Path(app.env.doc2path(doc)).resolve()] = doc
    for node in doctree.findall(nodes.reference):
        if isinstance(node, addnodes.download_reference) or node.get("internal"):
            continue
        uri = node.get("refuri", "")
        parsed = urlsplit(uri)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (source.parent / unquote(parsed.path)).resolve()
        if target.is_dir():
            target = target / "README.md"
        if not target.is_relative_to(root) or not target.is_file():
            raise ExtensionError(f"{source}: broken local link {uri!r}")
        if target in reverse:
            destination = app.builder.get_target_uri(reverse[target])
        elif target.suffix in (".md", ".rst"):
            raise ExtensionError(
                f"{source}: guide is not in website navigation: {target}"
            )
        else:
            relative = target.relative_to(root).as_posix()
            app.env.site_source_files[relative] = str(target)
            app.env.note_dependency(str(target), docname=docname)
            destination = "source/" + quote(public_source_path(relative)) + ".html"
        fragment = parsed.fragment
        if (
            destination.startswith("source/")
            and fragment.startswith("L")
            and fragment[1:].isdigit()
        ):
            fragment = "L-" + fragment[1:]
        node["refuri"] = relative_uri(app.builder.get_target_uri(docname), destination)
        if fragment:
            node["refuri"] += "#" + fragment
        node["internal"] = True


def initialize(app):
    app.env.site_source_files = {}
    directory = Path(app.srcdir) / "_static/vendor"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ExtensionError(f"Vendored documentation asset changed: {name}")


def source_pages(app):
    for relative, filename in sorted(app.env.site_source_files.items()):
        text = Path(filename).read_text(encoding="utf-8")
        try:
            lexer = get_lexer_for_filename(filename)
        except ClassNotFound:
            lexer = TextLexer()
        public = public_source_path(relative)
        page = "source/" + public + ".html"
        download = relative_uri(quote(page), "_downloads/repository/" + quote(public))
        body = (
            f"<h1>Source: <code>{html.escape(relative)}</code></h1>"
            "<p>This is the actual file from the checkout used to build this website. "
            f'<a href="{download}" download>Download the file</a>.</p>'
            + highlight(
                text,
                lexer,
                HtmlFormatter(linenos="inline", lineanchors="L", anchorlinenos=True),
            )
        )
        yield page.removesuffix(".html"), {"title": relative, "body": body}, "page.html"


def copy_sources(app, exception):
    if exception is not None:
        return
    for relative, filename in app.env.site_source_files.items():
        destination = (
            Path(app.outdir) / "_downloads/repository" / public_source_path(relative)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, destination)


def setup(app):
    app.add_directive("mermaid", Mermaid)
    app.connect("builder-inited", initialize)
    app.connect("source-read", read_source)
    app.connect("doctree-resolved", resolve_links)
    app.connect("html-collect-pages", source_pages)
    app.connect("build-finished", copy_sources)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": False}
