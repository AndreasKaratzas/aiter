# Read and maintain the website

The website brings the architecture, runtime, kernel development and delivery guides into one searchable place. Most pages render the Markdown maintained beside the code. Editing those guides updates the website; there is no second copy to keep in sync.

## Build and read locally

Run these commands from the repository root with Python 3.11 or newer. The documentation environment does not need AITER, Torch or a GPU.

```bash
python -m venv /tmp/aiter-docs-env
/tmp/aiter-docs-env/bin/python -m pip install -r requirements/docs/build.txt
/tmp/aiter-docs-env/bin/python -m docs.website build --output /tmp/aiter-site
python -m http.server --bind 127.0.0.1 --directory /tmp/aiter-site 8000
```

Open `http://localhost:8000`. Search runs locally. Diagrams render without a CDN; use **Expand** to read a large diagram, zoom inside it, or download its SVG. Press Escape to close the expanded view.

The builder starts fresh, treats Sphinx warnings as errors and checks every local link and HTML anchor. It replaces only a previous output directory created by this builder. A failed build leaves the previous site available.

## Check the actual browser result

```bash
/tmp/aiter-docs-env/bin/python -m pip install -r requirements/docs/browser.txt
/tmp/aiter-docs-env/bin/python -m playwright install --with-deps chromium
/tmp/aiter-docs-env/bin/python -m docs.website check \
  --site /tmp/aiter-site --output /tmp/aiter-site-check
```

This opens every HTML page in Chromium at desktop and phone widths with external network requests blocked. It checks local assets, JavaScript errors, Mermaid output, page and table clipping, search, mobile navigation and diagram controls. Wide tables become labeled rows on phones; the browser checks each label and the available reading width. Additional checks cover dark mode and enlarged text. The output directory retains a JSON report and screenshots for visual review. External article links are listed but are not fetched; the check does not certify their contents.

A passing browser report establishes rendering and navigation. GPU examples need their own operator or framework tests; the [engineering evidence](../notes.md) records those scopes separately.

## Add or change a guide

For a guide maintained outside the Sphinx source tree, add its canonical path to [the guide map](website/guides.json), create the corresponding wrapper page, and add that page to the appropriate toctree in [the site index](index.rst). The wrapper contains no copied prose. Relative links resolve from the maintained file's actual directory.

Links to registered guides open rendered pages. Links to source files open a highlighted copy of the exact local file, with an exact download. They do not redirect an uncommitted change to a different GitHub branch. Every local target must exist; an unregistered Markdown guide is a build error.

Use fenced `mermaid` blocks or the RST `mermaid` directive for diagrams. Keep the first diagram small, and put detailed class or sequence diagrams next to the explanation they support. Mermaid's pinned standalone renderer and license live in `docs/_static/vendor/`; its manifest binds their bytes. The build rejects an altered vendor asset.

The focused CPU tests also use GNU tar to reproduce the Linux Pages action's archive exclusions. Run them with:

```bash
/tmp/aiter-docs-env/bin/python -m unittest discover -s docs/website/tests -v
```

See [documentation coverage](DOCUMENTATION_AUDIT_REPORT.md) for the treatment of older material, and [publication](DEPLOYMENT.md) for the separate remote workflow.
