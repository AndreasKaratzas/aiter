# SPDX-License-Identifier: MIT
"""Real Chromium acceptance: offline rendering, navigation and responsive layout."""

import functools
import hashlib
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

KEY_PAGES = (
    "index.html",
    "understand/model.html",
    "use/runtime.html",
    "use/examples.html",
    "deliver/ci.html",
    "use/frameworks.html",
    "extend/kernel-manager.html",
    "extend/test-fixtures.html",
    "deliver/vllm-e2e.html",
    "extend/testing.html",
    "deliver/vllm-tests.html",
    "deliver/vllm-operators.html",
    "deliver/vllm-upstream.html",
    "extend/vllm-benchmarks.html",
    "extend/triton.html",
    "deliver/scripts.html",
    "deliver/workflows.html",
)


class WebsiteFailure(ValueError):
    """A rendered website acceptance condition failed."""


def require(condition, message):
    if not condition:
        raise WebsiteFailure(message)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def layout_issues(page):
    """Detect clipping at cells and their ancestors, not only document overflow."""
    return page.evaluate(r"""() => {
        const issues = [];
        if (document.documentElement.scrollWidth > innerWidth + 1)
            issues.push('document overflows horizontally');
        const article = document.querySelector('article');
        if (!article) return issues;
        const area = article.getBoundingClientRect();
        for (const table of article.querySelectorAll('table.docutils')) {
            const stacked = table.classList.contains('mobile-stacked') && innerWidth <= 640;
            if (stacked) {
                const head = table.tHead;
                if (!head || head.rows.length !== 1) {
                    issues.push('mobile table lost its accessible header');
                    continue;
                }
                for (const header of [head, ...head.rows[0].cells]) {
                    const style = getComputedStyle(header);
                    if (header.closest('[aria-hidden="true"], [hidden]') ||
                        style.display === 'none' || style.visibility !== 'visible' ||
                        (header.tagName === 'TH' && header.getAttribute('role') !== 'columnheader'))
                        issues.push('mobile table lost its accessible header');
                }
                const labels = [...table.tHead.rows[0].cells].map(cell => cell.textContent.trim().replace(/\s+/g, ' '));
                for (const row of [...table.tBodies].flatMap(body => [...body.rows])) {
                    [...row.cells].forEach((cell, index) => {
                        const label = getComputedStyle(cell, '::before');
                        if (cell.dataset.label !== labels[index] ||
                            label.content !== JSON.stringify(labels[index]) ||
                            label.display === 'none' || label.visibility !== 'visible' ||
                            Number(label.opacity) === 0 || !(parseFloat(label.fontSize) > 0))
                            issues.push(`missing mobile table label: ${labels[index]}`);
                        if (cell.getBoundingClientRect().width < table.getBoundingClientRect().width - 4)
                            issues.push('mobile table still has narrow columns');
                    });
                }
            }
            for (const cell of table.querySelectorAll('th, td')) {
                if (stacked && cell.closest('thead')) continue;
                const box = cell.getBoundingClientRect();
                const label = cell.textContent.trim().slice(0, 70);
                if (box.left < Math.max(0, area.left) - 2 ||
                    box.right > Math.min(innerWidth, area.right) + 2)
                    issues.push(`table cell escapes article: ${label}`);
                if (cell.scrollWidth > cell.clientWidth + 2)
                    issues.push(`table cell content clips: ${label}`);
                for (let parent = cell.parentElement; parent && parent !== article;
                     parent = parent.parentElement) {
                    const style = getComputedStyle(parent);
                    if (['hidden','clip','auto','scroll'].includes(style.overflowX)) {
                        const bounds = parent.getBoundingClientRect();
                        if (box.left < bounds.left - 2 || box.right > bounds.right + 2)
                            issues.push(`table cell clipped by ${parent.tagName}: ${label}`);
                    }
                }
            }
        }
        return [...new Set(issues)];
    }""")


def table_regressions(page, result):
    """Challenge the real layout checker with broken phone labels and headers."""
    header = page.locator("table.mobile-stacked thead").first
    require(header.count() == 1, "Expected a wide table for mobile regression checks")
    cases = (
        (
            "empty labels",
            'table.mobile-stacked td::before { content: "" !important; }',
            "missing mobile table label",
        ),
        (
            "invisible labels",
            "table.mobile-stacked td::before { visibility: hidden !important; }",
            "missing mobile table label",
        ),
        (
            "removed headers",
            "table.mobile-stacked thead { display: none !important; }",
            "accessible header",
        ),
        ("inaccessible headers", None, "accessible header"),
        (
            "hidden body cell",
            "table.mobile-stacked tbody tr:first-child td:nth-child(2) { display: none !important; }",
            "narrow columns",
        ),
    )
    for name, css, expected in cases:
        original = header.get_attribute("aria-hidden")
        style = None
        try:
            if css:
                style = page.add_style_tag(content=css)
            else:
                header.evaluate("node => node.setAttribute('aria-hidden', 'true')")
            require(
                any(expected in issue for issue in layout_issues(page)),
                f"Browser checker accepted {name}",
            )
        finally:
            if style:
                style.evaluate("node => node.remove()")
            if original is None:
                header.evaluate("node => node.removeAttribute('aria-hidden')")
            else:
                header.evaluate(
                    "(node, value) => node.setAttribute('aria-hidden', value)", original
                )
        require(not layout_issues(page), f"Table was not restored after {name}")
        result["interactions"].append(f"mobile: rejected {name} and restored the table")


def inspect_page(page, base, relative, label, blocked, result):
    errors, responses = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: (
            errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on(
        "response",
        lambda response: (
            responses.append(f"{response.status} {response.url}")
            if response.status >= 400
            else None
        ),
    )
    response = page.goto(base + "/" + relative, wait_until="networkidle")
    require(response and response.status == 200, f"Page failed: {relative}")
    page.wait_for_function(
        "['ready','error'].includes(document.documentElement.dataset.diagrams)",
        timeout=30000,
    )
    require(
        page.locator("html").get_attribute("data-diagrams") == "ready",
        f"Diagram error: {relative}",
    )
    require(
        not errors and not responses and not blocked,
        f"{relative}: {errors}, {responses}, {blocked}",
    )
    issues = layout_issues(page)
    if relative == "index.html":
        require(
            not page.locator(".diagram-viewport").evaluate_all(
                "nodes => nodes.some(node => node.scrollWidth > node.clientWidth + 2)"
            ),
            "The small homepage diagram must fit without horizontal scrolling",
        )
    require(not issues, f"{label} {relative}: {issues}")
    diagrams = page.locator(".architecture-diagram").count()
    require(
        page.locator(".architecture-diagram svg").count() == diagrams,
        f"Missing diagram: {relative}",
    )
    result["pages"].append(
        {"page": relative, "viewport": label, "diagrams": diagrams, "status": "PASS"}
    )
    return diagrams


def screenshot(page, output, name, result, full=False):
    page.screenshot(path=str(output / name), full_page=full)
    result["screenshots"].append(name)


def diagram_controls(page, label, output, result):
    diagram = page.locator(
        '.architecture-diagram[data-diagram-type="classDiagram"]'
    ).first
    require(diagram.count(), "Architecture page has no class diagram")
    opener = diagram.get_by_role("button", name="Expand diagram")
    painted = "element => [...element.querySelectorAll('rect,path,polygon,text')].map(node => [getComputedStyle(node).fill, getComputedStyle(node).stroke])"
    original_paint = diagram.locator("svg").evaluate(painted)
    opener.click()
    dialog = page.get_by_role("dialog", name="Expanded diagram")
    require(dialog.is_visible(), "Expanded class diagram did not open")
    require(
        dialog.locator("svg").evaluate(painted) == original_paint,
        "Expanded SVG lost its original styles",
    )
    initial = dialog.locator("svg").evaluate(
        "element => element.getBoundingClientRect().width"
    )
    dialog.get_by_role("button", name="Zoom in").click()
    enlarged = dialog.locator("svg").evaluate(
        "element => element.getBoundingClientRect().width"
    )
    require(enlarged > initial, "Diagram zoom did not enlarge the SVG")
    screenshot(page, output, f"{label}-expanded-class-diagram.png", result)
    if label == "mobile":
        canvas = dialog.locator(".diagram-dialog-canvas")
        position = canvas.evaluate(
            "element => {element.scrollLeft = element.scrollWidth; element.scrollTop = element.scrollHeight; return {x:element.scrollLeft, y:element.scrollTop};}"
        )
        require(
            position["x"] > 0 and position["y"] > 0,
            "Mobile class diagram did not pan to its far side",
        )
        require(
            canvas.evaluate(
                "element => {const area=element.getBoundingClientRect(); return [...element.querySelectorAll('g.node.default')].some(node => {const box=node.getBoundingClientRect(); return box.left < area.right && box.right > area.left && box.top < area.bottom && box.bottom > area.top;});}"
            ),
            "Panning reached no class nodes",
        )
        # Center a reached leaf so its name and members are readable in the evidence.
        centered = canvas.evaluate("""element => {
            const area = element.getBoundingClientRect();
            const nodes = [...element.querySelectorAll('g.node.default')]
                .map(node => ({node, box:node.getBoundingClientRect()}))
                .filter(({box}) => box.left < area.right && box.right > area.left && box.top < area.bottom && box.bottom > area.top)
                .sort((a,b) => b.box.bottom - a.box.bottom);
            if (!nodes.length) return false;
            const {node, box} = nodes[0];
            element.scrollLeft += box.left - area.left - (element.clientWidth - box.width) / 2;
            const visible = node.getBoundingClientRect();
            return visible.left >= area.left && visible.right <= area.right;
        }""")
        require(centered, "A reached class could not be read across the phone viewport")
        screenshot(page, output, "mobile-class-diagram-far-side.png", result)
        result["interactions"].append(
            "mobile: pan class diagram to opposite horizontal and vertical edges"
        )
    dialog.get_by_role("button", name="Fit", exact=True).click()
    screenshot(page, output, f"{label}-class-diagram-overview.png", result)
    page.keyboard.press("Escape")
    page.locator("dialog").wait_for(state="detached")
    require(
        opener.evaluate("element => element === document.activeElement"),
        "Diagram did not return keyboard focus",
    )
    with page.expect_download() as downloaded:
        diagram.get_by_role("button", name="SVG", exact=True).click()
    asset = downloaded.value
    require(asset.suggested_filename.endswith(".svg"), "Diagram download is not SVG")
    saved = output / f"{label}-class-diagram.svg"
    asset.save_as(saved)
    require("<svg" in saved.read_text(), "Downloaded diagram has no SVG")
    result["interactions"].append(
        f"{label}: class diagram expansion, zoom, SVG download, Escape and focus return"
    )


def check_browser(directory: Path, output: Path):
    directory, output = directory.resolve(), output.resolve()
    require(
        not output.is_relative_to(directory),
        "Browser evidence must be outside the built site",
    )
    output.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(directory))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    pages = sorted(
        path.relative_to(directory).as_posix() for path in directory.rglob("*.html")
    )
    inventory = {
        path.relative_to(directory)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    inventory_bytes = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
    (output / "site-inventory.json").write_bytes(inventory_bytes)
    result = {
        "site_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "status": "RUNNING",
        "site": str(directory),
        "pages": [],
        "interactions": [],
        "screenshots": [],
    }
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--no-sandbox"])
            result["browser"] = browser.version
            for label, width, height, scheme, text_scale in (
                ("desktop", 1440, 1000, "light", 1),
                ("mobile", 390, 844, "light", 1),
                ("dark", 1440, 1000, "dark", 1),
                ("large-text", 1440, 1000, "light", 2),
            ):
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    color_scheme=scheme,
                    reduced_motion="reduce",
                )
                if text_scale == 2:
                    context.add_init_script(
                        "document.addEventListener('DOMContentLoaded', () => document.documentElement.style.fontSize = '200%')"
                    )
                blocked = []

                def route_request(route, blocked=blocked):
                    if route.request.url.startswith(
                        base + "/"
                    ) or route.request.url.startswith("data:"):
                        route.continue_()
                    else:
                        blocked.append(route.request.url)
                        route.abort()

                context.route("**/*", route_request)
                selected = pages if label in ("desktop", "mobile") else KEY_PAGES
                for relative in selected:
                    page = context.new_page()
                    diagrams = inspect_page(
                        page, base, relative, label, blocked, result
                    )
                    if relative == "deliver/vllm-upstream.html" and label == "mobile":
                        table_regressions(page, result)
                    if relative in KEY_PAGES:
                        name = f"{label}-{relative.removesuffix('.html').replace('/', '-')}.png"
                        screenshot(
                            page, output, name, result, full=relative == "index.html"
                        )
                    if (
                        relative == "understand/model.html"
                        and diagrams
                        and label in ("desktop", "mobile", "dark")
                    ):
                        diagram_controls(page, label, output, result)
                    page.close()
                if label in ("desktop", "mobile"):
                    page = context.new_page()
                    page.goto(base + "/search.html?q=RMSNorm", wait_until="networkidle")
                    page.wait_for_selector("#search-results li a", timeout=30000)
                    matches = page.locator("#search-results li a").all_text_contents()
                    require(matches, "Search returned no RMSNorm results")
                    result["interactions"].append(
                        f"{label}: local search returned {len(matches)} results"
                    )
                    if label == "mobile":
                        page.goto(base + "/index.html", wait_until="networkidle")
                        page.locator(
                            'label.nav-overlay-icon[for="__navigation"]'
                        ).click()
                        require(
                            page.locator("#__navigation").is_checked(),
                            "Mobile sidebar did not open",
                        )
                        page.locator(".sidebar-drawer").locator(
                            'a[href="use/runtime.html"]'
                        ).click()
                        page.wait_for_url("**/use/runtime.html")
                        require(
                            not page.locator("#__navigation").is_checked(),
                            "Mobile sidebar did not close after navigation",
                        )
                        result["interactions"].append(
                            "mobile: sidebar opens, navigates and closes"
                        )
                    page.close()
                require(not blocked, f"Off-site requests: {blocked}")
                context.close()
            browser.close()
        observed_inventory = {
            path.relative_to(directory)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        require(
            observed_inventory == inventory,
            "Built site changed during browser acceptance",
        )
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = str(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        (output / "browser-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
