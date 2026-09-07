# SPDX-License-Identifier: MIT
"""Documentation configuration; no AITER, Torch or GPU imports."""

project = "AITER"
author = "AMD ROCm contributors"
copyright = "2026, Advanced Micro Devices, Inc."
extensions = ["myst_parser", "docs.website.extension"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "website",
    "README.md",
    "DEPLOYMENT.md",
    "migrations",
    "DOCUMENTATION_AUDIT_REPORT.md",
    "newsletter",
    "examples",
    "autotuning_pipeline.md",
    "triton_comms.md",
    "isa_kernel_optimization.md",
    "aiter_container_nonroot_setup.md",
]
html_theme = "furo"
html_title = "AITER documentation"
html_logo = "assets/aiter_logo_mini.png"
html_favicon = "assets/aiter_logo_mini.png"
html_use_index = False
html_static_path = ["_static"]
html_css_files = ["site.css"]
html_js_files = ["vendor/mermaid.min.js", "site.js"]
html_show_sourcelink = False
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#166d80",
        "color-brand-content": "#126577",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7ed4dc",
        "color-brand-content": "#7ed4dc",
    },
    "navigation_with_keys": True,
}
myst_heading_anchors = 6
myst_fence_as_directive = ["mermaid"]
myst_all_links_external = True
myst_enable_extensions = ["colon_fence", "deflist"]
nitpicky = True
