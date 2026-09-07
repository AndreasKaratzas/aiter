# Documentation coverage and evidence

The maintained website renders the code-adjacent architecture, runtime, build, tuning, testing and delivery guides listed in [the guide map](website/guides.json). Its introduction and API pages explain a narrower entry point for new readers. The [website checks](README.md) cover every generated HTML page, including local source pages and nested relative links.

## Current guides

The architecture describes responsibilities and dependency direction. Runtime and API guides describe accepted layouts, dtypes and lifecycle rules. The framework page traces specific installed framework bridges; it does not turn a small operator test into a claim that an entire model was qualified. Delivery guides describe the implemented controls, with infrastructure and publication status recorded separately in [engineering evidence](../notes.md).

The specialist tuning, Iris communication, non-root container and native code-object guides have been revised against their current source paths and interfaces. They distinguish preparation from execution, estimates from guarantees, and experimental artifacts from qualified ones. The Iris and code-object experiments are not presented as passing product qualification results.

## Historical and experimental material

The [May 2026 newsletter](newsletter/2026-05.md) is published under **Historical material**, with its original reporting clearly separated from current guidance. Its hardware, releases, roadmap and model scores are not independently certified by the current website build.

The [ISA helper scripts](examples/isa_optimization/README.md) are available for inspection as experimental utilities. Their sample Dockerfile is not a supported product image. Former instructions to overwrite installed or source-tree code objects have been removed from the maintained guide.

This page replaces an older static audit report. A historical sentence saying a check passed is not a substitute for a fresh build, browser report or operation test result.

## What the checks establish

| Check | Establishes | Does not establish |
|---|---|---|
| Warning-fatal Sphinx build | All selected guides parse and their declared references resolve. | GPU example correctness. |
| Local link and anchor audit | Local pages, assets, downloads and fragments exist in this build. | Availability or accuracy of an external article. |
| Offline Chromium checks | Actual diagrams, local search, navigation and responsive layouts work in the checked viewports. | Every browser or assistive technology behaves identically. |
| Human screenshot review | The selected desktop, phone and expanded diagram views are readable. | Automatic verification of every factual sentence. |
| Operator and framework tests | Their exact declared numerical and integration scopes pass in the recorded environment. | Untested models, devices, versions or publication readiness. |

When a source guide changes, rebuild the website and rerun its browser checks. When an API example changes, validate its corresponding operator or framework test as well. Keep the resulting evidence with the change rather than editing an old acceptance record.
