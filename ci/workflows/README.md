# Edit workflows by their owner

Workflow definitions live here in real directories. The files in `.github/workflows/` are generated copies because GitHub only discovers entrypoints directly in that directory. Follow the [complete source index](index.md) to find the source, its stable GitHub filename and the application it calls.

| Directory | Responsibility |
| --- | --- |
| `common/` | Reusable profile execution shared by product, client and release workflows |
| `host/` | CPU checks, documentation, workflow validation and repository maintenance |
| `product/` | Product qualification, specialized kernels, communication and tuning |
| [clients/vllm/](clients/vllm/README.md) | Fresh installation, operator/model qualification, measurements and disaggregation |
| `clients/sglang/` | SGLang downstream model execution |
| `clients/atom/`, `clients/kimi/`, `clients/flash-attention/` | The respective specialized integrations |
| `clients/common/` | The rolling canary entrypoint shared across client adapters |
| `release/` | Wheel builds, qualification, images, publication and channel history |

These are complete YAML sources, not an additional templating language. Reusable jobs still call stable `./.github/workflows/NAME.yaml` entrypoints. Runtime execution, evidence and Docker cleanup remain owned by [ci/pipelines](../pipelines/README.md); reviewed test selections remain in the product and client catalogs. Specialized jobs retain the procedures shown in their sources. Generation does not claim that they have all migrated to one execution protocol.

## Make a change

Edit the source file, then run from the repository root:

```bash
python -m ci.workflows --write
python -m ci.workflows --check
```

Commit the source and generated output together. The generator prepends a comment with the canonical path and source SHA-256, then copies the YAML body without parsing or reformatting it. Expressions, triggers, permissions, job/check names and reusable-workflow calls retain their source spelling. There is no second independently edited workflow definition.

`registry.json` is the single source-to-entrypoint map. Add one record and a nested source when adding a workflow. Preserve existing generated filenames when moving sources, so reusable calls, status-check names and external links continue to work. For an intentional workflow removal, remove its source, registry record and generated entrypoint explicitly; the generator refuses to delete unknown files for you.

New entrypoint names use their owner prefix: `common-`, `host-`, `product-`, `client-` or `release-`. The existing `product-run-profile.yaml` is the sole compatibility exception: its source now belongs to `common/run-profile.yaml`, while callers keep the established filename.

`--check` performs no writes. It rejects stale or missing generated output and indexes, undeclared sources/entrypoints, duplicate mappings, noncanonical paths, symlinks, missing application modules and missing or nested local workflow calls. All inputs and output paths are checked before generation writes. `--root /path/to/reviewed-controls` selects an explicit independent tree; missing files are not borrowed from the executing checkout.

The existing `python -m ci.pipelines workflows --check` and `--write-index` commands delegate to this application for compatibility. `--write-index` now updates the complete generated outputs, so prefer the clearer `ci.workflows --write` command for new instructions.

## Verify execution syntax separately

Host checks and the Actionlint workflow run generation checking. Actionlint then validates the generated YAML using its normal GitHub schema and expression checks. Unit tests exercise round-trip preservation, drift and path rejection, and actual CLI failure behavior. The generator is a materializer and ownership validator, not a YAML parser or an independent permission-review service.

GitHub documents that [workflow subdirectories are unsupported](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow). Do not put inactive nested YAML below `.github/workflows` or edit its generated files directly. Hierarchical sources belong here; [nested scripts](../../.github/scripts/README.md) and shared controllers hold reusable execution behavior.
