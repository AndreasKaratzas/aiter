# Test and tune the AITER library

Start with [qualification.yaml](qualification.yaml) for declared product profiles and affected framework checks. The [product catalog](../../../ci/qualification/catalog.json) names test groups and their requirements; `python -m ci coverage --client aiter` shows the actual selected paths.

[legacy.yaml](legacy.yaml) retains the broad existing library workflow. Its standard and communication jobs call the [same reusable job](../reusable/library-area.yaml), then the [library-area controller](../../../ci/pipelines/product.py). The [area declaration](../../../ci/pipelines/product_areas.json) and existing weighted splitter own file selection. Historical driver exclusions, tuning and performance bookkeeping remain explicit; this path is not silently treated as strict release qualification.

The remaining files have narrower responsibilities:

- [triton.yaml](triton.yaml), [fmha.yaml](fmha.yaml) and [opus.yaml](opus.yaml) retain their specialized kernel suites.
- [extended.yaml](extended.yaml) carries the existing extended workload.
- [tuning.yaml](tuning.yaml) performs operator tuning; [tuning-validation.yaml](tuning-validation.yaml) checks the declared tuning suites.
- [network.yaml](network.yaml) diagnoses connectivity to package and source services.

Change recurring times in [library schedules](../schedules/library/README.md). GPU and compiler requirements belong to the selected test or workload, not the folder name.

Return to the [workflow source guide](../README.md).
