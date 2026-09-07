# SGLang serving adapters

`kimi_accuracy.sh` and `kimi_perf.sh` run the specialized Kimi SGLang serving checks with their workflow's explicit model, environment and GPU allocation. Their results describe those workloads rather than all SGLang model support.

The [framework guide](../../../../docs/use/frameworks.md) explains integration scope, and the [script index](../../README.md) links each direct caller. Shared installation and cleanup belong in the controller rather than another shell bootstrap.
