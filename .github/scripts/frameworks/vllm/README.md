# vLLM serving adapters

`kimi_accuracy.sh` and `kimi_perf.sh` run the specialized Kimi vLLM serving checks with the model, environment and devices supplied by their workflow. They are retained workload adapters, separate from the fresh candidate installation and declared operator/model profiles.

See the [vLLM client guide](../../../../ci/clients/vllm/README.md) for current qualification, or the [script index](../../README.md) for direct callers. Accuracy and throughput runs have different assertions and must retain separate results.
