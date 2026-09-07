# Framework integration workflows

Choose the consumer you are working on:

- [vLLM](vllm/README.md): fresh installation, operator and model checks, measurements and disaggregated serving.
- [SGLang](sglang/README.md): upstream model cases and their candidate-AITER installation path.
- [ATOM](atom/README.md): retained upstream integration and disaggregation jobs.
- [FlashAttention](flash-attention/README.md): its existing downstream integration.
- [Cross-framework work](common/README.md): rolling comparisons and Kimi workloads that use both vLLM and SGLang.

A framework workflow chooses an execution protocol. Maintained local profiles keep their tests, models and acceptance criteria in reviewed [client declarations](../../../ci/clients/registry.json) and [framework tests](../../../tests/frameworks/README.md). Retained integrations such as FlashAttention and ATOM still select upstream tests through their declared workflow steps; those are separate execution paths. Recurring invocations have a matching home under [framework schedules](../schedules/frameworks/README.md).

Return to the [workflow source guide](../README.md).
