# vLLM tests

Choose the feature you want to check. Operator tests exercise one framework adapter; model tests load real weights and check generation or serving behavior. Both must observe the AITER implementation they claim to exercise.

| Area | Acceptance condition |
| --- | --- |
| [Imports](imports/test_imports.py) | Public vLLM entry points and selected AITER bridges import from the selected environment before any model runs. |
| [Operator bridges](operators/README.md) | Normalization, quantization, position encoding, attention and MoE bridges reach AITER and agree with independent tensor references. |
| [OpenAI server](entrypoints/test_openai_server.py) | Completion and chat streams match complete responses; the server recovers after invalid input and its GPU worker executes AITER. |
| [Language quality](evaluation/test_gsm8k.py) | Qwen3 answers pinned GSM8K arithmetic questions through the real server: 32 daily questions and 128 disjoint extended questions, with a fixed exact-answer threshold. |
| [Qwen3 language model](models/language/test_qwen3.py) | Greedy tokens match Transformers, and every prompt-token likelihood stays within the declared BF16 cross-engine bounds. |
| [Image grounding](models/multimodal/test_image_grounding.py) | Real Qwen2.5-VL weights answer “red” and “blue” for controlled images with identical text. |
| [Batching](generation/batching/test_batch_equivalence.py) | Batched and individual Llama generations produce identical complete token sequences. |
| [Speculation](generation/speculative_decoding/test_ngram.py) | GPU n-gram proposals are accepted and preserve ordinary greedy decoding. |
| [Prefix cache](attention/prefix_cache/test_reuse.py) | A repeated prompt reuses cached tokens; clearing the cache removes reuse without changing the answer. |
| [Chunked prefill](execution/scheduling/test_chunked_prefill.py) | The V1 scheduler actually splits prefill into bounded steps while preserving the unchunked output. |
| [Graph replay](execution/cuda_graph/test_decode.py) | Single and paired requests replay graphs containing AITER kernels and match eager generation. |
| [Tensor parallelism](distributed/tensor_parallel/test_tp2.py) | Two worker processes on distinct GPUs execute AITER and match a one-GPU generation. |
| [Online FP8](quantization/test_online_fp8.py) | Online per-channel FP8 weights execute AITER linear kernels and preserve a controlled repeated-sequence continuation. |
| [Image ordering](models/multimodal/test_image_order.py) | Extended: reused and reordered images keep their corresponding answers. |
| [FP8 likelihoods](quantization/test_likelihood.py) | Extended: every recorded prompt-token likelihood is compared with BF16, with actual FP8 weights and AITER GEMM observed. |
| [Mixed speculation](generation/speculative_decoding/test_mixed_batch.py) | Extended: a batch produces both accepted and rejected drafts while preserving all target token sequences. |

Yes, the speculative tests run with AITER enabled. The engine sets `VLLM_ROCM_USE_AITER=1`, explicitly selects AITER normalization and unified attention, and requires successful calls and kernel launches from its GPU worker. The test also checks speculative metrics and output tokens. Setting a flag alone cannot pass it. These are real integration tests, not mock-based unit tests.

The daily `vllm-nightly` profile has 19 workload groups and 85 cases. Weekly `vllm-extended` includes those same groups plus four extended groups, for 89 cases. The nightly controller runs the separate one-case import gate first. The [upstream selection guide](../../../ci/clients/vllm/upstream/README.md) explains how the port relates to vLLM's CI areas and which scope remains unported.

The [model runtime guide](runtime/README.md) explains engine settings, verified weights, worker observations and commands. Shared GPU markers, package paths, model admission and process cleanup belong to [tests/common](../../common/README.md). Framework tensor references and bridge tracing belong to `frameworks/common`.

Local discovery skips unavailable prerequisites and expensive model cases. Qualification uses `--require-capabilities`; model groups also use `--run-e2e`. A selected qualification case must run. Test filename ordering does not enforce the import prerequisite; the pipeline does.
