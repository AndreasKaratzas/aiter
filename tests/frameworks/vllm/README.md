# vLLM tests

Choose the feature you want to check. Operator tests exercise one framework adapter; model tests load real weights and check generation or serving behavior. Both must observe the AITER implementation they claim to exercise.

| Area | Acceptance condition |
| --- | --- |
| [Imports](imports/test_imports.py) | Public vLLM entry points and selected AITER bridges import from the selected environment before any model runs. |
| [Operator bridges](operators/README.md) | Normalization, quantization, position encoding, attention and MoE bridges reach AITER and agree with independent tensor references. |
| [OpenAI server](entrypoints/test_openai_server.py) | Completion and chat streams match complete responses; batched request ordering, usage and token log probabilities remain consistent; invalid input and model names do not prevent recovery. Every path observes AITER in its GPU worker. |
| [Language quality](evaluation/test_gsm8k.py) | Qwen3 answers pinned GSM8K arithmetic questions through the real server: 32 daily questions and 128 disjoint extended questions, with a fixed exact-answer threshold. |
| [Qwen3 language model](models/language/test_qwen3.py) | Greedy tokens match Transformers, and every prompt-token likelihood stays within the declared BF16 cross-engine bounds. |
| [GPT-OSS experts](models/moe/test_gpt_oss.py) | Real GPT-OSS-20B packed MXFP4 weights execute explicitly selected native CK or AITER Triton experts and unified attention; all eight arithmetic/queue tasks require exact final answers. This is model behavior, separate from GPQA and tensor-kernel references. |
| [GPQA Diamond](evaluation/test_gpqa.py) | Separate GPT-OSS quality evaluation on 32 or all 198 real questions with a predeclared accuracy floor; requires publisher-approved dataset bytes. Kernel tests cannot substitute for this result. |
| [DeepSeek MLA](attention/mla/test_deepseek.py) | DeepSeek-V2-Lite-Chat executes MLA and sparse experts; every prompt likelihood is compared with independent eager Transformers. |
| [Dense FlashAttention](attention/flash/test_decoder.py) | The same Qwen prompts exercise unified attention and FlashAttention against an independent model reference. |
| [Speech recognition](models/speech/test_qwen_asr.py) | Qwen3-ASR transcribes 4 or 16 original LibriSpeech recordings against human transcripts, with observed encoder FA and decoder attention. |
| [Chart understanding](models/multimodal/test_chartqa.py) | Qwen2.5-VL answers 16 original human-authored ChartQA questions, using the declared exact-label or numeric criterion. |
| [Publisher FP8 checkpoint](quantization/test_checkpoint_fp8.py) | Real Qwen block-FP8 weights execute AITER GEMM and agree within declared likelihood bounds with independently dequantized eager Transformers. |
| [Image grounding](models/multimodal/test_image_grounding.py) | Real Qwen2.5-VL weights answer “red” and “blue” for controlled images with identical text. |
| [Batching](generation/batching/test_batch_equivalence.py) | Batched and individual Llama generations produce identical complete token sequences. |
| [Speculation](generation/speculative_decoding/test_ngram.py) | GPU n-gram proposals are accepted and preserve ordinary greedy decoding. |
| [Prefix cache](attention/prefix_cache/test_reuse.py) | A repeated prompt reuses cached tokens; clearing the cache removes reuse without changing the answer. |
| [Chunked prefill](execution/scheduling/test_chunked_prefill.py) | The V1 scheduler actually splits prefill into bounded steps while preserving the unchunked output. |
| [Graph replay](execution/cuda_graph/test_decode.py) | Single and paired requests replay graphs containing AITER kernels and match eager generation. |
| [Tensor parallelism](distributed/tensor_parallel/test_tp2.py) | Two worker processes on distinct GPUs execute AITER and match a one-GPU generation. |
| [Online FP8](quantization/test_online_fp8.py) | Online per-channel FP8 weights execute AITER linear kernels and preserve a controlled repeated-sequence continuation. |
| [Long and ragged contexts](models/language/test_context.py) | Extended: Llama and Qwen preserve every generated token for mixed 2048/321/65-token inputs across observed chunked and unchunked scheduling. |
| [FP16 models](models/language/test_context.py) | Extended: both families load actual FP16 parameters and agree with independent FP16 Transformers token likelihoods and greedy output. |
| [Multimodal HTTP](entrypoints/multimodal/test_chat.py) | Extended: real Qwen images sent through the OpenAI chat API retain red/blue grounding and observed vision/decoder AITER execution. |
| [Image ordering](models/multimodal/test_image_order.py) | Extended: reused and reordered images keep their corresponding answers. |
| [FP8 likelihoods](quantization/test_likelihood.py) | Extended: every recorded prompt-token likelihood is compared with BF16, with actual FP8 weights and AITER GEMM observed. |
| [Mixed speculation](generation/speculative_decoding/test_mixed_batch.py) | Extended: a batch produces both accepted and rejected drafts while preserving all target token sequences. |

Yes, the speculative tests run with AITER enabled. The engine sets `VLLM_ROCM_USE_AITER=1`, explicitly selects AITER normalization and unified attention, and requires successful calls and kernel launches from its GPU worker. The test also checks speculative metrics and output tokens. Setting a flag alone cannot pass it. These are real integration tests, not mock-based unit tests.

The daily `vllm-nightly` and weekly `vllm-extended` profiles combine the numerical operator matrices with bounded real-model scenarios. `vllm-operators` runs the standard tensor numerical and admission regressions and requires no model assets. The strict `vllm-hipblaslt` profile is separate: its optional preshuffled FP8 route has no valid library solutions in the reviewed gfx950/ROCm 7.2.3 environment and is not counted as passed numerical coverage. See the [client guide](../../../ci/clients/vllm/README.md#evaluate-the-optional-hipblaslt-path). `vllm-e2e` selects publicly provisionable model scenarios; `vllm-gpqa` separately selects the gated evaluation. The import gate runs before workload execution. Use the generated inventory below for current group and case counts; it reports declarations, not completed GPU runs. The [upstream selection guide](../../../ci/clients/vllm/upstream/README.md) explains the relationship to upstream areas.

The [model runtime guide](runtime/README.md) explains engine settings, verified weights, worker observations and commands. Shared GPU markers, package paths, model admission and process cleanup belong to [tests/common](../../common/README.md). Framework tensor references and bridge tracing belong to `frameworks/common`.

Local discovery skips unavailable prerequisites and expensive model cases. Qualification uses `--require-capabilities`; model groups also use `--run-e2e`. A selected qualification case must run. Test filename ordering does not enforce the import prerequisite; the pipeline does.

## Find a model feature and its actual cases

The [coverage catalog](../../../ci/clients/vllm/coverage.json) joins each concrete pytest selector to its pinned model, family, dtype, topology, execution mode, context capacity, feature and oracle. It covers seven pinned assets, including dense BF16, publisher FP8, packed MXFP4 experts, MLA and speech. This is selector coverage, not an execution or supported-model claim.

```bash
python -m ci coverage --client vllm
python -m ci.clients.vllm.coverage --profile vllm-e2e
python -m ci.clients.vllm.coverage --feature long-context
python -m ci.clients.vllm.coverage --family qwen3 --dtype float16
python -m ci.clients.vllm.coverage --tensor-parallel 2
```

GPT-OSS and DeepSeek declarations apply to the exact retained checkpoint and selected paths. They do not establish whole-family support. GPQA execution additionally requires approved data and cannot be inferred from model generation or kernel results. Learned drafts, hybrid recurrent models, multilingual or streaming ASR, TTS, video, pooling/reranking, KV transfer and TP4/TP8 remain explicit gaps. The byte-level checkpoint registry is separate from feature declarations; neither records a fabricated run.

The [dataset and scoring guide](evaluation/README.md) documents original data, licensing, prerequisites, answer parsing, numerical references and the distinction between pytest case counts and evaluated questions or utterances.
