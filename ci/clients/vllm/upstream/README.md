# Upstream selections and the AITER port

The [selector inventory](buildkite-selectors.json) records the exact commands, exclusions, markers, shard arguments, hardware choices, optional flags and AMD mirror overrides at vLLM revision `5690b02c03832a4ac3af231d3ecebe649c188095`. It also records the referenced model configuration lists and the SHA256 of each source YAML file. These are selections to inspect, not tests that AITER claims to have executed.

The current vLLM checkout has both `.buildkite/test_areas/*.yaml` and a separate `.buildkite/test-amd.yaml`. The inventory retains both forms. They overlap and must not be added together as executed coverage. For LM Eval, Language Models, Multimodal Models and Spec Decode, the area-file inventory includes every step, including steps whose individual labels have a different prefix. Matching labels in other area files are also retained. This explains why the complete area counts exceed an initial label-only search.

| Requested area | Area-file selections | Separate AMD-file selections | What the upstream selection runs |
| --- | ---: | ---: | --- |
| Entrypoints Integration | 9 | 16 | Offline LLM, server and scale-out, OpenAI completion/chat, generate/tool protocols, Responses, speech, multimodal and pooling; API server shards and clean-process exclusions are retained. |
| LM Eval | 28 | 11 | GSM8K configuration lists across device/model sizes, distributed and speculative configurations, legacy lm-eval harness, GPQA and other accuracy jobs. Optional and hardware-specific jobs remain explicit. |
| Qwen3 | 6 | 5 | Matching Qwen3 accuracy and EPLB jobs. There is no standalone group named Qwen3 in this checkout; Qwen3 also appears inside language, vision and speculative selectors. |
| Language Models | 8 | 11 | Core/slow markers, hybrid models, a separate Granite compatibility job, extended generation, perplexity, pooling and MTEB. |
| Multimodal Models | 12 | 14 | Core and extended generation, Qwen/Gemma partitions, processor shards, tensor schemas, perplexity, accuracy and pooling. |
| Quantized Models | 1 | 2 | `models/quantization`, distinct from the broader low-level `quantization/` and fusion jobs. |
| Spec Decode | 16 | 11 | N-gram/suffix, draft models, Eagle, speculators/MTP, acceptance-rate jobs and related selectors outside the main area file. |
| V1 | 9 | 13 | Engine, attention, sampling/logits, core/KV/metrics, CPU utilities, end-to-end execution and hybrid chunked prefill. |

The AITER [group declarations](../groups.json) deliberately select a bounded port. Every selected engine or server test enables AITER and requires actual normalization and attention observations from the worker that generated the answer. Operator checks have their own direct numerical assertions. The strict optional `vllm-hipblaslt` profile is excluded from daily and extended runs; the reviewed gfx950/ROCm 7.2.3 library exposes no valid solutions for its preshuffled row/channel FP8 route, so those two tests remain failed rather than numerical support claims. Enabling a flag, importing a module or successfully starting the engine is insufficient.

| Area | Daily checks | Extended checks | Scope excluded from this port |
| --- | --- | --- | --- |
| Entrypoints | Real loopback OpenAI completion and chat, streaming equivalence, model listing, invalid-request recovery | Qwen image grounding through real OpenAI chat requests; batched completion accounting/logprobs and unknown-model recovery also run | Speech HTTP transport, pooling, Responses, tool protocols and scale-out server topologies |
| LM Eval | First 32 pinned GSM8K test questions, four training demonstrations, exact final answers | Next 128 GSM8K test questions and 16 real human ChartQA questions; separate GPT-OSS GPQA32/198 declaration | Full GSM8K leaderboard scores, MMLU and large-model matrices; GPQA is a separate approved-data profile |
| Qwen3 / language | Real Qwen3-1.7B greedy tokens and every prompt log probability against Transformers eager attention; existing Llama batching | FP8 likelihood drift, independent FP16 likelihoods on Llama/Qwen and 2048-token ragged-context scheduling | Other Qwen3 sizes, additional MoE/hybrid families, embedding/classification/MTEB |
| Multimodal | Qwen2.5-VL-3B grounds red and blue image pixels | Reorder and reuse those images across three batches; answers must remain attached to each image | Other vision families, video and pooling; Qwen3-ASR recorded speech has its own4/16-utterance tests |
| Quantized models | Actual online FP8 Llama, publisher block-FP8 Qwen and packed MXFP4 GPT-OSS weights with observed AITER paths | Every recorded prompt-token likelihood compared with BF16 | Other prequantized formats, AWQ/GPTQ/NVFP4 and general accuracy equivalence |
| Spec Decode | GPU N-gram proposals, nonzero acceptance and exact greedy target tokens | Mixed prompts with both accepted and rejected proposals; every target output must match | Learned draft models, Eagle, MTP, DFlash and DSpark |
| V1 | Actual 128-token prefill scheduling, prefix reuse, graph replay and TP2 output equivalence | Daily scenarios remain required | The complete upstream engine, scheduler, KV-transfer and distributed unit suites |

The current local scope also includes DeepSeek-V2-Lite MLA and expert execution, dense-model FA versus unified attention, real Qwen3-ASR recordings, and original chart questions. Every model case retains its own oracle and exact checkpoint identity. GPT-OSS kernel tests and its gated GPQA evaluation are separate selections.

Use `python -m ci coverage --client vllm` for current profile/group/path counts, and `python -m ci.clients.vllm.coverage --profile vllm-e2e` for concrete model cases. A pytest case can evaluate many questions, prompts or worker operations; those quantities are reported separately. These declarations do not establish that all selected tests have run on any particular hardware.

To refresh the inventory from a reviewed vLLM checkout, use an interpreter with PyYAML installed:

```bash
python -m ci.clients.vllm.upstream.inventory \
  --repo /path/to/vllm \
  --output ci/clients/vllm/upstream/buildkite-selectors.json
```

Review the changed commands and model lists before changing the port. The inventory command reads YAML and Git identity; it does not import upstream test modules, download models or launch their shell commands.

The [feature-to-case catalog](../coverage.json) is the maintained local execution map; the Buildkite inventory remains the pinned upstream survey. `python -m ci.clients.vllm.coverage` validates exact local selector/group/model closure and reports declared cases. It lists learned drafts, hybrid recurrent models, additional MoE and speech families, video, pooling, KV transfer and larger topologies as gaps. GPQA data admission is explicit and distinct from model or kernel execution. Adding a model name to a matrix is not a hardware execution result.
