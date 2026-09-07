# Model quality and independent references

Quality tests load the exact checkpoint from the [model registry](../../../../ci/clients/vllm/models.json), execute vLLM, and require actual AITER observations from the worker that produced each answer. A model startup or a tensor-kernel pass cannot satisfy an evaluation. Results retain every selected row in the denominator, including empty, truncated or incorrectly formatted responses.

## Datasets and criteria

| Dataset | Selection | Criterion | Input ownership |
|---|---|---|---|
| GSM8K | 32 daily and 128 disjoint extended test questions, with four training demonstrations | At least 50% exact final numeric answers | Original row positions and source hashes in [gsm8k.json](fixtures/gsm8k.json) |
| GPQA Diamond | 32 smoke or all 198 questions, deterministic choice order | At least 50% correct final choices, maximum 4096 generated tokens, medium reasoning | Metadata only in [gpqa.json](fixtures/gpqa.json); publisher-approved offline CSV required |
| LibriSpeech test-clean | 4 smoke or 16 original recordings, 1–15 seconds each | Word error rate at most 15%, corpus numerator/denominator | Original FLAC bytes and human transcripts in [librispeech](fixtures/librispeech/README.md) |
| ChartQA | First 16 human-authored test rows | At least 50% correct; exact normalized label or 5% relative numeric error | Metadata only in [chartqa.json](fixtures/chartqa.json); original test Parquet required |

These are bounded regression criteria. They do not reproduce full leaderboard scores or estimate confidence intervals. The speech smoke set overlaps the 16-recording set; GPQA smoke overlaps Diamond. GSM8K's extended rows are disjoint. A passing pytest case reports its actual question/utterance count separately.

## Provision datasets explicitly

Model files are provisioned with the existing shared model command. Dataset tests never download implicitly. Public ChartQA bytes can be cached separately; GPQA additionally requires the publisher's approved account and terms. Do not distribute GPQA plaintext questions or circumvent its access gate.

```bash
hf download HuggingFaceM4/ChartQA \
  data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet \
  --repo-type dataset --revision b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5

# Run only after the publisher approves your account.
hf download Idavidrein/gpqa gpqa_diamond.csv \
  --repo-type dataset --revision 633f5ee89ab8ad4522a9f850766b73f62147ffdd

python -m pytest -c tests/pytest.ini \
  tests/frameworks/vllm/evaluation/test_gpqa.py \
  --run-e2e --require-capabilities
```

GPQA admission verifies the publisher revision's exact file size and Git blob identity before parsing; the result additionally records SHA-256. Its retained report contains row hashes, selected letters, correctness and source/model identity. It excludes questions, request bodies and free-form model responses. A missing approved dataset fails required qualification and cannot count as a passing model test.

ChartQA verifies the full Parquet hash before extracting original image bytes into the private test directory. No ChartQA images are bundled in this repository. The publisher declares GPL-3.0 for that dataset. The retained LibriSpeech slice preserves original encoded audio and records the original Parquet hash, row indices and attribution under CC-BY4.0. Audio decoding requires SoundFile; ChartQA extraction requires PyArrow.

## Reference computations

Dense BF16/FP16 and DeepSeek MLA comparisons use independent Transformers eager attention and the same verified weights. Every prompt token must align and have a finite likelihood; both maximum and mean error bounds apply. Controlled greedy-token tests additionally require exact output sequences. A bounded prompt comparison does not establish general answer quality.

The publisher Qwen FP8 checkpoint reference decodes each 128×128 weight block in FP32 using its stored inverse scale, casts the resulting weights to BF16, and runs the ordinary eager Transformers model. This is independent of AITER quantization and GEMM. The test separately requires real FP8 parameters and AITER blockscale execution. Quantized model bounds are explicit in the test and do not claim universal quantized/BF16 equivalence.

GPT-OSS model behavior and GPQA accuracy remain separate from [GPT-OSS tensor adapter tests](../operators/moe/test_quantized_gpt_oss.py). Packed weight dtype, selected expert method, actual expert calls and attention execution must be observed. A reference discrepancy remains a failed result until its cause is resolved; registering the case does not assert that the selected backend passed it.

The [GPT-OSS behavior fixture](fixtures/gpt_oss_semantics.json) contains eight independently specified arithmetic, queue, expert-dispatch and chunk-count tasks. It is an authored regression fixture, not a public quality dataset. Native CK and AITER Triton are selected explicitly and tested separately; each must return every final integer exactly through the real OpenAI server. Reasoning text cannot supply the answer, and truncation fails. Worker identities prove logical FP4 encoding and actual packed storage, including Triton's logical tensor views.

An earlier GPT-OSS dequantized Transformers likelihood comparison failed. Independent non-AITER vLLM also exceeded that dense-model tolerance, and native versus Triton experts produced different prompt likelihoods. Those diagnostics remain failed comparisons; they are not relabeled as passed numerical equivalence. Strict tensor tests instead compare the actual selected activation arithmetic and reductions, while the model behavior cases check their stated final-answer contract. GPQA still requires its own approved data and accuracy result.

The shared client controller invokes `python -m ci.clients.vllm.datasets` for selected public ChartQA input, using the existing provisioning stage and retaining its exact-byte receipt. This helper has no automatic GPQA admission. Offline tests reverify data before evaluating rows.
