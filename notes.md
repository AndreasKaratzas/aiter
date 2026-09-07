# Engineering record

This file records the local implementation, the defects found during review and the evidence retained from testing. [ARCHITECTURE.md](ARCHITECTURE.md) explains the design; [rollout.md](rollout.md) gives the order for reviewing it in separate PRs.

The working branch is `akaratza_aiter_implementation`, based on `456b92780c8b650c1e3e4b0fa1ca21f0d1fb363d`. The latest request authorizes publishing this branch in an AITER fork and deploying its documentation. Implementation and local acceptance precede that publication. The proposal and manual have separate repositories. The September 7 proposal rewrite is recorded in the proposal repository; the manual was not changed in this round.

## September 7: GitHub workflow navigation

The editable GitHub configuration now lives together under `.github/`. [Start with its directory guide](.github/README.md), then choose [vLLM](.github/workflow-sources/frameworks/vllm/README.md), [SGLang](.github/workflow-sources/frameworks/sglang/README.md) or the AITER library. `repository` replaces the ambiguous workflow label `host`; `frameworks` replaces `clients`; `library` replaces `product`. These names describe what is being checked. Existing profile identifiers such as `host` and `product-nightly`, and the Python `ci.clients` application, keep their established interfaces.

| Files | Change and reason |
| --- | --- |
| `.github/workflow-sources/` | Move all 58 canonical YAML files and their registry out of the Python generator package. Group execution by repository, library, framework and release; mirror those groups under `schedules/`. Give every directory a guide. |
| `.github/workflows/`, `ci/workflows/` | Generate the flat files GitHub requires, with explicit source and directory-guide comments. Use descriptive filename prefixes and update every local reusable call. Keep a short starting page here and the complete mapping beside the sources. |
| `.github/actions/common/validate-workflows/`, `.github/workflow-sources/reusable/` | Separate a shared validation step from complete shared jobs. The action checks the root checkout and propagates validation failure; it does not install dependencies or select tests. |
| `.github/scripts/`, `ci/pipelines/scripts.{py,json}` | Move 16 adapters into the same repository/library/framework naming scheme, update their callers and retain all 25 scripts in the checked inventory. |
| `ci/ownership/owners.json`, `.github/CODEOWNERS`, `ci/qualification/catalog.json` | Update review ownership and test-selection paths along with the physical moves. Framework schedules have framework ownership. |
| `docs/website/`, current directory guides, `docs/migrations/paths.json` | Preserve existing website guide URLs, add grouped directory navigation and map historical filenames to their current destinations. |

The migration keeps the same 134 job identifiers and 19 cron selections. The `Checks` and `Documentation` workflow names remain because existing callers use them. Schedule display names now explain their work, and `Host checks` is displayed as `Repository checks`. The workflow-lint trigger also includes the new source and action directories. QA compares the complete job definitions with the previous commit, allowing only the explicit path, display-name and shared-step changes.

Both clean Python 3.10 and 3.12 environments pass the final complete CPU suite: 677 tests and 1,349 subtests per interpreter. The first 3.10 attempt loaded an unfinished QA matcher that mistook a job named `run` for a shell command; its 19 false failures are retained separately, and the corrected complete suite passes. The independent workflow comparison records zero unapproved changes across all 58 workflows. Generation, script inventory, ownership, Actionlint and architecture checks pass; the architecture check still accounts for 1,058 Python files and 9,100 dependency edges. Reports are retained under `/tmp/aiter-workflow-layout-*`, including the final interpreter XML files and `qa/final-semantic-audit.json`.

This round changes orchestration and navigation, not kernels or model selections. The existing GPU acceptance records below retain their original scope; they are not counted again as execution of the renamed workflows. The GitHub default-branch rule for scheduled runs still applies.

## September 7: upstream proposal, numerical coverage and pipeline execution

The proposal is maintained separately in `AndreasKaratzas/aiter-proposal`. Its five pages now compare the proposed design with upstream ROCm/AITER at `24a62b1c122f23645a19b9d8b0abd4750c59359b`. They describe the real wrapper/JIT/native flow, recipe evaluation, tuning/code-object handling, existing test and release jobs, and documentation gaps. They do not cite this implementation branch as evidence of upstream behavior or completed proposal work.

| Files | Change and reason |
| --- | --- |
| `ci/workflows/schedules/`, `ci/workflows/common/run-profile.yaml`, generated `.github/workflows/` | Separate 19 cron selections from reusable execution. Directory-owned YAML is authoritative; GitHub's flat entrypoints remain generated and checked. |
| `ci/pipelines/bootstrap.py`, `ci/pipelines/runner.py`, `ci/qualification/coverage.py` | Translate explicit job inputs into the existing source/wheel/client controllers and Docker executor. Preserve source/control admission and reports. The inventory connects profiles, test areas, exact selectors, hardware, models and prerequisites. |
| `tests/frameworks/vllm/operators/`, `ci/clients/vllm/{groups,profiles}.json` | Retain the original adapter cases and add required boundary suites for normalization, rotary positions, routing, FP8 scaling/GEMM/BMM, unified attention and FlashAttention. Real MLA and GPT-OSS expert/projection cases extend the operation coverage. |
| `tests/frameworks/vllm/{models,attention,quantization,evaluation,runtime}/`, `ci/clients/vllm/{models,coverage}.json` | Add actual GPT-OSS/MXFP4, DeepSeek/MLA, Qwen checkpoint-FP8, speech and human-authored chart questions, with pinned inputs and observed AITER execution. GPQA has distinct smoke/full selections and explicit dataset admission. |
| `tests/common/hardware.py`, `tests/frameworks/common/runtime.py` | Separate hardware capability from implementation eligibility; preserve introspectable adapter signatures and count successful calls rather than attempts or Triton warmups. |
| `csrc/kernels/quant_kernels.cu`, `aiter/ops/triton/_triton_kernels/{quant/quant.py,attention/unified_attention_sparse_mla.py}` | Correct defects exposed by the expanded numerical/model checks, described below. |
| `csrc/blas/hipbsolgemm.cu`, `tests/unit/build/test_hipblaslt_tuning.py`, `tests/integration/runtime/test_blas_tuning.py` | Reject empty or invalid online-tuning candidates before launch, with deterministic native-body tests and real GPU numerical/process-lifetime regressions. |

The seven main boundary files contain 12,864 collected cases; FP8 BMM adds 128. These are numerical input combinations, not 12,992 distinct model tests or a complete supported-hardware matrix. Each case computes an independent reference and observes the selected implementation. Daily and extended profiles select these areas explicitly, and `vllm-operators` makes them runnable without model downloads. Models and quality datasets have separate groups and acceptance conditions.

### Defects found during expansion

Dynamic tensor/token FP8 quantization returned a zero scale for all-zero inputs on some native paths, unlike the existing grouped quantizer. The shared non-FP4 scale calculation now applies the same `1e-10` maximum floor before division by the destination maximum. This also defines behavior for nonzero maxima below that floor. The reproduced outputs were finite; the observed defect was the zero scale and infinite reciprocal, not an observed NaN output. A 64-case direct native regression includes INT8's toward-zero conversion and mixed zero/nonzero rows. The original FP8 failure and passing fresh-cache run remain separate.

DeepSeek's real BF16 batched FP4 projection exposed an input-width assumption in `_mxfp4_quant_op`: the scale bitcast and `ldexp` operation require FP32. Promote FP16/BF16 inputs before both operations; the conversion is exact and existing FP32 callers retain their arithmetic. Independently decoded FP4 projection tests and the original 36 product quantization cases check the change.

Sparse MLA produced NaNs when every selected cache index was `-1`. The existing mathematical reference defines that empty selection as zero. Its final normalization now uses a finite denominator for the zero accumulator, leaving populated selections unchanged. Tests cover both decode and multiple queries, with scattered physical cache rows.

The tests also exposed a harness defect: wrapping an adapter with a bare `*args, **kwargs` signature made vLLM believe its MLA implementation did not accept descale arguments. The observation helper now preserves the real signature and records successful returns. This is a test-harness correction, not a change to MLA's argument contract.

FP8 attention rounds an unnormalized softmax numerator before multiplying by V. The new paged and MLA tests use an analytical elementwise error bound for that declared arithmetic, including FP8 subnormals and final BF16 rounding. Initial FP16-style tolerance failures remain retained. The correction changes the independent precision model; it does not alter attention output to satisfy a test or hide failures behind an average.

Native source changes require a new JIT cache or an explicit rebuild: the module repository's architecture/ABI checks do not automatically invalidate every source edit. An initial post-fix run reused the earlier quantization module and correctly retained its failure. Fresh-cache execution passed the 1,536 tensor/token FP8 cases. Evidence directories distinguish these inputs rather than treating the cache hit as a build of the change.

The final optional hipBLASLt review found a fourth native defect. vLLM enables online tuning for that route, and the tuner ran all 32 allocated candidate slots even when the library returned zero or fewer candidates. It now checks the returned count and each candidate's success state before reading algorithm data, excludes failed launches, and rejects an empty winner set after cleanup. The caller releases its descriptors before raising an informative error and cannot write a tuning record for a failed selection. Independent actual-function tests cover empty, partial, full and invalid lists; real BF16 tuning and a subprocess-lifetime regression pass. The optional FP8 numerical cases still report no available solution, but now produce two ordinary test failures instead of terminating the process. Earlier crashes remain retained. The `blas-tuning` product group connects this regression to fast, nightly, extended and feature profiles without claiming optional FP8 numerical support.

Local execution in this round uses gfx950 and the selected vLLM development environment at `5690b02c`. Hardware labels recognize gfx942, gfx950 and applicable gfx1250 bring-up paths; the latter is CDNA5/MI450-family work, not a claim about an unverified MI455 SKU. CK groups retain explicit target restrictions. No gfx942/gfx1250 execution or new official-nightly installation is inferred from local gfx950 results.

### Workflow execution after the second review

The review also removed substantial inline execution logic from three older pipelines. `ci/pipelines/product.py` now owns the standard and multi-GPU legacy driver areas, their exact shard inventory, candidate-wheel installation, import check and retained results. A reusable `common/product-area.yaml` preserves the two eight-shard matrices and their job/artifact identities. The canonical product workflow falls from 1,510 to 933 lines compared with the start of this round; its build, tuning and performance procedures remain explicit. The legacy drivers retain their historical exclusions and are labeled accordingly, rather than being mistaken for strict release qualification.

`ci/clients/sglang/downstream.py` now owns upstream checkout/patching, container setup, candidate replacement, import admission, model execution and cleanup. The actual current SGLang AMD branch was cloned and all patches applied successfully at `947e6b87d3ef7612528ff0fb30803c20a70118f9`. Docker and the large downstream models were not executed in this local review.

The [vLLM disaggregation application](ci/clients/vllm/disaggregation/README.md) separates case selection, wheel admission, Slurm submission and log/accuracy collection. It preserves ten active upstream model/topology cases at `a42850c85f815ab3123a541bc92fb9b35892335e`, with immutable executor/router images. Independent tests caught a commented-command parser error and ZIP path aliases/file-directory conflicts; those are corrected. Actual subprocess tests with fake Slurm executables check success, failure, owned-job cancellation and accounting failures. The actual pinned vLLM checkout also passed the common bootstrap and runner selection phase, exporting all ten exact cases; `/tmp/aiter-disaggregation-real-selection/{bootstrap,phase}.json` and `/tmp/aiter-disaggregation-real-matrix.txt` retain that metadata-phase result. The real Spur cluster and its GPU containers remain a separate execution requirement.

### Current local acceptance

These results describe separate, identified executions. The case counts below must not be added to older historical runs, and the ordinary installed-wheel checks overlap the source regressions.

| Scope | Result | Retained evidence |
| --- | --- | --- |
| Standard vLLM operator selection | **13,100 PASS** on gfx950: 13,093 numerical/mixed-behavior cases and seven admission-only checks | Root: `operator-regression-final.xml`, `unified-refactored-final.xml`, `bmm-flash-reviewed.xml` under `/tmp/aiter-kernel-expansion/`; independent QA: `observed-final.xml`, `native-moe-full-final.xml`, `fp8-final.xml` under `/tmp/aiter-kernel-expansion-qa/` |
| Direct product regressions | **144 PASS**, including 64 native zero-scale cases and the original FP4/activation checks | `product-regression-final.xml` and `product-fp4.xml` under `/tmp/aiter-kernel-expansion/` |
| Installed ordinary wheel | **122 PASS** against isolated installed AITER, with fresh native/Triton caches and all four production fixes | `/tmp/aiter-kernel-expansion/installed-guard-regression.xml` and `wheel-guard-audit.json` |
| Real models | **11 distinct PASS**: eight new cases and three existing shared-runtime regressions | `/tmp/aiter-model-expansion/model-acceptance-audit.json` |
| Complete host suite | **668 tests and 1,034 subtests PASS** on each of clean Python 3.10 and 3.12; no skips, errors or failures | `/tmp/aiter-platform-host-complete-handoff.json`, `/tmp/aiter-platform-host-complete-python310.xml` and `/tmp/aiter-platform-host-complete-python312.xml` |
| Disaggregation review | **43 host tests PASS**, included in the host total above | `/tmp/aiter-disaggregation-independent-final.xml` |
| Native online-tuning regression | **Two GPU checks PASS**, plus deterministic CPU tests of the actual tuner body | `/tmp/aiter-kernel-expansion-qa/hipblaslt-guard-handoff.json` and `hipblaslt-guard-final-metadata.json` |
| Optional hipBLASLt FP8 | **Two strict failures: no valid solutions** after the crash fix; earlier exit-139 and eight direct-probe failures retained | `/tmp/aiter-kernel-expansion-qa/hipb-probe.json`, `hipblaslt-identity.json` and `hipblaslt-guard-final.xml` |
| Official GPQA | **Data prerequisite failed; zero questions executed** | `/tmp/aiter-model-expansion/gpqa-prerequisite/result.xml` |

The final ordinary wheel SHA-256 is `ec31cecf10cd18a0dde64069bcb53b22625d791c2689106458cce537b2819be6`. Its 12,766 archive/RECORD entries were checked, and all four changed native/Triton sources match their checkout bytes exactly. The isolated installation passes 122 checks in 68.92 seconds, including BF16 online tuning and the clean unsupported-FP8 subprocess exit. This wheel carries sources and has no native prebuild or SDK/release qualification. The earlier wheel `75cce0f5c47b7bdd407c5fabf8ac6bceec14ffa02e6b46f65ecfa915fabec0d8` and its 120 passing checks remain retained separately, along with its initial failed attempt that omitted the copied CI helpers.

The real-model audit has SHA-256 `affe15151d1b0330dd9ed0cd51b70f782c8d4fc107c396f0c7c9ec41d715a21e`. Each accepted case retains its input identities, dependency observations and successful AITER work. GPT-OSS passes eight fixed answer checks with each of its explicitly selected native CK and AITER Triton expert backends. The original full-model likelihood comparison failed: dequantized Hugging Face, native CK and the alternative vLLM path do not share the same intermediate arithmetic. Those failed comparisons remain retained and are not claimed as equivalent numerical results. The new native CK kernel checks separately verify stage-one tensors, exact intermediate FP8 bytes and a stage-two reduction bound at GPT-OSS's padded dimensions.

The new speech smoke/extended cases transcribe four and sixteen real LibriSpeech utterances. Their word-error rates are 1/93 and 8/277; the smaller set is a subset of the larger set. The chart case answers 11 of 16 fixed human-authored questions correctly. Qwen checkpoint-FP8 and DeepSeek MoE/MLA pass their predeclared likelihood bounds, and dense Qwen checks both attention choices. These are bounded quality checks, not full public benchmark scores. The three existing regressions cover speculative n-gram decoding, HTTP request accounting and GSM8K.

The optional hipBLASLt path is still represented by two strict numerical tests in the explicit `vllm-hipblaslt` profile and a manual fresh-nightly choice. Default daily/extended profiles do not enable that optional route. The local library reports hipBLASLt 1.2.2, with exact runtime bytes recorded; no fallback or expected-failure marker turns the missing solution into a pass. Official GPQA similarly has required 32-question and 198-question selections, but needs publisher-approved dataset access before it can establish GPT-OSS quality.

The generated catalog declares 21 standard operator groups, 35 daily groups and 46 extended groups, with minimum case counts of 13,100, 13,117 and 13,130. These are selections, not evidence of a new full nightly execution. The local tests use the selected development framework; the revised Docker bootstrap, a new official-nightly installation, other GPU targets and the multi-node cluster must each be accepted in their actual environment before any supported release claim.

Independent collection and XML reconstruction in `/tmp/aiter-model-expansion/catalog-operator-audit-final.json` (SHA-256 `e33f84a7178884a3d045ffdb75808307344644c03a9653b5b58d91d6e37a056f`) checks that every current framework case has exactly one declared owner, matching architecture/GPU/model requirements and an exact minimum count. The six passing operator XML files cover the 13,100 standard node IDs without duplicates or nonpassing results. The optional and gated profiles remain outside that passing set.

## September 7: earlier workflow directories and broader vLLM workloads

Workflow authors now start in [ci/workflows/](ci/workflows/README.md). Its 38 YAML sources have common, host, product, client and release directories; vLLM, SGLang and the other clients have their own homes. The generator preserves the stable flat filenames GitHub requires and checks that every generated copy matches its source. It does not interpret another template language. A contributor edits the source and runs `python -m ci.workflows --write`; CI rejects forgotten regeneration.

| Files | Change and reason |
| --- | --- |
| `ci/workflows/`, `ci/pipelines/workflows.py`, `.github/workflows/` | One authoritative source registry, directory navigation, generated entrypoints and a compatibility command for existing callers. Existing job names and reusable calls retain their identities. |
| `ci/ownership/owners.json`, `.github/CODEOWNERS`, `ci/qualification/catalog.json`, `ci/architecture/policy.json` | Route source ownership to delivery or client maintainers, recognize the new paths and keep workflow tooling independent of GPU imports. Shared CI changes still broaden qualification. |
| `tests/frameworks/vllm/models/language/test_context.py`, `tests/frameworks/vllm/runtime/` | Exercise long prompts with mixed lengths and actual FP16 weights on the pinned Llama and Qwen models. Record scheduled work, parameter types and AITER execution. |
| `tests/frameworks/vllm/entrypoints/test_openai_server.py` | Check batched HTTP response order, token accounting, per-token scores and recovery after an unknown-model request. Compare the batch with individual requests and observe AITER work in the server process. |
| `tests/frameworks/vllm/entrypoints/multimodal/test_chat.py` | Send red and blue images through the OpenAI-compatible chat endpoint. Require the correct color answer and observe both vision and decoder AITER calls for each request. |
| `benchmarks/vllm/models/`, `ci/pipelines/benchmarks.py`, `ci/clients/vllm/observation.py` | Select explicit model workloads, share execution observations and retain raw results for each case. The workflow installs vLLM, checks the candidate AITER import, then measures the selected workload. |
| `README.md`, `ARCHITECTURE.md`, `docs/website/guides.json`, `docs/index.rst` | Make the editable workflow sources discoverable in the checkout and website, with the same maintained guides in both places. |

The first new GPU attempt passed both long-context families, Llama's precision comparison and the new server case. Qwen's cross-format comparison failed: one FP16 prompt-token log probability differed from BF16 by 1.391, while the mean difference was 0.0268 and greedy outputs matched. A separate Transformers FP16 reference agreed with the AITER-enabled FP16 result (maximum error 0.0312, mean 0.00243). The corrected test compares matching number formats against that independent reference; it does not enlarge the original tolerance to erase the failure. The first attempt remains under `/tmp/aiter-vllm-expansion/context/`, and the diagnostic reference is under `/tmp/aiter-vllm-expansion/fp16-reference/`.

The current declarations contain 19 daily workload groups with 86 cases and 26 extended groups with 95 cases, each preceded by the import gate. Within that selection, 18 model groups contain 22 real-checkpoint cases. Those are selection counts, not a claim that this entire expanded profile was rerun. This round exercised six new end-to-end cases on gfx950: two long-context cases, two corrected FP16/reference cases, batched text serving and image chat. They used the current AITER source and vLLM source revision `5690b02c` in the existing development environment; this was not a new wheel build or official-nightly installation. The precision rerun passed both cases in 161.40 seconds. The final image-chat run passed in 49.54 seconds; each image request produced 32 observed vision calls and 72 decoder attention launches. Its initial attempt failed GPU discovery because both visibility masks were set; the failed setup and subsequent passing runs remain separate.

The benchmark catalog has nine scenarios across five profiles. The configured daily 19:45 UTC run selects both smoke cases; Sunday 22:15 UTC selects all eight extended cases. These schedules require the configured executor image and GPU workers. Local development results do not establish that a remote nightly service or release channel is operating.

Independent QA found that benchmark reports needed stronger binding between the selected workload, actual worker observations, subprocess completion and aggregate summaries. Schema 2 now checks those connections and requires the temporary observation hooks to be restored before timing. QA passed 56 focused tests, 34 subtests and 36 independent attempts to admit inconsistent evidence. These checks supplement GPU execution; they cannot replace it.

All eight extended benchmark cases produced accepted measurements, including eager/graph execution, prefill, mixed prompt lengths, long context, FP16 and two-GPU execution. The original parent process nevertheless exited with an error: sorted JSON changed dictionary order, and its checker incorrectly treated an equivalent reconstructed command as different. The fix builds commands in the workload model's explicit field order. Read-only reconstruction then accepted the unchanged eight-case evidence, and a fresh two-GPU graph suite completed with the corrected parent and exit status zero in 111.77 seconds. The original failure remains recorded. Each extended case has three measured samples; these are descriptive measurements, not statistically established speedups or release performance gates.

| Final local check | Result and retained evidence |
| --- | --- |
| Host integrity | 561 tests and 841 subtests pass. `/tmp/aiter-workflow-publication-host.xml` |
| Workflow generation and syntax | All 38 sources match their generated entrypoints; Actionlint passes. The 14 workflow tests also pass independently on Python 3.10 and 3.12. `/tmp/aiter-workflow-schedules-checks.json` |
| Repository boundaries | Architecture checks pass for 1,048 Python files and 8,986 dependency edges; the catalog validates 68 groups and 20 profiles. `/tmp/aiter-workflow-publication-architecture.json` |
| New model scenarios | Original long-context and batched serving evidence is in `context/`; the corrected precision cases are in `precision-2/`; image chat is in `multimodal-server-final/`, under `/tmp/aiter-vllm-expansion/`. `/tmp/aiter-http-vision-final-review.json` independently checks the latter. |
| Benchmark acceptance | The eight-case evidence is in `/tmp/aiter-vllm-expansion/benchmark-extended-2/`; `benchmark-extended-corrected-reconstruction.json` records its later reconstruction. `benchmark-topology-corrected/` and `benchmark-topology-corrected-execution.json` retain the fresh passing controller run. |
| Independent benchmark QA | `/tmp/aiter-benchmark-expansion-final-review.json` records the closed findings and the exact reviewed files. |
| Website rendering | The first expanded site passes 520 browser page/viewport checks. Visual inspection of the subsequent build caught a short table heading wrapping inside a word; its browser run was stopped and retained. The corrected publication build and full browser review retain their outputs in `/tmp/aiter-workflow-publication-site-2/` and `/tmp/aiter-workflow-publication-browser-2/`. |

Earlier artifact reports below remain evidence for their original inputs. This round changes automation, benchmarks and test code; it does not establish new gfx942 results, GPU Docker qualification or release publication.

## Package ownership, consumer coverage and documentation

This round addresses navigation through the code and the missing connections between framework workloads, builds and delivery. Three engineers own runtime/build changes, delivery/benchmarks and framework QA. Each reviews failures before the final artifact run; earlier passing artifacts remain evidence for their original inputs.

| Change | Where to find it | Why it matters |
| --- | --- | --- |
| Operation ownership | `aiter/ops/{attention,moe,gemm,position,quantization}/` and `aiter/testing/` | Padding, expert dispatch, MLA, packing and test helpers have named homes. Five downstream module imports retain identity-preserving aliases. |
| Installed kernel resources | `aiter/kernels/data/` | Source and wheel use the same path. The resource manifest, byte checks and explicit admission still govern execution. |
| Build efficiency | `aiter/codegen/vendor/ck_layernorm/` and `build_backend/` | Bounded source batches reduce repeated compiler work. Explicit module prebuilds enter the normal build plan instead of relying on incidental checkout binaries. |
| Workflow ownership | `.github/scripts/{common,host,product,clients,release}/` and `ci/pipelines/` | Scripts have an owner and checked callers. GitHub requires workflow entry YAML in one directory; prefixes and an index make those entry points navigable. |
| vLLM feature coverage | `tests/frameworks/vllm/` and `ci/clients/vllm/` | Daily serving, evaluation, Qwen3 and scheduling groups connect to installed-nightly execution; the extended profile adds broader numerical and model checks. |
| Real model measurements | `benchmarks/vllm/` | A pinned checkpoint, an explicit token workload and raw repeated measurements establish model execution cost. The measured region has observation hooks removed. |
| Readable guides and examples | `.github/instructions/aiter-ops-triton.instructions.md`, READMEs, `examples/` and `docs/` | Guides explain concrete responsibilities in normal paragraphs. The website renders the maintained files and tests the resulting pages in a browser. |

The final readability pass also splits the dense CI overview into guides owned by `ci/qualification/`, `ci/clients/vllm/` and `ci/release/`. It corrects a stale reference to seven model groups: the complete model-only selection now has 15 groups and 16 cases. The website gives these guides, and the existing wheel-builder guide, their own navigation entries.

The vLLM selection inventory records the inspected source revision and exact upstream Buildkite commands in `ci/clients/vllm/upstream/buildkite-selectors.json`. Qwen3 is selected by several upstream areas in that checkout; it is not a standalone “all” job. AITER's ports intentionally declare smaller, repeatable workloads with named acceptance conditions. Their coverage does not imply that the whole upstream suite ran.

The ordinary n-gram case runs with `VLLM_ROCM_USE_AITER=1`, explicitly selected AITER normalization and attention, and observed successful AITER work in its engine process. It compares all generated greedy tokens and requires actual proposed and accepted drafts. It is an integration test using real weights, rather than a mock-based unit test. Serving, graph and distributed cases likewise inspect the relevant process or execution event.

Development found a useful numerical comparison issue in Qwen3. A BF16 vLLM run differed slightly from Transformers on prompt likelihoods despite identical greedy outputs. An independent AITER-disabled vLLM baseline differed more, so QA matched per-prompt scheduling and documented a bound based on that cross-engine behavior. The initial tighter failures remain available; no production result was changed to fit the oracle. The held-out extended GSM8K slice is separate from the daily slice used during development.

The full LayerNorm comparison reduced a cold build from 504.79 to 196.80 seconds with eight workers: 2.57 times faster, or 61% less elapsed time. The generator preserves all 1,440 explicit instantiations while combining 392 source fragments into 49 bounded batches. Both binaries passed the same 64 supported numerical cases. Peak memory for an individual compiler child rose from 1,196,728 to 1,381,844 KiB; this is not aggregate parallel memory. This is one local comparison for one kernel family. Non-include flags and the set of include directories match, but the existing compiler helper varies system-include ordering between processes. The measurements therefore do not establish a whole-library speedup or identical ordered compiler commands. `/tmp/aiter-layernorm-unity-full/comparison.json` retains the comparison and its limitations.

The numerical review also found a private dynamic-only CK adapter selecting a mode absent from its generator. It now rejects that unsupported mode before launch; the public dynamic quantization interfaces continue to use their existing Triton implementation. A fresh guarded native build passed 66 checks: the 64 supported numerical cases and two rejection checks. Initial attempts using unsupported private shapes and modes remain separate diagnostic evidence.

### Current installed artifact

The frozen source is `/tmp/aiter-ownership-acceptance/build`, inventory `85aed6ad750040e405a2691ab5a57c3ef7e16dc73ea46218af7e5595742dbd60`. The separately copied controls are `control-1`, inventory `8cad69414d6336c7320bed68fdb37bba21afa0aada8cb9bfde9906de930f942f`. They include the stronger mobile-table checker reviewed after the product freeze. Later documentation edits do not change the candidate's production or test inputs.

The source archive `amd_aiter-0.1.0.dev2026090701.tar.gz` has SHA-256 `5b546fb87c5c5d53ee3ebc4dfeeff894d8e1dd63ca5f195004d01768d84b479b`. Building that archive with a fresh native SDK produced `amd_aiter-0.1.0.dev2026090701-cp312-cp312-linux_x86_64.whl`, SHA-256 `dd024e6961be2c3b958a181ed846e3c3e48aeb5034ea6554668a94f811083005`. This ordinary wheel includes the SDK and compiles missing legacy specializations when needed.

| Check | Result and retained evidence |
| --- | --- |
| Clean host environments | Python 3.10 and 3.12 each pass 542 cases and 775 subtests without Torch, Triton or client frameworks installed. `host-python310-1-report.json`, `host-python312-1-report.json` |
| Installed product features | All nine groups pass: 542 host cases, 775 subtests and 47 numerical GPU cases. `product-report-2.json`, report digest `b7b4f4e634d67d6c7e9818693fd2a63a6c17bbc082850a7595a0e5cefd00e1aa` |
| Installed runtime and packaging | 267 cases pass with no skipped or failed cases. Independent review verifies 246 imported module origins and the copied suite. `installed-independent-audit.json` |
| Public compatibility | All 601 exports, 547 signatures and five module-identity aliases pass. The same four documented enum annotation qualifications remain. `public-installed-audit.json` |
| Fresh native SDK | 36 checks pass using two GPUs. The earlier one-GPU attempt failed its explicit cross-device prerequisite and remains recorded. `native-runtime-2.log` |
| Source and archive integrity | All 12,771 wheel RECORD entries, 12,758 source-to-archive-to-wheel mappings, 3,055 managed resource files and four fresh SDK libraries verify. `archive-independent-audit.json` |
| Runnable examples | Eight Python command variants execute numerical checks on gfx950; the C++ and Rust SDK examples also pass. The commands and their expected results are in `examples/README.md`. |

Evidence filenames in this table are relative to `/tmp/aiter-ownership-acceptance`. These selections overlap; their counts should not be added into a unique repository-wide total. The first product attempt stopped before collection because the fresh environment lacked pytest. The second installs the declared host test requirements and passes; the failed attempt remains available. The public API's first independent probe hid all GPUs and failed architecture discovery; its subsequent visible-GPU probe verifies the installed interface.

The installed wheel excludes repository CI, tests, caches and inherited build receipts. The source archive also carries SCM-selected developer material, including some `.claude`, GitHub and documentation files. It is a buildable source distribution, not a minimal SDK-only source bundle. Removing those incidental archive inputs remains a release packaging review item; their presence is not represented as an installed-package boundary failure.

### Selecting a prebuilt family

`PREBUILD_MODULES=module_gemm_a8w8_blockscale_cktile` selects that family through the build controller. It also includes the required core module. It cannot be combined with a numbered prebuild profile. The resulting wheel carries `aiter/jit/prebuild.json`, which records the requested and resolved jobs and the exact compiled bytes. The vLLM disaggregation workflow checks this selection before consuming the wheel.

A separate source-archive build produced the named-only wheel under `named-wheels-2/`, SHA-256 `e086b88dff0908b2d119b9b938781a277353f028347a6a8ef746b16d1961be22`. Its receipt matches exactly two loaded native modules. Four installed FP16/BF16 GEMM cases pass against independently dequantized FP32 matrix multiplication with runtime compilation forbidden. Independent review also matches all 66 compiled instantiations to the regenerated source and verifies every installed payload byte. `named-gpu.json`, `named-acceptance-execution.json` and `named-independent-audit.json` retain the evidence. This is a different artifact from the ordinary SDK wheel used for framework qualification.

The selected build also exposed an existing scheduling limit: the prebuild helper divides the compiler budget by a five-worker ceiling even for this two-module plan. The first attempt was stopped and retained as incomplete. A fresh attempt with an explicit larger budget completed in 437.10 seconds. This is a build success, not a speedup comparison. Making the worker budget depend on the actual selected plan remains a compilation-controller improvement; `named-worker-budget-limitation.json` records the affected code and both attempts.

### Current website review

The warning-fatal website build produces 198 HTML pages. Chromium checks 430 page/viewport combinations, local links, navigation, search, diagrams, dark mode and enlarged text. Narrow screens show wide tables as labeled rows; simple desktop tables retain their ordinary layout. The browser checker deliberately removes labels, hides headers and conceals body content, and requires every corrupted version to fail before restoring the page.

The accepted renderer run is `/tmp/aiter-ownership-docs-accepted-browser`; an independent 36-view review is `/tmp/aiter-responsive-table-independent-control1/report.json`. A second complete 430-view run, `/tmp/aiter-ownership-docs-publication-review-browser`, includes the revised installation instructions and initial acceptance record. A clean export of the staged files also builds all 198 pages without initialized submodules. Serving that export under the `/aiter/` project prefix passes 51 focused browser views and verifies all 197 source downloads; `/tmp/aiter-pages-prefix-review` and `/tmp/aiter-pages-prefix-downloads.json` retain those checks.

An earlier independent table report was accidentally overwritten during a root rerun. Its original findings were reproduced separately using the frozen pre-fix checker, rather than presenting the replacement as the original evidence. The final publication build will include the later acceptance-record edits and retain its own complete browser review.

The deployment review found a separate defect that local rendering alone could not expose. Source pages and downloads used literal `.github` directories; GitHub's pinned Pages archive action excludes that name. An actual archive of the old site therefore lost workflow and script targets despite its passing browser report. `/tmp/aiter-pages-archive-before-fix/report.json` retains the reproduced failure.

The website extension now gives hidden source directories visible public paths and escapes literal tildes to prevent name collisions. Headings keep the original repository path, and downloads keep the original bytes. Link validation rejects hidden public paths before upload, including a hidden symlink that points to a visible file. Nine website unit tests pass, including an actual Pages-style archive with workflow names, spaces, `#` and `?`. Independent review found and fixed the latter characters' relative-URL handling; its retained 256-path adversarial check verifies unique names, unchanged downloads and all 768 local links after archiving. `/tmp/aiter-pages-public-path-review-2.json` records that review. These changes affect documentation generation and checking; the library and framework runner remain frozen.

The repaired full site also passes the actual Pages archive step: all 433 files and 51,346 local links are preserved across 202 pages. `/tmp/aiter-pages-archive-fixed/report.json` records the complete before/after byte comparison. Browser review runs against that extracted archive, rather than relying only on the original build directory.

### Fresh installed vLLM acceptance

The complete extended pipeline passes all 23 workload groups and its separate import gate: 89 workload cases plus one import case, with no failures or skips. Eight operator groups contain 73 cases; 15 model groups contain 16 cases. Independent reconstruction also proves that the 19 daily groups use exactly the same definitions inside this extended selection. This is one actual extended run, not a second daily execution.

The environment freshly installed official vLLM `0.28.1rc1.dev451+g1970f3ed4.rocm723`, commit `1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27`, and then candidate AITER wheel `0701`. The official vLLM wheel has SHA-256 `d0c1bdfd51c5f34ce19a3e78eeb30affc6add38dd96a3d19b3dda4df6a1c2a65`; the installer retains all 219 distribution identities. No preinstalled framework environment supplied the qualification results.

| Model check | Observed result |
| --- | --- |
| Daily GSM8K | 22 of 32 answers correct: 68.75% |
| Separate extended GSM8K slice | 93 of 128 answers correct: 72.66%; independent review matches every question, demonstration and answer to the pinned inputs |
| Qwen3 reference | Both greedy sequences match Transformers; all 116 prompt likelihoods satisfy the declared bounds. Observed maximum error 0.338979 and mean 0.033109, against limits 0.75 and 0.06 |
| FP8 likelihood comparison | All 82 prompt likelihoods pass, with maximum error 0.509858 and mean 0.083526, against limits 1.0 and 0.15 |
| Standard n-gram speculation | 48 proposed and accepted tokens; generated target tokens match |
| Mixed speculative requests | 134 of 168 proposed tokens accepted, 34 rejected; all three 64-token outputs match target execution |
| Tensor parallelism | Two distinct engine ranks, processes and GPU identities; both 64-token outputs match the reference |
| Serving, vision and scheduling | Streaming and bad-request recovery, image grounding and ordering, batching, prefix reuse, graph replay, chunked prefill and online FP8 all pass their recorded assertions |

The report is `/tmp/aiter-ownership-acceptance/nightly-1/report.json`, file SHA-256 `0a58d6fabe9d7d19725f8751315a2f5f14265aaa8561bde7c628dfe00586c1f7`. Its canonical pipeline digest is `36b3d258bfc79f44266e45f6eebf23f945454bdc54b85fa02d256e49b27898c9`; the workload report digest is `833cb3a3cc99ef12851bfc0aee9a757ac17bc4be3c9a0c6e44608efcbcc6d2d8`. `nightly-delivery-handoff.json` and `nightly-daily-subset-proof.json` retain reconstruction and subset checks. The separate operator audit verifies all 73 cases, exact imported bytes, fresh native builds and Triton objects in `nightly-1-operators-independent-audit.json`. The final independent model audit verifies all 15 groups and 16 cases in `nightly-model-final-audit/report.json`, SHA-256 `f17d1c766162f825cbde495169edc60749d7488b28718511a9e8a6d4e266f5e9`.

This run uses the verified private glibc 2.39 userspace on local gfx950 hardware. vLLM's released `amd-aiter==0.1.21.post1` requirement is the sole explicit candidate replacement; its original `pip check` exit 1 remains recorded, and final dependency and payload checks pass under that declared policy. Package metadata was not rewritten. This establishes the bounded local integration results, including real AITER execution. It does not qualify a Docker image, gfx942, general model quality or a supported release channel.

### Real-model measurement

The benchmark executes pinned Llama 3.2 1B weights on one gfx950 GPU, using BF16 and eager execution. Each synchronous batch contains two requests, each with 128 synthetic input tokens and 64 generated tokens. Two warmups precede nine recorded measurements. Median complete-batch latency is 329.37 ms; median output throughput is 388.62 tokens/s. The relative interquartile range is 0.42%. These are observations for this workload and machine, not a previous-release comparison or a time-to-first-token measurement.

The local measurement uses the same already qualified private nightly environment, with its exact candidate and all 219 installed distribution identities checked before and afterward. It has separate model and compilation caches. Observation hooks establish AITER execution before timing and are removed for measurement. This run does not claim a second installation or OCI execution; the standalone CI benchmark controller composes its own fresh installation and import gate.

Evidence is `/tmp/aiter-ownership-acceptance/benchmark-1/`, with measurement-report SHA-256 `659ceca9817c7202bb982f59fc007b1cd433681eb7a27a35490ac79196abd984`. `benchmark-reconstruction.json` independently reconstructs the raw samples and retains 1,043 imported module identities.

The separate QA review also passes: `/tmp/aiter-ownership-acceptance/benchmark-independent-audit.json`, SHA-256 `d408576a1ce0fdd36aa77a59c1b4648bd21aeb95b0200172912a38014e16eb16`. It reconstructs all nine samples and their statistics, checks the model and installed package bytes, and confirms that observation hooks were disabled during timing.

### Publication record

This source checkpoint records the reviewed local implementation. The [implementation branch](https://github.com/AndreasKaratzas/aiter/tree/akaratza_aiter_implementation) retains the code; the [documentation workflow](https://github.com/AndreasKaratzas/aiter/actions/workflows/host-docs.yml) records each subsequent build and deployment with its exact commit, checked HTML and browser evidence. The website is [AndreasKaratzas.github.io/aiter](https://AndreasKaratzas.github.io/aiter/). A remote publication result belongs to that workflow run and must be checked after deployment.

Historical results further down this file remain evidence for their original artifacts. Publishing this development branch does not promote a supported wheel or container release.

## September 6: feature areas and fresh consumer nightlies

This round expands the test system after the two-model resource review. The earlier frozen wheel remains evidence for its own inputs. Current-source development checks and the final installed-candidate attempt are recorded separately below.

| Responsibility | Files and change |
| --- | --- |
| vLLM feature areas | `tests/frameworks/vllm/`: imports; normalization, quantization, position, attention and expert operator bridges; separate batching, speculation, prefix-cache, graphs, tensor-parallel and online-FP8 model areas |
| Model execution | `tests/frameworks/vllm/runtime/`: shared protocol, owned engine processes and per-worker observation; feature modules own their assertions |
| Product feature selection | `ci/qualification/catalog.json`: eight bounded numerical groups for position, routing, recurrent state, paged addressing, dense GEMM and sampling; connected to fast, nightly, extended and dedicated feature profiles |
| Selection visibility | `ci/qualification/coverage.py`: inspect exact file/node selectors, unused membership and unselected files without importing tests or claiming execution |
| Fresh consumer installation | `ci/clients/vllm/nightly.py`, `ci/pipelines/nightly.py`, `.github/workflows/client-vllm-nightly.yaml`: resolve the official ROCm wheel, install dependencies into a private environment, install the candidate last, gate imports and run the feature profile |
| Installed identity | `ci/qualification/nightly.py`, `substitution.py`, `environments.py`, `probe.py`: retain dependency observations and package origins; bind the explicit consumer-pin replacement to the selected artifact |
| Execution lifecycle | `ci/pipelines/process.py`: shared subprocess ownership, timeout and cleanup for local nightly and container orchestration |

The GPU development runs exposed real defects. vLLM forwards an absent attention threshold as `None`; the AITER public entry now normalizes it to zero before its integer custom-op boundary. A positive threshold retains its existing meaning: skip short sequences and leave their output untouched. It is not a harmless optimization hint. Tests exercise both meanings and invalid inputs.

The CK unquantized MoE path also accepted raw FP16/BF16 weights despite its preshuffled kernel layout, producing incorrect results. The affected path now rejects unprepared weights before sorting or native loading. Tests exercise vLLM's actual preparation and numerical result, plus one- and two-weight invalid layouts. [The migration note](docs/migrations/ck-moe-layout.md) explains the compatibility tightening; other raw-capable providers are not reclassified.

Independent QA found that the retained local-expert test logged numerical mismatches without asserting the return. It now uses raising tensor assertions. Two additional fault-injection cases corrupt actual kernel results and prove that the acceptance checks reject them. The static coverage inventory was also corrected to include both `test_*.py` and `*_test.py` filenames. Selected files still do not imply complete shape or numerical coverage.

A later independent review found that the paged-attention addressing test checked only the output mean. Equal and opposite errors could therefore pass. It now checks every element against the constant-value reference with the same tolerances. A new case executes the real kernel, corrupts two elements while preserving the mean, and verifies rejection. All 13 paging cases pass with fresh compilation caches. This adds one case to the bounded product selection, bringing it to 47 GPU cases. `/tmp/aiter-paging-elementwise-development/audit.json` retains the focused development check; complete installed acceptance is recorded below.

Development operator execution passed 73 numerical/interface cases plus one import gate. The operator suite contains 66 numerical checks through vLLM adapters, six layout-rejection checks and one direct AITER attention regression. The initial bounded product run passed 44 cases; the repaired routing group added two corruption checks, and the later paging review added one. These counts overlap across profiles and must not be added into a repository-wide unique total.

All seven model scenarios passed in the mutable source environment: image grounding; GPU n-gram verification; ordinary batching; prefix reuse; graph replay; two-GPU tensor parallelism; online per-channel FP8. Worker observation records successful AITER calls, launched kernels, real graph replays and participating ranks, not merely enabled options. The FP8 case observed 64 FP8 projection weights and 2,048 AITER preshuffled GEMM calls on a controlled repeated-sequence continuation. This does not establish general FP8 quality. The optional HIPBMM route failed to obtain a valid heuristic solution for one projection and remains a retained rejected trial; the passing case uses the default tuned route.

Development records are `/tmp/aiter-vllm-operators-final-dev.{log,xml}`, `/tmp/aiter-feature-development/`, `/tmp/aiter-moe-routing-review.{log,xml}` and `/tmp/aiter-vllm-features-dev/handoff.json`. Early cache, harness and numerical-oracle failures remain separate. Mutable-source runs cannot qualify later package bytes, even when their numerical assertions passed.

The current official nightly requires manylinux glibc 2.39, newer than this container's glibc 2.35. A private userspace built from signed Ubuntu package metadata supplies the actual 2.39 loader and libraries without changing the host. Its platform receipt is `/tmp/aiter-nightly-platform/smoke.json`. No wheel was retagged. This is a local development platform, not an approved executor image. An initial fresh installation retained 218 consumer/dependency artifacts and correctly stopped at vLLM's exact released-AITER pin. Candidate testing now handles that one intentional replacement explicitly while retaining the original failed dependency-check output.

### Frozen product acceptance

The candidate is `/tmp/aiter-feature-acceptance/build`, with maintained-input inventory `20dda541a7db24ee376eb594b23d376c606b90661270f6c702e719c49f062684`. The new wheel is `amd_aiter-0.1.0.dev2026090606-cp312-cp312-linux_x86_64.whl`, SHA-256 `54a64c24d43bd212a063dd79ca33929bca5e4bdb7b23609ed54d1decc81e3a7e`. Its source archive has SHA-256 `3ed11e8c2646afb2b887f2995eaf147930063f3f0d3959143e096ea4fe219cde`. The wheel was built from that archive with a fresh native SDK; it can still compile missing legacy specializations.

| Check | Observed result |
| --- | --- |
| Complete installed product feature profile | PASS: 498 host cases, 701 subtests and 47 GPU cases across nine fresh group attempts. `product-report-6.json` and `product-handoff-6.json` |
| Clean host environments | PASS on Python 3.10 and 3.12: 498 cases and 701 subtests each, without Torch, Triton, vLLM or SGLang installed. `host-python310-6-report.json`, `host-python312-6-report.json`; both use `control-6` after the final paging and navigation changes |
| Installed package selection | 243 cases pass in the initial 244-case run. The remaining generator fixture needed the newly required `preshuffle_mode=True`; its corrected installed case passes separately. The original failed run and an intermediate wrong-keyword correction remain retained. `installed.log`, `corrected-codegen-2.xml` |
| Fresh native SDK | 36 checks pass with two GPUs, plus one executed C API test. An initial one-GPU invocation failed its cross-device prerequisite and is retained separately. `native-sdk-two-devices.log` and `native-c-api.log` |
| Public interface | All 601 public exports and 547 signatures resolve; the same four documented annotation qualifications differ. Parameter names and defaults remain unchanged. `public-compatibility-audit.json` |
| Independent archive verification | 12,754 wheel RECORD hashes, 842 source Python modules, 655 data files, 823 native sources, 3,055 managed-resource files and all 24 dependency inputs match their frozen sources. `independent-archive-audit.json` |

Paths in the table are relative to `/tmp/aiter-feature-acceptance/`. The current complete product report digest is `090943b151c72524a4eb3e6d76013f7c07690bfd94a0acb4aedd57b6f6c48e38`. Its reviewed controls are `control-6`, inventory `cadd91b57331f51403bb99ef4daf2d430805cbadfbdab66d486c10da4f90c4a7`. All nine groups ran again after the paging-oracle fix: 545 cases in total, with 701 additional host subtests, 72 distinct initially empty caches and checked package/resource identities before and after execution. Independent QA reconstructed the result from retained logs and hashes. The run took 225 seconds on two gfx950 GPUs.

Earlier attempts remain separate. The first passed all 46 selected GPU cases but correctly failed a host subtest: the source map listed an empty retired `csrc/gemm_a6w6/` directory. After removing that entry, the complete `control-2` attempt passed, report digest `a2a9e201b66a7a96fa5a55a26813e96e1c3b4aeac9b8fb27c8c608097a6cbd03`. Later review strengthened paging and expanded host checks, producing the new complete run above. No previous pass was copied into the current report.

The artifact receipt records observed inputs and environment; it is not a complete transitive build attestation. Its first CLI invocation mistakenly supplied a Python script where a JSON recipe was expected. The successful build outputs stayed unchanged; the failed invocation and corrected observation-only receipt remain separate in `build.log` and `receipt.log`.

### Nightly installation review

The first attempt with wheel 06 installed the consumer and candidate successfully, then rejected duplicate metadata reported after public imports. Setuptools had added its bundled dependency directory to Python's search path. Those bundled files belong to setuptools; they are not separate top-level pip installations. The observer now inventories explicit private installation roots and rejects actual duplicate top-level distributions. Dependency-policy checks use that same inventory. Independent tests also prove that a bundled copy cannot satisfy a missing declared installation.

The corrected observer passes against the retained installation in a separately labeled diagnostic attempt, including actual public imports, exact candidate origin, dependency policy and stable inventories across fresh subprocesses. This diagnostic does not claim a fresh installation or model execution. The next full attempt, `nightly-2`, used a fresh environment and frozen `control-3`, inventory `e8598771a77e7b1c87a67c31a9566c8d42e8e4a12e41774666b370aa1aecf411`.

The complete copied clean-host gate passes against `control-3`: 494 cases and 701 subtests. An independent scope audit confirms that its inventory changes do not alter the earlier accepted product profile or enter its numerical execution path; the product report keeps its original control identity. `control3-product-scope-audit.json` records that comparison. A separately sealed, explicitly reused-environment import diagnostic also passes its actual installed-candidate runner and report checker.

Process review also proved that successful or failing parent processes could leave background children behind. The shared executor now cleans up its process group on every exit, and both live execution and reconstructed reports reject cleanup failures. Real child-process tests cover success, nonzero exit and timeouts.

`nightly-2` passed installation, both import gates and model provisioning, then failed native compilation: the private interpreter lacked development headers at the path used by Torch's extension builder. Its `INCLUDEPY` setting pointed elsewhere and was misleading. The run was stopped; completed failures and unfinished groups remain a failed report. No numerical result from that attempt qualifies a model.

The corrected platform, `/tmp/aiter-nightly-platform-2`, includes the matching Python headers. A real C++ extension compile and import proved that the private include path supplies them. The controller now checks `sysconfig.get_path("include")` and `get_path("platinclude")`, compiles a header probe before installing the consumer, and retains the compiler and header identities. The original platform fails this prerequisite; the corrected platform passes. Seven actual attention cases also pass in a separate diagnostic using the corrected platform and fresh native caches, but reusing the previous dependency installation. That diagnostic is not fresh-nightly acceptance.

The new complete attempt, `nightly-3`, uses another fresh environment, corrected platform and `control-4`, inventory `49dd76aebd10bd2e5c67abd2e1ce8c85259afe755438ee29d7bc961b2ebbb5e5`. The independently checked product report retains its original controls: `control4-product-scope-audit.json` verifies that the nightly-specific changes do not alter its selection or numerical execution path.

Python 3.10 review caught newer-interpreter conveniences in the new controls and a race in the process-test fixture: a correctly terminated child could disappear between two `/proc` reads. The fixture now accepts an already-reaped child and still rejects a live child. A deliberately disabled cleanup was rejected on both Python 3.10 and 3.12. Failed attempts remain retained. The next host checks passed against `control-5`, inventory `f5e66aacd275814537cc5aa2fb2aa611c63cc23140bea077dd7ad40184e650f9`. Its only change from `control-4` is that CPU fixture.

Final host and product checks use `control-6`, inventory `cadd91b57331f51403bb99ef4daf2d430805cbadfbdab66d486c10da4f90c4a7`, which also includes the stronger paging test and website navigation. Both clean host interpreters pass all 498 tests and 701 subtests again. `control6-nightly-scope-audit.json` proves that the nightly pipeline and all its GPU-test bytes remain identical to `control-4`; the rolling-nightly report retains that actual control identity.

### Accepted current-nightly run

`nightly-3/report.json` is PASS. This attempt installed the official ROCm vLLM nightly `0.28.1rc1.dev451+g1970f3ed4.rocm723`, revision `1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27`, in a fresh private environment and installed candidate wheel 06 last. The upstream wheel SHA-256 is `d0c1bdfd51c5f34ce19a3e78eeb30affc6add38dd96a3d19b3dda4df6a1c2a65`. Torch is `2.12.0+git6bbd260`; the retained inventory contains 219 primary installed distributions.

The separate import gate passed one case. All 15 workload groups then passed 80 cases: 73 operator cases and seven real model scenarios. There were no failed or skipped cases. The import report digest is `800d163c8a398830f17776210b8ecb49bdcd851ddc9ea473e4479c419cffc865`; the workload report digest is `a9a1d4a01ec07ddf01e21f2d70e177286fa4e918d5bfacfeeacd7101d704e196`. Installation, native-header preflight, final dependency checks, package/resource identities and complete report reconstruction belong to this same attempt.

| Model scenario | Observed result on the installed nightly |
| --- | --- |
| Image grounding | Real Qwen2.5-VL weights answer “red” and “blue” for distinct images; 32 AITER vision-attention calls are observed |
| Batching | Both batched 64-token sequences equal their individual generations |
| Graph execution | Single and paired requests each replay a captured AITER attention graph 63 times and match eager generation |
| Online FP8 | 64 FP8 projection tensors; 32 AITER and 32 Torch linear implementations; 2,048 AITER GEMM calls; controlled continuation matches BF16 |
| Prefix cache | Cached-token counts are 0, 64 and 0 for cold, warm and reset requests; all complete outputs agree |
| GPU n-gram speculation | 16 draft rounds, 48 proposed and 48 accepted tokens; the complete 64-token output agrees with ordinary greedy decoding |
| Tensor parallelism | Two worker processes on distinct GPUs execute AITER normalization and attention; generation agrees with the one-GPU baseline |

The operator audit, `nightly-3-operators-independent-audit.json`, checks all exact case identities, 12,750 installed wheel payload files, pre/post observations and fresh native/Triton artifacts. Its scope is the 73 operator cases and import gate. Independent model review also passes: `nightly-3/models-independent-audit.json` reconstructs all seven feature cases and 11 actual engine executions, rehashes 1,220 imported Python/native files and checks all seven private checkpoint views against their pinned manifests. Requests, generated outputs, worker identities and execution observations remain under `nightly-3/workloads-run/`. The complete pipeline independently reconstructs in `nightly-3-reconstruction.json`.

This is successful local gfx950 integration with the resolved rolling nightly. It uses the verified private glibc 2.39 userspace and matching Python headers, rather than an executed OCI image. vLLM's released `amd-aiter==0.1.21.post1` pin was deliberately replaced by the exact candidate; raw `pip check` exit 1 is retained and independently explained by that sole exception. No package metadata was altered. This result does not activate a supported release channel, qualify gfx942 or claim general model quality, every projection backend, HTTP serving, draft-model speculation or multi-node execution.

### Website review for the feature round

The new operator guide has its own navigation entry and linked source pages. Tables use short area labels; each small model/operator flow fits a phone screen without internal scrolling. The website build treats warnings and broken local links as errors. The retained browser review covers desktop and mobile pages, additional dark and enlarged-text views, navigation, local search and diagram controls. Separate screenshots inspect the complete small flows, including content below the first viewport.

The accepted site and its rendered evidence are `/tmp/aiter-feature-docs-accepted`, `/tmp/aiter-feature-docs-accepted-browser` and `/tmp/aiter-feature-docs-accepted-mobile`. The local preview serves the same files from `/tmp/aiter-docs-preview`. These are generated local artifacts; the canonical guides remain beside their owning code.

## September 6: kernel resources and real model coverage

This round follows the architecture and website review below. The earlier artifact remains evidence for its frozen source, not qualification of the subsequent kernel-resource, dependency or model-test changes. The starting inventory is `/tmp/aiter-artifacts-e2e-before.json`.

The review counted 206 operator test files but only four framework test files. None of those four loaded a complete model. Operator and bridge tests had useful coverage, but did not answer whether image conditioning or speculative verification worked through a real vLLM engine.

| Responsibility | Current owner | Change from the previous layout |
| --- | --- | --- |
| Precompiled resources | `kernels/`, `aiter/kernels/` | Separate immutable imported bytes from catalog validation and explicit cache admission; retain target, byte hashes, selection references and known availability gaps |
| BLAS tuning bridges | `csrc/blas/`, `aiter/tuning/` | Remove the standalone `gradlib` packaging island; retain the public bridge and its actual native dependencies |
| Shared test infrastructure | `tests/common/` | Centralize hardware markers, model verification, process cleanup and package-origin handling |
| Real vLLM model cases | `tests/frameworks/vllm/e2e/` | Exercise image-grounded generation and speculative output equivalence with pinned weights and observed AITER dispatch |
| Client qualification declarations | `ci/clients/registry.json`, each client's `groups.json` and `profiles.json` | Resolve selected reviewed controls; keep client extension separate from required release policy |
| Dependency inputs | `requirements/` | Give runtime, build, host/GPU test, documentation and framework environments one maintainable home |
| Profiler analysis | `benchmarks/traces/` | Preserve useful trace filtering while removing invented groups of eight unrelated events and root log-output clutter |
| Structural checks | `ci/architecture/` | Reject retired paths, scattered requirement files, library-to-test imports and data-layer imports of GPU/admission code |

The kernel move preserves all 3,053 original resource files byte for byte; `/tmp/aiter-kernel-resource-move-audit.json` records the independent comparison. The inventory contains 2,940 `.co` files, two additional historical ELF variants and 110 CSV tables. Two gfx942 MLA-prefill table selections refer to absent binaries. Those remain visible as unavailable selections, rather than being silently qualified. Twenty-eight other apparent missing paths are declared MI300/MI308 alternatives. Imported provenance states what is unknown: the file inventory does not reconstruct original builds or prove every operator's argument ABI.

The first real speculative probe produced valid output and accepted drafts but did not execute AITER RMSNorm. The framework's default IR priority selected another implementation even with its AITER flag enabled. The tests now request the public AITER IR priorities and unified-attention backend explicitly, then assert actual successful AITER calls and launches. This finding is retained; an environment flag is not dispatch evidence.

Initial real model runs pass on gfx950: the Llama baseline and GPU n-gram engines produce the same 64 greedy tokens with 48 drafts proposed and accepted; Qwen2.5-VL changes its answer from Red to Blue when only image pixels change. These runs establish bounded BF16, eager, single-GPU engine behaviour. They do not establish HTTP serving, tensor parallelism, every speculative method, speed improvements or gfx942 execution. Their retained records are `/tmp/aiter-vllm-e2e-spec.xml`, `/tmp/aiter-vllm-e2e-mm.xml` and `/tmp/aiter-vllm-e2e-accepted-evidence/`. Connected final qualification is recorded after implementation freeze.

Review also challenged the input and process boundaries. A hash-checked Hugging Face snapshot could still contain undeclared files that influence model loading. Tests now construct an owned view containing only declared files and verify it again after execution. Marker arguments are validated so a misspelling cannot broaden eligibility. Required runs reject selected skips. The complete host gate collects both unittest classes and pytest functions; importing a pytest test file through unittest discovery alone does not execute those functions. Independent inventory comparison also found that pytest's default directory exclusions skipped the entire `tests/unit/build` suite during parent-directory discovery. The suite now declares its exclusions explicitly, and a subprocess regression checks that parent discovery includes build tests.

The first frozen round remains rejected under `/tmp/aiter-kernel-acceptance/`. The copied host profile found build requirement files hidden by the root `build` ignore pattern and a tuning-driver registry hidden by a local JSON-output pattern. Equality with a Git-visible file list had not established completeness against the actual maintained inputs. The root ignore is now limited to `/build/`; the registry is explicitly included; architecture and freeze checks cover Python, JSON and requirement inputs. That first wheel is diagnostic even where its numerical checks pass. Its model groups failed prerequisite setup because the runner's private XDG cache could not see the provisioned Hugging Face cache. The next complete run declares that model cache explicitly. A nested marker regression also exposed a difference between the two installed pytest versions; the explicit `strict_markers` configuration passes both. The failed inputs and reports were not edited.

### Corrected local acceptance

The corrected candidate and reviewed controls are separate directories under `/tmp/aiter-kernel-acceptance-2/`. `freeze.json` binds their maintained inputs and pinned vendor files with inventory SHA-256 `c2df19690ce08ea91aa1c8089e17435e4f42469e9574a6e8d500d8b98146b447`. The branch, HEAD and CK pin remain the same; the copies and failed earlier reports have not been edited.

| Check | Result and retained record |
| --- | --- |
| Complete copied host profile | PASS, 434 tests and 559 subtests in a clean environment containing pytest 9.0.2 and tabulate 0.10.0, with no Torch, Triton, vLLM or SGLang. `host-report.json` |
| Installed runtime and packaging selection | PASS, 244 cases; all 227 observed AITER module origins resolve to the selected installation or its owned compiled artifacts. `installed.xml`, `installed-origins.json` |
| Native SDK and external C consumer | PASS, 36 native API checks and an executed C client. `native-sdk.log`, `native-c-api.log` and `handoff.json` |
| Public entry points and documented example | All 601 exports and 547 callable signatures preserved; the actual README HIP RMSNorm example matches the reference from the installed wheel. `installed-public.json`, `installed-readme.json` |
| Installed BLAS tuning command | One real BF16-to-FP32 shape, M=8, N=32, K=128, completes with zero reported mismatch ratio. `tuning/`; this validates the command, not a performance claim |
| Real vLLM image-conditioned answers | PASS, pinned Qwen2.5-VL produces Red and Blue from the corresponding pixels; generation observes 32 AITER vision-attention calls and 72 unified-attention launches |
| Real vLLM speculative generation | PASS, pinned Llama baseline and GPU n-gram runs produce identical 64-token output; 16 draft rounds propose and accept 48 tokens; both runs observe AITER normalization and attention |
| Complete model profile | PASS, both required groups in one sealed plan with fresh per-attempt caches, owned verified model views and distinct candidate/control roots. `e2e-report.json`, `e2e-run/` |
| Independent wheel/source-archive audit | PASS, 12,754 RECORD hashes, 842 source Python modules plus the exact generated version module, 655 package data files, 823 native files, 3,055 kernel-resource files including the manifest, four SDK entries and all 24 dependency inputs. `independent-archive-audit.json` |

All paths in that table are relative to `/tmp/aiter-kernel-acceptance-2/`. The host report digest is `a09d4e1c06f843958a8a21045bef57e76dc988af535e55e221e9a4aaa83a67b8`; the model report digest is `b05616e20fc628a01f99571189fd727ec40a03fd4872af307e744349e58cda45`. Counts overlap across selections and must not be added into a unique-test total.

The wheel is `amd_aiter-0.1.0.dev2026090605-cp312-cp312-linux_x86_64.whl`, SHA-256 `04695bf4fb2bcaf07bd9408ffa62210e96d41ef18065df012f4ae3a5587cc86c`. Its source archive has SHA-256 `5d35a4e50df1b36a1267759f6c7e5bdf7d05770408d198a941e2f77f58f75bab`. This is an ordinary wheel with the native SDK. It can compile missing legacy specializations; the Qwen run actually compiled its dynamic vision-attention specialization in a new cache.

Independent QA rehashed both private model views and 1,113 unique imported AITER/vLLM files, checked all three engine requests/results, reconstructed the token and dispatch assertions and confirmed engine exit. `e2e-independent-audit.json` retains that review. `qa-installed-audit.json` separately checks the installed JUnit results, all 554 copied suite files and observed module bytes against the wheel or owned cache.

The real-model environment is explicitly a development environment: observed vLLM source revision `5690b02c03832a4ac3af231d3ecebe649c188095`, Torch `2.12.0+git6bbd260` and ROCm `7.2.53211`. Its observed setuptools 84.0.0 exceeds two declared vLLM dependency bounds; those observations remain in the environment evidence. There is no immutable executor image in this container. The passing engine cases therefore establish the recorded local behaviour, not an approved release environment or GPU container qualification. Their BF16, eager, single-GPU scope remains explicit.

Final documentation review corrected host prerequisites, the location of client groups, container dependency installation and the wheel's `aiter_meta` resource payload. Those explanatory changes and this acceptance record are kept separate from the frozen product inputs. A subsequent CI-only correction clears inherited `PYTHONOPTIMIZE`, `PYTEST_ADDOPTS` and `PYTEST_PLUGINS`, observes the selected interpreter before running a group, and rejects optimized execution. The original model reports keep their original controls; their runs had no such overrides. This control change passed a separate copied host qualification against the unchanged candidate: 438 tests and 565 subtests. `/tmp/aiter-interpreter-acceptance/freeze.json` identifies the new controls, and its `host-report.json` has digest `e15a426b3addd3bd230e6940d80a854ae85346b31341752f71d947401b59c546`. The actual selected interpreter records optimization level zero and no inherited policy overrides. The final input audit records the exact differences.

## September 6: architecture and website review

This round addresses weaknesses that the earlier numerical and packaging qualification did not settle. A browser audit found that the website omitted the canonical Markdown guides, clipped table columns on mobile and reduced class diagrams to unreadable text. A build that exits successfully does not establish that its pages are complete or usable.

The architecture review found duplicated provider defaults, parallel operation-dispatch switches, duplicated native-library selection, a large JIT module containing several lifecycles and repeated container execution in workflow shell. The chosen pattern is ports and adapters, with explicit composition for each application. The runtime, compiler/build and delivery applications keep separate controllers and ownership; their adapters implement the interfaces those controllers need.

| Work | Main files | Why it changes the design |
| --- | --- | --- |
| Dependency and placement rules | `ci/architecture/`, `tests/unit/architecture/` | A reviewable policy checks actual source imports and layout; tests challenge forbidden edges and narrow exceptions |
| Runtime composition and operation adapters | `aiter/runtime/composition.py`, `configuration.py`, `aiter/backends/dispatch.py`, `backends/triton/` | Defaults and operation registrations have one owner; support and preparation share the same table |
| Shared native lifecycle | `aiter/backends/native/` | Override, bundle and source selection use one artifact resolver and explicit compiler adapters |
| JIT application | `aiter/jit/service.py`, `composition.py`, `compiler.py`, `modules.py`, `bindings.py`, `dispatch.py`, `resolver.py` | The legacy facade forwards to focused services instead of combining configuration, compilation, loading and ABI conversion |
| Build planning | `build_backend/plan.py`, `controller.py` | Immutable selection is separate from executing compiler and staging phases |
| Container and pipeline organization | `docker/common/`, `docker/pytorch/`, `docker/vllm/`, `docker/sglang/`, `ci/pipelines/` | Consumer recipes share one installer and source/wheel/image qualification shares one executor |
| Documentation website | `docs/website/`, `docs/conf.py`, `docs/_static/` | Canonical guides become navigable pages with local diagram assets, responsive layout and actual browser validation |

Independent reviews produced concrete corrections:

- A native module rebuilt at the same path could keep returning its old Python or ctypes code. Selected bytes now have a digest-derived loader origin; successful rebuilds invalidate the selected repository entry, while old callers retain their handles.
- A build with CK disabled ignored the selected prebuild profile. Selection now intersects declared profile membership and dependency requirements. Aggregate builds also retain each source's HIP flags.
- Container timeouts and interrupts could leave daemon containers or client processes alive. The shared executor owns both lifecycles and records cleanup without hiding the original failure.
- The root CI command could seal one control checkout while reading another checkout's catalog. Catalogs, test paths, retained workflows and ownership policy now come from the selected controls; candidate files are checked against that policy.
- Linked source directories and aliased built-in imports could disappear from the architectural check. Both have explicit failure cases, alongside the existing relative-import and composition-boundary checks.
- Expanded SVGs lost Mermaid's scoped styles and displayed black boxes. Expansion now remaps SVG identifiers and their style/marker references together. Browser checks compare the original and expanded rendering, exercise pan and zoom, and inspect phone layouts.
- Documentation triggers omitted some maintained guides and linked source files. The website gate now runs for every pull request to main and every main push; it does not depend on a second, incomplete path inventory.

The preceding review's source and installed-artifact checks are recorded below. The September 5 reports remain historical evidence for their frozen candidate; they are not reused as acceptance of this round. The starting inventory is `/tmp/aiter-architecture-round2-before.json`.

| Earlier review check | Result and retained evidence |
| --- | --- |
| Independent host unit suite | PASS, 368 tests; no input files changed during execution. `/tmp/aiter-architecture-round2-host-final/` retains the log and before/after inventories |
| Architectural dependency and placement policy | PASS, 989 Python files and 8,558 dependency edges; 18 computed imports remain explicitly listed for review. `/tmp/aiter-architecture-round2-policy-final.json` |
| Source runtime, packaging, tuning and activation selection | PASS, 231 cases, no failures/errors/skips. `/tmp/aiter-service-source-accepted.xml` |
| Same selection against the isolated wheel, including installed SDK cases | PASS, 233 cases, no failures/errors/skips. `/tmp/aiter-service-acceptance-3/installed.xml`; the origin audit found no invalid module origins |
| Native SDK and external C consumer | PASS, 36 SDK cases and an executed C ABI client. `/tmp/aiter-service-sdk-final.log` and `/tmp/aiter-service-acceptance-3/native-c-client.json` |
| Existing Python entry points | PASS, all 601 exports and 547 signatures in source and installed environments. `/tmp/aiter-service-acceptance-3/handoff.json` links both observations |
| Documentation example | The actual README HIP RMSNorm example matches an independent Torch reference in source and installed environments; both JSON results are linked by the same handoff |
| Independent archive audit | PASS, 12,747 hashed RECORD entries, 835 source Python files, 817 native source files, 2,348 source-archive inputs, the CK helper and four SDK library entries. `/tmp/aiter-architecture-round2-wheel-audit-corrected.json` |
| CI configuration and formatting | Catalog validation, workflow inventory, ownership routing, Actionlint, scoped Ruff/Black and whitespace checks pass. Staffing remains explicitly unassigned |

These selections overlap; their counts are not a unique-test total. They exercise the current runtime, loader and packaging changes on gfx950. They do not replace the older complete nightly profile, qualify gfx942, or establish full-model serving and multi-node results.

The new wheel is `amd_aiter-0.1.0.dev2026090603-cp312-cp312-linux_x86_64.whl`, under `/tmp/aiter-service-acceptance-3/wheels/`, with SHA-256 `1f65b4492a5d740747158c747037ff984aacc0aad0e7b3d98767a0145b09852d`. It was built from source archive SHA-256 `50a0bdd9bc046594fee1cc4b48029d4b1e28d2d433645d9ae03388532f6fbbf7`. Its Python files match the current package, apart from the deliberately generated version module, whose exact contents and distribution metadata are checked separately. It is an ordinary wheel with the native SDK, not an all-kernel AOT package.

Failed diagnostic attempts remain available. One source selection incorrectly included two installed-only tests. An earlier snapshot script accidentally omitted the legitimate `aiter/dist` package while excluding root build outputs; that snapshot was replaced using explicit root-only exclusions and source-inventory equality. The review also corrected a real source-archive omission of `3rdparty/ck_helper`. The final handoff distinguishes those attempts from the passing runs. The first independent inventory audit flagged the deliberately generated version module; its corrected report records the exact permitted addition instead of relaxing source-file comparisons.

The final documentation output is `/tmp/aiter-documentation-site`; browser evidence is under `/tmp/aiter-documentation-browser`. Its report names the served-file inventory, every checked page and viewport, interactions and screenshots. This is the rendering acceptance record for the actual website, separate from GPU qualification. Human review of the refreshed desktop and phone homepage, class-diagram detail/overview, dark mode and large-text views confirmed that the earlier clipping and SVG-style defects were corrected. [The website guide](docs/README.md) reproduces the build and browser checks without a GPU or external rendering service.

The Docker process adapter, source/wheel/image controllers, consumer recipes and failure paths have host tests and independent review. This container has no Docker executable or daemon, so actual OCI builds, image inheritance and full-model canaries remain external execution gates. Specialized legacy kernel and multi-service workflows retain their platform procedures and are identified in the workflow index. No registry or GitHub publication was performed.

## Responsibility and review

An integration lead, a runtime architect, a delivery architect and an independent QA engineer worked with separate file ownership. Implementation was followed by independent review and new tests for the findings. The review changed production behavior: it caught incorrect native scalar widths, wrong package resolution, unsafe tuning-file writes, incomplete benchmark arguments and release evidence that could hide a failed retry.

| Area | Files to review | Responsibility |
| --- | --- | --- |
| Public Python package | `aiter/__init__.py`, `_compat_exports.json`, `_logging.py`, `__main__.py` | Lazy imports, compatibility exports and diagnostics without device initialization |
| Operation descriptions | `aiter/api/`, `_validation.py` | Tensor layout, dtype, operation semantics and the complete public domain inventory |
| Prepared execution | `aiter/runtime/`, `aiter/backends/` | Support decisions, explicit provider inventory, preparation, retained launchers, binding checks and explanation |
| Native SDK | `include/aiter/`, `csrc/runtime/`, `CMakeLists.txt`, `cmake/`, `examples/native/` | Versioned C ABI, installed CMake package, native RMSNorm and CK GEMM providers |
| Rust frontend | `bindings/rust/`, `tests/integration/sdk/run-rust.sh` | Owned C plan handles, typed errors and descriptors, explicit asynchronous lifetime obligations |
| Build and packaging | `build_backend/`, `setup.py`, `pyproject.toml`, `MANIFEST.in` | Metadata, isolated staging, source archives, wheel payloads and explicit prebuild options |
| Native generation | `aiter/codegen/`, `_build_layout.json` | Named generators, declared source/wheel resources, checked subprocesses and caller-selected output directories |
| Native and DSL loading | `aiter/jit/`, `aiter/aot/`, `aiter/ops/_native/` | Recipe resolution, cache identity, compilers and package-qualified launch bridges |
| Tuning | `aiter/tuning/` | Separate observations, approved selections and legacy table reads; offline search programs and reusable workload factories |
| Tests | `tests/unit/`, `tests/integration/`, `tests/frameworks/`, `tests/operators/` | Host rules, connected GPU behavior, actual client calls and the broad operator inventory |
| Benchmarks | `benchmarks/common/`, `operators/`, `models/` | Measurement protocols, operation timings and model-shape sweeps |
| Qualification | `ci/qualification/`, `ci/clients/` | Explainable dependency selection, exact environments, execution and independent evidence checking |
| Delivery | `ci/release/`, `docker/`, `.github/workflows/release-*` | Wheel receipts, tested image identities, release notes, channel history, rollback and cadence |
| Ownership | `ci/ownership/`, `.github/CODEOWNERS` | One reviewed ownership policy, generated path rules and visible staffing requirements |

Native source directories no longer carry first-party Python generators or runtime bridges. Shared workload factories moved out of test files, so installed tuning programs do not import a test suite. Framework tests have separate `pytorch`, `vllm`, `sglang` and `common` packages. Benchmarks have their own root directory and module entry points.

## Important QA findings

| Finding | Resolution |
| --- | --- |
| Generator scripts guessed parent directories and could import an unrelated AITER installation | `BuildContext` reads the source or installed resource manifest; generators run by registered package name with an explicit child environment |
| Resource overrides were ignored by several native bridges | Bridge include and template paths now use declared native/assembly resources; an unrelated-working-directory regression covers relocated resources |
| A moved assembly command contained an invalid `/-m` prefix | Evaluated recipe execution now reaches the registered generator; assembly and dynamic recipe families are exercised beyond `--help` |
| AOT launchers passed a float32 scalar using the C++ `double` ABI | Scalar widths and compiled symbol/signature metadata were corrected; actual normal and residual layernorm launches cover a tail width and nontrivial epsilon |
| BF16 assembly split-K rounded partial sums into BF16, and an explicit split could select a kernel without its semaphore | Accumulate in FP32 and round once; honor or reject exact split requests; validate descriptors and workspace; use a distinct private ABI and explicit output-capability metadata. See [the GEMM migration](docs/migrations/asm-gemm.md) |
| Ragged attention passed arguments in an obsolete generated ABI order; its old test used the wrong tensor representation | Corrected bridge argument order and a native consumer with FP16 packed-cache inputs and an independent CPU softmax reference |
| Native tests depended on hard-coded generated symbols, source-directory outputs or embedded Python path edits | Test-owned build commands use returned compiler metadata, declared package resources and explicit writable outputs; both source and installed-wheel profiles exercise them |
| Distributed utilities imported framework code and referenced undefined consumer globals | Torch supplies device properties and FP8 conversion; AITER owns process identifiers and environment access; dependency rules now cover distributed code |
| Prebuild worker failures could disappear in an unconsumed iterator | Workers return checked results; process failures propagate, and launch jobs do not pickle ctypes pointer objects |
| JIT compilation wrote locks and modules under the installed package | New artifacts and locks use an explicit JIT directory or the user's XDG cache; bundled AOT files remain input resources |
| Legacy CSV merging rewrote source measurements and selected the lowest reported time | Input tables remain unchanged; ambiguous shapes fail with both source locations; successful merges produce content-addressed cache files |
| A tuning result could attach one timing to several configurations or invent neighboring-shape selections | Each observation identifies one measured configuration and exact shape; retained traces and samples support reconstruction; results do not install runtime policy |
| A profiler timeout could leave GPU descendants running | Shared bounded process-group execution terminates descendants and retains the actual status and logs |
| Model benchmarks imported missing programs and tests supplied incomplete arguments | A lazy operation catalog loads the selected driver; integration tests use the production parser and verify the resulting CSV/plots |
| A benchmark summary could overflow despite finite individual samples | Nonfinite derived medians, quartiles and spreads are rejected |
| A native plan could accept wrong-device resources or outlive mutable code at a reused filename | Device/span/stream checks and exact-byte retained libraries; the C loader uses sealed memfd snapshots and preserves graph executable lifetime |
| Prepared execution could recompile or allocate during launch | Plans retain native symbols or compiled DSL launchers; graph tests forbid compiler and allocation entry points during execution |
| An ordinary wheel copied ignored historical JIT binaries and cache metadata | Stage source inputs separately from generated artifacts; source archives exclude local outputs, and only an explicit SDK/prebuild adds executable payloads |
| A broad build ignore rule hid the new unit/build tests from source identity | Explicit test-directory inclusion and an architecture check keep contributor sources visible to Git |
| Wheel repair invalidated native hashes | Native manifests and wheel RECORD are refreshed before final receipt creation; later mutation is rejected |
| A PASS summary could disagree with the individual test results | Evidence checking rereads retained JUnit, script logs, return codes, attempts and hashes |
| A failed publication retry could reuse an earlier PASS | Attempt input identities include delivery outcomes; changed retries remain visible failures while the last good channel remains available |
| A release candidate could replace the CI controller used on the host | Trusted controls and candidate source have separate checkouts and sealed identities; candidate build/test code runs in the selected environment |
| A source qualification reused an ambient compilation cache despite selecting the correct Python package | Every attempt owns fresh compiler caches; inherited resource overrides are removed, and observed builder resources are checked against the selected package layout |
| Choosing a writable cache suppressed an installed wheel's prebuilt extension | Cache storage and installed artifact selection are separate; only a declared installed layout can supply bundled modules, and a real compiled-extension test checks both layouts |
| An activation reference rounded intermediate values before selecting an MXFP4 exponent | The reference follows the fused operation's FP32 intermediates; independent FP64 boundary cases guard the scale decision without weakening tolerances |
| JIT and AOT maintained different lock loops; one could consume a dead builder's result or leave a lock after cleanup failed | Both entry points delegate to one lock lifecycle; real child-process tests cover owner death, waiter return values and failure cleanup |
| A Rust frontend could imply safe asynchronous lifetimes it cannot prove | Enqueue is explicitly unsafe; plans are neither Send nor Sync; ABI offset, compile-fail and real installed-consumer tests enforce the declared interface |

## Local environment

The container exposes eight gfx950 GPUs. GPU checks use `/app/vllm/.venv/bin/python`: Python 3.12, Torch `2.12.0+git6bbd260`, HIP `7.2.53211`, Triton `3.7.1+gitf0b55c07` and FlyDSL `0.3.2`. The CK submodule remains at its original pin, `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`. The shared environment's installed AITER wheel was not replaced.

Tests explicitly select the source checkout or an isolated installed package. SGLang bridge checks use pinned source `da76fa073f8e7df4bf30d7049dd4609d9ca3f23c` and isolated compatible dependencies. The local vLLM environment has a setuptools version outside its declared dependency range. Its observed kernel calls are useful development evidence, but do not qualify that environment as a supported release profile.

Use one visibility variable. Setting both HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to physical device numbers can filter the devices twice. Build outputs, installations, traces and logs are retained under `/tmp`; they are not source deliverables.

## Evidence retained during development

These checks establish the stated behavior at their recorded implementation point. A later source change requires affected checks to run again. Final source and installed-artifact qualification is recorded separately below, rather than borrowing an old wheel's result.

| Check | Evidence and scope |
| --- | --- |
| Python compatibility | All 601 baseline exports and all 547 callable argument/default lists preserved. Four annotation strings now use the canonical `aiter.jit.module_aiter_core` namespace; they are not byte-identical annotation strings. `/tmp/aiter-qa-exports-comparison.log` |
| Prepared runtime | 103 expanded GPU cases, plus 22 focused final MXFP4 cases; numerical, layout, alias, device, stream and capture checks. `/tmp/aiter-runtime-expanded.xml`, `/tmp/aiter-mxfp4-final.xml` |
| Connected execution | Shared-buffer RMSNorm → FP8 quantization → projection → RoPE → dense attention, including changing-input graph replay and compiler/allocation guards |
| Reorganized runtime suite | 93 prepared GPU integration cases after the code-generation move. `/tmp/aiter-qa-codegen-runtime.xml` |
| Framework paths | Eight actual PyTorch/vLLM/SGLang cases, including two vLLM MXFP4 leaf cases. `/tmp/aiter-framework-tree.xml`, `/tmp/aiter-framework-tree-sglang.xml` |
| Existing numerical behavior | 152 elementwise, quantization and RoPE cases passed without skips; logs and case inventories under `/tmp/aiter-qa-{elementwise,quant,rope}` |
| Eight-GPU communication | Native CustomAllreduce, FP16/BF16, repeated eager calls and changing-input graph replay with exact results on every rank. `/tmp/aiter-qa-collectives.log` |
| Cold CK path | Actual generation, native compilation and numerical dispatch after the package migration. `/tmp/aiter-codegen-dispatch.log` |
| Native attention benchmark | Explicit output build and FP16/BF16 dQ/dK/dV references; two host and two GPU cases. `/tmp/aiter-native-benchmark-harness.xml` |
| Shipped configuration tables | Read-only family resolution and collision controls; repair details in [the migration record](docs/migrations/tuning/README.md). `/tmp/aiter-shape-collision.xml` |
| Native AOT bridge | Two real normal/residual generated layernorm cases, width 137 and epsilon 0.37. `/tmp/aiter-codegen-aot.xml` |
| Packaging navigation | 32 package-origin, cold entrypoint, generator and compiler cases. `/tmp/aiter-native-resource-override.xml` adds relocated-resource checks |
| C SDK | 36 native assertions, C-only consumer, installed CMake consumer and graph/library lifetime checks |
| Rust SDK | Three ABI/metadata tests, two compile-fail documentation tests and real GPU RMSNorm/GEMM from an isolated installed consumer. Fresh lib64 build/install also passed. `/tmp/aiter-rust-reviewed.log`, `/tmp/aiter-rust-lib64.log` |
| Benchmark programs | Five actual GPU harness cases exercise the reorganized operation/model programs and production argument parser. `/tmp/aiter-benchmark-harness.xml` |
| Paired measurements | Six retained operation comparisons with independently reconstructed raw samples. `/tmp/aiter-benchmark-cli/report.json` |
| Real offline profiling | One gfx950 GEMM configuration, 250 retained kernel samples, checked profiler completion. `/tmp/aiter-qa-profiler-launcher/`; sweep-to-results path in `/tmp/aiter-triton-sweep-check/` |
| Profiling integrity | 18 host tests plus 62 subtests, including a real timeout descendant process. `/tmp/aiter-qa-profile-integrity.xml` |
| Release state | 62 focused controller/delivery tests plus 28 subtests; retries, rollback, exact identities and failure visibility. `/tmp/aiter-qa-delivery-final.xml` |
| Historical large AOT build | 6,646 accepted FlyDSL jobs and four native FMHA modules completed in an isolated build. The collector excluded 136 rows before job selection. `/tmp/aiter-aot-wheel-8tpp67iu/`. This predates the latest build-module migration and is not current-wheel qualification. |

The timing samples above validate measurement paths. They are not claims about product speed or full-model throughput. A framework bridge test proves an observed operator call and numerical result; it is not a real-weight model-serving qualification.

## Acceptance iterations

Acceptance uses separate frozen copies of the candidate and the CI controller. A failing attempt stays on disk with its original inputs. A repair creates a new candidate and new evidence; it does not change a failed report into a pass.

The first attempt found three packaging problems: an obsolete install-marker assertion, a required assembly-directory check that prevented a legitimate Triton-only import, and ignored historical JIT outputs leaking into an ordinary wheel. The fixture and resource checks were corrected. Wheel staging now excludes incidental generated artifacts before adding any explicitly requested SDK or prebuilt payload. The first wheel remains rejected even though its bounded numerical tests passed.

The second frozen candidate passed the product profile (seven groups, 453 recorded cases), the actual CPU source-archive-to-wheel build, and 148 checks against a fresh ordinary wheel installation. Independent QA verified the wheel's RECORD hashes, native payload, package origins and copied controller files. The wheel SHA-256 is `3af76bf3bf8629171477048861666620947024d5017bfc77bbe493d14c3e5e87`; its evidence is retained in `/tmp/aiter-acceptance-wheel-identity-2.json` and `/tmp/aiter-acceptance-qa-wheel-2.json`.

That same candidate failed the broader nightly profile. Eleven groups passed, including eight-GPU communication, paired measurements, the native SDK and Rust. The legacy BF16 GEMM driver reported incorrect values in its assembly split-K path while returning process status zero. The CI checker rejected the attempt by inspecting the numerical log. Independent FP64-reference checks reproduced the problem: the BF16 atomic reduction rounded every partial sum, with the mismatch fraction increasing as the number of splits increased. A separate explicit-split request exposed a missing semaphore check and an ignored dispatch argument. Those findings require a new wheel; the earlier 148 passing cases do not override them. The failed report remains `/tmp/aiter-acceptance-2-nightly-report.json`.

The final layout sweep also replaced the old AOT test program under the installed package. Its native consumer now uses the compiler's returned header and symbol, writes to a temporary output directory and checks 19 shapes against an independent exact CPU reference. The former print-only raw-HSACO demonstration is retired; the generated-launcher check does not claim to qualify that separate raw loader.

The repaired assembly driver passes its original numerical checks. Independent lifecycle tests cover separate streams, changing-input graph replay, writable-buffer alias rejection and the BF16-only kernel subset. The GEMM capability manifests explicitly declare FP32 support, and old binaries cannot satisfy the new private workspace ABI. The correction adds FP32 temporary storage and a conversion; its performance cost is recorded in the migration note rather than hidden by weakening accuracy requirements.

Native attention now checks 2,048 ragged-attention output values and 6,144 Gluon output values against independent CPU softmax references; the Gluon consumer checks both cold and repeated execution. Distributed helper tests block framework imports and exercise actual FP8 cache generation. These checks have persistent test files and qualification groups.

The third candidate passed the product profile (eight groups, 497 recorded cases) and the CPU packaging profile. Its fresh ordinary wheel, version `0.1.0.dev2026090503`, has SHA-256 `6013d8b4c86998f142978ac0886521cffcbc0ce601839768302ff1f44e69eed4`. QA checked all 12,723 hashed RECORD entries and compared the first-party Python and native sources with the frozen candidate. These are packaging results; this wheel is not the final accepted artifact.

Thirteen of fourteen nightly groups passed. The activation driver rejected an MXFP4 scale at an exponent boundary. Here the production kernel was correct: with BF16 gate `0.7421875` and value `5.96875`, the fused result is approximately `3.0011635`, while the old reference rounded an intermediate result to `3.0`. That premature rounding changed the selected exponent by one. The corrected reference preserves the documented FP32 intermediates, including the input-dtype gate-clamp behavior. It passes the original driver without changing tolerances. Eight independent FP64-oracle cases and fourteen focused operation cases now retain this distinction. See [the precision explanation](docs/migrations/activation-precision.md).

Review of that same run found a separate evidence problem: the source profile could load named extensions from an ambient cache. The source/control identities were correct, but that alone did not prove a cold build of every loaded native module. The third nightly report remains a failed report, and earlier source runs are not promoted into cold-source qualification. Its original evidence remains in `/tmp/aiter-acceptance-3-nightly-report.json` and `/tmp/aiter-acceptance-qa-nightly-3.json`.

The executor now removes inherited AITER resource settings and allocates fresh per-attempt JIT, AOT, Triton, Torch, FlyDSL and XDG caches. Its retained isolation record is checked against subprocess observations and already-loaded builder resources. Regression tests poison ambient paths and compare separate attempts in both source and wheel modes. An explicit writable JIT directory no longer disables a wheel's declared installed extension; source-local ignored binaries are not treated as installed artifacts. These changes require a fourth frozen candidate and a newly built wheel.

The build review also removed duplicated JIT/AOT lock orchestration. The AOT copy had ignored stale-owner recovery and discarded the waiter callback's return value; either copy could retain its lock if the final callback raised. Both now use one helper while keeping their public callback signatures. Four real process/lifecycle tests and four existing wrapper cases pass. The helper retries acquisition after a dead owner and releases its lock even when cleanup fails.

The fourth candidate passed all 309 independent host unit tests, the CPU packaging profile and the dedicated FlyDSL profile. The FlyDSL test compiles a real specialization, checks a cold run-only hit, rejects a missing specialization and rejects adoption after a JIT object has already executed. The package must select its bundle before compiler initialization; a later environment change cannot retarget an existing kernel object. QA verified the fourth wheel's 12,725 hashed RECORD entries, 813 first-party Python files, 817 native source files and four freshly built SDK libraries. Those wheel checks establish archive contents, not installed execution.

The fourth product profile passed nine groups and 554 recorded cases. Product and FlyDSL source runs deliberately used the same frozen directory for candidate and controller. Their recorded imports match that directory. Before claiming separated execution, the team exercised the full host profile through the runner's copied-control mode. This exposed two harness assumptions: a test looked for the candidate's cache module beside the copied tests, and the suite omitted reviewed workflow files needed by another test. The fixture now reads the selected candidate module, and the runner carries the reviewed `.github/` inputs with its controls. Failed copied-suite evidence remains under `/tmp/aiter-copied-suite-host-run`; snapshots and earlier reports were not edited. The corrected complete copied host profile passed all 315 recorded cases through planning, execution and independent checking. Its report is `/tmp/aiter-copied-suite-host-report-2.json`.

## September 5 acceptance

The fifth candidate uses separate frozen directories: `/tmp/aiter-acceptance-build-5` for candidate sources and `/tmp/aiter-acceptance-control-5` for reviewed controls. Its tracked-diff SHA-256 is `6b513836e7f7f89c5082385adb0217870617469c1971c38f289b08fde1ee2c69`; its untracked-file digest is `dbbc9382982d3c0e41b435e714855a9cbe9493b3cbd0a619691e06d36f406844`. Both share the original branch HEAD and CK pin recorded above. Each source group copies the reviewed controls and uses a fresh, recorded compilation cache.

| Check | Result and retained record |
| --- | --- |
| Independent host unit tests | PASS, 309 tests; `/tmp/aiter-acceptance-qa-unit-5.json` |
| Product profile | PASS, nine groups and 554 recorded cases/invocations; `/tmp/aiter-acceptance-5-product-report.json` |
| FlyDSL profile | PASS, two groups and 316 records; `/tmp/aiter-acceptance-5-flydsl-report.json` |
| CPU source-archive-to-wheel build | PASS; `/tmp/aiter-acceptance-5-packaging-report.json` and independent archive audit |
| Broader nightly profile | PASS, 15 groups and 576 records; independently reconstructed in `/tmp/aiter-acceptance-qa-nightly-5.json` |
| Fresh wheel contents | PASS; 12,725 RECORD hashes, 813 first-party Python files, 817 native source files and four fresh SDK binaries; `/tmp/aiter-acceptance-qa-wheel-5.json` |
| Expanded installed attempt | FAIL, 223 passing checks and two missing-SGLang prerequisites; `/tmp/aiter-acceptance-installed-5.xml` |
| Corrected installed framework checks | PASS, eight PyTorch/vLLM/SGLang cases; `/tmp/aiter-acceptance-qa-frameworks-5.json` |
| Installed FlyDSL profile | PASS, two groups and 316 records; `/tmp/aiter-acceptance-formal-flydsl-report-5.json` |
| Corrected complete wheel profile | PASS, five fresh groups/attempts and 56 records in `local-mi350x-rocm72-tabulate010`; `/tmp/aiter-acceptance-qa-formal-wheel-corrected-5.json` |

The wheel is `amd_aiter-0.1.0.dev2026090505-cp312-cp312-linux_x86_64.whl`, retained under `/tmp/aiter-acceptance-wheels-5/`. Its SHA-256 is `a09228cda7445533221843c678c86f7dc915bc3131b16e78dc8b119b9dbeddf7`. This is an ordinary wheel with the native SDK. The separate FlyDSL test exercises real compiled bundle handling; it does not turn this ordinary wheel into an all-kernel AOT distribution.

The expanded installed run attempted 225 cases. It passed 223 and failed the two SGLang prerequisites because that interpreter did not include the isolated client installation. The failed attempt remains `/tmp/aiter-acceptance-installed-5.xml`; it has not been relabeled as a passing suite. A corrected client run passed all eight PyTorch/vLLM/SGLang bridge cases in 8.93 seconds. An already-started SGLang-only rerun also passed its two cases; those overlap the eight and are not additional coverage. QA compared all 4,374 installed SGLang source/resource files with the retained pinned archive and verified AITER imports against the same wheel. The client audit is `/tmp/aiter-acceptance-qa-frameworks-5.json`.

The first formal wheel attempt passed its numerical GEMM checks but failed while rendering a Pandas summary: the private test environment lacked `tabulate`. The production qualification workflow already declares that dependency. Correcting the local environment changes the execution inputs, so the team retained the failed run and started a complete new five-group qualification with an explicit corrected environment identity. Passing groups were not copied from the failed attempt. The exact selected dependencies are retained in `/tmp/aiter-acceptance-corrected-wheel-environment-5.json`.

That complete corrected wheel profile passed: 22 activation cases, 25 assembly GEMM cases, two native attention cases, two SDK-provider cases and five legacy smoke invocations. Its independently reconstructed report is `/tmp/aiter-acceptance-formal-wheel-corrected-report-5.json`, digest `sha256:751cb409771161a2a44b72b67c4d337b202dc242fad771783c21108a59c89082`. Every observed AITER import came from the isolated installation of the same wheel. The installed normalization module compiled from its packaged sources in 1,984.5 seconds; the complete driver took 1,988.952 seconds and passed both numerical checks. Its fresh cache did not reuse the source run's library.

The complete source nightly report has digest `sha256:4c766a84b18be3ca4c9e5d07fcfd8a8114a6284f0ddd3381e129cd5248fbe1ed`. Its 576 records include individual test cases and script invocations, not a count of every assertion inside those scripts. Product, nightly and FlyDSL profiles overlap; their counts must not be added into a unique-test total. `/tmp/aiter-acceptance-5-delivery-handoff.json` retains each group's source, controls, observed resources and fresh cache paths.

The final source normalization driver built 393 generated C++ files and two direct sources with two compiler workers. Compilation took 1,989.7 seconds; the complete driver, including numerical checks, took 1,993.619 seconds and passed. This is cold compilation cost, not inference latency. The resulting `module_norm.so` has SHA-256 `d7a6017f635cea5d4381841761ac103f8340ec4279028d033e49dbd0249d898e`. It demonstrates the remaining first-use build cost of a legacy operation. The ordinary wheel's bundled native SDK does not make every legacy operator compiler-free.

Those source and installed-wheel reports used the observed development environments. They did not qualify every supported release cell. At the September 5 freeze, the working tree differed from the accepted candidate only in that acceptance record, rollout results and one SDK diagram syntax correction. `/tmp/aiter-final-source-delta-5.json` records that earlier file comparison. All 18 first-party Mermaid diagrams passed the parser, and Sphinx built with warnings treated as errors. The September 6 browser review demonstrated why those checks alone did not establish a complete, usable website. Subsequent architecture changes require new affected-code and installed-artifact evidence.

## Reproduce the current checks

From the repository root, run the complete host suite with its declared CPU test dependencies:

```sh
python -m pip install -r requirements/test/host.txt
python -m pytest tests/unit --require-capabilities
```

This collects both unittest classes and pytest functions, including the build suite. Pure metadata and policy subsets also have supplemental standard-library-only commands:

```sh
python -S -m unittest discover -s tests/unit/architecture -t tests -v
python -S -m unittest discover -s tests/unit/build -t tests -v
python -S -m unittest discover -s tests/unit/runtime -t tests -v
python -S -m unittest discover -s tests/unit/tuning -t tests -v
python -S -m unittest discover -s tests/unit/ci -t tests -v
python -m ci validate
python -m ci.ownership.policy check
```

In the selected ROCm environment, freeze the source before planning:

```sh
python -m ci plan --profile product-fast --architecture gfx950 --output /tmp/aiter-plan.json
python -m ci run --plan /tmp/aiter-plan.json --gpus 0,1 --output-dir /tmp/aiter-run
python -m ci check --plan /tmp/aiter-plan.json --results /tmp/aiter-run
```

Use separate control and build checkouts for artifact qualification. The planner seals both identities; the executor rejects changes after planning. See `python -m ci --help` and [the CI guide](ci/README.md) for explicit roots, wheels and profiles. Native C/Rust acceptance commands live in [the SDK guide](include/aiter/README.md) and [the Rust guide](bindings/rust/README.md).

## Release qualification is still a separate decision

This machine supplies gfx950 evidence, including the two bounded real vLLM engine cases recorded above. gfx942, other environment cells, broader serving canaries, multi-node failure tests and actual GPU image execution require their declared workers. There is no usable Docker engine in this container; user/mount namespace creation is denied. A client executable alone cannot execute the image qualification pipeline.

Prepared guarantees apply to the operations documented in the runtime guide. The complete legacy interface remains available and is accounted for in the domain catalog. Stateful attention, MoE and communicator interfaces retain their existing specialized semantics; a new plan/lifetime promise must receive its own implementation and tests before being advertised.

Ownership rules are active files, but actual upstream maintainers and backups still need to accept responsibilities. Registry access, immutable runner environments and protected publication settings must be configured before activating delivery. None were changed locally.
