# Shared test infrastructure

`tests/common` detects the available GPUs, checks package and model inputs, and manages test processes. Framework prompts, engine options and assertions stay with their framework tests.

Run the host suite with `python -m pytest tests/unit`. It collects both `unittest.TestCase` classes and pytest functions. Install `requirements/test/host.txt` first. The host suite needs no GPU packages; focused metadata checks also have separate `python -S -m unittest` commands.

The suite declares `tests/` as its helper import root. Import helpers as `common.paths` or `frameworks.common.runtime`; do not modify `sys.path` inside a test or depend on another installed package named `tests`.

## Declare what a test needs

```python
import pytest

pytestmark = [
    pytest.mark.gpu(min_count=1, min_memory_gib=16),
    pytest.mark.requires_arch("gfx942", "gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
]

def test_operation(gpu_device):
    # Import Torch and the operation here, after prerequisites were checked.
    ...
```

`requires_arch` lists the architectures that can run a case. `skip_arch("gfx942", reason=...)` documents a specific exclusion. Use these markers instead of repeating device checks. `gpu_hardware` returns the observed hardware; `gpu_device` selects the first visible device. The runner controls which devices a process can see.

The `bf16`, `fp8` and `mxfp4` capability names describe hardware eligibility. They do not promise that every backend implements that data type. A test must execute and check the operation it claims to cover.

| Architecture label | Hardware family | Selection rule |
| --- | --- | --- |
| `gfx942` | CDNA3, including MI300X and MI325X | BF16 and FP8 groups may be eligible; native MXFP4 groups require a different target. |
| `gfx950` | CDNA4, including MI350X and MI355X | BF16, FP8 and MXFP4 groups may be eligible. This is the hardware available for local execution here. |
| `gfx1250` | CDNA5 / MI450 bring-up target | Select only tests whose actual backend supports this target. A gfx950 CK kernel is not made compatible by adding a marker. |

Use the architecture label in test and pipeline declarations, because a product name alone does not identify the kernel instruction set. AMD's [supported-GPU table](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) identifies the gfx942/gfx950 families. The upstream [AITER gfx1250 bring-up report](https://github.com/ROCm/aiter/issues/2299) identifies the newer target and illustrates why individual native backends need separate enablement. Declaring eligibility or collecting a test on a host does not establish execution on that GPU. MI455-specific results are not claimed.

Collection rejects unknown markers and invalid arguments without importing Torch, vLLM or AITER. Hardware detection happens when a marked test starts. Framework tests also defer their framework imports until execution.

## Discovery and qualification have different requirements

An ordinary developer run skips a marked test when its GPU or framework is unavailable. Real model tests additionally require `--run-e2e`.

A qualification run passes `--require-capabilities`. Missing prerequisites then fail, and an explicit `pytest.skip` in a selected case also fails. A CPU machine or an empty model cache therefore cannot produce a passing model-qualification result. The CI catalog selects the cases.

## Model inputs and execution evidence

Each client owns its model declarations. For example, [`ci/clients/vllm/models.json`](../../ci/clients/vllm/models.json) names exact revisions, file sizes and hashes for both tests and benchmarks. `ci.common.checkpoints` validates the declared bytes; `models.py` connects that validation to test provisioning. Tests consume previously provisioned files and do not download models themselves.

Before an engine starts, the fixture copies only declared files into a private model directory and checks the copied bytes. Extra files in the Hugging Face cache cannot change the loader's inputs. After execution, the fixture checks the whole private directory again, including unexpected files. Hugging Face blob symlinks are accepted as inputs; the private copy contains ordinary files.

Small evidence files go beneath `--e2e-evidence-dir`; large model copies remain under pytest's temporary directory. CI retains requests, model receipts, process logs, outputs and observed AITER calls. `process.py` gives each model engine a deadline and cleans up its process group.
