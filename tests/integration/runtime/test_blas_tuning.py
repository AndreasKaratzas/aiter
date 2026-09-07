# SPDX-License-Identifier: MIT
"""Online tuning succeeds or rejects unavailable algorithms without a crash."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("fp8"),
]


def test_hipblaslt_online_tuning_bf16_matches_matmul(tmp_path, monkeypatch):
    import torch

    import aiter
    from common.paths import assert_package_origin

    assert_package_origin()
    monkeypatch.chdir(tmp_path)  # No earlier tuning CSV can bypass this call.
    monkeypatch.setenv("HIP_ONLINE_TUNING", "1")
    torch.manual_seed(1436)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16).T
    aiter.hipb_create_extension()
    try:
        actual = aiter.hipb_mm(x, w, -1)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual.float(), (x.double() @ w.double()).float(), rtol=0.02, atol=0.04
        )
        assert (tmp_path / "hip_online_tuning_res.csv").is_file()
    finally:
        aiter.hipb_destroy_extension()


def test_hipblaslt_online_tuning_unavailable_fp8_does_not_kill_process(tmp_path):
    # This is a process-lifetime regression, not an FP8 support claim. Future
    # libraries may support this shape; those must return a correct tensor.
    # The separately selected vllm-hipblaslt numerical tests require success.
    script = r"""
import json
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
from common.paths import assert_package_origin

assert_package_origin()
torch.manual_seed(1437)
x = (torch.randn(32, 512, device="cuda") * 0.5).to(torch.float8_e4m3fn)
w = (torch.randn(512, 512, device="cuda") * 0.5).to(torch.float8_e4m3fn)
xs = torch.linspace(0.125, 0.75, 32, device="cuda").view(32, 1)
ws = torch.linspace(0.25, 1.25, 512, device="cuda").view(1, 512)
bias = torch.linspace(-0.25, 0.25, 512, device="cuda").to(torch.bfloat16)
prepared = shuffle_weight(w).T
aiter.hipb_create_extension()
try:
    try:
        actual = aiter.hipb_mm(x, prepared, -1, bias, torch.bfloat16,
                              xs, ws, bpreshuffle=True)
    except RuntimeError as error:
        assert "hipBLASLt online tuning found 0 valid solutions" in str(error), str(error)
        status = "clean_no_solution_rejection"
    else:
        torch.cuda.synchronize()
        expected = (x.double() * xs.double()) @ (w.double() * ws.T.double()).T + bias.double()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.012, atol=0.025)
        status = "supported_numerical_result"
finally:
    aiter.hipb_destroy_extension()
print("TUNING_RESULT=" + json.dumps({"status": status}))
"""
    env = dict(os.environ, HIP_ONLINE_TUNING="1")
    # Keep the caller's admitted package and test-helper origins in the child.
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "subprocess.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        line.removeprefix("TUNING_RESULT=")
        for line in result.stdout.splitlines()
        if line.startswith("TUNING_RESULT=")
    ]
    assert len(records) == 1, result.stdout
    assert json.loads(records[0])["status"] in {
        "clean_no_solution_rejection",
        "supported_numerical_result",
    }
