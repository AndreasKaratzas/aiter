# SPDX-License-Identifier: MIT
"""Pinned topology selection, wheel overlays and actual local Slurm-process boundaries."""

import copy
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from ci.clients.vllm.disaggregation import artifacts, cluster, logs, selection


IMAGE = "reviewed/image@sha256:" + "a" * 64
PIPELINE = Path(__file__).with_name("fixtures") / "vllm-disaggregation.yaml"


def test_exact_pinned_pipeline_selects_only_ten_router_cases_as_argv_and_env():
    pipeline = PIPELINE.read_bytes()
    assert (
        hashlib.sha256(pipeline).hexdigest()
        == selection.configuration()["source"]["sha256"]
    )
    cases = selection.cases(pipeline, IMAGE)
    assert len(cases) == len({item["case_slug"] for item in cases}) == 10
    assert {item["model_name"] for item in cases} == {
        "DeepSeek-V3",
        "DeepSeek-R1-MXFP4",
        "Kimi-K2.5-MXFP4",
        "Kimi-K2.6-MXFP4",
        "MiniMax-M3-MXFP8",
    }
    assert {item["topology"] for item in cases} == {"1P1D-TP8", "2P2D-TP8"}
    for case in cases:
        assert case["argv"] == [
            "bash",
            ".buildkite/amd-disagg/run-slurm-disagg-test.sh",
        ]
        assert case["environment"]["IMAGE"] == IMAGE
        assert case["environment"]["ROUTER_TYPE"] == "vllm-router"
        assert case["environment"]["WAIT"] == "1"
        assert case["nodes"] == (2 if case["topology"] == "1P1D-TP8" else 4)
        assert selection.select_case(cases, case["case_slug"]) == case
    with pytest.raises(ValueError, match="exactly one"):
        selection.select_case(cases, "not-declared")
    with pytest.raises(ValueError, match="exactly one"):
        selection.select_case(cases + [cases[0]], cases[0]["case_slug"])


@pytest.mark.parametrize(
    "image",
    [
        "unpinned:latest",
        "name@sha256:" + "z" * 64,
        "name with space@sha256:" + "a" * 64,
    ],
)
def test_selection_rejects_unpinned_or_ambiguous_images(image):
    with pytest.raises(ValueError, match="pinned"):
        selection.cases(PIPELINE.read_bytes(), image)


def test_changed_source_and_missing_case_cannot_shrink_reviewed_selection(monkeypatch):
    original = PIPELINE.read_bytes()
    with pytest.raises(ValueError, match="reviewed revision"):
        selection.cases(original + b"\nchanged", IMAGE)
    # Rebind the synthetic source hash only to reach the structural contract;
    # production always checks the real pinned source first.
    changed = original.replace(
        b"DeepSeek-V3-PD-1P1D-TP8-MoRIIO-vllm-router",
        b"DeepSeek-V3-PD-1P1D-TP8-MoRIIO-unselected",
    )
    config = copy.deepcopy(selection.configuration())
    config["source"]["sha256"] = hashlib.sha256(changed).hexdigest()
    monkeypatch.setattr(selection, "configuration", lambda: config)
    with pytest.raises(ValueError, match="upstream case differs"):
        selection.cases(changed, IMAGE)


def wheel_pair(tmp_path, *, change=None, extra=None):
    native = {
        name: ("compiled " + name).encode()
        for name in ("module_aiter_core", "module_gemm_a8w8_blockscale_cktile")
    }
    receipt = {
        "schema_version": 1,
        "prebuild_profile": 0,
        "flydsl_requested": False,
        "requested_modules": ["module_gemm_a8w8_blockscale_cktile"],
        "native": [
            {
                "name": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in native.items()
        ],
    }
    if change:
        change(receipt)
    entries = {
        "aiter/__init__.py": b"candidate package",
        "aiter/kernels/data/gfx950/test.co": b"declared code object",
        "aiter/jit/prebuild.json": json.dumps(receipt).encode(),
        **{f"aiter/jit/{name}.so": data for name, data in native.items()},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_path / "amd_aiter-1.0-py3-none-any.whl", "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(tmp_path / "flydsl-1.0-py3-none-any.whl", "w") as archive:
        archive.writestr("flydsl/__init__.py", b"compiler package")
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return entries


def test_overlay_admits_exact_native_receipt_and_preserves_both_packages(tmp_path):
    inputs = tmp_path / "wheels"
    wheel_pair(inputs)
    template = tmp_path / "template.sh"
    template.write_text("export OWNED=1\n")
    out = tmp_path / "overlay"
    record = artifacts.overlay(inputs, out, template)
    assert len(record) == 2 and all(
        hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest() == x["sha256"]
        for x in record
    )
    assert (out / "aiter/__init__.py").read_bytes() == b"candidate package"
    assert (out / "flydsl/__init__.py").read_bytes() == b"compiler package"
    assert (tmp_path / "aiter_cluster_env.sh").read_bytes() == template.read_bytes()
    with pytest.raises(ValueError, match="new"):
        artifacts.overlay(inputs, out, template)


@pytest.mark.parametrize(
    "member",
    [
        "aiter/__init__.py",
        "aiter/__init__.py/",
        "aiter/./__init__.py",
        "aiter//__init__.py",
        "../escape",
        "/absolute",
        "dir\\escape",
    ],
)
def test_overlay_rejects_collisions_and_normalized_path_aliases_before_writing(
    tmp_path, member
):
    inputs = tmp_path / "wheels"
    wheel_pair(inputs, extra={member: b"overwrite candidate"})
    template = tmp_path / "template"
    template.write_text("unchanged")
    with pytest.raises(ValueError):
        artifacts.overlay(inputs, tmp_path / "overlay", template)
    assert not (tmp_path / "overlay").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(schema_version=True),
        lambda r: r.update(prebuild_profile=False),
        lambda r: r.update(requested_modules=[]),
        lambda r: r["native"].pop(),
        lambda r: r["native"][0].update(sha256="0" * 64),
        lambda r: r["native"][0].update(size_bytes=1),
        lambda r: r["native"][0].update(name="../foreign"),
    ],
)
def test_prebuild_receipt_does_not_admit_wrong_mode_identity_or_corrupt_bytes(
    tmp_path, change
):
    wheel_pair(tmp_path, change=change)
    with pytest.raises((ValueError, KeyError)):
        artifacts.validate(tmp_path)


def test_malformed_and_duplicate_wheels_are_rejected(tmp_path):
    wheel_pair(tmp_path)
    aiter = tmp_path / "amd_aiter-1.0-py3-none-any.whl"
    with zipfile.ZipFile(aiter, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("aiter/__init__.py", b"duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.validate(tmp_path)
    aiter.write_bytes(b"not a ZIP")
    with pytest.raises(zipfile.BadZipFile):
        artifacts.validate(tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[vllm_disagg] PASS: exact_match=0.4 threshold=0.5",
        "[vllm_disagg] FAIL: exact_match=0.9 threshold=0.5",
        "[vllm_disagg] PASS: exact_match=nan threshold=0.5",
        "[vllm_disagg] PASS: exact_match=1 threshold=nan",
        "[vllm_disagg] PASS: exact_match=inf threshold=0.5",
        "[vllm_disagg] PASS: exact_match=1.1 threshold=0.5",
    ],
)
def test_accuracy_verdict_requires_finite_consistent_score(text):
    with pytest.raises(ValueError):
        logs.verdict(text)


def test_accuracy_boundary_and_job_identity_are_not_inferred_from_unrelated_logs(
    tmp_path,
):
    assert (
        logs.verdict("[vllm_disagg] PASS: exact_match=0.5 count=8 threshold=0.5")[
            "status"
        ]
        == "PASS"
    )
    assert (
        logs.verdict("[vllm_disagg] FAIL: exact_match=0.49 threshold=0.5")["status"]
        == "FAIL"
    )
    submit = tmp_path / "submit.log"
    submit.write_text("Submitted batch job 42\n")
    (tmp_path / "vllm-disagg-pd-41.log").write_text(
        "[vllm_disagg] PASS: exact_match=1 threshold=0.5"
    )
    with pytest.raises(ValueError, match="42"):
        logs.collect(submit, tmp_path, tmp_path / "results")
    submit.write_text("Submitted batch job 42\nSubmitted batch job 43\n")
    with pytest.raises(ValueError, match="exactly one"):
        logs.collect(submit, tmp_path, tmp_path / "results")


@pytest.mark.parametrize(
    "pool,present,count",
    [
        ("n1,n1", ["n1"], 2),
        ("n1", ["n1"], 2),
        ("n1,n2", ["n1"], 2),
        ("n1,n 2", ["n1", "n 2"], 2),
        ("n1,n2", ["n1", "n2"], 3),
    ],
)
def test_invalid_slurm_pool_cannot_borrow_unselected_nodes(pool, present, count):
    with pytest.raises(ValueError):
        cluster.exclusions(pool, present, count)


def test_slurm_restriction_is_precise_and_not_reapplied(tmp_path):
    assert cluster.exclusions("n1,n2", ["n3", "n2", "n1"], 2) == ["n3"]
    script = tmp_path / "run.slurm"
    script.write_text("#!/bin/bash\n#SBATCH --exclusive\nrun\n")
    cluster.restrict_nodes(script, ["n3"])
    assert script.read_text().count("#SBATCH --exclude=n3\n") == 1
    with pytest.raises(ValueError, match="already"):
        cluster.restrict_nodes(script, ["n3"])


def executable(path, body):
    path.write_text("#!" + sys.executable + "\n" + body + "\n")
    path.chmod(0o755)


@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("accounting_failure", [False, True])
def test_real_local_submission_retains_failure_and_cancels_only_its_job(
    tmp_path, monkeypatch, returncode, accounting_failure
):
    source = tmp_path / "source with spaces"
    directory = source / ".buildkite/amd-disagg"
    directory.mkdir(parents=True)
    (directory / "run_xPyD_disagg.slurm").write_text(
        "#!/bin/bash\n#SBATCH --exclusive\n"
    )
    script = directory / "run-slurm-disagg-test.sh"
    script.write_text(
        '#!/bin/bash\nsbatch --job-name="$MODEL_NAME"\nexit ' + str(returncode) + "\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "calls.jsonl"
    executable(binaries / "sinfo", "print('n1\\nn2\\nn3')")
    executable(
        binaries / "sbatch",
        "import os, json, sys\nfrom pathlib import Path\np=Path(os.environ['LOG_ROOT'])\n(p/'vllm-disagg-pd-42.log').write_text('[vllm_disagg] PASS: exact_match=0.75 threshold=0.5')\nwith open(os.environ['CALLS'],'a') as f:f.write(json.dumps({'argv':sys.argv,'model':os.environ['MODEL_NAME']})+'\\n')\nprint('Submitted batch job 42')",
    )
    executable(
        binaries / "scancel",
        "import os,json,sys\nwith open(os.environ['CALLS'],'a') as f:f.write(json.dumps({'cancel':sys.argv[1:]})+'\\n')",
    )
    if accounting_failure:
        executable(binaries / "sacct", "raise SystemExit(9)")
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CALLS", str(calls))
    # Accounting is outside this test's fake cluster; no real Slurm tools run.
    original_which = cluster.shutil.which
    monkeypatch.setattr(
        cluster.shutil,
        "which",
        lambda name: (
            None if name == "sacct" and not accounting_failure else original_which(name)
        ),
    )
    env = {
        **os.environ,
        "CALLS": str(calls),
        "DISAGG_SCRIPTS_STAGE": str(tmp_path / "shared logs"),
        "SPUR_NODE_POOL": "n1,n2",
        "GITHUB_RUN_ID": "unit",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    model = "literal$(touch SHOULD_NOT_EXIST); value"
    case = {
        "case_slug": "bounded",
        "nodes": 2,
        "argv": ["bash", ".buildkite/amd-disagg/run-slurm-disagg-test.sh"],
        "environment": {"MODEL_NAME": model},
    }
    config = {"source": {"revision": "a" * 40}, "router_image": IMAGE}
    output = tmp_path / "execution"
    if returncode:
        with pytest.raises(ValueError, match="Process bash failed"):
            cluster.run(source, case, config, env, output)
    else:
        assert cluster.run(source, case, config, env, output)["status"] == "PASS"
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert records[0]["model"] == model and records[0]["argv"][1:] == [
        "--job-name=" + model
    ]
    assert records[1:] == ([{"cancel": ["42"]}] if returncode else [])
    assert not (source / "SHOULD_NOT_EXIST").exists()
    receipt = json.loads((output / "submission/001-bash.execution.json").read_text())
    assert (
        receipt["returncode"] == returncode
        and not receipt["timed_out"]
        and not receipt["cleanup_problems"]
    )
    assert json.loads((output / "submitted.json").read_text())["job_ids"] == ["42"]
    if accounting_failure:
        diagnostics = list(output.rglob("diagnostics_error.json"))
        assert len(diagnostics) == 1, (
            "Failed optional accounting needs retained diagnostics"
        )
        assert diagnostics[0].read_text()


@pytest.mark.parametrize("member", ["aiter", "aiter/jit", "flydsl"])
def test_overlay_rejects_file_entries_that_replace_implicit_directories(
    tmp_path, member
):
    inputs = tmp_path / "wheels"
    wheel_pair(inputs, extra={member: b"file cannot replace an owned directory"})
    with pytest.raises(ValueError):
        artifacts.validate(inputs)


def test_select_phase_uses_retained_git_commands_and_exports_complete_matrix(
    tmp_path, monkeypatch
):
    from ci.clients.vllm.disaggregation import __main__ as controller

    source = tmp_path / "pinned upstream"
    pipeline = source / selection.configuration()["source"]["pipeline"]
    pipeline.parent.mkdir(parents=True)
    pipeline.write_bytes(PIPELINE.read_bytes())
    binaries = tmp_path / "bin"
    binaries.mkdir()
    revision = selection.configuration()["source"]["revision"]
    executable(
        binaries / "git",
        "import sys\nif sys.argv[1:] == ['rev-parse','HEAD']:print("
        + repr(revision)
        + ")\nelif sys.argv[1:] != ['diff','--exit-code','HEAD','--','.buildkite/amd-disagg']:raise SystemExit(8)",
    )
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    controls = Path(controller.__file__).resolve().parents[4]
    output = tmp_path / "select"
    github_output = tmp_path / "github-output"
    result = controller.execute(
        "select",
        source=source,
        controls=controls,
        output=output,
        env={"BASE_IMAGE": IMAGE, "GITHUB_OUTPUT": str(github_output)},
    )
    assert len(result["matrix"]["include"]) == 10
    assert all(
        "argv" not in row and "environment" not in row
        for row in result["matrix"]["include"]
    )
    assert (
        json.loads(github_output.read_text().removeprefix("matrix="))
        == result["matrix"]
    )
    receipt = json.loads((output / "source/001-rev-parse.execution.json").read_text())
    assert (
        receipt["command"] == ["git", "rev-parse", "HEAD"]
        and receipt["returncode"] == 0
    )
    with pytest.raises(ValueError, match="new and external"):
        controller.execute(
            "select",
            source=source,
            controls=controls,
            output=output,
            env={"BASE_IMAGE": IMAGE},
        )
