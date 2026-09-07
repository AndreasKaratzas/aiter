import copy
import os
import re
import signal
import sys
import tempfile
import time
import unittest
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.clients.vllm.nightly import ROOT, require_platform, resolve, wheel_revision
from ci.common.json import digest, load_json, write_json
from ci.pipelines.nightly import (
    clean_environment,
    execute,
    installation_commands,
    run,
    stages,
    validate_install_report,
    validate_request,
)
from ci.pipelines.process import Process
from ci.qualification.controls import copy_controls


class NightlyResolution(unittest.TestCase):
    def setUp(self):
        self.revision = "a" * 40
        self.filename = "vllm-0.28.1rc1.dev451+gaaaaaaaaa.rocm723-cp312-cp312-manylinux_2_39_x86_64.whl"
        self.url = "https://wheels.vllm.ai/rocm/" + self.revision + "/" + self.filename
        self.pages = {
            ROOT: '<a href="rocm723/">rocm723</a>',
            ROOT + "rocm723/vllm/": f'<a href="{self.url}">wheel</a>',
        }

    def resolution(self):
        return resolve(read=self.pages.__getitem__)

    def test_official_immutable_wheel_keeps_full_commit_and_source_indices(self):
        actual = self.resolution()
        self.assertEqual(actual["revision"], self.revision)
        self.assertEqual(actual["wheel_url"], self.url)
        self.assertEqual(len(actual["indices"]), 2)
        self.assertEqual(
            actual, resolve(variant="rocm723", read=self.pages.__getitem__)
        )

    def test_ambiguous_variant_does_not_guess(self):
        self.pages[ROOT] += '<a href="rocm722/">older</a>'
        with self.assertRaisesRegex(ValueError, "choose one"):
            self.resolution()
        self.assertEqual(
            resolve(variant="rocm723", read=self.pages.__getitem__)["variant"],
            "rocm723",
        )

    def test_cuda_unpinned_off_origin_wrong_python_and_arch_are_rejected(self):
        for url in (
            self.url.replace("rocm723", "cu130"),
            self.url.replace(self.revision, "nightly"),
            self.url.replace("wheels.vllm.ai", "example.invalid"),
            self.url.replace("cp312", "cp313"),
            self.url.replace("x86_64", "aarch64"),
        ):
            with self.subTest(url=url):
                self.pages[ROOT + "rocm723/vllm/"] = f'<a href="{url}">wheel</a>'
                with self.assertRaises(ValueError):
                    self.resolution()

    def test_multiple_wheels_and_unadvertised_stale_variant_fail(self):
        self.pages[ROOT + "rocm723/vllm/"] *= 2
        with self.assertRaises(ValueError):
            self.resolution()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolve(variant="rocm721", read=self.pages.__getitem__)

    def test_insufficient_libc_or_python_fails_before_install(self):
        actual = self.resolution()
        with patch(
            "ci.clients.vllm.nightly.platform.libc_ver", return_value=("glibc", "2.35")
        ), patch(
            "ci.clients.vllm.nightly.sys.version_info", (3, 12)
        ), self.assertRaisesRegex(
            ValueError, "2.39"
        ):
            require_platform(actual)
        with patch(
            "ci.clients.vllm.nightly.sys.version_info", (3, 11)
        ), self.assertRaisesRegex(ValueError, "Python 3.12"):
            require_platform(actual)

    def test_short_version_suffix_or_untrusted_host_is_not_source_identity(self):
        self.assertEqual(wheel_revision(self.url), self.revision)
        self.assertIsNone(
            wheel_revision(self.url.replace(self.revision, self.revision[:9]))
        )
        self.assertIsNone(wheel_revision(self.url.replace("https:", "http:")))
        self.assertIsNone(
            wheel_revision(
                self.url.replace("wheels.vllm.ai", "wheels.vllm.ai.evil.invalid")
            )
        )

    def test_installs_candidate_last_without_dependency_resolution_then_checks(self):
        commands = installation_commands(
            self.resolution(),
            Path("/artifacts/candidate.whl"),
            Path("/control"),
            Path("/out"),
        )
        self.assertIn(self.url, commands[0])
        self.assertIn("--report", commands[0])
        self.assertEqual(commands[1][-1], "/artifacts/candidate.whl")
        self.assertIn("--no-deps", commands[1])
        self.assertEqual(commands[2], ["-m", "pip", "check"])
        self.assertTrue(
            all("/workspace" not in part for command in commands for part in command)
        )

    def test_unhashed_dependency_or_pypi_fallback_cannot_be_certified(self):
        report = {
            "version": "1",
            "install": [
                {
                    "metadata": {"name": "vllm"},
                    "download_info": {
                        "url": self.url,
                        "archive_info": {"hashes": {"sha256": "b" * 64}},
                    },
                }
            ],
        }
        validate_install_report(report, self.resolution())
        for change in ("hash", "url", "duplicate"):
            bad = copy.deepcopy(report)
            if change == "hash":
                bad["install"][0]["download_info"]["archive_info"] = {}
            elif change == "url":
                bad["install"][0]["download_info"]["url"] = "https://pypi.org/cuda.whl"
            else:
                bad["install"] *= 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_install_report(bad, self.resolution())

    def test_ambient_python_pip_and_native_redirects_do_not_enter_venv(self):
        hostile = {
            name: "hostile"
            for name in (
                "PYTHONPATH",
                "PYTHONOPTIMIZE",
                "PIP_EXTRA_INDEX_URL",
                "UV_INDEX_URL",
                "PYTEST_ADDOPTS",
                "AITER_ASM_DIR",
                "VLLM_ROCM_USE_AITER",
                "LD_PRELOAD",
            )
        }
        with patch.dict(os.environ, hostile, clear=True):
            environment = clean_environment(Path("/control"), Path("/out"))
        self.assertEqual(environment["PYTHONPATH"], "/control")
        self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
        self.assertFalse(any(value == "hostile" for value in environment.values()))


class NightlyFailureHistory(unittest.TestCase):
    def test_real_failed_installer_keeps_failure_and_never_starts_import_or_models(
        self,
    ):
        self.failed_attempt("install")

    def test_real_import_failure_blocks_every_operator_and_model(self):
        self.failed_attempt("import")

    def failed_attempt(self, stage):
        class FixtureProcess(Process):
            def command(self, arguments, **kwargs):
                # Fixture commands are real Python failures; no pip installation is exercised here.
                if arguments[:2] == ["-m", "venv"]:
                    arguments = [*arguments, "--without-pip"]
                return super().command(arguments, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls, source, output, artifacts = (
                root / name for name in ("control", "source", "evidence", "artifacts")
            )
            for directory in (controls, source, output, artifacts):
                directory.mkdir()
            (controls / "ci/qualification").mkdir(parents=True)
            (controls / "ci/__init__.py").touch()
            (controls / "ci/qualification/__init__.py").touch()
            (controls / "ci/qualification/nightly.py").write_text(
                'raise ImportError("missing vllm public API")\n'
            )
            write_json(output / "consumer-install.json", {})
            identity = {"revision": "a" * 40}
            artifact = {"filename": "candidate.whl", "sha256": "b" * 64}
            receipt = SimpleNamespace(
                source=SimpleNamespace(to_dict=lambda: identity),
                artifact=SimpleNamespace(to_dict=lambda: artifact),
            )
            request = {
                "schema_version": 2,
                "through": "workloads",
                "workload_profile": "vllm-nightly",
                "candidate_substitution": {
                    "schema_version": 1,
                    "classification": "rolling-candidate-only",
                    "consumer": "vllm",
                    "dependency": "amd-aiter",
                    "module": "aiter",
                },
                "source": identity,
                "control_source": identity,
                "artifact": artifact,
                "architecture": "gfx950",
                "gpus": "0,1",
                "variant": None,
                "executor_image": None,
            }
            request["request_digest"] = digest(request)
            (controls / "ci/clients/vllm").mkdir(parents=True)
            write_json(
                controls / "ci/clients/vllm/nightly.json",
                request["candidate_substitution"],
            )
            write_json(output / "request.json", request)
            with patch("ci.pipelines.nightly.Process", FixtureProcess), patch(
                "ci.pipelines.nightly.collect_source_identity",
                return_value=receipt.source,
            ), patch("ci.pipelines.nightly.load_receipt", return_value=receipt), patch(
                "ci.pipelines.nightly.verify_wheel"
            ), patch(
                "ci.pipelines.nightly.resolve", return_value={}
            ), patch(
                "ci.pipelines.nightly.require_platform"
            ), patch(
                "ci.pipelines.nightly.preflight"
            ), patch(
                "ci.pipelines.nightly.installation_commands",
                return_value=[
                    [
                        "-c",
                        (
                            'raise SystemExit("incompatible nightly dependency")'
                            if stage == "install"
                            else 'print("fixture installed")'
                        ),
                    ]
                ],
            ), patch(
                "ci.pipelines.nightly.validate_install_report"
            ), patch(
                "ci.pipelines.nightly.run_plan"
            ) as workloads, patch(
                "ci.pipelines.nightly.provision_models"
            ) as models, self.assertRaisesRegex(
                ValueError, "failed"
            ):
                execute(
                    output / "request.json",
                    controls=controls,
                    source=source,
                    artifacts=artifacts,
                )
            self.assertEqual(load_json(output / "status.json")["stage"], stage)
            self.assertEqual(load_json(output / "status.json")["status"], "FAIL")
            records = sorted((output / "installation/venv").glob("*.execution.json"))
            self.assertEqual(len(records), 1 if stage == "install" else 2)
            self.assertNotEqual(load_json(records[-1])["returncode"], 0)
            self.assertIn(
                (
                    "incompatible nightly dependency"
                    if stage == "install"
                    else "missing vllm public API"
                ),
                "\n".join(
                    path.read_text()
                    for path in (output / "installation/venv").glob("*.log")
                ),
            )
            workloads.assert_not_called()
            models.assert_not_called()
            self.assertFalse((output / "import.json").exists())

    def test_workflow_download_matches_the_builders_python312_artifact(self):
        import fnmatch

        root = Path(__file__).resolve().parents[3]
        workflow = (root / ".github/workflows/client-vllm-nightly.yaml").read_text()
        builder = (root / ".github/workflows/release-build-wheels.yaml").read_text()
        downloads = re.findall(
            r"uses: actions/download-artifact@([^\s]+)[^\n]*\n\s+with:\n\s+pattern: ([^\n]+)",
            workflow,
        )
        self.assertEqual(len(downloads), 1)
        self.assertRegex(downloads[0][0], r"^[a-f0-9]{40}$")
        self.assertIn("aiter-whl-packages-py${{ matrix.python_version }}", builder)
        self.assertTrue(
            fnmatch.fnmatch("aiter-whl-packages-py3.12-20260906", downloads[0][1])
        )
        self.assertFalse(
            fnmatch.fnmatch("aiter-whl-packages-py3.10-20260906", downloads[0][1])
        )

    def test_request_rejects_duplicate_gpu_alias_and_path_escape(self):
        request = {
            "schema_version": 2,
            "through": "workloads",
            "workload_profile": "vllm-nightly",
            "candidate_substitution": {
                "schema_version": 1,
                "classification": "rolling-candidate-only",
                "consumer": "vllm",
                "dependency": "amd-aiter",
                "module": "aiter",
            },
            "source": {},
            "control_source": {},
            "artifact": {"filename": "candidate.whl"},
            "architecture": "gfx950",
            "gpus": "0,1",
            "variant": None,
            "executor_image": None,
        }
        for key, value in (
            ("gpus", "0,00"),
            ("artifact", {"filename": "../other.whl"}),
            ("schema_version", True),
            ("through", "none"),
            ("workload_profile", "product-fast"),
        ):
            bad = dict(request, **{key: value})
            bad["request_digest"] = digest(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(bad)


class NightlyStageScope(unittest.TestCase):
    def test_extended_scope_is_explicit_and_daily_remains_bounded(self):
        self.assertEqual(
            stages({"through": "workloads", "workload_profile": "vllm-extended"}),
            [("vllm-import", "imports"), ("vllm-extended", "workloads")],
        )
        root = Path(__file__).resolve().parents[3]
        workflow = (root / ".github/workflows/client-vllm-nightly.yaml").read_text()
        self.assertIn("cron: '15 20 * * 0'", workflow)
        self.assertIn('--workload-profile "$WORKLOAD_PROFILE"', workflow)

    def test_import_diagnostic_cannot_include_model_stage(self):
        self.assertEqual(stages({"through": "imports"}), [("vllm-import", "imports")])
        self.assertEqual(
            stages({"through": "workloads", "workload_profile": "vllm-nightly"}),
            [("vllm-import", "imports"), ("vllm-nightly", "workloads")],
        )

    def test_missing_candidate_receipt_is_retained_without_inventing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("source", "controls"):
                (root / name).mkdir()
            with self.assertRaises(FileNotFoundError):
                run(
                    source=root / "source",
                    controls=root / "controls",
                    wheel=root / "missing.whl",
                    output=root / "out",
                    image=None,
                    variant=None,
                    gpus="0,1",
                )
            record = load_json(root / "out/pipeline-status.json")
            self.assertEqual(record["status"], "FAIL")
            self.assertEqual(record["stage"], "admission")
            self.assertNotIn("artifact", record)


class NightlyInterpreterBoundary(unittest.TestCase):
    def test_copied_bootstrap_imports_installed_payload_without_checkout_shadowing(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls = root / "controls"
            (controls / "ci").mkdir(parents=True)
            (controls / "ci/__init__.py").touch()
            (controls / "aiter").mkdir()
            (controls / "aiter/__init__.py").write_text(
                'raise RuntimeError("checkout AITER imported")\n'
            )
            (controls / "sitecustomize.py").write_text(
                f"from pathlib import Path\nPath({str(root / 'poisoned')!r}).touch()\n"
            )
            (controls / "ci/observe.py").write_text(
                "import aiter, json, sys\nfrom pathlib import Path\n"
                "assert aiter.VALUE == 42\n"
                "assert Path(aiter.__file__).is_relative_to(Path(sys.prefix))\n"
                "print(json.dumps({'origin': aiter.__file__}))\n"
            )
            private = root / "private"
            venv.EnvBuilder(with_pip=False).create(private)
            installed = (
                private
                / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/aiter"
            )
            installed.mkdir()
            (installed / "__init__.py").write_text("VALUE = 42\n")
            bootstrap = copy_controls(controls, root / "suite")
            environment = clean_environment(bootstrap, root / "out")
            result = Process(
                root / "evidence", executable=str(private / "bin/python")
            ).command(["-m", "ci.observe"], cwd=bootstrap, env=environment, timeout=30)
            self.assertIn(str(installed), result)
            self.assertFalse((root / "poisoned").exists())
            self.assertFalse((bootstrap / "aiter").exists())
            self.assertFalse((bootstrap / "sitecustomize.py").exists())

    def test_timeout_kills_term_ignoring_descendant_after_leader_exits(self):
        self.assert_descendant_cleanup(None)

    def test_successful_leader_cannot_leave_an_active_background_writer(self):
        self.assert_descendant_cleanup(0)

    def test_failed_leader_cannot_leave_an_active_background_writer(self):
        self.assert_descendant_cleanup(1)

    def assert_descendant_cleanup(self, exit_code):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "process.py"
            script.write_text(
                "import os, signal, time\nfrom pathlib import Path\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                f"    Path({str(root / 'child')!r}).write_text(str(os.getpid()))\n"
                "    while True: time.sleep(1)\n"
                + (
                    "while True: time.sleep(1)\n"
                    if exit_code is None
                    else f"while not Path({str(root / 'child')!r}).exists(): time.sleep(0.01)\nraise SystemExit({exit_code})\n"
                )
            )
            try:
                process = Process(root / "evidence", executable=sys.executable)
                if exit_code == 0:
                    process.command(["-c", script.read_text()], timeout=1)
                else:
                    with self.assertRaisesRegex(ValueError, "failed"):
                        process.command(["-c", script.read_text()], timeout=1)
                pid = int((root / "child").read_text())
                state = Path(f"/proc/{pid}/stat")
                deadline = time.monotonic() + 2
                while (
                    state.exists()
                    and state.read_text().split()[2] != "Z"
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                if state.exists():
                    self.assertEqual(state.read_text().split()[2], "Z")
                record = next((root / "evidence").glob("*.execution.json"))
                self.assertEqual(load_json(record)["timed_out"], exit_code is None)
                if exit_code is not None:
                    self.assertEqual(load_json(record)["returncode"], exit_code)
            finally:
                if (root / "child").exists():
                    try:
                        os.kill(int((root / "child").read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
