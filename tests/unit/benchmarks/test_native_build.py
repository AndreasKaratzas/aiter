# SPDX-License-Identifier: MIT
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiter.codegen import BuildContext
from benchmarks.native.mha import build
from benchmarks.native.mha.build import host_command, main


class NativeBenchmarkBuildTests(unittest.TestCase):
    def test_build_output_and_native_resources_are_explicit(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        native = Path(directory.name) / "native"
        ck = Path(directory.name) / "ck"
        native.mkdir()
        ck.mkdir()
        context = BuildContext.load(overrides={"native": native, "ck": ck})
        output = Path("/separate build/output")
        command = host_command(
            "bwd_v3", output, context, Path("/rocm/bin/hipcc"), "gfx950"
        )
        self.assertIn(f"-I{native}/include", command)
        self.assertNotIn(f"-I{ck}/include", command)
        self.assertEqual(command[command.index("-L") + 1], str(output))
        self.assertEqual(command[-1], str(output / "bwd.exe"))
        self.assertIn("-Wl,-rpath,$ORIGIN", command)
        forward = host_command(
            "fwd", output, context, Path("/rocm/bin/hipcc"), "gfx950"
        )
        self.assertIn(f"-I{ck}/include", forward)
        self.assertIn("-DCK_TILE_FMHA_FWD_SPLITKV_API=1", forward)

    def test_previous_build_cannot_supply_successful_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as caught:
                main(["bwd_v3", "--output", directory])
            self.assertEqual(caught.exception.code, 2)

    def test_successful_exit_requires_every_nonempty_requested_artifact(self):
        for contents in ({}, {"libmha_bwd.so": b"library", "bwd.exe": b""}):
            with self.subTest(
                contents=contents
            ), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "build"

                def run(command, *, contents=contents, output=output, **options):
                    for name, data in contents.items():
                        (output / name).write_bytes(data)

                with (
                    patch.object(build.subprocess, "run", side_effect=run),
                    self.assertRaisesRegex(RuntimeError, "nonempty artifact"),
                ):
                    main(["bwd_v3", "--output", str(output)])
                self.assertEqual(
                    json.loads((output / "build.json").read_text())["status"], "failed"
                )

    def test_failed_compiler_retains_its_log_and_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "build"

            def run(command, **options):
                options["stdout"].write("compiler failed with a useful diagnostic\n")
                raise subprocess.CalledProcessError(3, command)

            with (
                patch.object(build.subprocess, "run", side_effect=run),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                main(["bwd_v3", "--output", str(output)])
            self.assertIn("useful diagnostic", (output / "build-0.log").read_text())
            self.assertEqual(
                json.loads((output / "build.json").read_text())["status"], "failed"
            )

    def test_copied_application_uses_candidate_and_private_cache_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory(prefix="native builder ") as directory:
            root = Path(directory)
            application = root / "application"
            candidate = root / "candidate"
            package = candidate / "aiter"
            (package / "codegen").mkdir(parents=True)
            (package / "jit").mkdir()
            application.mkdir()
            selected = BuildContext.load()
            for name in ("build.py", "compile.py"):
                shutil.copy(Path(build.__file__).with_name(name), application / name)
            (package / "__init__.py").write_text("")
            (package / "jit/__init__.py").write_text("")
            shutil.copy(
                selected.package / "codegen/context.py", package / "codegen/context.py"
            )
            (package / "codegen/__init__.py").write_text(
                "from .context import BuildContext\n"
            )
            (package / "_build_layout.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "installed",
                        "resources": {
                            name: os.path.relpath(path, package)
                            for name, path in selected.resources.items()
                        },
                    }
                )
            )
            # A controlled compiler double observes the real subprocess boundary;
            # numerical native compilation is covered by the GPU integration test.
            (package / "jit/core.py").write_text("""import json, os
from pathlib import Path
def get_user_jit_dir():
    return os.environ["AITER_JIT_DIR"]
def get_args_of_build(name):
    return {"srcs": [], "flags_extra_cc": [], "flags_extra_hip": [], "blob_gen_cmd": [],
            "extra_include": [], "extra_ldflags": [], "verbose": False,
            "third_party": [], "flags_extra_hip_per_source": {}}
def build_module(**recipe):
    cache = Path(get_user_jit_dir())
    (cache / (recipe["md_name"] + ".so")).write_bytes(b"new library")
    print(json.dumps({"architecture": os.environ["GPU_ARCHS"], "cache": str(cache),
                      "package": __file__, "pythonpath": os.environ["PYTHONPATH"]}))
""")
            compiler = root / "compiler"
            compiler.write_text(
                f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\nPath(sys.argv[-1]).write_bytes(b'new executable')\n"
            )
            compiler.chmod(0o755)
            output = root / "output"
            env = dict(
                os.environ,
                PYTHONPATH=str(candidate),
                GPU_ARCHS="gfx942",
                AITER_JIT_DIR=str(root / "unrelated cache"),
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(application / "build.py"),
                    "bwd_v3",
                    "--output",
                    str(output),
                    "--architecture",
                    "gfx950",
                    "--compiler",
                    str(compiler),
                ],
                env=env,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                result.stdout
                + result.stderr
                + "\n".join(path.read_text() for path in output.glob("*.log")),
            )
            observation = json.loads((output / "build-0.log").read_text())
            self.assertEqual(observation["architecture"], "gfx950")
            self.assertEqual(observation["cache"], str(output / ".jit"))
            self.assertEqual(observation["pythonpath"], str(candidate))
            self.assertTrue(observation["package"].startswith(str(package)))
            self.assertFalse((root / "unrelated cache").exists())
            receipt = json.loads((output / "build.json").read_text())
            self.assertEqual(receipt["status"], "built")
            self.assertEqual(set(receipt["artifacts"]), {"libmha_bwd.so", "bwd.exe"})
            retry = subprocess.run(
                [
                    sys.executable,
                    str(application / "compile.py"),
                    "--api",
                    "bwd_v3",
                    "--output",
                    str(output),
                ],
                env=env,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(retry.returncode, 2)
            self.assertIn("already contains a library build", retry.stderr)
