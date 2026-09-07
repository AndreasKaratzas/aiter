"""The selected interpreter's real extension include paths must compile before installation."""

import os
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from ci.common.json import load_json, write_json
from ci.pipelines.toolchain import preflight, verify_preflight


class PythonDevelopmentPreflight(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.include = self.root / "include"
        self.include.mkdir()
        self.platform = self.root / "platform"
        self.platform.mkdir()
        stack.enter_context(
            patch(
                "ci.pipelines.toolchain.sysconfig.get_path",
                side_effect=lambda name: str(
                    self.include if name == "include" else self.platform
                ),
            )
        )
        self.header = self.include / "Python.h"
        self.header.write_text('#include "pyconfig.h"\n')
        (self.platform / "pyconfig.h").write_text(
            f"#define PY_MAJOR_VERSION {sys.version_info.major}\n#define PY_MINOR_VERSION {sys.version_info.minor}\n"
        )

    def output(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_existing_includepy_cannot_hide_missing_actual_extension_headers(self):
        self.header.unlink()
        with patch(
            "ci.pipelines.toolchain.sysconfig.get_config_var",
            return_value="/usr/include/python3.12",
        ), self.assertRaisesRegex(ValueError, "headers missing"):
            preflight(self.output("missing"), dict(os.environ))

    def test_real_compiler_checks_transitive_platform_headers_and_version(self):
        output = self.output("accepted")
        record = preflight(output, dict(os.environ))
        self.assertEqual(record["include_paths"]["platinclude"], str(self.platform))
        self.assertEqual(verify_preflight(output), record)
        (self.platform / "pyconfig.h").unlink()
        with self.assertRaisesRegex(ValueError, "failed"):
            preflight(self.output("missing-config"), dict(os.environ))
        (self.platform / "pyconfig.h").write_text(
            "#define PY_MAJOR_VERSION 2\n#define PY_MINOR_VERSION 7\n"
        )
        with self.assertRaisesRegex(ValueError, "failed"):
            preflight(self.output("wrong-version"), dict(os.environ))

    def test_reconstruction_rejects_cleanup_failure_and_changed_header(self):
        output = self.output("accepted")
        preflight(output, dict(os.environ))
        path = output / "preflight/001--std=c++17.execution.json"
        execution = load_json(path)
        write_json(path, dict(execution, cleanup_problems=["unreaped compiler child"]))
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            verify_preflight(output)
        write_json(path, execution)
        self.header.write_text("#error changed header\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_preflight(output, live=True)

    def test_relocated_container_evidence_does_not_require_host_toolchain_paths(self):
        output = self.output("executor")
        record = preflight(output, dict(os.environ))
        relocated = self.root / "downloaded"
        shutil.copytree(output, relocated)
        self.header.unlink()
        shutil.rmtree(output)
        self.assertEqual(verify_preflight(relocated), record)
        (relocated / "preflight/Python.h").write_text("modified retained header")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_preflight(relocated)
