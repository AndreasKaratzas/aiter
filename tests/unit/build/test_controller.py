# SPDX-License-Identifier: MIT
import multiprocessing
import os
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import Mock, patch

from build_backend.controller import BuildController
from build_backend.options import BuildOptions
from build_backend.plan import BuildPlan, NativeBuildJob, selected_modules


class ControllerTests(unittest.TestCase):
    def test_named_selection_retains_core_and_rejects_missing_dependencies(self):
        profiles = {"module_opaque": (), "module_aiter_core": (), "module_other": (1,)}
        options = BuildOptions(False, True, 0, ("module_opaque",))
        self.assertEqual(
            selected_modules(profiles, {"module_opaque"}, options),
            ("module_opaque", "module_aiter_core"),
        )
        for modules, ck in ((("module_unknown",), True), (("module_opaque",), False)):
            with self.subTest(modules=modules, ck=ck), self.assertRaises(ValueError):
                selected_modules(
                    profiles, {"module_opaque"}, BuildOptions(False, ck, 0, modules)
                )

    def test_named_prebuild_is_native_only_and_configuration_is_unambiguous(self):
        with patch.dict(os.environ, {"PREBUILD_MODULES": "module_demo"}, clear=True):
            options = BuildOptions.from_environment()
            self.assertTrue(options.builds_kernels)
            self.assertEqual(options.modules, ("module_demo",))
        for settings in (
            {"PREBUILD_MODULES": "module_demo,module_demo"},
            {"PREBUILD_MODULES": "module_demo,"},
            {"PREBUILD_MODULES": "module_demo", "PREBUILD_KERNELS": "1"},
            {"PREBUILD_MODULES": "module_demo", "AITER_TRITON_ONLY": "1"},
        ):
            with self.subTest(settings=settings), patch.dict(
                os.environ, settings, clear=True
            ), self.assertRaises(ValueError):
                BuildOptions.from_environment()
        phases = Mock()
        plan = BuildPlan(0, (self.job(),), 2, 1, 1, requested_modules=("module_demo",))
        BuildController(phases).execute(plan)
        phases.initialize.assert_called_once_with(plan)
        phases.native.assert_called_once_with(self.job(), plan)
        phases.flydsl.assert_not_called()
        phases.pretune.assert_not_called()

    def job(self):
        return NativeBuildJob.from_arguments(
            {"md_name": "module_demo", "srcs": ["native.cu"]}
        )

    def test_ck_policy_and_mode_selection_are_intersected(self):
        profiles = {
            "module_opaque": (2, 3),
            "module_ck": (2, 3),
            "module_unselected": (1,),
        }
        self.assertEqual(
            selected_modules(profiles, {"module_ck"}, BuildOptions(False, False, 3)),
            ("module_opaque",),
        )
        self.assertEqual(
            selected_modules(profiles, {"module_ck"}, BuildOptions(False, True, 2)),
            ("module_opaque", "module_ck"),
        )

    def test_job_identity_and_plan_types_cannot_disagree_with_execution(self):
        with self.assertRaises(ValueError):
            NativeBuildJob("module_first", '{"md_name":"module_second"}')
        with self.assertRaises(ValueError):
            NativeBuildJob(
                "module_first", '{"md_name":"module_first","md_name":"module_first"}'
            )
        with self.assertRaises(ValueError):
            BuildPlan(True, (), 1, 1, 1)
        with self.assertRaises(ValueError):
            BuildPlan(1, (self.job(), self.job()), 1, 1, 1)

    def test_adapters_receive_fresh_arguments_and_jobs_survive_spawn(self):
        job = self.job()
        job.arguments()["srcs"].clear()
        self.assertEqual(job.arguments()["srcs"], ["native.cu"])
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as workers:
            self.assertEqual(
                workers.submit(job.arguments).result(timeout=20), job.arguments()
            )

    def test_failed_phase_never_reaches_tuning(self):
        for failed in ("flydsl", "native", None):
            events = []

            class Phases:
                def __init__(self, failure, events):
                    self.failure = failure
                    self.events = events

                def record(self, name):
                    self.events.append(name)
                    if self.failure == name:
                        raise RuntimeError(name)

                def initialize(self, plan):
                    self.record("initialize")

                def flydsl(self, plan):
                    self.record("flydsl")

                def native(self, job, plan):
                    self.record("native")

                def pretune(self, plan):
                    self.record("pretune")

            controller = BuildController(Phases(failed, events))
            plan = BuildPlan(1, (self.job(),), 1, 1, 1, "gemm")
            if failed:
                with self.assertRaisesRegex(RuntimeError, failed):
                    controller.execute(plan)
                self.assertNotIn("pretune", events)
            else:
                controller.execute(plan)
                self.assertEqual(events, ["initialize", "flydsl", "native", "pretune"])
