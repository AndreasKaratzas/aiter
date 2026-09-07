# SPDX-License-Identifier: MIT
"""Hardware eligibility must not turn a mixed or unknown device set into support."""

import unittest

from common.hardware import Device, Hardware


class ArchitectureEligibilityTests(unittest.TestCase):
    def test_architecture_capabilities(self):
        expected = {
            "gfx942": {"bf16", "fp8"},
            "gfx950": {"bf16", "fp8", "mxfp4"},
            "gfx1250": {"bf16", "fp8", "mxfp4"},
            "gfx1201": set(),
            "unknown": set(),
        }
        for architecture, capabilities in expected.items():
            hardware = Hardware((Device(0, architecture, 16 * 2**30),), "observed")
            for capability in ("bf16", "fp8", "mxfp4"):
                with self.subTest(architecture=architecture, capability=capability):
                    self.assertEqual(
                        hardware.supports(capability), capability in capabilities
                    )

    def test_every_selected_device_must_be_eligible(self):
        devices = (Device(0, "gfx950", 16 * 2**30), Device(1, "gfx942", 16 * 2**30))
        self.assertFalse(Hardware(devices).supports("mxfp4"))
        self.assertTrue(Hardware(devices).supports("fp8"))
        self.assertFalse(Hardware().supports("fp8"))
        with self.assertRaisesRegex(ValueError, "Unknown GPU capability"):
            Hardware(devices).supports("fp4-every-backend")
