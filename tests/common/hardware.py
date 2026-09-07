# SPDX-License-Identifier: MIT
"""Lazy GPU observations. Importing this module never imports Torch."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Device:
    index: int
    architecture: str
    memory_bytes: int


@dataclass(frozen=True)
class Hardware:
    devices: tuple[Device, ...] = ()
    rocm_version: str | None = None
    unavailable: str | None = None

    def supports(self, capability: str) -> bool:
        """Hardware eligibility only; tests still verify the actual operation."""
        if capability not in CAPABILITIES:
            raise ValueError(f"Unknown GPU capability: {capability}")
        return bool(self.devices) and all(
            device.architecture in CAPABILITIES[capability] for device in self.devices
        )


CAPABILITIES = {
    "bf16": frozenset(("gfx90a", "gfx942", "gfx950")),
    "fp8": frozenset(("gfx942", "gfx950")),
    "mxfp4": frozenset(("gfx950",)),
}


@lru_cache(maxsize=1)
def observe() -> Hardware:
    """Observe visible devices at test execution time, after runner isolation."""
    try:
        import torch

        if not torch.version.hip:
            return Hardware(unavailable="The selected Torch build is not ROCm.")
        devices = tuple(
            Device(index, properties.gcnArchName.split(":")[0], properties.total_memory)
            for index in range(torch.cuda.device_count())
            for properties in (torch.cuda.get_device_properties(index),)
        )
        return Hardware(
            devices,
            torch.version.hip,
            None if devices else "No ROCm GPU is visible to this test process.",
        )
    except (ImportError, RuntimeError, OSError) as error:
        return Hardware(unavailable=f"ROCm prerequisites unavailable: {error}")
