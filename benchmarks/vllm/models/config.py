"""Explicit bounded model benchmark workload configuration."""

from dataclasses import dataclass
from pathlib import Path


def default_manifest():
    return Path(__file__).resolve().parents[3] / "ci/clients/vllm/models.json"


@dataclass(frozen=True)
class Workload:
    batch_size: int = 2
    input_tokens: int = 128
    output_tokens: int = 64
    warmup: int = 2
    repeats: int = 9
    seed: int = 101

    def validate(self):
        limits = {
            "batch_size": (1, 32),
            "input_tokens": (1, 4096),
            "output_tokens": (1, 1024),
            "warmup": (1, 100),
            "repeats": (3, 1000),
            "seed": (0, 2**32 - 1),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
