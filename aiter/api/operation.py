# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The descriptor interface implemented by each operator domain."""

from typing import Protocol, runtime_checkable

from .tensor import TensorSpec


@runtime_checkable
class Operation(Protocol):
    """Describe named reads and writes independently of execution providers."""

    def inputs(self) -> dict[str, TensorSpec]: ...
    def outputs(self) -> dict[str, TensorSpec]: ...
    def to_dict(self) -> dict: ...
    def fingerprint(self) -> str: ...
