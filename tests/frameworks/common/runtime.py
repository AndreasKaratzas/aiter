# SPDX-License-Identifier: MIT
"""Shared prerequisites, reference math and call tracing for framework adapters."""

from importlib.util import find_spec

from common.paths import assert_package_origin


def require_framework(name: str) -> None:
    assert find_spec(name) is not None, f"The {name} profile requires {name} installed."


def rocm():
    import torch

    assert torch.cuda.is_available(), "This group requires a ROCm GPU."
    assert torch.version.hip, "This group qualifies ROCm execution."
    assert_package_origin()
    return torch


def rmsnorm_reference(x, weight, epsilon, *, dtype=None):
    import torch

    values = x.float()
    return (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
        * weight.float()
    ).to(dtype or x.dtype)


def trace_call(monkeypatch, owner, attribute):
    """Observe a real adapter call without replacing its computation."""
    operation = getattr(owner, attribute)
    calls = []

    def observed(*args, **kwargs):
        calls.append(attribute)
        return operation(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, observed)
    return calls


def trace_triton_kernel(monkeypatch, name):
    """Observe the leaf launch across AITER's historical module aliases."""
    from triton.runtime.jit import JITFunction

    operation = JITFunction.run
    calls = []

    def observed(kernel, *args, **kwargs):
        if kernel.fn.__name__ == name:
            calls.append(name)
        return operation(kernel, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", observed)
    return calls
