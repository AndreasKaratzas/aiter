# SPDX-License-Identifier: MIT
"""Tracing must preserve callable contracts inspected by framework adapters."""

import inspect
import sys
from types import SimpleNamespace

import pytest

from frameworks.common.runtime import trace_call, trace_triton_kernel


def test_trace_retains_signature_return_and_wrapped_identity(monkeypatch):
    sentinel = object()

    def operation(value, *, q_scale=None, kv_scale=None):
        return value, q_scale, kv_scale

    owner = SimpleNamespace(operation=operation)
    calls = trace_call(monkeypatch, owner, "operation")
    assert inspect.signature(owner.operation) == inspect.signature(operation)
    assert inspect.unwrap(owner.operation) is operation
    assert owner.operation(sentinel, q_scale=0.75, kv_scale=1.25) == (
        sentinel,
        0.75,
        1.25,
    )
    assert calls == ["operation"]


def test_trace_preserves_original_exception(monkeypatch):
    failure = RuntimeError("original operation failed")

    def operation(*, value):
        raise failure

    owner = SimpleNamespace(operation=operation)
    calls = trace_call(monkeypatch, owner, "operation")
    with pytest.raises(RuntimeError) as caught:
        owner.operation(value=3)
    assert caught.value is failure
    assert calls == [], "A failed operation must not count as successful AITER use."


def test_triton_trace_counts_only_successful_execution(monkeypatch):
    failure = RuntimeError("original launch failed")
    result = object()

    class JITFunction:
        def run(self, *, warmup=False, fail=False):
            if fail:
                raise failure
            return result

    original = JITFunction.run
    module = SimpleNamespace(JITFunction=JITFunction)
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", module)
    kernel = JITFunction()
    kernel.fn = SimpleNamespace(__name__="selected_leaf")
    calls = trace_triton_kernel(monkeypatch, "selected_leaf")
    assert inspect.signature(JITFunction.run) == inspect.signature(original)
    assert inspect.unwrap(JITFunction.run) is original
    with pytest.raises(RuntimeError) as caught:
        kernel.run(fail=True)
    assert caught.value is failure and calls == []
    assert kernel.run(warmup=True) is result and calls == []
    assert kernel.run() is result and calls == ["selected_leaf"]
