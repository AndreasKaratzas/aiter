# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Resolve support and prepare an implementation before execution begins."""

import importlib.metadata
import platform

from .._validation import ValidationError, canonical_digest
from ..api import (
    BlockScaleQuantize,
    DenseAttention,
    FP8BlockScaleGemm,
    MXFP4Gemm,
    MXFP4Quantize,
    RMSNorm,
    RotaryEmbedding,
)
from ..tuning import DispatchManifest
from .plan import ExecutionPlan, expected_bindings, tensor_spec, validate_bindings
from .provider import ExecutionPolicy, UnsupportedOperation, backend_registry


class Runtime:
    """A device-bound planner with explicit backend and compilation policy."""

    def __init__(self, device=0, *, policy=None, manifest=None, backends=None):
        if type(device) is not int or device < 0:
            raise ValidationError("device must be a nonnegative integer")
        if backends is None:
            from .composition import default_backends

            backends = default_backends()
        registry = backend_registry(backends)
        if policy is None:
            policy = ExecutionPolicy(backend_order=tuple(registry))
        if not isinstance(policy, ExecutionPolicy):
            raise ValidationError("policy must be an ExecutionPolicy")
        unknown = set(policy.backend_order) - set(registry)
        if unknown:
            raise ValidationError(
                f"policy names unavailable backends: {sorted(unknown)}"
            )
        if manifest is not None and not isinstance(manifest, DispatchManifest):
            raise ValidationError("manifest must be a DispatchManifest")
        import torch

        if not torch.version.hip or not torch.cuda.is_available():
            raise UnsupportedOperation("prepared AITER execution requires a ROCm GPU")
        if device >= torch.cuda.device_count():
            raise ValidationError("device index is unavailable")
        self._device = device
        self._policy = policy
        properties = torch.cuda.get_device_properties(device)
        self._target = properties.gcnArchName.split(":")[0]
        packages = {}
        for package in ("triton", "flydsl"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        self._environment = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "packages": packages,
            "gpu": {
                "architecture": properties.gcnArchName,
                "name": properties.name,
                "compute_units": properties.multi_processor_count,
                "memory_bytes": properties.total_memory,
            },
        }
        self._environment_digest = canonical_digest(self._environment)
        if (
            manifest is not None
            and manifest.environment_digest != self.environment_digest
        ):
            raise UnsupportedOperation(
                "dispatch manifest was qualified in a different runtime environment"
            )
        self._manifest = manifest
        self._torch = torch
        self._backends = registry

    @property
    def device(self):
        return self._device

    @property
    def target(self):
        return self._target

    @property
    def policy(self):
        return self._policy

    @property
    def environment_digest(self):
        return self._environment_digest

    @property
    def environment(self):
        """Copy of the device and dependency observations used to bind tuning."""
        import copy

        return copy.deepcopy(self._environment)

    def capabilities(self, operation):
        """Return support decisions without compiling or launching a kernel."""
        expected_bindings(operation)
        return {
            name: {"supported": result.supported, "reason": result.reason}
            for name, provider in self._backends.items()
            for result in (provider.supports(operation, self.target),)
        }

    def prepare(self, operation, bindings, *, backend=None):
        specs = expected_bindings(operation)
        validate_bindings(
            bindings, specs, self.device, self._torch, operation.outputs()
        )
        selection = None
        if self._manifest is not None:
            selection = self._manifest.select(operation.fingerprint(), self.target)
            if backend is not None and backend != selection.backend:
                raise UnsupportedOperation(
                    "requested backend conflicts with the pinned dispatch manifest"
                )
            backend = selection.backend
        candidates = self.policy.backend_order if backend is None else (backend,)
        if any(name not in self._backends for name in candidates):
            raise ValidationError(
                f"unknown backend; choose from {tuple(self._backends)}"
            )
        if backend is not None and backend not in self.policy.backend_order:
            raise UnsupportedOperation(f"backend {backend} is excluded by policy")
        rejected = []
        for name in candidates:
            provider = self._backends[name]
            support = provider.supports(operation, self.target)
            if not support.supported:
                rejected.append(f"{name}: {support.reason}")
                continue
            with self._torch.cuda.device(self.device):
                if self._torch.cuda.is_current_stream_capturing():
                    raise ValidationError("prepare must run before graph capture")
                kernel = provider.prepare(operation, bindings, self.target, self.policy)
            if (
                selection is not None
                and kernel.artifact_digest != selection.artifact_digest
            ):
                raise UnsupportedOperation(
                    "prepared artifact differs from the pinned dispatch manifest"
                )
            return ExecutionPlan(
                operation,
                name,
                self.target,
                self.device,
                kernel,
                self._torch,
                specs,
                tuple(operation.outputs()),
                self._manifest.digest if self._manifest is not None else None,
                self.environment_digest,
            )
        raise UnsupportedOperation("; ".join(rejected))

    def prepare_gemm(self, x, w, x_scale, w_scale, out, *, backend=None):
        request = FP8BlockScaleGemm(
            tensor_spec(x),
            tensor_spec(w),
            tensor_spec(x_scale),
            tensor_spec(w_scale),
            tensor_spec(out).dtype,
        )
        return self.prepare(
            request,
            {"x": x, "w": w, "x_scale": x_scale, "w_scale": w_scale, "out": out},
            backend=backend,
        )

    def prepare_rmsnorm(self, x, weight, out, *, epsilon=1e-6, backend=None):
        request = RMSNorm(tensor_spec(x), tensor_spec(weight), epsilon)
        return self.prepare(
            request, {"x": x, "weight": weight, "out": out}, backend=backend
        )

    def prepare_quantize(self, x, out, scales, *, backend=None):
        request = BlockScaleQuantize(tensor_spec(x), tensor_spec(out).dtype)
        return self.prepare(
            request, {"x": x, "out": out, "scales": scales}, backend=backend
        )

    def prepare_mxfp4_quantize(self, x, out, scales, *, backend=None):
        request = MXFP4Quantize(tensor_spec(x))
        return self.prepare(
            request, {"x": x, "out": out, "scales": scales}, backend=backend
        )

    def prepare_mxfp4_gemm(self, x, w, x_scale, w_scale, out, *, backend=None):
        request = MXFP4Gemm(
            tensor_spec(x),
            tensor_spec(w),
            tensor_spec(x_scale),
            tensor_spec(w_scale),
            tensor_spec(out).dtype,
        )
        return self.prepare(
            request,
            {"x": x, "w": w, "x_scale": x_scale, "w_scale": w_scale, "out": out},
            backend=backend,
        )

    def prepare_rope(self, x, freqs, out, *, style="neox", backend=None):
        request = RotaryEmbedding(tensor_spec(x), tensor_spec(freqs), style)
        return self.prepare(
            request, {"x": x, "freqs": freqs, "out": out}, backend=backend
        )

    def prepare_attention(
        self, q, k, v, out, lse, *, causal=False, scale=None, backend=None
    ):
        request = DenseAttention(
            tensor_spec(q), tensor_spec(k), tensor_spec(v), causal, scale
        )
        return self.prepare(
            request, {"q": q, "k": k, "v": v, "out": out, "lse": lse}, backend=backend
        )
