# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixed execution plans and validation of borrowed tensor bindings."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .._validation import ValidationError
from ..api import DType, Operation, TensorSpec
from .provider import PreparedKernel

_DTYPE_NAMES = {
    "torch.uint8": DType.UINT8,
    "torch.float8_e4m3fn": DType.FP8_E4M3FN,
    "torch.float8_e4m3fnuz": DType.FP8_E4M3FNUZ,
    "torch.float16": DType.FP16,
    "torch.bfloat16": DType.BF16,
    "torch.float32": DType.FP32,
}


def tensor_spec(tensor):
    try:
        dtype = _DTYPE_NAMES[str(tensor.dtype)]
    except (AttributeError, KeyError) as error:
        raise ValidationError("unsupported tensor dtype") from error
    return TensorSpec(tuple(tensor.shape), tuple(tensor.stride()), dtype)


def expected_bindings(operation):
    if not isinstance(operation, Operation):
        raise ValidationError("operation must describe inputs, outputs and semantics")
    inputs, outputs = operation.inputs(), operation.outputs()
    if not inputs or not outputs or set(inputs) & set(outputs):
        raise ValidationError("operation needs distinct named inputs and outputs")
    if "out" not in outputs:
        raise ValidationError("operation must name its primary output 'out'")
    if any(
        not isinstance(spec, TensorSpec)
        for spec in (*inputs.values(), *outputs.values())
    ):
        raise ValidationError("operation bindings must be TensorSpec values")
    return (*inputs.items(), *outputs.items())


def validate_bindings(bindings, specs, device, torch, output_names=("out",)):
    """Check metadata and memory ranges without reading GPU tensor contents."""
    if not isinstance(bindings, Mapping):
        raise ValidationError("bindings must be a mapping of names to tensors")
    required = {name for name, _ in specs}
    if set(bindings) != required:
        raise ValidationError(f"bindings must contain exactly {sorted(required)}")
    ranges = {}
    for name, spec in specs:
        tensor = bindings[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a Torch tensor")
        if tensor.device.type != "cuda" or tensor.device.index != device:
            raise ValidationError(f"{name} must reside on cuda:{device}")
        if tensor.requires_grad:
            raise ValidationError(f"{name}: prepared execution is inference-only")
        if tensor_spec(tensor) != spec:
            raise ValidationError(
                f"{name}: shape, strides, dtype or layout differs from the plan"
            )
        if name in output_names:
            extent = 1
            for stride, dimension in sorted(zip(spec.strides, spec.shape)):
                if dimension == 1:
                    continue
                if stride < extent:
                    raise ValidationError(f"{name} must not overlap itself")
                extent += (dimension - 1) * stride
        pointer = tensor.data_ptr()
        if pointer % 16:
            raise ValidationError(f"{name} requires 16-byte pointer alignment")
        span = 1 + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(spec.shape, spec.strides)
        )
        ranges[name] = (pointer, pointer + span * tensor.element_size())
    for output in output_names:
        out_start, out_end = ranges[output]
        for name, (start, end) in ranges.items():
            if name != output and start < out_end and out_start < end:
                raise ValidationError(f"{output} must not overlap {name}")


@dataclass(frozen=True)
class ExecutionPlan:
    """Reuse prepared code with matching buffers and an explicit GPU stream.

    Success means that work was enqueued. The caller keeps all buffers and
    this plan alive until execution completes, including graph replays.
    Concurrent calls may share a plan but must use independent output storage.
    The plan owns no input, output or scratch tensors and allocates none.
    """

    operation: object
    backend: str
    target: str
    device: int
    _kernel: PreparedKernel = field(repr=False)
    _torch: object = field(repr=False)
    _specs: tuple = field(repr=False)
    _output_names: tuple = field(repr=False)
    manifest_digest: str | None = None
    environment_digest: str | None = None

    def execute(self, bindings, *, stream=None):
        validate_bindings(
            bindings, self._specs, self.device, self._torch, self._output_names
        )
        if stream is not None:
            if not isinstance(stream, self._torch.cuda.Stream):
                raise ValidationError("stream must be a Torch CUDA/HIP stream")
            if stream.device.index != self.device:
                raise ValidationError("stream device differs from the plan")
        with self._torch.cuda.device(self.device):
            if stream is None:
                stream = self._torch.cuda.current_stream(self.device)
            if not self._kernel.capture_safe:
                # An explicit stream can differ from the thread's current stream.
                # Capture eligibility belongs to the stream receiving this launch.
                with self._torch.cuda.stream(stream):
                    if self._torch.cuda.is_current_stream_capturing():
                        raise ValidationError(
                            "prepared kernel does not support graph capture"
                        )
            self._kernel.launch(bindings, stream.cuda_stream)
        return bindings["out"]

    def explain(self):
        return {
            "request_id": self.operation.fingerprint(),
            "manifest_digest": self.manifest_digest,
            "environment_digest": self.environment_digest,
            "operator": self.operation.to_dict(),
            "provider": self.backend,
            "kernel": self._kernel.name,
            "target": self.target,
            "device": self.device,
            "artifact_digest": self._kernel.artifact_digest,
            "library_path": self._kernel.library_path,
            "workspace_bytes": self._kernel.workspace_bytes,
            "capture_safe": self._kernel.capture_safe,
            "host_runtime": "python",
            "output_layout": dict(self._specs)["out"].layout,
            "execution": "enqueue; caller retains buffers and plan through completion",
        }
