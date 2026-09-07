# SPDX-License-Identifier: MIT
"""Legacy operator decorators; all build/load decisions go through JitService."""

import functools
import logging
import types
import typing
from collections.abc import Callable
from typing import Any, Optional

from .bindings import _pybind_develop_hooks
from .bindings import ctypes_caller as make_ctypes_caller
from .composition import get_service
from .configuration import AITER_LOG_MORE
from .utils.torch_guard import torch_compile_guard

logger = logging.getLogger("aiter")


def compile_ops(
    _md_name: str,
    fc_name: str | None = None,
    gen_func: Callable[..., dict[str, Any]] | None = None,
    gen_fake: Callable[..., Any] | None = None,
    ffi_type: str = "pybind",
    develop: bool = False,
):
    def decorator(func):
        loadName = fc_name if fc_name is not None else func.__name__

        if ffi_type == "ctypes":
            ctypes_caller = make_ctypes_caller(
                func, loadName, _md_name, load_library=get_service().load_ctypes
            )

            @functools.wraps(func)
            def ctypes_wrapper(*args, **kwargs):
                return ctypes_caller(*args, **kwargs)

            @torch_compile_guard(device="cuda", calling_func_=func)
            def ctypes_custom_wrapper(*args, **kwargs):
                return ctypes_wrapper(*args, **kwargs)

            return ctypes_custom_wrapper

        elif ffi_type == "pybind":
            func.arg_checked = False

            @functools.wraps(func)
            def wrapper(*args, custom_build_args=None, **kwargs):

                if custom_build_args is None:
                    custom_build_args = {}
                if gen_func is not None:
                    custom_build_args = {
                        **custom_build_args,
                        **gen_func(*args, **kwargs),
                    }
                module = get_service().load_pybind(_md_name, custom_build_args)

                if isinstance(module, types.ModuleType):
                    op = getattr(module, loadName)
                else:
                    return None

                def check_args():
                    import inspect
                    import re

                    import torch

                    enum_types = ["ActivationType", "QuantType", "MlaVersion"]

                    if not op.__doc__.startswith("Members:"):
                        doc_str = op.__doc__.split("\n")[0]
                        doc_str = re.sub(r"<(.*?)\:.*?>", r"\g<1>", doc_str)
                        doc_str = doc_str.replace("list[", "List[")
                        doc_str = doc_str.replace("tuple[", "Tuple[")
                        doc_str = doc_str.replace("collections.abc.Sequence[", "List[")
                        doc_str = doc_str.replace("typing.SupportsInt", "int")
                        doc_str = doc_str.replace("typing.SupportsFloat", "float")
                        doc_str = re.sub(r"\s*\|\s*typing\.SupportsIndex", "", doc_str)
                        pattern = r"([\w\.]+(?:\[[^\]]+\])?)\s*\|\s*None"
                        doc_str = re.sub(pattern, r"Optional[\1]", doc_str)
                        for el in enum_types:
                            doc_str = re.sub(
                                f" (module_)?aiter.*{el} ", f" {el} ", doc_str
                            )
                        doc_str = re.sub(
                            r"(?:[\w.]+\.)?aiter_tensor_t",
                            "aiter_tensor_t",
                            doc_str,
                        )
                        try:
                            aiter_tensor_t = (
                                get_service()
                                .load_pybind("module_aiter_core")
                                .aiter_tensor_t
                            )
                        except (
                            Exception  # noqa: BLE001  blanket catch is intentional here
                        ):
                            aiter_tensor_t = object
                        # Every name the doc_str rewriting above can emit has to
                        # be bound here. `from aiter import *` in the exec below
                        # is not a reliable source: it only ever supplied these
                        # by accident, via submodules that happened to do
                        # `from typing import ...` at module scope.
                        namespace = {
                            "List": list,
                            "Tuple": tuple,
                            "Optional": Optional,
                            "torch": torch,
                            "typing": typing,
                            "aiter_tensor_t": aiter_tensor_t,
                        }

                        exec(  # noqa: S102
                            f"from aiter import*\ndef {doc_str}: pass",
                            namespace,
                        )
                        foo = namespace[doc_str.split("(")[0]]
                        sig = inspect.signature(foo)
                        func.__signature__ = sig
                        ann = {k: v.annotation for k, v in sig.parameters.items()}
                        ann["return"] = sig.return_annotation
                        _tensor_types = (torch.Tensor,)
                        if aiter_tensor_t is not object:
                            _tensor_types = (torch.Tensor, aiter_tensor_t)

                        def _is_tensor_like(obj):
                            return isinstance(obj, _tensor_types)

                        def _is_tensor_type(tp):
                            return tp is torch.Tensor or (
                                aiter_tensor_t is not object and tp is aiter_tensor_t
                            )

                        callargs = inspect.getcallargs(func, *args, **kwargs)
                        for el, arg in callargs.items():
                            expected_type = ann[el]
                            got_type = type(arg)
                            origin = typing.get_origin(expected_type)
                            sub_t = typing.get_args(expected_type)

                            if origin is None:
                                if _is_tensor_type(expected_type) and _is_tensor_like(
                                    arg
                                ):
                                    pass
                                elif not isinstance(arg, expected_type) and not (
                                    any(el in str(expected_type) for el in enum_types)
                                    and isinstance(arg, int)
                                ):
                                    raise TypeError(
                                        f"{loadName}: {el} needs to be {expected_type} but got {got_type}"
                                    )
                            elif origin is list:
                                if not isinstance(arg, list):
                                    raise TypeError(
                                        f"{loadName}: {el} needs to be List[{sub_t}] but got {arg}"
                                    )
                            elif origin is typing.Union or origin is types.UnionType:
                                if (
                                    arg is not None
                                    and not _is_tensor_like(arg)
                                    and not isinstance(arg, sub_t)
                                ):
                                    raise TypeError(
                                        f"{loadName}: {el} needs to be Optional[{sub_t}] but got {arg}"
                                    )
                            else:
                                raise TypeError(f"Unsupported type: {expected_type}")

                        func_hints = typing.get_type_hints(func)
                        if ann["return"] is None:
                            func_hints["return"] = None

                        tensor_like_types = {torch.Tensor}
                        if aiter_tensor_t is not object:
                            tensor_like_types.add(aiter_tensor_t)

                        enum_type_objs = tuple(
                            namespace[el] for el in enum_types if el in namespace
                        )

                        def canonicalize_hint(hint):
                            if hint in tensor_like_types:
                                return ("tensor",)
                            if hint in enum_type_objs:
                                return int

                            origin = typing.get_origin(hint)
                            if origin in (list, list):
                                return (
                                    "list",
                                    tuple(
                                        canonicalize_hint(arg)
                                        for arg in typing.get_args(hint)
                                    ),
                                )
                            if origin is tuple:
                                return (
                                    "tuple",
                                    tuple(
                                        canonicalize_hint(arg)
                                        for arg in typing.get_args(hint)
                                    ),
                                )
                            if origin in (typing.Union, types.UnionType):
                                return (
                                    "union",
                                    tuple(
                                        sorted(
                                            (
                                                canonicalize_hint(arg)
                                                for arg in typing.get_args(hint)
                                            ),
                                            key=repr,
                                        )
                                    ),
                                )
                            return hint

                        canonical_ann = {
                            key: canonicalize_hint(value) for key, value in ann.items()
                        }
                        canonical_func_hints = {
                            key: canonicalize_hint(value)
                            for key, value in func_hints.items()
                        }

                        if canonical_ann != canonical_func_hints:
                            logger.warning(
                                f"type hints mismatch, override to --> {doc_str}"
                            )
                    return True

                if not func.arg_checked:
                    func.arg_checked = check_args()

                if AITER_LOG_MORE == 2:
                    from ..test_common import log_args

                    log_args(func, *args, **kwargs)
                # develop=True: torch.Tensor -> pybind aiter_tensor_t before C++ (activation, CAR, ...).
                if develop:
                    convert, tensor_cls, raw_stream, current_device = (
                        _pybind_develop_hooks()
                    )

                    args = tuple(
                        convert(a) if isinstance(a, tensor_cls) else a for a in args
                    )
                    if kwargs:
                        kwargs = {
                            k: convert(v) if isinstance(v, tensor_cls) else v
                            for k, v in kwargs.items()
                        }

                    module._set_current_hip_stream(raw_stream(current_device()))
                return op(*args, **kwargs)

            @torch_compile_guard(device="cuda", gen_fake=gen_fake, calling_func_=func)
            def custom_wrapper(*args, **kwargs):
                return wrapper(*args, **kwargs)

            return custom_wrapper

        else:
            raise ValueError(
                f"Unknown ffi_type: {ffi_type!r}, expected 'ctypes' or 'pybind'"
            )

    return decorator
