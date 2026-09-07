# SPDX-License-Identifier: MIT
"""Torch/native argument marshaling, independent of compilation orchestration."""

import types
import typing

from .configuration import AITER_LOG_MORE


def _is_union(origin):
    """Check for both typing.Union (Optional[X]) and types.UnionType (X | None)."""
    return origin is typing.Union or origin is types.UnionType


# Per-parameter conversion kinds, resolved once per op from the type hints (see
# _ensure_loaded) so the per-call loop is an int compare instead of a fresh
# typing.get_origin/get_args round trip. _ARG_SCALAR covers int/float/anything
# else: with c_func.argtypes declared, ctypes converts the raw Python value.
(
    _ARG_TENSOR,
    _ARG_OPT_TENSOR,
    _ARG_OPT_INT,
    _ARG_OPT_STR,
    _ARG_STR,
    _ARG_BOOL,
    _ARG_SCALAR,
) = range(7)


def ctypes_caller(func, fc_name, md_name, *, load_library):
    """Build a ctypes-based caller for a torch-free .so module.

    Type-hint to C ABI mapping
    -------------------------------------------------------
    Python annotation     | ctypes type          | C type
    ----------------------|----------------------|---------
    Tensor                | POINTER(aiter_tensor_t) | aiter_tensor_t*
    Optional[Tensor]      | POINTER(aiter_tensor_t) | aiter_tensor_t* (NULL if None)
    int                   | c_int64              | int64_t
    Optional[int]         | c_int64              | int64_t (-1 if None)
    str                   | c_char_p             | char* (.encode())
    Optional[str]         | c_char_p             | char* (NULL if None)
    bool                  | c_int                | int   (0 / 1)
    float                 | c_float              | float
    (other)               | c_void_p             | void*
    (auto-appended)       | c_void_p             | hipStream_t
    -------------------------------------------------------
    """
    import ctypes
    import inspect

    import torch

    from ..utility.dtypes import aiter_tensor_t, torch_to_aiter

    # Avoid constructing a Python Stream object on every ctypes invocation.
    # Keep the public API fallback for torch versions without the private raw
    # stream getter, and preserve the first tensor's device selection.
    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if raw_stream is None:

        def raw_stream(device_index):
            return torch.cuda.current_stream(device_index).cuda_stream

    current_device = torch.cuda.current_device

    _cache = {}
    _arg_checked = False
    _sig = inspect.signature(func)
    _hints = typing.get_type_hints(func)

    def _ensure_loaded():
        if _cache:
            return
        lib = load_library(md_name)
        c_func = getattr(lib, fc_name)

        def _opt_sym(name, argtypes=(), restype=None):
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.argtypes = list(argtypes)
                fn.restype = restype
            return fn

        abi_fn = _opt_sym("aiter_ctypes_abi_version", restype=ctypes.c_int)
        ctypes_abi_version = abi_fn() if abi_fn else 1
        ctypes_status_mode = ctypes_abi_version >= 2
        err_getter = _opt_sym("aiter_get_last_error", restype=ctypes.c_char_p)
        err_clear = _opt_sym("aiter_clear_last_error")

        ret_hint = _hints.get("return")
        ctypes_data_return = ctypes_status_mode and ret_hint is int

        if ctypes_status_mode or ret_hint is int:
            c_func.restype = ctypes.c_int
        elif ret_hint is float:
            c_func.restype = ctypes.c_float
        else:
            c_func.restype = None

        # `argtypes` and `kinds` are produced by the SAME pass over the type
        # hints, so the ctypes type a parameter is declared as and the branch
        # caller() takes for it can never drift apart. Hints are static, so this
        # runs once per op instead of on every call.
        argtypes = []
        kinds = []
        has_tensor = False
        for pname in _sig.parameters:
            hint = _hints.get(pname)
            origin = typing.get_origin(hint)
            type_args = typing.get_args(hint)
            if hint is torch.Tensor:
                argtypes.append(ctypes.POINTER(aiter_tensor_t))
                kinds.append(_ARG_TENSOR)
                has_tensor = True
            elif _is_union(origin) and torch.Tensor in type_args:
                argtypes.append(ctypes.POINTER(aiter_tensor_t))
                kinds.append(_ARG_OPT_TENSOR)
                has_tensor = True
            elif _is_union(origin) and int in type_args:
                argtypes.append(ctypes.c_int64)
                kinds.append(_ARG_OPT_INT)
            elif _is_union(origin) and str in type_args:
                argtypes.append(ctypes.c_char_p)
                kinds.append(_ARG_OPT_STR)
            elif hint is str:
                argtypes.append(ctypes.c_char_p)
                kinds.append(_ARG_STR)
            elif hint is bool:
                argtypes.append(ctypes.c_int)
                kinds.append(_ARG_BOOL)
            elif hint is int:
                argtypes.append(ctypes.c_int64)
                kinds.append(_ARG_SCALAR)
            elif hint is float:
                argtypes.append(ctypes.c_float)
                kinds.append(_ARG_SCALAR)
            else:
                argtypes.append(ctypes.c_void_p)
                kinds.append(_ARG_SCALAR)
        # hipStream_t: the caller always appends the current stream to the args, so the
        # argtypes must always declare it -- otherwise ctypes takes the variadic path
        # (ffi_prep_cif_var) for torch-free modules whose params are all non-tensor, which
        # fails on stricter libffi builds.
        argtypes.append(ctypes.c_void_p)  # hipStream_t
        c_func.argtypes = argtypes

        # Positional fast path in caller(): with no *args/**kwargs/keyword-only
        # parameters, the values are already in parameter order, so inspect's
        # binding machinery has nothing to figure out. Anything else (kwargs,
        # too few args) falls back to _sig.bind so error messages and binding
        # semantics stay exactly as they were.
        params = list(_sig.parameters.values())
        fast_ok = all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
        empty = inspect.Parameter.empty

        _cache["lib"] = lib
        _cache["c_func"] = c_func
        _cache["err_getter"] = err_getter
        _cache["err_clear"] = err_clear
        _cache["ctypes_status_mode"] = ctypes_status_mode
        _cache["ctypes_data_return"] = ctypes_data_return
        _cache["has_tensor"] = has_tensor
        _cache["kinds"] = tuple(kinds)
        _cache["names"] = tuple(_sig.parameters)
        _cache["defaults"] = tuple(
            None if p.default is empty else p.default for p in params
        )
        _cache["n_params"] = len(params)
        _cache["n_required"] = sum(1 for p in params if p.default is empty)
        _cache["fast_ok"] = fast_ok
        # A NULL aiter_tensor_t* carries no state, so one instance is reused for
        # every omitted Optional[Tensor] instead of allocating per call.
        _cache["null_tensor"] = ctypes.POINTER(aiter_tensor_t)()

    def _check_args_before_convert(bound_args, hints):
        for pname, value in bound_args.items():
            hint = hints.get(pname)
            origin = typing.get_origin(hint)
            type_args = typing.get_args(hint)

            if hint is torch.Tensor:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects torch.Tensor, "
                        f"got {type(value).__name__}"
                    )
            elif _is_union(origin) and torch.Tensor in type_args:
                if value is not None and not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects Optional[torch.Tensor], "
                        f"got {type(value).__name__}"
                    )
            elif _is_union(origin) and int in type_args:
                if value is not None and not isinstance(value, int):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects Optional[int], "
                        f"got {type(value).__name__}"
                    )
            elif _is_union(origin) and str in type_args:
                if value is not None and not isinstance(value, str):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects Optional[str], "
                        f"got {type(value).__name__}"
                    )
            elif hint is str:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects str, got {type(value).__name__}"
                    )
            elif hint is bool:
                if not isinstance(value, (bool, int)):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects bool, got {type(value).__name__}"
                    )
            elif hint is int:
                if not isinstance(value, int):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects int, got {type(value).__name__}"
                    )
            elif hint is float and not isinstance(value, (float, int)):
                raise TypeError(
                    f"{fc_name}: '{pname}' expects float, "
                    f"got {type(value).__name__}"
                )

    def caller(*args, **kwargs):
        nonlocal _arg_checked
        _ensure_loaded()
        c_func = _cache["c_func"]
        err_getter = _cache.get("err_getter")
        err_clear = _cache.get("err_clear")
        ctypes_status_mode = _cache.get("ctypes_status_mode", False)
        ctypes_data_return = _cache.get("ctypes_data_return", False)

        if AITER_LOG_MORE == 2:
            from ..test_common import log_args

            log_args(func, *args, **kwargs)

        kinds = _cache["kinds"]
        n_params = _cache["n_params"]
        n_args = len(args)
        if (
            _cache["fast_ok"]
            and not kwargs
            and _cache["n_required"] <= n_args <= n_params
        ):
            # Already in parameter order; only the omitted tail needs defaults.
            defaults = _cache["defaults"]
            values = args if n_args == n_params else args + defaults[n_args:]
        else:
            # kwargs, a missing required arg, or an exotic signature -- let
            # inspect do the binding (and raise the usual TypeError).
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            values = tuple(bound.arguments.values())

        if not _arg_checked:
            _check_args_before_convert(dict(zip(_cache["names"], values)), _hints)
            _arg_checked = True

        c_args = []
        aiter_refs = []
        tensor_device = None
        add_arg = c_args.append
        keep_alive = aiter_refs.append

        null_tensor = _cache["null_tensor"]

        for kind, value in zip(kinds, values):
            if kind == _ARG_SCALAR:
                # int / float / anything else: c_func.argtypes drives the
                # conversion, so the raw Python value goes straight through.
                add_arg(value)
            elif kind == _ARG_TENSOR:
                if tensor_device is None:
                    tensor_device = value.get_device()
                at = torch_to_aiter(value)
                keep_alive(at)
                add_arg(ctypes.byref(at))
            elif kind == _ARG_OPT_TENSOR:
                if value is not None:
                    if tensor_device is None:
                        tensor_device = value.get_device()
                    at = torch_to_aiter(value)
                    keep_alive(at)
                    add_arg(ctypes.byref(at))
                else:
                    add_arg(null_tensor)
            elif kind == _ARG_OPT_INT:
                add_arg(value if value is not None else -1)
            elif kind == _ARG_OPT_STR:
                add_arg(value.encode() if value is not None else None)
            elif kind == _ARG_STR:
                add_arg(value.encode())
            else:  # _ARG_BOOL
                add_arg(1 if value else 0)

        if tensor_device is None:
            tensor_device = current_device()
        c_args.append(ctypes.c_void_p(raw_stream(tensor_device)))
        if err_clear is not None:
            err_clear()
        ret = c_func(*c_args)

        err_msg = None
        if ctypes_status_mode and not ctypes_data_return and ret != 0:
            err_msg = f"ctypes status={ret}"
        if err_getter is not None:
            raw = err_getter()
            if raw:
                err_msg = raw.decode(errors="replace")
        if err_msg is not None:
            if err_clear is not None:
                err_clear()
            raise RuntimeError(f"{fc_name} failed: {err_msg}")

        if ctypes_data_return:
            return ret
        if ctypes_status_mode:
            return None
        return ret

    return caller


_pybind_develop_hooks_cache = None


def _pybind_develop_hooks():
    """Everything the develop=True pybind path needs, resolved once.

    All four are per-call on that path -- the converter runs once per tensor
    argument -- and importing them inside the wrapper meant a sys.modules round
    trip each time for names that never change. aiter.utility.dtypes imports back
    into this module, so binding them at import time is not an option either.
    """
    global _pybind_develop_hooks_cache
    if _pybind_develop_hooks_cache is None:
        import torch

        from ..utility.dtypes import torch_to_aiter_pybind

        # Hands back the same handle as current_stream().cuda_stream without
        # building the Python Stream object to carry it. Private, so fall back to
        # the public spelling rather than assume a torch version floor.
        raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        if raw_stream is None:

            def raw_stream(_device_index):
                return torch.cuda.current_stream().cuda_stream

        _pybind_develop_hooks_cache = (
            torch_to_aiter_pybind,
            torch.Tensor,
            raw_stream,
            torch.cuda.current_device,
        )
    return _pybind_develop_hooks_cache
