# SPDX-License-Identifier: MIT
"""Load a requested kernel module without changing process import precedence."""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from aiter.codegen import BuildContext


def load_kernel_module(path):
    source = Path(path).resolve(strict=True)
    package = BuildContext.load().package
    if source.is_relative_to(package):
        relative = source.relative_to(package).with_suffix("")
        module = importlib.import_module("aiter." + ".".join(relative.parts))
        if Path(module.__file__).resolve() != source:
            raise ImportError(
                f"kernel module origin differs from requested source: {source}"
            )
        return module
    # External standalone kernels may import installed packages normally. Their
    # directory is deliberately not installed as a global Python search root.
    name = "_aiter_kernel_" + hashlib.sha256(str(source).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load kernel source: {source}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def write_launcher_sources(params, backend, suffix, out_name, explicit_path=None):
    """Publish complete rendered launch sources into an owned artifact directory."""
    import os
    import tempfile

    from aiter.jit.cache import cache_directory

    templates = BuildContext.load().resource(
        "native", "cpp_itfs/gluon_aot_tools/extra", backend
    )
    rendered = {
        template.suffix: template.read_text().format(**params)
        for template in sorted(templates.glob("compile.*"))
    }
    if not rendered:
        raise FileNotFoundError(f"no {backend} launcher templates in {templates}")
    digest = hashlib.sha256()
    for extension, content in rendered.items():
        digest.update(extension.encode())
        digest.update(content.encode())
    prefix = (
        Path(explicit_path)
        if explicit_path is not None
        else cache_directory() / "aot" / "codegen" / digest.hexdigest() / out_name
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, content in rendered.items():
        output = prefix.with_suffix(f".{suffix}{extension}")
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, delete=False
        ) as temporary:
            temporary.write(content)
            path = temporary.name
        os.replace(path, output)
        outputs.append(output)
    return outputs


def native_scalar_type(type_name, fallback):
    """Use the device argument width, not Python's scalar parsing width."""
    # Triton's Python launcher accepts Python floats as C doubles, then casts
    # them before enqueue. These launchers pass argument storage directly to HIP.
    if type_name == "fp32":
        return "float"
    return fallback(type_name)
