"""Describe supported builder interpreters and dependency inputs without a shell."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from ci.common.json import require


@dataclass(frozen=True)
class Environment:
    python: str
    flavor: str
    image: str = ""

    def __post_init__(self):
        require(self.python in ("3.10", "3.12"), "unsupported builder Python")
        require(self.flavor in ("legacy", "manylinux"), "unknown builder flavor")
        require(
            not self.image.startswith("-") and not any(c.isspace() for c in self.image),
            "invalid container image",
        )

    @property
    def container(self) -> str:
        return "aiter_build_" + self.python

    @property
    def python_bin(self) -> str:
        if self.flavor == "legacy":
            return "python3"
        abi = self.python.replace(".", "")
        return f"/opt/python/cp{abi}-cp{abi}/bin/python"

    @property
    def rocm_tag(self) -> str:
        tag = self.image.split("@", 1)[0].rsplit(":", 1)[-1]
        require(
            re.fullmatch(r"rocm[0-9]+\.[0-9]+(?:\.[0-9]+)?", tag),
            "manylinux image needs a ROCm tag or explicit dependency index; immutable references need explicit version inputs",
        )
        return tag


def architectures(value: str) -> str:
    values = value.split(";")
    require(
        values and all(re.fullmatch(r"gfx[0-9a-f]+", item) for item in values),
        "GPU_ARCHS must be semicolon-separated gfx targets",
    )
    require(len(values) == len(set(values)), "GPU architectures are duplicated")
    return value


def torch_dependency(environment: Environment, pin: str, index: str) -> tuple[str, str]:
    require(
        not pin or re.fullmatch(r"[0-9][a-zA-Z0-9_.+!-]*", pin),
        "invalid torch version pin",
    )
    if not index:
        index = "https://download.pytorch.org/whl/" + environment.rocm_tag
    parsed = urlparse(index)
    require(
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
        and not any(c.isspace() for c in index),
        "torch index must be an HTTPS URL without credentials",
    )
    return ("torch==" + pin if pin else "torch<2.13"), index


def wheel_version(
    base: str, *, date_stamp: bool = False, rocm_tag: str | None = None
) -> str:
    """Append local build labels with one PEP 440 '+' separator."""
    base = base.removeprefix("v")
    if date_stamp:
        base = re.sub(r"\.post[0-9]+$", "", base)
    require(
        re.fullmatch(r"[0-9][a-zA-Z0-9.!+_-]*", base) and base.count("+") <= 1,
        "invalid base package version",
    )
    labels = []
    if date_stamp:
        labels.append(datetime.now(timezone.utc).strftime("%Y%m%d"))
    if rocm_tag is not None:
        require(
            re.fullmatch(r"rocm[0-9]+\.[0-9]+(?:\.[0-9]+)?", rocm_tag),
            "invalid ROCm version label",
        )
        labels.extend((rocm_tag, "manylinux_2_28"))
    if labels:
        base += ("." if "+" in base else "+") + ".".join(labels)
    return base


def append_output(path: Path | None, name: str, value: str) -> None:
    require("\n" not in value and "\r" not in value, "workflow output must be one line")
    if path is not None:
        with path.open("a") as output:
            output.write(f"{name}={value}\n")
