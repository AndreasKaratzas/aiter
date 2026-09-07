"""Resolve the official ROCm nightly index without PyPI/CUDA fallback."""

from __future__ import annotations

import hashlib
import platform
import re
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

from ci.common.json import require

ROOT = "https://wheels.vllm.ai/rocm/nightly/"
REVISION = re.compile(r"/rocm/([0-9a-f]{40})/vllm-[^/]+\.whl$")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(
                value for name, value in attrs if name == "href" and value
            )


def wheel_revision(url: str) -> str | None:
    parsed = urlparse(url)
    match = REVISION.fullmatch(unquote(parsed.path))
    if parsed.scheme == "https" and parsed.netloc == "wheels.vllm.ai" and match:
        return match.group(1)
    return None


def fetch(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        require(
            response.geturl().startswith("https://wheels.vllm.ai/"),
            "nightly index redirected off its official origin",
        )
        data = response.read(1024 * 1024 + 1)
        require(len(data) <= 1024 * 1024, "nightly index exceeds size limit")
        return data.decode("utf-8")


def resolve(*, variant: str | None = None, read=fetch) -> dict:
    index = read(ROOT)
    parser = Links()
    parser.feed(index)
    available = sorted(
        {
            path.rstrip("/")
            for path in parser.links
            if re.fullmatch(r"rocm[0-9]{3}/", path)
        }
    )
    require(bool(available), "official ROCm nightly index has no variants")
    if variant is None:
        require(
            len(available) == 1, "multiple nightly ROCm variants; choose one explicitly"
        )
        variant = available[0]
    require(variant in available, "requested ROCm nightly variant is unavailable")
    url = ROOT + variant + "/vllm/"
    body = read(url)
    parser = Links()
    parser.feed(body)
    candidates = []
    for href in parser.links:
        link = urljoin(url, href)
        revision = wheel_revision(link)
        if revision:
            filename = unquote(urlparse(link).path.rsplit("/", 1)[1])
            if re.fullmatch(
                r"vllm-.+\." + variant + r"-cp312-cp312-manylinux_2_[0-9]+_x86_64\.whl",
                filename,
            ):
                candidates.append((link, revision, filename))
    require(
        len(candidates) == 1,
        "nightly index must resolve one exact Python 3.12 ROCm x86_64 wheel",
    )
    link, revision, filename = candidates[0]
    return {
        "schema_version": 1,
        "channel": ROOT,
        "index_url": url,
        "variant": variant,
        "wheel_url": link,
        "revision": revision,
        "filename": filename,
        "indices": [
            {
                "url": ROOT,
                "html": index,
                "sha256": hashlib.sha256(index.encode()).hexdigest(),
            },
            {
                "url": url,
                "html": body,
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            },
        ],
    }


def require_platform(resolution: dict) -> None:
    require(
        sys.version_info[:2] == (3, 12),
        "official ROCm nightly requires Python 3.12; refusing CUDA fallback",
    )
    require(
        sys.platform == "linux" and platform.machine() == "x86_64",
        "nightly wheel requires Linux x86_64",
    )
    libc, version = platform.libc_ver()
    minimum = int(re.search(r"manylinux_2_([0-9]+)_", resolution["filename"]).group(1))
    require(
        libc == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, minimum),
        f"resolved nightly wheel requires glibc >= 2.{minimum}; observed {libc} {version}",
    )
