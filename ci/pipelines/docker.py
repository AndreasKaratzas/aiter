"""One Docker process boundary for build, execution and immutable inspection."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ci.common.json import digest, require, write_json
from ci.pipelines.process import Process

IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


class Docker(Process):
    def __init__(self, evidence: Path, *, executable: str = "docker"):
        super().__init__(evidence, executable=executable)

    def inspect(self, reference: str) -> dict:
        data = json.loads(self.command(["image", "inspect", reference]))
        require(
            isinstance(data, list) and len(data) == 1,
            "Docker inspection must identify one image",
        )
        image = data[0]
        require(
            isinstance(image, dict) and IMAGE_ID.fullmatch(image.get("Id", "")),
            "invalid inspected image ID",
        )
        return image

    def resolve(self, reference: str) -> dict:
        self.command(["pull", reference])
        return self.inspect(reference)

    def run(
        self,
        image_id: str,
        *,
        controls: Path,
        source: Path,
        evidence: Path,
        request: str,
        artifacts: Path | None = None,
        timeout: int = 21600,
        controller: str = "ci.pipelines.container",
    ) -> None:
        require(
            IMAGE_ID.fullmatch(image_id), "execute only an inspected immutable image ID"
        )
        require(
            controller
            in {
                "ci.pipelines.container",
                "ci.pipelines.nightly",
                "ci.pipelines.benchmarks",
            },
            "unknown container controller",
        )
        mounts = [
            (controls, "/control", True),
            (source, "/workspace", True),
            (evidence, "/evidence", False),
        ]
        if artifacts is not None:
            mounts.append((artifacts, "/artifacts", True))
        name = (
            "aiter-qualification-"
            + digest({"output": str(evidence), "image": image_id}).split(":")[1][:24]
        )
        arguments = [
            "run",
            "--name",
            name,
            "--rm",
            "--entrypoint",
            "python3",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add",
            "video",
            "--shm-size",
            "16g",
            "--ipc=host",
        ]
        for origin, target, readonly in mounts:
            require(
                ":" not in str(origin) and "\n" not in str(origin),
                "invalid Docker mount path",
            )
            arguments.extend(["-v", f"{origin}:{target}" + (":ro" if readonly else "")])
        arguments.extend(
            [
                "-e",
                "PYTHONPATH=/control",
                "-e",
                "PYTHONNOUSERSITE=1",
                "-e",
                "AITER_CI_EXECUTOR_IMAGE=" + image_id,
                "-w",
                "/control",
                image_id,
                "-m",
                controller,
                "--request",
                "/evidence/" + request,
            ]
        )
        self.managed_container(arguments, name=name, timeout=timeout)

    def managed_container(
        self, arguments: list[str], *, name: str, timeout: int
    ) -> None:
        require(
            re.fullmatch(r"[A-Za-z0-9_.-]+", name), "invalid managed container name"
        )
        try:
            self.command(arguments, timeout=timeout)
        except BaseException:
            cleanup = []
            try:
                self.command(["rm", "--force", "--volumes", name], timeout=120)
            except (ValueError, OSError) as error:
                cleanup.append(str(error))
            write_json(
                self.evidence / (name + ".cleanup.json"),
                {"container": name, "problems": cleanup},
            )
            raise
