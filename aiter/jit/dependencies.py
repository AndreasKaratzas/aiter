# SPDX-License-Identifier: MIT
"""Provision explicitly requested, pinned native build dependencies."""

import logging
import os

from .configuration import HIP_KITTENS_DIR, bd_dir
from .synchronization import mp_lock

logger = logging.getLogger("aiter")


def clone_3rdparty(third_party: str) -> None:
    def MainFunc():
        if not os.path.exists(dir_path):
            import subprocess

            def check_git_version(required_major, required_minor):
                try:
                    output = subprocess.check_output(
                        ["git", "--version"], text=True
                    ).strip()
                    import re

                    m = re.search(r"(\d+)\.(\d+)", output)
                    if m:
                        major, minor = int(m.group(1)), int(m.group(2))
                        return (major > required_major) or (
                            major == required_major and minor >= required_minor
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to check git version: {e}")
                return False

            logger.info(f"Cloning 3rdparty {third_party} to {dir_path}")
            # Check git version for --revision flag support (>=2.49)
            if not check_git_version(2, 49):
                logger.warning(
                    "Your git version does not support the --revision flag (requires >=2.49). Slow path is used for cloning 3rdparty."
                )
                subprocess.call(
                    [
                        "git",
                        "clone",
                        "-q",
                        third_party_info["url"],
                        dir_path,
                    ]
                )
                subprocess.call(
                    [
                        "git",
                        "-C",
                        dir_path,
                        "reset",
                        "-q",
                        "--hard",
                        third_party_info["commit"],
                    ]
                )
                subprocess.call(
                    [
                        "git",
                        "-C",
                        dir_path,
                        "submodule",
                        "update",
                        "-q",
                        "--init",
                        "--recursive",
                    ]
                )
            else:
                # Save current git config value for advice.detachedHead, set to false
                prev_detached_head = None
                try:
                    try:
                        prev_detached_head = subprocess.check_output(
                            ["git", "config", "--get", "advice.detachedHead"], text=True
                        ).strip()
                    except subprocess.CalledProcessError:
                        prev_detached_head = None  # not set before
                    # Set to false before clone
                    subprocess.call(
                        ["git", "config", "--global", "advice.detachedHead", "false"]
                    )

                    subprocess.call(
                        [
                            "git",
                            "clone",
                            "-q",
                            f"--revision={third_party_info['commit']}",
                            "--depth=1",
                            "--recurse-submodules",
                            third_party_info["url"],
                            dir_path,
                        ]
                    )
                finally:
                    # Restore config after clone
                    if prev_detached_head is not None:
                        subprocess.call(
                            [
                                "git",
                                "config",
                                "--global",
                                "advice.detachedHead",
                                prev_detached_head,
                            ]
                        )
                    else:
                        subprocess.call(
                            [
                                "git",
                                "config",
                                "--global",
                                "--unset",
                                "advice.detachedHead",
                            ]
                        )

    if third_party == "HipKittens":
        dir_path = HIP_KITTENS_DIR
        third_party_info = {
            "url": "https://github.com/HazyResearch/HipKittens.git",
            "commit": "d3cd9b31cb0ff611ff64b5701f57ccdeb7712f39",
        }
    elif third_party == "ComposableKernel":
        # TODO: ComposableKernel will be supported in the future
        pass

    if "third_party_info" in locals():
        lock_path = f"{bd_dir}/lock_3rdparty_clone_{third_party}"
        mp_lock(lockPath=lock_path, MainFunc=MainFunc)
