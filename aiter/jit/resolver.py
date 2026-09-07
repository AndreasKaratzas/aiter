# SPDX-License-Identifier: MIT
"""Resolve reviewed recipe data into fresh native build arguments."""

import logging
import os

from .configuration import (
    AITER_CONFIG_DIR,
    AITER_CONFIGS,
    AITER_CSRC_DIR,
    BUILD_CONTEXT,
    CK_3RDPARTY_DIR,
    CK_DIR,
    is_experimental_enabled,
)
from .recipes import defaults, load_recipes
from .utils.chip_info import get_gfx

logger = logging.getLogger("aiter")


def _build_recipe_context():
    from aiter.jit.recipes import RecipeContext

    def hip_major():
        import torch

        return int(torch.version.hip.split(".")[0])

    def torch_fp8():
        import torch

        return hasattr(torch, "float8_e4m3fnuz")

    return RecipeContext(
        resources={
            **{name: str(path) for name, path in BUILD_CONTEXT.resources.items()},
            "native": AITER_CSRC_DIR,
            "configs": AITER_CONFIG_DIR,
            "ck": CK_DIR,
        },
        environment=dict(os.environ),
        config=lambda name: getattr(AITER_CONFIGS, name),
        target=get_gfx,
        hip_major=hip_major,
        torch_fp8=torch_fp8,
    )


class RecipeResolver:
    def __init__(self, catalog=None, context=None):
        self.catalog = catalog or load_recipes()
        self.context = context

    def resolve(self, ops_name, exclude=None):
        catalog = self.catalog
        context = self.context or _build_recipe_context()
        if ops_name != "all":
            if ops_name not in catalog.names:
                logger.warning("Build recipe is not registered: %s", ops_name)
                return defaults()
            return catalog.resolve(ops_name, context)

        ck_available = os.path.isdir(CK_3RDPARTY_DIR)
        if not ck_available:
            logger.info(
                "[CK-free] Auto-excluding %s CK-dependent modules",
                len(catalog.requires("ck")),
            )
        return catalog.resolve_all(
            context,
            exclude=exclude or (),
            experimental=is_experimental_enabled(),
            ck_available=ck_available,
        )
