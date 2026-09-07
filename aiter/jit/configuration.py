# SPDX-License-Identifier: MIT
"""Selected package resources and the legacy tuning-table configuration adapter."""

import functools
import logging
import os
import sys
from pathlib import Path

from aiter.codegen import BuildContext

from .cache import cache_directory
from .utils.chip_info import get_gfx

this_dir = str(Path(__file__).resolve().parent)
AITER_REBUILD = int(os.environ.get("AITER_REBUILD", "0"))
ENABLE_CK = int(os.environ.get("ENABLE_CK", "1")) != 0
AITER_DISABLE_KERNARG_PRELOAD = (
    int(os.environ.get("AITER_DISABLE_KERNARG_PRELOAD", "0")) != 0
)


def is_experimental_enabled() -> bool:
    # Mirror the C++ side (atoi(...) != 0): treat unset and "0" as disabled,
    # any other integer value as enabled. Non-numeric strings are treated as
    # disabled to avoid accidentally turning on experimental code paths.
    val = os.environ.get("AITER_ENABLE_EXPERIMENTAL", "0")
    try:
        return int(val) != 0
    except ValueError:
        return False


aiter_lib = None


logger = logging.getLogger("aiter")

PY = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_CONTEXT = BuildContext.load()
AITER_ROOT_DIR = str(BUILD_CONTEXT.package.parent)
AITER_CONFIG_DIR = str(BUILD_CONTEXT.resource("configs"))
AITER_LOG_MORE = int(os.getenv("AITER_LOG_MORE", "0"))
AITER_LOG_TUNED_CONFIG = int(os.getenv("AITER_LOG_TUNED_CONFIG", "0"))


# config_env start here
AITER_CONFIG_GEMM_A4W4 = os.getenv(
    "AITER_CONFIG_GEMM_A4W4",
    f"{AITER_CONFIG_DIR}/a4w4_blockscale_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A6W6 = os.getenv(
    "AITER_CONFIG_GEMM_A6W6",
    f"{AITER_CONFIG_DIR}/a6w6_blockscale_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A8W8 = os.getenv(
    "AITER_CONFIG_GEMM_A8W8",
    f"{AITER_CONFIG_DIR}/a8w8_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    f"{AITER_CONFIG_DIR}/a8w8_bpreshuffle_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A8W8_BLOCKSCALE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    f"{AITER_CONFIG_DIR}/a8w8_blockscale_tuned_gemm.csv",
)

AITER_CONFIG_FMOE = os.getenv(
    "AITER_CONFIG_FMOE",
    f"{AITER_CONFIG_DIR}/tuned_fmoe.csv",
)

AITER_CONFIG_FHMOE = os.getenv(
    "AITER_CONFIG_FHMOE",
    f"{AITER_CONFIG_DIR}/tuned_fhmoe.csv",
)

AITER_CONFIG_GROUPED_FMOE = os.getenv(
    "AITER_CONFIG_GROUPED_FMOE",
    f"{AITER_CONFIG_DIR}/tuned_grouped_fmoe.csv",
)

AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    f"{AITER_CONFIG_DIR}/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)

AITER_CONFIG_A8W8_BATCHED_GEMM = os.getenv(
    "AITER_CONFIG_A8W8_BATCHED_GEMM",
    f"{AITER_CONFIG_DIR}/a8w8_tuned_batched_gemm.csv",
)

AITER_CONFIG_BF16_BATCHED_GEMM = os.getenv(
    "AITER_CONFIG_BF16_BATCHED_GEMM",
    f"{AITER_CONFIG_DIR}/bf16_tuned_batched_gemm.csv",
)

# fp8 e8m0 mxscale (block-scale) batched-GEMM tuned config. Its own family
# (scale type baked into the filename, matching the a8w8_/bf16_ split) so a
# future fp32 rowwise-scale variant lands in a separate CSV and never collides
# on key. The scale type is identified by the filename alone. The
# per-model tuned data currently lives under model_configs/ (e.g.
# dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv), merged in at runtime by
# get_config_file; this canonical path may not exist on disk.
AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE = os.getenv(
    "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
    f"{AITER_CONFIG_DIR}/batched_gemm_a8w8_blockscale_mxscale_tuned.csv",
)

AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
    f"{AITER_CONFIG_DIR}/" "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv",
)

AITER_CONFIG_GEMM_BF16 = os.getenv(
    "AITER_CONFIG_GEMM_BF16",
    f"{AITER_CONFIG_DIR}/bf16_tuned_gemm.csv",
)

AITER_CONFIG_GDR_DECODE = os.getenv(
    "AITER_CONFIG_GDR_DECODE",
    f"{AITER_CONFIG_DIR}/gdr_decode_tuned.csv",
)

# K5 opt BV tuned config. Per-model tuned rows live under model_configs/
# (qwen3_5_*_chunk_gdn_h_opt_tuned.csv) and get merged into this canonical file by
# get_config_file. It ships header-only: with no per-model table present
# get_config_file returns this path as-is, and the opt AOT reads it, so it has to
# be a readable csv rather than a missing path.
AITER_CONFIG_GDN_K5_OPT = os.getenv(
    "AITER_CONFIG_GDN_K5_OPT",
    f"{AITER_CONFIG_DIR}/chunk_gdn_h_opt_tuned.csv",
)


class AITER_CONFIG:
    def __init__(self, config_dir=None):
        self.config_dir = str(config_dir or AITER_CONFIG_DIR)

    @property
    def AITER_CONFIG_GEMM_A4W4_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A4W4",
            AITER_CONFIG_GEMM_A4W4,
            "a4w4_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A6W6_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A6W6",
            AITER_CONFIG_GEMM_A6W6,
            "a6w6_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm"
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE,
            "a8w8_bpreshuffle_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
            "a8w8_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_FMOE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_FMOE", AITER_CONFIG_FMOE, "tuned_fmoe"
        )

    @property
    def AITER_CONFIG_FHMOE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_FHMOE", AITER_CONFIG_FHMOE, "tuned_fhmoe"
        )

    @property
    def AITER_CONFIG_GROUPED_FMOE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GROUPED_FMOE",
            AITER_CONFIG_GROUPED_FMOE,
            "tuned_grouped_fmoe",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
            "a8w8_blockscale_bpreshuffle_tuned_gemm",
        )

    @property
    def AITER_CONFIG_A8W8_BATCHED_GEMM_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_A8W8_BATCHED_GEMM",
            AITER_CONFIG_A8W8_BATCHED_GEMM,
            "a8w8_tuned_batched_gemm",
        )

    @property
    def AITER_CONFIG_BF16_BATCHED_GEMM_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_BF16_BATCHED_GEMM",
            AITER_CONFIG_BF16_BATCHED_GEMM,
            "bf16_tuned_batched_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_BF16_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_BF16", AITER_CONFIG_GEMM_BF16, "bf16_tuned_gemm"
        )

    @property
    def AITER_CONFIG_GDR_DECODE_FILE(self):
        return AITER_CONFIG_GDR_DECODE

    @property
    def AITER_CONFIG_GDN_K5_OPT_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GDN_K5_OPT",
            AITER_CONFIG_GDN_K5_OPT,
            "chunk_gdn_h_opt_tuned",
        )

    @property
    def AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
            AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE,
            "batched_gemm_a8w8_blockscale_mxscale_tuned",
        )

    @property
    def AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
            AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE,
            "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned",
        )

    def update_config_files(self, file_path: str, merge_name: str):
        paths = file_path.split(os.pathsep) if file_path else []
        if len(paths) <= 1:
            return file_path
        from pathlib import Path

        from aiter.jit.utils.chip_info import gfx_from_cu_num
        from aiter.tuning.tables import merge_tables

        untuned_name = "untuned".join(merge_name.rsplit("tuned", 1))
        return merge_tables(
            paths,
            schema=Path(self.config_dir) / f"{untuned_name}.csv",
            cache=cache_directory() / "tuning",
            name=merge_name,
            infer_arch=gfx_from_cu_num,
        )

    # Cache is keyed on (self, env_name, ...); this object is a
    # process-lifetime singleton, so the retained reference is not a leak.
    @functools.lru_cache(maxsize=20)  # noqa: B019
    def get_config_file(self, env_name, default_file, tuned_file_name):
        config_env_file = os.getenv(env_name)
        if Path(default_file).parent == Path(AITER_CONFIG_DIR):
            default_file = str(Path(self.config_dir) / Path(default_file).name)

        if not config_env_file:
            model_config_dir = Path(f"{self.config_dir}/model_configs/")
            op_tuned_file_list = sorted(
                p
                for p in model_config_dir.glob(f"*{tuned_file_name}*.csv")
                if (p.is_file() and "untuned" not in p.name)
            )

            if not op_tuned_file_list:
                config_file = default_file
            else:
                paths = [str(p) for p in op_tuned_file_list]
                if Path(default_file).is_file():
                    paths.insert(0, default_file)
                tuned_files = os.pathsep.join(paths)
                logger.info(
                    f"merge tuned file under model_configs/ and configs/ {tuned_files}"
                )
                config_file = self.update_config_files(tuned_files, tuned_file_name)
        else:
            config_file = self.update_config_files(config_env_file, tuned_file_name)
            # print(f"get config file from environment ", config_file)
        return config_file


AITER_CONFIGS = AITER_CONFIG()
# config_env end here

# All native and package-data locations come from the selected package's
# versioned build layout. Explicit overrides fail at use if unavailable.
AITER_META_DIR = str(BUILD_CONTEXT.resource("metadata"))
AITER_CSRC_DIR = str(BUILD_CONTEXT.resource("native"))
AITER_GRADLIB_DIR = str(BUILD_CONTEXT.resource("native", "blas"))
# Triton-only wheels intentionally omit assembly payloads. Resolve the location
# here; readers require the selected resource only when an assembly op is used.
AITER_ASM_DIR = str(BUILD_CONTEXT.resource("kernels", required=False))
OPUS_GEN_CO_DIR = str(
    BUILD_CONTEXT.resource("native", "opus_gemm/gen_co", required=False)
)
CK_3RDPARTY_DIR = str(BUILD_CONTEXT.resource("ck", required=False))
CK_HELPER_DIR = str(BUILD_CONTEXT.resource("ck_helper", required=False))
CK_DIR = CK_3RDPARTY_DIR
HIP_KITTENS_DIR = str(BUILD_CONTEXT.resource("hip_kittens", required=False))


@functools.lru_cache(maxsize=1)
def get_asm_dir():
    from aiter.kernels import prepare_native

    target = get_gfx()
    return str(
        Path(prepare_native(context=BUILD_CONTEXT, targets=(target,)).root) / target
    )


@functools.lru_cache(maxsize=1)
def get_user_jit_dir() -> str:
    path = cache_directory()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


bd_dir = str(cache_directory() / "build")
