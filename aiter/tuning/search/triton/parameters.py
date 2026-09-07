# SPDX-License-Identifier: MIT
"""Pure configuration parsing and search-space pruning."""

config_parms_key = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "cache_modifier",
    "NUM_KSPLIT",
]


def validate_configuration(values):
    if len(values) != len(config_parms_key) or any(
        type(value) is not int for value in values
    ):
        raise ValueError("configuration requires ten integer parameters")
    # Zero stages/waves may request compiler defaults; dimensions, grouping,
    # warp count and split count always describe positive extents.
    for index in (0, 1, 2, 3, 4, 9):
        if values[index] <= 0:
            raise ValueError(f"{config_parms_key[index]} must be positive")
    for index in (0, 1, 2, 4):
        if values[index] & (values[index] - 1):
            raise ValueError(f"{config_parms_key[index]} must be a power of two")
    if any(values[index] < 0 for index in (5, 6, 7)):
        raise ValueError(
            "compiler stage, wave and instruction hints cannot be negative"
        )
    if values[8] not in (0, 1):
        raise ValueError("cache_modifier must be 0 (.cg) or 1 (default)")


def get_config_list(argv: list[str]) -> list[dict | None]:
    if len(argv) % len(config_parms_key):
        raise ValueError("each configuration must contain all ten parameters")
    config_argv = argv
    num_config_parms_key = len(config_parms_key)
    config_list = []
    while len(config_argv) >= num_config_parms_key:
        values = [int(value) for value in config_argv[:num_config_parms_key]]
        validate_configuration(values)
        config_list.append(
            {config_parms_key[i]: values[i] for i in range(num_config_parms_key)}
        )
        config_list[-1]["cache_modifier"] = (
            ".cg" if config_list[-1]["cache_modifier"] == 0 else None
        )
        config_argv = config_argv[num_config_parms_key:]

    if len(config_list) == 0:
        config_list = [None]

    return config_list


def get_input_shape(argv: list[str]) -> list[int]:
    return [int(v) for v in argv]


def get_input_shape_and_config_list(
    argv: list[str], shape_size: int = 3
) -> tuple[list[int], list[dict | None]]:
    if len(argv) < shape_size + 1:
        raise ValueError(
            f"expected {shape_size} positive dimensions before configuration parameters"
        )
    input_shape = get_input_shape(argv[1 : shape_size + 1])
    if any(value <= 0 for value in input_shape):
        raise ValueError("input dimensions must be positive")
    config_list = get_config_list(argv[shape_size + 1 :])
    return input_shape, config_list


def read_screen_file(filename, case_data):
    """Read completed measurements; fail on malformed or nonfinite timings."""
    import math
    from pathlib import Path

    path = Path(filename)
    if not path.is_file():
        return
    configuration = None
    for line in path.read_text().splitlines():
        if line.startswith("screencase "):
            if configuration is not None:
                raise ValueError(f"missing timing before next case in {path}")
            configuration = line.removeprefix("screencase ").strip()
            values = configuration.split()
            if len(values) != len(config_parms_key):
                raise ValueError(
                    f"each screen measurement needs exactly one explicit configuration in {path}"
                )
            get_config_list(values)
        elif line.strip().endswith("(us)") and configuration is not None:
            latency = float(line.split()[0])
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid latency in {path}")
            case_data.append([latency, configuration])
            configuration = None
    if configuration is not None:
        raise ValueError(f"unfinished measurement in {path}")


def pre_pruning_rules(M: int, N: int, K: int, config_list: list[int], verbose: bool):
    (
        _BLOCK_SIZE_M,
        _BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        _num_warps,
        num_stages,
        _waves_per_eu,
        _matrix_instr_nonkdim,
        _cache_modifier,
        NUM_KSPLIT,
    ) = config_list
    # remove cases
    if BLOCK_SIZE_K >= 2 * (K // NUM_KSPLIT):
        if verbose:
            print(
                f"Remove case {config_list} because BLOCK_SIZE_K >= 2 * (K // NUM_KSPLIT)"
            )
        return True
    if NUM_KSPLIT > 1 and GROUP_SIZE_M > 1:
        if verbose:
            print(
                f"Remove case {config_list} because NUM_KSPLIT > 1 and GROUP_SIZE_M > 1"
            )
        return True
    if BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages > 1:  # k_itr == 1 case
        if verbose:
            print(
                f"Remove case {config_list} because BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages > 1"
            )
        return True
    if BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1:  # k_itr > 1 case
        if verbose:
            print(
                f"Remove case {config_list} because BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1"
            )
        return True
    return False


def driver_help(argv):
    if len(argv) == 1 or argv[1:] in (["--help"], ["-h"]):
        print("Usage: DRIVER M N K [CONFIG ...]")
        print("Each configuration has ten integer parameters, in this order:")
        print(" ".join(config_parms_key))
        print("Omit CONFIG to profile the currently selected implementation.")
        return True
    return False
