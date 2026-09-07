# SPDX-License-Identifier: MIT
"""Logging configuration, independent of the GPU runtime."""

import logging
import os

logger = logging.getLogger("aiter")


def getLogger():
    if not logger.handlers:
        # Configure log level from environment variable
        # Valid values: DEBUG, INFO (default), WARNING, ERROR
        log_level_str = os.getenv("AITER_LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        if log_level_str not in valid_levels:
            print(
                f"\033[93m[aiter] Warning: Invalid AITER_LOG_LEVEL '{log_level_str}', "
                f"using 'INFO'. Valid values: {', '.join(valid_levels)}\033[0m"
            )
            log_level_str = "INFO"

        log_level = getattr(logging, log_level_str)
        logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        if int(os.environ.get("AITER_LOG_MORE", "0")):
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)03d - %(processName)s:%(process)d - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                fmt="[%(name)s] %(message)s",
            )
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)

        logger.addHandler(console_handler)
        logger.propagate = False

    return logger
