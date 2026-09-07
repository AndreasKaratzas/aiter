"""Reject numerical failures that legacy script drivers only write to logs."""

import re
from pathlib import Path

from ci.common.json import require


def validate_script_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    failure = re.search(
        r"\[checkAllclose[^\]\r\n]*(?:failed!|catastrophic!|skipped:\s*empty mask)[^\]\r\n]*\]",
        text,
        re.IGNORECASE,
    )
    require(
        failure is None, "legacy script logged a failed or skipped numerical comparison"
    )
