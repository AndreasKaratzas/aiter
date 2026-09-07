# SPDX-License-Identifier: MIT
"""One missing-prerequisite policy for developer and qualification runs."""


def require_available(condition, reason, request):
    """Skip optional local discovery; fail an explicitly required qualification."""
    if condition:
        return
    import pytest

    if request.config.getoption("require_capabilities"):
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)
