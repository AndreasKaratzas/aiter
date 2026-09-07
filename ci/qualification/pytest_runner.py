"""Check the import origin inside the same process that executes GPU tests."""

import sys

import pytest

from ci.qualification.probe import probe


class ExecutionEnvironment:
    def pytest_sessionstart(self, session):
        probe()

    def pytest_sessionfinish(self, session, exitstatus):
        probe()


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[ExecutionEnvironment()]))
