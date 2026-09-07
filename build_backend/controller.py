# SPDX-License-Identifier: MIT
"""Execute a build plan through phase adapters with fail-fast ordering."""

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .plan import BuildPlan


class BuildPhases(Protocol):
    def initialize(self, plan): ...
    def flydsl(self, plan): ...
    def native(self, job, plan): ...
    def pretune(self, plan): ...


class BuildController:
    def __init__(self, phases):
        self.phases = phases

    def execute(self, plan):
        if not isinstance(plan, BuildPlan):
            raise TypeError("execute requires a BuildPlan")
        if not plan.mode and not plan.native_jobs:
            return
        self.phases.initialize(plan)
        if plan.mode:
            self.phases.flydsl(plan)
        with ThreadPoolExecutor(max_workers=plan.native_workers) as executor:
            list(
                executor.map(
                    lambda job: self.phases.native(job, plan), plan.native_jobs
                )
            )
        if plan.pretune_modules:
            self.phases.pretune(plan)
