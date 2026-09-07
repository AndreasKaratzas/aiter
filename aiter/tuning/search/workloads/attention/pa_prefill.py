# SPDX-License-Identifier: MIT
"""Shared attention.pa_prefill workload generation and numerical references."""

import random

import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
