# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import numpy as np
import torch

from aiter import logger


def tensor_dump(x: torch.Tensor, name: str, dir="./"):
    x_cpu = x.cpu().view(torch.uint8)
    filename = f"{dir}/{name}.bin"
    x_cpu.numpy().tofile(filename)
    logger.info(f"saving {filename} {x.shape}, {x.dtype}")

    with open(f"{dir}/{name}.meta", "w") as f:
        f.writelines([f"{el}\n" for el in [x.shape, x.dtype]])


def tensor_load(filename: str):
    DWs = np.fromfile(filename, dtype=np.uint32)
    metafile = ".".join(filename.split(".")[:-1]) + ".meta"
    with open(metafile) as fh:
        shape, dtype = [eval(line.strip()) for line in fh]
    return torch.tensor(DWs).view(dtype).view(shape)
