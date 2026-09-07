# SPDX-License-Identifier: MIT
"""A bounded eight-rank check of the existing native custom-allreduce path."""

import datetime
import json
from pathlib import Path


def _rank_check(rank, world_size, store_port, result_dir):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    communicator = None
    store = dist.TCPStore(
        "127.0.0.1",
        store_port,
        world_size=None,
        is_master=False,
        timeout=datetime.timedelta(seconds=120),
    )
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    checks = []
    try:
        communicator = CustomAllreduce(
            dist.group.WORLD, torch.device("cuda", rank), max_size=1024 * 1024
        )
        assert not communicator.disabled, "AITER custom allreduce was disabled"
        assert communicator._ptr, "The native communicator was not initialized"
        for dtype in (torch.float16, torch.bfloat16):
            pattern = (torch.arange(2048, device=rank) % 16).reshape(2, 1024) / 16
            x = (pattern + rank).to(dtype)
            expected = (pattern * world_size + sum(range(world_size))).to(dtype)
            assert communicator.should_custom_ar(x)
            for _ in range(3):
                out = communicator.custom_all_reduce(x)
                assert out is not None, "AITER silently selected a fallback"
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
            checks.append(f"{dtype}:eager")

            stream = torch.cuda.Stream(device=rank)
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with communicator.capture(), torch.cuda.graph(graph, stream=stream):
                captured = communicator.custom_all_reduce(x)
            assert captured is not None
            for increment in (0, 1, 2):
                x.copy_((pattern + rank + increment).to(dtype))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    captured, expected + world_size * increment, rtol=0, atol=0
                )
            checks.append(f"{dtype}:capture")
            del graph
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        if communicator is not None:
            communicator.close()
            assert not getattr(communicator, "_ptr", 0)
        dist.destroy_process_group()
    Path(result_dir, f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "checks": checks, "closed": True}) + "\n"
    )


def test_eight_rank_custom_allreduce_eager_and_capture(tmp_path):
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    assert (
        torch.version.hip and torch.cuda.device_count() >= 8
    ), "This test requires eight visible ROCm GPUs"
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, wait_for_workers=False
    )
    mp.spawn(
        _rank_check,
        args=(8, store.port, str(tmp_path)),
        nprocs=8,
        join=True,
    )
    for rank in range(8):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["rank"] == rank and result["closed"]
        assert result["checks"] == [
            "torch.float16:eager",
            "torch.float16:capture",
            "torch.bfloat16:eager",
            "torch.bfloat16:capture",
        ]
