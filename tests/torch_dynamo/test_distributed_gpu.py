# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``torch.distributed`` collectives of GPU tensors run end to end over NCCL (RCCL on ROCm), in a single process: a
one-rank ``gloo`` process group bootstraps the NCCL communicator of the program.
"""

import os
import tempfile

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed._functional_collectives as fc  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402
from dace.libraries import comm  # noqa: E402


@pytest.mark.gpu
@pytest.mark.torch
def test_collectives_of_gpu_tensors():
    """An all-reduce and an all-gather of a strided view (packed into a contiguous buffer) match eager torch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    path = tempfile.mktemp(prefix="dace_pg_")
    dist.init_process_group("gloo", init_method="file://" + path, rank=0, world_size=1)
    try:
        group = dist.group.WORLD

        def f(x, y):
            reduced = fc.all_reduce(x * 2, "sum", group)
            gathered = fc.all_gather_tensor(y.t(), 0, group)
            return reduced + torch.sin(y), gathered * 3

        backend = DaceBackend()
        x, y = torch.randn(4, 3, device="cuda"), torch.randn(4, 3, device="cuda")
        result = torch.compile(f, backend=backend, fullgraph=True)(x, y)
        torch.testing.assert_close(result, (x * 2 + torch.sin(y), y.t() * 3))

        collectives = (comm.Allreduce, comm.Allgather)
        libnodes = [n for n, _ in backend.last_sdfg.all_nodes_recursive() if isinstance(n, collectives)]
        assert {type(n) for n in libnodes} == set(collectives)
        assert all(n.implementation == "NCCL" for n in libnodes)
    finally:
        dist.destroy_process_group()
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    test_collectives_of_gpu_tensors()
