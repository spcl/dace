# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``distributed_compile`` builds on rank 0 and loads on the other ranks; a failure on any rank raises on every rank.

Run under MPI::

    mpirun -n 2 python -m pytest --with-mpi -m mpi tests/codegen/distributed_compile_test.py
"""

import numpy as np
import pytest

import dace
from dace.sdfg import utils


def _world():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
    return comm


def _scale_sdfg(name: str, factor: float) -> dace.SDFG:

    @dace.program
    def scale(a: dace.float64[8], b: dace.float64[8]):
        b[:] = a * factor

    sdfg = scale.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


@pytest.mark.mpi
@pytest.mark.parametrize("others_hold_sdfg", [False, True])
def test_distributed_compile(others_hold_sdfg):
    comm = _world()
    rank = comm.Get_rank()
    sdfg = _scale_sdfg("distributed_compile_scale", 2.0)
    func = utils.distributed_compile(sdfg if rank == 0 or others_hold_sdfg else None, comm)
    a = np.arange(8, dtype=np.float64) + rank
    b = np.zeros(8)
    func(a=a, b=b)
    assert np.allclose(b, a * 2)


@pytest.mark.mpi
def test_distributed_compile_failure_on_rank_0():
    """Rank 0 raises its error, the other ranks a ``DistributedCompilationError`` (instead of waiting)."""
    comm = _world()
    rank = comm.Get_rank()
    sdfg = _scale_sdfg("distributed_compile_invalid", 2.0)
    sdfg.start_state.add_access("undefined_container")  # Fails validation
    expected = dace.sdfg.InvalidSDFGError if rank == 0 else utils.DistributedCompilationError
    with pytest.raises(expected):
        utils.distributed_compile(sdfg if rank == 0 else None, comm)


@pytest.mark.mpi
def test_distributed_compile_rank_dependent_sdfg():
    """An SDFG that differs from rank 0's cannot use rank 0's build: every rank raises."""
    comm = _world()
    rank = comm.Get_rank()
    sdfg = _scale_sdfg("distributed_compile_rank_dependent", 2.0 if rank == 0 else 3.0)
    with pytest.raises(Exception) as info:
        utils.distributed_compile(sdfg, comm)
    if rank == 0:
        assert isinstance(info.value, utils.DistributedCompilationError)
        assert "rank 1: ValueError: The SDFG distributed_compile_rank_dependent differs" in str(info.value)
    else:
        assert isinstance(info.value, ValueError)


if __name__ == "__main__":
    test_distributed_compile(False)
    test_distributed_compile(True)
    test_distributed_compile_failure_on_rank_0()
    test_distributed_compile_rank_dependent_sdfg()
