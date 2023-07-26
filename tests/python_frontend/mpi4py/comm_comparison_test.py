# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests comparison operators with communicator objects. """
import dace
import numpy as np
import pytest


@pytest.mark.mpi
def test_eq_commworld_0():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    @dace.program
    def eq_commworld_0(out: dace.bool[1]):
        out[0] = comm == MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    eq_commworld_0(res)
    assert res[0] == (comm == MPI.COMM_WORLD)


@pytest.mark.mpi
def test_eq_commworld_1():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm2 = comm.Dup()

    @dace.program
    def eq_commworld_1(out: dace.bool[1]):
        out[0] = comm2 == MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    eq_commworld_1(res)
    assert res[0] == (comm2 == MPI.COMM_WORLD)


@pytest.mark.mpi
def test_eq_commworld_2():

    from mpi4py import MPI

    @dace.program
    def eq_commworld_2(out: dace.bool[1]):
        out[0] = MPI.COMM_NULL == MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    eq_commworld_2(res)
    assert res[0] == (MPI.COMM_NULL == MPI.COMM_WORLD)


@pytest.mark.mpi
def test_noteq_commworld_0():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    @dace.program
    def noteq_commworld_0(out: dace.bool[1]):
        out[0] = comm != MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    noteq_commworld_0(res)
    assert res[0] == (comm != MPI.COMM_WORLD)


@pytest.mark.mpi
def test_noteq_commworld_1():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm2 = comm.Dup()

    @dace.program
    def noteq_commworld_1(out: dace.bool[1]):
        out[0] = comm2 != MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    noteq_commworld_1(res)
    assert res[0] == (comm2 != MPI.COMM_WORLD)


@pytest.mark.mpi
def test_noteq_commworld_2():

    from mpi4py import MPI

    @dace.program
    def noteq_commworld_2(out: dace.bool[1]):
        out[0] = MPI.COMM_NULL != MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    noteq_commworld_2(res)
    assert res[0] == (MPI.COMM_NULL != MPI.COMM_WORLD)


@pytest.mark.mpi
def test_is_commworld_0():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    @dace.program
    def is_commworld_0(out: dace.bool[1]):
        out[0] = comm is MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    is_commworld_0(res)
    assert res[0] == (comm is MPI.COMM_WORLD)


@pytest.mark.mpi
def test_is_commworld_1():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm2 = comm.Dup()

    @dace.program
    def is_commworld_1(out: dace.bool[1]):
        out[0] = comm2 is MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    is_commworld_1(res)
    assert res[0] == (comm2 is MPI.COMM_WORLD)


@pytest.mark.mpi
def test_is_commworld_2():

    from mpi4py import MPI

    @dace.program
    def is_commworld_2(out: dace.bool[1]):
        out[0] = MPI.COMM_NULL is MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    is_commworld_2(res)
    assert res[0] == (MPI.COMM_NULL is MPI.COMM_WORLD)


@pytest.mark.mpi
def test_isnot_commworld_0():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    @dace.program
    def isnot_commworld_0(out: dace.bool[1]):
        out[0] = comm is MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    isnot_commworld_0(res)
    assert res[0] == (comm is MPI.COMM_WORLD)


@pytest.mark.mpi
def test_isnot_commworld_1():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm2 = comm.Dup()

    @dace.program
    def isnot_commworld_1(out: dace.bool[1]):
        out[0] = comm2 is not MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    isnot_commworld_1(res)
    assert res[0] == (comm2 is not MPI.COMM_WORLD)


@pytest.mark.mpi
def test_isnot_commworld_2():

    from mpi4py import MPI

    @dace.program
    def isnot_commworld_2(out: dace.bool[1]):
        out[0] = MPI.COMM_NULL is not MPI.COMM_WORLD
    
    res = np.zeros((1,), dtype=np.bool_)
    isnot_commworld_2(res)
    assert res[0] == (MPI.COMM_NULL is not MPI.COMM_WORLD)


if __name__ == "__main__":
    test_eq_commworld_0()
    test_eq_commworld_1()
    test_eq_commworld_2()
    test_noteq_commworld_0()
    test_noteq_commworld_1()
    test_noteq_commworld_2()
    test_is_commworld_0()
    test_is_commworld_1()
    test_is_commworld_2()
    test_isnot_commworld_0()
    test_isnot_commworld_1()
    test_isnot_commworld_2()
