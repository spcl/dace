import dace
import numpy as np
import pytest

from dace.data.distributed import ProcessGrid
from dace.libraries.mpi.nodes.dummy import Dummy
from dace.libraries.mpi.nodes.bcast import Bcast


@pytest.mark.mpi
def test_create_cart_bcast_uses_process_grid_descriptor():
    MPI = pytest.importorskip('mpi4py.MPI')

    @dace.program
    def pgrid_bcast(A: dace.int32[10]):
        pgrid = MPI.COMM_WORLD.Create_cart([1, 1])
        if pgrid != MPI.COMM_NULL:
            pgrid.Bcast(A)

    sdfg = pgrid_bcast.to_sdfg()
    process_grids = sdfg.process_grids

    assert not hasattr(sdfg, '_pgrids')
    assert len(process_grids) == 1
    pgrid_name, pgrid = next(iter(process_grids.items()))
    assert isinstance(pgrid, ProcessGrid)
    assert pgrid.name == pgrid_name
    assert sdfg.arrays[pgrid_name] is pgrid

    bcasts = [node for state in sdfg.states() for node in state.nodes() if isinstance(node, Bcast)]
    assert len(bcasts) == 1
    assert bcasts[0].grid == pgrid_name


@pytest.mark.mpi
def test_create_cart_subgrid_bcast_uses_descriptor_name():
    MPI = pytest.importorskip('mpi4py.MPI')

    @dace.program
    def subgrid_bcast(A: dace.int32[10], rank: dace.int32):
        pgrid = MPI.COMM_WORLD.Create_cart([2, 1])
        if pgrid != MPI.COMM_NULL:
            sgrid = pgrid.Sub([False, True])
            pgrid.Bcast(A)
        B = np.empty_like(A)
        B[:] = rank % 10
        if pgrid != MPI.COMM_NULL:
            sgrid.Bcast(B)
        A[:] = B

    sdfg = subgrid_bcast.to_sdfg()
    process_grids = sdfg.process_grids

    assert len(process_grids) == 2
    assert all(pgrid.name == pgrid_name for pgrid_name, pgrid in process_grids.items())

    init_code = sdfg.init_code['frame'].as_string
    dummy_fields = '\n'.join(field for state in sdfg.states() for node in state.nodes() if isinstance(node, Dummy)
                             for field in node.fields)
    for pgrid_name in process_grids:
        assert f'{pgrid_name}_comm' in init_code
        assert f'{pgrid_name}_comm' in dummy_fields

    bcasts = [node for state in sdfg.states() for node in state.nodes() if isinstance(node, Bcast)]
    assert len(bcasts) == 2
    assert all(bcast.grid in process_grids for bcast in bcasts)

    sdfg.expand_library_nodes()
    tasklet_code = '\n'.join(node.code.as_string for state in sdfg.states() for node in state.nodes()
                             if isinstance(node, dace.nodes.Tasklet))
    for pgrid_name in process_grids:
        assert f'{pgrid_name}_comm' in tasklet_code


if __name__ == "__main__":
    test_create_cart_bcast_uses_process_grid_descriptor()
    test_create_cart_subgrid_bcast_uses_descriptor_name()
