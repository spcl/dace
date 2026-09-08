# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.data.distributed import ProcessGrid
from dace.libraries.mpi.nodes.dummy import Dummy
from dace.libraries.mpi.nodes.bcast import Bcast


def _incoming_descriptor_name(sdfg, node, connector):
    states = [state for state in sdfg.states() if node in state.nodes()]
    assert len(states) == 1

    edges = list(states[0].in_edges_by_connector(node, connector))
    assert len(edges) == 1
    return edges[0].data.data


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
    assert _incoming_descriptor_name(sdfg, bcasts[0], '_grid') == pgrid_name


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
        assert f'__state->{pgrid_name}' in init_code
        assert f'MPI_Comm {pgrid_name};' in dummy_fields

    bcasts = [node for state in sdfg.states() for node in state.nodes() if isinstance(node, Bcast)]
    assert len(bcasts) == 2
    assert all(_incoming_descriptor_name(sdfg, bcast, '_grid') in process_grids for bcast in bcasts)

    sdfg.expand_library_nodes()
    tasklet_code = '\n'.join(node.code.as_string for state in sdfg.states() for node in state.nodes()
                             if isinstance(node, dace.nodes.Tasklet))
    assert '_grid' in tasklet_code
    assert '_comm' not in tasklet_code


@pytest.mark.mpi
def test_nextgen_create_cart_bcast_installs_one_process_grid():
    """
    The same program through the next-generation frontend, asserted on ITS
    SDFG. The tests above build the classic frontend's, so they only see
    nextgen through the callback-discrepancy check -- and a grid-creating call
    that lowers without a callback can still bind the wrong descriptor.

    The frontend types the name a grid-creating call binds (so that
    ``pgrid.Bcast(A)`` resolves through the ``ProcessGrid`` registrations),
    while the grid itself is installed by the replacement. If that declaration
    were left in place, ``add_pgrid`` would install the real grid BESIDE it and
    the collective would be wired to an uninitialized communicator.
    """
    MPI = pytest.importorskip('mpi4py.MPI')
    from dace.frontend.python import nextgen
    from dace.sdfg.analysis.schedule_tree import treenodes as tn

    @dace.program
    def nextgen_pgrid_bcast(A: dace.int32[10]):
        pgrid = MPI.COMM_WORLD.Create_cart([1, 1])
        if pgrid != MPI.COMM_NULL:
            pgrid.Bcast(A)

    tree = nextgen.parse_program(nextgen_pgrid_bcast, np.zeros(10, dtype=np.int32))
    assert not [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]

    sdfg = tree.as_sdfg()
    sdfg.validate()
    assert len(sdfg.process_grids) == 1
    pgrid_name, pgrid = next(iter(sdfg.process_grids.items()))
    assert pgrid.name == pgrid_name
    assert sdfg.arrays[pgrid_name] is pgrid

    # Only an INSTALLED descriptor emits its init/exit code and its fields.
    assert f'__state->{pgrid_name}' in sdfg.init_code['frame'].as_string
    dummy_fields = '\n'.join(field for state in sdfg.states() for node in state.nodes() if isinstance(node, Dummy)
                             for field in node.fields)
    assert f'MPI_Comm {pgrid_name};' in dummy_fields

    bcasts = [node for state in sdfg.states() for node in state.nodes() if isinstance(node, Bcast)]
    assert len(bcasts) == 1
    assert _incoming_descriptor_name(sdfg, bcasts[0], '_grid') == pgrid_name


@pytest.mark.mpi
def test_nextgen_subgrid_is_linked_to_its_parent():
    """A sub-grid through the next-generation frontend: two installed grids,
    with the sub-grid's parent link naming the parent's container."""
    MPI = pytest.importorskip('mpi4py.MPI')
    from dace.frontend.python import nextgen
    from dace.sdfg.analysis.schedule_tree import treenodes as tn

    @dace.program
    def nextgen_subgrid_bcast(A: dace.int32[10]):
        pgrid = MPI.COMM_WORLD.Create_cart([2, 1])
        if pgrid != MPI.COMM_NULL:
            sgrid = pgrid.Sub([False, True])
            sgrid.Bcast(A)

    tree = nextgen.parse_program(nextgen_subgrid_bcast, np.zeros(10, dtype=np.int32))
    assert not [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]

    sdfg = tree.as_sdfg()
    sdfg.validate()
    assert len(sdfg.process_grids) == 2
    assert all(pgrid.name == pgrid_name for pgrid_name, pgrid in sdfg.process_grids.items())
    subgrids = [pgrid for pgrid in sdfg.process_grids.values() if pgrid.is_subgrid]
    assert len(subgrids) == 1
    assert subgrids[0].parent_grid in sdfg.process_grids
    assert list(subgrids[0].shape) == [1]  # ``color`` keeps the second dimension

    bcasts = [node for state in sdfg.states() for node in state.nodes() if isinstance(node, Bcast)]
    assert len(bcasts) == 1
    assert _incoming_descriptor_name(sdfg, bcasts[0], '_grid') == subgrids[0].name


if __name__ == "__main__":
    test_create_cart_bcast_uses_process_grid_descriptor()
    test_create_cart_subgrid_bcast_uses_descriptor_name()
    test_nextgen_create_cart_bcast_installs_one_process_grid()
    test_nextgen_subgrid_is_linked_to_its_parent()
