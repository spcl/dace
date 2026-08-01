# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for method-replacement calls whose receiver is a compile-time Python
OBJECT rather than a repository container (``commworld.Bcast(A)``, where
``commworld`` is an ``mpi4py`` communicator from the program's closure).

The registry keys these on the receiver's class
(``replaces_method('Intracomm', 'Bcast')``) and the implementations resolve the
object back through the visitor's globals, so the frontend records the object
on the emitted ``ReplacementCallNode`` and the expansion publishes it under the
receiver name. The tests cover both ends: the tree carries no interpreter
fallback, and it still converts to a valid SDFG.

Also covered here: an opaque HANDLE one replacement produces and another
consumes by name (``dace.comm.Subarray`` into ``dace.comm.Redistribute``).
Those are Python-object-typed like a fallback result, but they are exactly
what the next replacement takes, and the registry state they install
(``sdfg.subarrays``) has to exist in the build-time viability trial too.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import from_schedule_tree

mpi4py = pytest.importorskip('mpi4py')
from mpi4py import MPI  # noqa: E402 -- guarded by importorskip above

P = dace.symbol('P', dace.int32)
commworld = MPI.COMM_WORLD


def _lowered(program, *args):
    """The program's schedule tree, asserted callback-free."""
    tree = nextgen.parse_program(program, *args)
    reasons = [node.reason for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]
    assert not reasons, f'unexpected interpreter fallbacks: {reasons}'
    return tree


def test_communicator_method_records_its_receiver_object():

    @dace.program
    def comm_bcast(A: dace.int32[10]):
        commworld.Bcast(A)

    tree = _lowered(comm_bcast)
    calls = [node for node in tree.preorder_traversal() if isinstance(node, tn.ReplacementCallNode)]
    assert len(calls) == 1
    assert calls[0].qualname == 'Bcast'
    assert calls[0].receiver == 'commworld'
    assert calls[0].receiver_object is commworld
    # The receiver names a closure object, not a container, so it is
    # deliberately absent from the data arguments.
    assert 'commworld' not in calls[0].data_arguments
    from_schedule_tree(tree).validate()


@pytest.mark.parametrize('collective', ['alltoall', 'send_recv'])
def test_communicator_collectives_convert_to_an_sdfg(collective):

    @dace.program
    def alltoall(A: dace.int32[10], B: dace.int32[10]):
        commworld.Alltoall(A, B)

    @dace.program
    def send_recv(A: dace.int32[10]):
        commworld.Send(A, 1, 0)
        commworld.Recv(A, 0, 0)

    program = {'alltoall': alltoall, 'send_recv': send_recv}[collective]
    from_schedule_tree(_lowered(program)).validate()


def test_process_grid_from_a_communicator_method():
    """``Create_cart`` on the communicator produces a grid container, whose own
    methods then resolve through the ordinary container-receiver path."""

    @dace.program
    def sub_grid_bcast(A: dace.int32[10]):
        pgrid = commworld.Create_cart([2, 2])
        sgrid = pgrid.Sub([True, False])
        sgrid.Bcast(A)

    tree = _lowered(sub_grid_bcast)
    calls = [node for node in tree.preorder_traversal() if isinstance(node, tn.ReplacementCallNode)]
    assert [node.qualname for node in calls] == ['Create_cart', 'Sub', 'Bcast']
    assert calls[0].receiver_object is commworld
    assert calls[1].receiver_object is None  # A container receiver from here on
    from_schedule_tree(tree).validate()


def test_nonblocking_requests_through_a_communicator():

    @dace.program
    def isend_irecv(rank: dace.int32, size: dace.int32):
        src = (rank - 1) % size
        dst = (rank + 1) % size
        req = np.empty((2, ), dtype=MPI.Request)
        sbuf = np.full((1, ), rank, dtype=np.int32)
        req[0] = commworld.Isend(sbuf, dst, tag=0)
        rbuf = np.empty((1, ), dtype=np.int32)
        req[1] = commworld.Irecv(rbuf, src, tag=0)
        MPI.Request.Waitall(req)
        return rbuf

    from_schedule_tree(_lowered(isend_irecv)).validate()


def test_replacement_handle_is_consumed_by_the_next_replacement():
    """``dace.comm.Subarray`` produces an opaque handle that
    ``dace.comm.Redistribute`` takes by name. The build-time viability trial
    replays the producer so its ``sdfg.subarrays`` entry exists, and the real
    expansion releases the frontend's placeholder so the producer installs the
    handle under the name the consumer looks up."""

    @dace.program
    def matrix_2d_2d(A: dace.int32[4 * P, 16]):
        a_grid = dace.comm.Cart_create([2, P // 2])
        b_grid = dace.comm.Cart_create([P // 2, 2])
        B = np.empty_like(A, shape=(16, 4 * P))
        a_arr = dace.comm.Subarray((8 * P, 8 * P), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((8 * P, 8 * P), B, process_grid=b_grid)
        rdistr = dace.comm.Redistribute(A, a_arr, B, b_arr)
        return B

    tree = _lowered(matrix_2d_2d)
    sdfg = from_schedule_tree(tree)
    sdfg.validate()
    assert 'a_arr' in sdfg.subarrays and 'b_arr' in sdfg.subarrays


if __name__ == '__main__':
    test_communicator_method_records_its_receiver_object()
    test_communicator_collectives_convert_to_an_sdfg('alltoall')
    test_communicator_collectives_convert_to_an_sdfg('send_recv')
    test_process_grid_from_a_communicator_method()
    test_nonblocking_requests_through_a_communicator()
    test_replacement_handle_is_consumed_by_the_next_replacement()
