# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
:func:`dace.libraries.comm.schedule.overlap_communication` moves nonblocking starts up and waits down in a schedule
tree, keeps the order in which communication starts, and puts every communication node in a state of its own.

The test that runs the program needs MPI::

    mpirun -n 2 python -m pytest --with-mpi -m mpi tests/library/comm/overlap_schedule_test.py
"""

import numpy as np
import pytest

import dace
from dace import nodes, subsets
from dace.libraries import comm
from dace.libraries.comm.schedule import overlap_communication
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _map(name: str, code: str, inputs: dict[str, str], output: str) -> tn.MapScope:
    """``output[i] = code(inputs...)`` for ``i`` in ``0:n``."""
    tasklet = nodes.Tasklet(name, set(inputs), {"__out"}, code)
    in_memlets = {conn: Memlet(f"{data}[i]") for conn, data in inputs.items()}
    body = tn.TaskletNode(tasklet, in_memlets, {"__out": Memlet(f"{output}[i]")})
    entry = nodes.MapEntry(nodes.Map(name + "_map", ["i"], subsets.Range([(0, "n - 1", 1)])))
    return tn.MapScope(node=entry, children=[body])


def _allreduce(name: str, inbuf: str, outbuf: str, request: str) -> tn.LibraryCall:
    node = comm.Allreduce(name)
    node.implementation = "MPI"
    node.nonblocking = True
    node.add_out_connector("_request")
    return tn.LibraryCall(
        node=node,
        in_memlets={"_inbuffer": Memlet(f"{inbuf}[0:n]")},
        out_memlets={"_outbuffer": Memlet(f"{outbuf}[0:n]"), "_request": Memlet(f"{request}[0]")},
    )


def _wait(name: str, buffer: str, request: str) -> tn.LibraryCall:
    node = comm.Wait(name)
    node.implementation = "MPI"
    node.remove_out_connector("_stat_tag")
    node.remove_out_connector("_stat_source")
    node.add_in_connector("_in_buffer")
    node.add_out_connector("_out_buffer")
    return tn.LibraryCall(
        node=node,
        in_memlets={"_request": Memlet(f"{request}[0]"), "_in_buffer": Memlet(f"{buffer}[0:n]")},
        out_memlets={"_out_buffer": Memlet(f"{buffer}[0:n]")},
    )


def _tree(name: str, children: list[tn.ScheduleTreeNode]) -> tn.ScheduleTreeRoot:
    n = dace.symbol("n")
    containers = {}
    for array in ("x", "out", "other"):
        containers[array] = dace.data.Array(dace.float64, [n])
    for array in ("y", "z", "reduced", "reduced2"):
        containers[array] = dace.data.Array(dace.float64, [n], transient=True)
    for request in ("request", "request2"):
        containers[request] = dace.data.Array(dace.opaque("MPI_Request"), [1], transient=True)
    return tn.ScheduleTreeRoot(
        name=name,
        containers=containers,
        symbols={"n": dace.int64},
        constants={},
        arg_names=["x", "out", "other"],
        children=children,
    )


def _program_tree(name: str) -> tn.ScheduleTreeRoot:
    """``out = allreduce(2 x) + (x + 1)``, with the independent ``x + 1`` after the wait."""
    return _tree(
        name,
        [
            _map("double", "__out = 2 * __in", {"__in": "x"}, "y"),
            _allreduce("allreduce", "y", "reduced", "request"),
            _wait("wait", "reduced", "request"),
            _map("independent", "__out = __in + 1", {"__in": "x"}, "z"),
            _map("combine", "__out = __a + __b", {"__a": "reduced", "__b": "z"}, "out"),
        ],
    )


def _labels(scope: tn.ScheduleTreeScope) -> list[str]:
    labels = []
    for child in scope.children:
        if isinstance(child, tn.StateBoundaryNode):
            labels.append("|")
        elif isinstance(child, tn.LibraryCall):
            labels.append(child.node.label)
        elif isinstance(child, tn.MapScope):
            labels.append(child.node.map.label[: -len("_map")])
        elif isinstance(child, tn.AssignNode):
            labels.append(f"assign {child.name}")
    return labels


def test_overlap_moves_independent_work_between_start_and_wait():
    stree = _program_tree("overlap_order")
    overlap_communication(stree)
    assert _labels(stree) == ["double", "|", "allreduce", "|", "independent", "|", "wait", "|", "combine"]


def test_overlap_moves_work_from_before_the_start():
    """Independent work before the producer of the reduced data moves between the start and the wait, behind the
    start and its producer."""
    stree = _tree(
        "overlap_before",
        [
            _map("independent", "__out = __in + 1", {"__in": "x"}, "z"),
            _map("double", "__out = 2 * __in", {"__in": "x"}, "y"),
            _allreduce("allreduce", "y", "reduced", "request"),
            _wait("wait", "reduced", "request"),
            _map("combine", "__out = __a + __b", {"__a": "reduced", "__b": "z"}, "out"),
        ],
    )
    overlap_communication(stree)
    assert _labels(stree) == ["double", "|", "allreduce", "|", "independent", "|", "wait", "|", "combine"]


def test_overlap_keeps_the_order_of_starts():
    """Two reductions: the second starts before the first wait, but not before the first start."""
    stree = _tree(
        "overlap_two",
        [
            _allreduce("first", "x", "reduced", "request"),
            _wait("first_wait", "reduced", "request"),
            _allreduce("second", "other", "reduced2", "request2"),
            _wait("second_wait", "reduced2", "request2"),
            _map("combine", "__out = __a + __b", {"__a": "reduced", "__b": "reduced2"}, "out"),
        ],
    )
    overlap_communication(stree)
    assert _labels(stree) == ["first", "|", "second", "|", "first_wait", "|", "second_wait", "|", "combine"]


def test_overlap_respects_dependencies():
    """A start stays after the writer of its input and the definition of a symbol it uses; a wait stays before the
    readers of its buffer and the writers of the sent data."""
    count = tn.AssignNode(name="m", value=CodeBlock("n"), edge=InterstateEdge(assignments={"m": "n"}))
    node = comm.Allreduce("allreduce")
    node.implementation = "MPI"
    node.nonblocking = True
    node.add_out_connector("_request")
    start = tn.LibraryCall(
        node=node,
        in_memlets={"_inbuffer": Memlet("y[0:m]")},
        out_memlets={"_outbuffer": Memlet("reduced[0:m]"), "_request": Memlet("request[0]")},
    )
    stree = _tree(
        "overlap_dependencies",
        [
            count,
            _map("double", "__out = 2 * __in", {"__in": "x"}, "y"),
            start,
            _wait("wait", "reduced", "request"),
            _map("overwrite", "__out = __in", {"__in": "x"}, "y"),  # Writes the input of the reduction
            _map("read", "__out = __in", {"__in": "reduced"}, "z"),
        ],
    )
    overlap_communication(stree)
    assert _labels(stree) == ["assign m", "double", "|", "allreduce", "|", "wait", "|", "overwrite", "read"]


@pytest.mark.mpi
def test_overlapped_program():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    size, rank = world.Get_size(), world.Get_rank()
    stree = _program_tree("overlap_program")
    overlap_communication(stree)
    sdfg = stree.as_sdfg()
    # The allreduce, the independent map and the wait run in this order, in states of their own
    order = []
    for state in sdfg.all_states():
        labels = {n.label for n in state.nodes() if isinstance(n, (comm.Allreduce, comm.Wait, nodes.MapEntry))}
        order.extend(sorted(labels & {"allreduce", "independent_map", "wait"}))
    assert order == ["allreduce", "independent_map", "wait"]

    compiled = None
    for r in range(size):  # The ranks share the build folder
        if r == rank:
            compiled = sdfg.compile()
        world.Barrier()
    x = np.arange(6, dtype=np.float64) + rank
    out = np.zeros(6)
    compiled(x=x, out=out, other=np.zeros(6), n=6)
    everyone = sum(np.arange(6, dtype=np.float64) + r for r in range(size))
    assert np.allclose(out, 2 * everyone + x + 1)


if __name__ == "__main__":
    test_overlap_moves_independent_work_between_start_and_wait()
    test_overlap_moves_work_from_before_the_start()
    test_overlap_keeps_the_order_of_starts()
    test_overlap_respects_dependencies()
    test_overlapped_program()
