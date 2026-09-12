# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every MPI library node must resolve its communicator through the shared
``resolve_comm`` helper: it reads an optional ``_comm`` (raw ``opaque(MPI_Comm)``)
input connector first, else an optional ``_grid`` process-grid connector, else
falls back to ``MPI_COMM_WORLD``.  Neither connector is declared in the node's
fixed ``inputs`` -- the caller wires them dynamically (``add_in_connector`` +
edge) and the expansion picks them up via ``expanded_input_connectors``.

These tests assert at the code-generation level (no MPI runtime needed): they
expand each node's tasklet and check the emitted C uses the resolved connector
token, NOT a hardcoded ``MPI_COMM_WORLD``.  They cover the nodes that previously
hardcoded the world communicator (allgather / scatter / redistribute) plus the
uniformity fix on barrier, and confirm backward compatibility (no connector ->
``MPI_COMM_WORLD``).
"""
from unittest import mock

import dace
import numpy as np
import pytest
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
from dace.libraries.mpi.nodes.comm_f2c import CommF2c
from dace.libraries.mpi.nodes.redistribute import Redistribute
import dace.frontend.python.replacements.mpi as comm_repl

# Comm-argument spellings of the wired connectors; used to prove a connector is
# (or is not) the communicator of an emitted MPI call.  Matching the bare name is
# ambiguous (e.g. ``int _commsize;`` contains ``_comm``), so we look for the
# connector immediately in an MPI call's comm-argument position.
_COMM_AS_COMM = ("(_comm", "_comm,", "_comm)", "_comm;")
_GRID_AS_COMM = ("(_grid", "_grid,", "_grid)", "_grid;")

###############################################################################
# Helpers
###############################################################################


class _MockProgramVisitor:
    """Minimal ProgramVisitor stand-in so ``_cart_create`` can allocate grids.
    Names are distinctive (``__testgrid*``) to avoid colliding with grids a real
    frontend program (e.g. redistribute) already created in the SDFG."""

    def __init__(self):
        self._counter = 0

    def get_target_name(self, output_index=None, default=None):
        self._counter += 1
        return default or f'__testgrid{self._counter}'

    def __getattr__(self, name):
        value = mock.MagicMock()
        setattr(self, name, value)
        return value


def _expanded_code(node, state, sdfg):
    """Expand a single MPI library node and return (emitted C code, in_connectors)."""
    expansion = node.implementations[node.default_implementation]
    tasklet = expansion.expansion(node, state, sdfg)
    return tasklet.code.as_string, tasklet.in_connectors


def _wire_comm(sdfg, state, node):
    """Wire a ``_comm`` connector fed by a ``CommF2c`` (mirrors the dace-fortran
    ``_wire_user_comm`` path): a Fortran integer handle -> ``opaque(MPI_Comm)``."""
    sdfg.add_scalar('_fh', dace.int32, transient=False)
    sdfg.add_scalar('_usercomm', dace.dtypes.opaque("MPI_Comm"), transient=True)
    f2c = CommF2c('_commf2c_')
    state.add_node(f2c)
    state.add_edge(state.add_read('_fh'), None, f2c, '_fcomm', Memlet(data='_fh', subset='0'))
    cw = state.add_access('_usercomm')
    state.add_edge(f2c, '_comm', cw, None, Memlet(data='_usercomm', subset='0'))
    node.add_in_connector('_comm', dace.dtypes.opaque("MPI_Comm"))
    state.add_edge(cw, None, node, '_comm', Memlet(data='_usercomm', subset='0'))


def _wire_grid(sdfg, state, node):
    """Wire a ``_grid`` connector fed by a cartesian process grid (mirrors the
    ``dace.comm.Bcast(..., grid=...)`` frontend path)."""
    grid = comm_repl._cart_create(_MockProgramVisitor(), sdfg, state, [1, 2])
    node.add_in_connector('_grid')
    state.add_edge(state.add_read(grid), None, node, '_grid', Memlet(data=grid))


###############################################################################
# SDFG builders (one fresh node per variant so resolution is independent)
###############################################################################


def _build_allgather():
    n, p = dace.symbol("n"), dace.symbol("p")
    sdfg = dace.SDFG("allgather_comm")
    state = sdfg.add_state("s")
    sdfg.add_array("inA", [n], dace.float64)
    sdfg.add_array("outA", [n * p], dace.float64)
    node = mpi.nodes.allgather.Allgather("allgather")
    state.add_memlet_path(state.add_access("inA"), node, dst_conn="_inbuffer", memlet=Memlet.simple("inA", "0:n"))
    state.add_memlet_path(node, state.add_access("outA"), src_conn="_outbuffer", memlet=Memlet.simple("outA", "0:n*p"))
    return sdfg, state, node


def _build_scatter():
    n, p = dace.symbol("n"), dace.symbol("p")
    sdfg = dace.SDFG("scatter_comm")
    state = sdfg.add_state("s")
    sdfg.add_array("inbuf", [n * p], dace.float64)
    sdfg.add_array("outbuf", [n], dace.float64)
    sdfg.add_array("root", [1], dace.int32)
    node = mpi.nodes.scatter.Scatter("scatter")
    state.add_memlet_path(state.add_access("inbuf"), node, dst_conn="_inbuffer", memlet=Memlet.simple("inbuf", "0:n*p"))
    state.add_memlet_path(state.add_access("root"), node, dst_conn="_root", memlet=Memlet.simple("root", "0:1"))
    state.add_memlet_path(node,
                          state.add_access("outbuf"),
                          src_conn="_outbuffer",
                          memlet=Memlet.simple("outbuf", "0:n"))
    return sdfg, state, node


def _build_barrier():
    sdfg = dace.SDFG("barrier_comm")
    state = sdfg.add_state("s")
    node = mpi.nodes.barrier.Barrier("barrier")
    state.add_node(node)
    return sdfg, state, node


def _build_redistribute():
    P = dace.symbol('P', dace.int32)

    @dace.program
    def matrix_2d_2d(A: dace.int32[4 * P, 16]):
        a_grid = dace.comm.Cart_create([2, P // 2])
        b_grid = dace.comm.Cart_create([P // 2, 2])
        B = np.empty_like(A, shape=(16, 4 * P))
        a_arr = dace.comm.Subarray((8 * P, 8 * P), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((8 * P, 8 * P), B, process_grid=b_grid)
        dace.comm.Redistribute(A, a_arr, B, b_arr)
        return B

    sdfg = matrix_2d_2d.to_sdfg(simplify=False)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, Redistribute):
                return sdfg, state, node
    raise AssertionError("Redistribute node not found in generated SDFG")


_BUILDERS = {
    "allgather": _build_allgather,
    "scatter": _build_scatter,
    "barrier": _build_barrier,
    "redistribute": _build_redistribute,
}

###############################################################################
# Tests
###############################################################################


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_default_comm_is_world(name):
    """No ``_comm`` / ``_grid`` wired -> the node falls back to MPI_COMM_WORLD
    (backward compatible with existing callers/tests)."""
    sdfg, state, node = _BUILDERS[name]()
    code, _ = _expanded_code(node, state, sdfg)
    assert "MPI_COMM_WORLD" in code
    for pat in _COMM_AS_COMM + _GRID_AS_COMM:
        assert pat not in code


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_comm_connector_is_honored(name):
    """A wired ``_comm`` connector is used verbatim; no hardcoded world comm
    remains, and the connector is threaded into the expanded tasklet."""
    sdfg, state, node = _BUILDERS[name]()
    _wire_comm(sdfg, state, node)
    code, in_connectors = _expanded_code(node, state, sdfg)
    assert "MPI_COMM_WORLD" not in code
    assert any(pat in code for pat in _COMM_AS_COMM)
    assert "_comm" in in_connectors


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_grid_connector_is_honored(name):
    """A wired ``_grid`` process-grid connector is used as the communicator; no
    hardcoded world comm remains, and the connector is threaded into the tasklet."""
    sdfg, state, node = _BUILDERS[name]()
    _wire_grid(sdfg, state, node)
    code, in_connectors = _expanded_code(node, state, sdfg)
    assert "MPI_COMM_WORLD" not in code
    assert any(pat in code for pat in _GRID_AS_COMM)
    assert "_grid" in in_connectors


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_comm_takes_priority_over_grid(name):
    """When both are wired, ``_comm`` wins (matches ``resolve_comm`` ordering):
    every emitted MPI call uses ``_comm``, never the grid."""
    sdfg, state, node = _BUILDERS[name]()
    _wire_grid(sdfg, state, node)
    _wire_comm(sdfg, state, node)
    code, in_connectors = _expanded_code(node, state, sdfg)
    assert "MPI_COMM_WORLD" not in code
    assert any(pat in code for pat in _COMM_AS_COMM)
    assert "_comm" in in_connectors and "_grid" in in_connectors
    for pat in _GRID_AS_COMM:
        assert pat not in code


if __name__ == "__main__":
    for _n in _BUILDERS:
        test_default_comm_is_world(_n)
        test_comm_connector_is_honored(_n)
        test_grid_connector_is_honored(_n)
        test_comm_takes_priority_over_grid(_n)
    print("OK")
