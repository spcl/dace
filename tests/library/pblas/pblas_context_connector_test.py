# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The BLACS context a PBLAS node runs on is a dataflow value, not process-global
init state.

``dace.libraries.pblas.nodes.node.scalapack_grid_code`` resolves it the way the
MPI library nodes resolve their communicator: the connector NAME is the
discriminator, the connector is wired dynamically rather than declared in the
node's fixed inputs, and a node with nothing wired keeps the old behaviour.

  * ``_context`` -- an integer BLACS context, as produced by ``BlacsGridInit``.
  * ``_comm`` -- a raw ``opaque(MPI_Comm)``, as produced by ``CommF2c``; the node
    builds the ``Py`` x ``Px`` grid on it.
  * neither -- the ``Py`` x ``Px`` grid over ``MPI_COMM_WORLD``, which is the grid
    the environment's ``__dace_init_`` used to build.

These assert the emitted C++, so they need neither MPI nor ScaLAPACK: what has to
hold is that the grid is built where a runtime value can reach it, and that
nothing about it is left in the initializer, where a symbol is frozen at the first
call and every grid after the first would silently be the first one.
"""
import dace
import pytest

from dace import Memlet
from dace.libraries.mpi.nodes.comm_f2c import CommF2c
from dace.libraries.pblas.environments import (intel_mkl_mpich, intel_mkl_openmpi, ref_mpich, ref_openmpi)
from dace.libraries.pblas.nodes.gridinit import BlacsGridInit
from dace.libraries.pblas.nodes.pgemm import Pgemm
from dace.libraries.pblas.nodes.pgemv import Pgemv

#: Every ScaLAPACK environment, whichever MPI and BLAS it links.
ENVIRONMENTS = (intel_mkl_mpich.IntelMKLScaLAPACKMPICH, intel_mkl_openmpi.IntelMKLScaLAPACKOpenMPI,
                ref_mpich.ScaLAPACKMPICH, ref_openmpi.ScaLAPACKOpenMPI)

#: Reference and MKL spell the same BLACS calls differently.
IMPLEMENTATIONS = ('MKLMPICH', 'MKLOpenMPI', 'ReferenceMPICH', 'ReferenceOpenMPI')

GM, GN, GK = (dace.symbol(s, positive=True) for s in ('GM', 'GN', 'GK'))
LMx, LNy, LKx, LKy = (dace.symbol(s, positive=True) for s in ('LMx', 'LNy', 'LKx', 'LKy'))

###############################################################################
# Helpers
###############################################################################


def _expanded_code(node, state, sdfg, implementation):
    """The C++ ``node`` expands to under ``implementation``."""
    return node.implementations[implementation].expansion(node, state, sdfg).code.as_string


def _wire_comm(sdfg, state, node):
    """A ``_comm`` connector fed by a ``CommF2c``: a Fortran integer handle turned
    into an ``opaque(MPI_Comm)``, exactly as the MPI nodes take one."""
    sdfg.add_scalar('fcomm', dace.int32)
    sdfg.add_scalar('usercomm', dace.dtypes.opaque('MPI_Comm'), transient=True)
    f2c = CommF2c('_commf2c_')
    state.add_edge(state.add_read('fcomm'), None, f2c, '_fcomm', Memlet(data='fcomm', subset='0'))
    comm_node = state.add_access('usercomm')
    state.add_edge(f2c, '_comm', comm_node, None, Memlet(data='usercomm', subset='0'))
    node.add_in_connector('_comm', dace.dtypes.opaque('MPI_Comm'))
    state.add_edge(comm_node, None, node, '_comm', Memlet(data='usercomm', subset='0'))


def _wire_context(sdfg, state, node):
    """A ``_context`` connector fed by a ``BlacsGridInit`` over that communicator."""
    _wire_comm(sdfg, state, node)
    sdfg.add_scalar('prows', dace.int32)
    sdfg.add_scalar('pcols', dace.int32)
    sdfg.add_scalar('context', dace.int32, transient=True)
    gridinit = BlacsGridInit('_blacs_gridinit_')
    gridinit.add_in_connector('_comm', dace.dtypes.opaque('MPI_Comm'))
    state.add_edge(state.add_read('usercomm'), None, gridinit, '_comm', Memlet(data='usercomm', subset='0'))
    state.add_edge(state.add_read('prows'), None, gridinit, '_prows', Memlet(data='prows', subset='0'))
    state.add_edge(state.add_read('pcols'), None, gridinit, '_pcols', Memlet(data='pcols', subset='0'))
    context_node = state.add_access('context')
    state.add_edge(gridinit, '_context', context_node, None, Memlet(data='context', subset='0'))
    node.add_in_connector('_context', dace.int32)
    state.add_edge(context_node, None, node, '_context', Memlet(data='context', subset='0'))
    return gridinit


###############################################################################
# SDFG builders (one fresh node per variant so resolution is independent)
###############################################################################


def _build_pgemm():
    sdfg = dace.SDFG('pgemm_context')
    state = sdfg.add_state('s')
    sdfg.add_array('A', (LMx, LKy), dace.float64)
    sdfg.add_array('B', (LKx, LNy), dace.float64)
    sdfg.add_array('C', (LMx, LNy), dace.float64)
    sdfg.add_array('a_bsizes', (2, ), dace.int32)
    sdfg.add_array('b_bsizes', (2, ), dace.int32)
    node = Pgemm('pgemm', GM, GN, GK)
    state.add_edge(state.add_read('A'), None, node, '_a', Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(state.add_read('B'), None, node, '_b', Memlet.from_array('B', sdfg.arrays['B']))
    state.add_edge(state.add_read('a_bsizes'), None, node, '_a_block_sizes', Memlet.simple('a_bsizes', '0:2'))
    state.add_edge(state.add_read('b_bsizes'), None, node, '_b_block_sizes', Memlet.simple('b_bsizes', '0:2'))
    state.add_edge(node, '_c', state.add_write('C'), None, Memlet.from_array('C', sdfg.arrays['C']))
    return sdfg, state, node


def _build_pgemv():
    sdfg = dace.SDFG('pgemv_context')
    state = sdfg.add_state('s')
    sdfg.add_array('A', (LMx, LNy), dace.float64)
    sdfg.add_array('x', (GN, ), dace.float64)
    sdfg.add_array('y', (GM, ), dace.float64)
    sdfg.add_array('a_bsizes', (2, ), dace.int32)
    sdfg.add_array('b_bsizes', (2, ), dace.int32)
    node = Pgemv('pgemv', m=GM, n=GN)
    state.add_edge(state.add_read('A'), None, node, '_a', Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(state.add_read('x'), None, node, '_b', Memlet.from_array('x', sdfg.arrays['x']))
    state.add_edge(state.add_read('a_bsizes'), None, node, '_a_block_sizes', Memlet.simple('a_bsizes', '0:2'))
    state.add_edge(state.add_read('b_bsizes'), None, node, '_b_block_sizes', Memlet.simple('b_bsizes', '0:2'))
    state.add_edge(node, '_c', state.add_write('y'), None, Memlet.from_array('y', sdfg.arrays['y']))
    return sdfg, state, node


_BUILDERS = {'pgemm': _build_pgemm, 'pgemv': _build_pgemv}

###############################################################################
# Tests
###############################################################################


@pytest.mark.parametrize('name', list(_BUILDERS))
@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_default_grid_is_the_world_grid_the_symbols_describe(name, implementation):
    """Nothing wired: the node builds the same ``Py`` x ``Px`` grid over
    ``MPI_COMM_WORLD`` that ``__dace_init_`` used to build, so an SDFG that knows
    nothing about the connectors keeps working."""
    sdfg, state, node = _BUILDERS[name]()
    code = _expanded_code(node, state, sdfg, implementation)

    assert 'Csys2blacs_handle(MPI_COMM_WORLD)' in code
    assert '__prows = Py, __pcols = Px' in code
    assert 'gridinit' in code
    assert '_context' not in code


@pytest.mark.parametrize('name', list(_BUILDERS))
@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_comm_connector_carries_the_grid(name, implementation):
    """A wired ``_comm`` is the communicator the grid is built on, in place of the
    world communicator."""
    sdfg, state, node = _BUILDERS[name]()
    _wire_comm(sdfg, state, node)
    code = _expanded_code(node, state, sdfg, implementation)

    assert 'Csys2blacs_handle(_comm)' in code
    assert 'Csys2blacs_handle(MPI_COMM_WORLD)' not in code


@pytest.mark.parametrize('name', list(_BUILDERS))
@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_context_connector_takes_priority(name, implementation):
    """A wired ``_context`` is used as-is: the node builds no grid of its own and
    reads neither the world communicator nor the ``Px`` / ``Py`` symbols."""
    sdfg, state, node = _BUILDERS[name]()
    _wire_context(sdfg, state, node)
    code = _expanded_code(node, state, sdfg, implementation)

    assert '__ctxt = _context;' in code
    assert 'gridinit' not in code
    assert 'Csys2blacs_handle' not in code
    assert 'Px' not in code and 'Py' not in code


@pytest.mark.parametrize('name', list(_BUILDERS))
@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_grid_position_comes_from_the_resolved_context(name, implementation):
    """``numroc_`` needs the node's position in the grid it is actually running on,
    so the row/column counts are read back from the resolved context rather than
    from the one field the initializer used to fill."""
    sdfg, state, node = _BUILDERS[name]()
    _wire_context(sdfg, state, node)
    code = _expanded_code(node, state, sdfg, implementation)

    assert 'gridinfo' in code
    assert '&__nprow, &__npcol, &__myprow, &__mypcol)' in code
    assert 'scalapack_myprow' not in code and 'scalapack_context' not in code


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_gridinit_node_builds_the_grid_on_its_communicator(implementation):
    """``BlacsGridInit`` is the ScaLAPACK ``CommF2c``: a communicator and a runtime
    shape in, a context value out."""
    sdfg, state, node = _build_pgemm()
    gridinit = _wire_context(sdfg, state, node)
    code = _expanded_code(gridinit, state, sdfg, implementation)

    assert 'Csys2blacs_handle(_comm)' in code
    assert '__prows = _prows, __pcols = _pcols' in code
    assert 'gridinit' in code
    assert '_context = __ctxt;' in code
    assert 'Csys2blacs_handle(MPI_COMM_WORLD)' not in code


@pytest.mark.parametrize('environment', ENVIRONMENTS, ids=lambda env: env.__name__)
def test_initializer_builds_no_grid(environment):
    """The bug this design replaces: ``Cblacs_gridinit`` in ``__dace_init_`` reads
    ``Px`` / ``Py``, a ``CompiledSDFG`` initializes once, and so every grid after
    the first silently ran against the first grid's context."""
    init = environment.init_code

    assert 'gridinit' not in init
    assert 'Px' not in init and 'Py' not in init


@pytest.mark.parametrize('environment', ENVIRONMENTS, ids=lambda env: env.__name__)
def test_every_grid_is_freed_before_mpi_goes_down(environment):
    """A grid built in the dataflow still owns a communicator. Freeing it is
    illegal once MPI is finalized, so the guard the process-global grid had must
    cover the whole set."""
    finalize = environment.finalize_code

    assert 'MPI_Finalized' in finalize
    assert '_grids' in finalize
    assert 'gridexit(' in finalize
    assert finalize.index('MPI_Finalized') < finalize.index('gridexit(')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
