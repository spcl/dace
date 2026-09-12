# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the PBLAS GEMV library node. """

import dace
import numpy as np
import pytest

from dace import Memlet
from dace.frontend.python.replacements.mpi import _distr_matmult
from dace.libraries.mpi.nodes.comm_f2c import CommF2c
from dace.libraries.pblas.nodes.gridinit import BlacsGridInit
from dace.libraries.pblas.nodes.pgemm import Pgemm
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils

# Symbols

# Process grid
P, Px, Py = (dace.symbol(s, positive=True) for s in ('P', 'Px', 'Py'))
# Global sizes
GM, GN, GK, GR, GS, GT = (dace.symbol(s, positive=True) for s in ('GM', 'GN', 'GK', 'GR', 'GS', 'GT'))
# Local sizes
LMx, LMy, LNx, LNy, LKx, LKy = (dace.symbol(s, positive=True) for s in ('LMx', 'LMy', 'LNx', 'LNy', 'LKx', 'LKy'))
LRx, LRy, LSx, LSy, LTx, LTy = (dace.symbol(s, positive=True) for s in ('LRx', 'LRy', 'LSx', 'LSy', 'LTx', 'LTy'))

rng = np.random.default_rng(42)


def mismatch(val, ref, rank, label: str) -> str:
    """What actually came back, for a wrong product that only appears on the self-hosted runner.

    ``assert np.allclose(...)`` reports nothing but ``False``. This failure has never reproduced
    off that machine, so the message has to carry the evidence itself: how much of the array is
    wrong, by how much, and the level MPI came up at. Below ``MPI_THREAD_FUNNELED`` Open MPI drops
    the locking around its shared-memory transport and ScaLAPACK's panel broadcasts come back
    corrupted differently on every call, which looks exactly like a plausible wrong product.
    """
    from mpi4py import MPI

    level = MPI.Query_thread() if MPI.Is_initialized() else None
    wrong = ~np.isclose(val, ref)
    where = tuple(int(axis[0]) for axis in np.nonzero(wrong)) if wrong.any() else ()
    seen = val[where] if where else None
    want = ref[where] if where else None
    return (f'{label}: rank {rank} got {int(wrong.sum())} of {wrong.size} entries wrong, '
            f'max |diff| {float(np.abs(val - ref).max()):.6e}, first at {where} '
            f'({seen} instead of {want}), MPI thread level {level} '
            f'(funneled is {MPI.THREAD_FUNNELED})')


# NOTE: The test passes with MKLMPICH, ReferenceMPICH, and ReferenceOpenMPI. It segfaults with MKLOpenMPI.
@pytest.mark.scalapack
def test_pgemm():

    from mpi4py import MPI

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    # DaCe programs
    @dace.program
    def pdgemm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy]):
        return dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))

    @dace.program
    def gemm(alpha: dace.float64, beta: dace.float64, C: dace.float64[LMx, LNy], A: dace.float64[LMx, LKy],
             B: dace.float64[LKx, LNy]):
        C[:] = alpha * dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK)) + beta * C

    @dace.program
    def k2mm(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy],
             C: dace.float64[LNx, LRy], D: dace.float64[LMx, LRy]):
        tmp = dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))
        D[:] = alpha * dace.distr.MatMult(tmp, C, (GM, GR, GN)) + beta * D

    @dace.program
    def k3mm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy], C: dace.float64[LNx, LRy], D: dace.float64[LRx, LSy],
             E: dace.float64[LMx, LSy]):
        tmp1 = dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))
        tmp2 = dace.distr.MatMult(tmp1, C, (GM, GR, GN))
        E[:] = dace.distr.MatMult(tmp2, D, (GM, GS, GR))

    def optimize(program):
        if rank == 0:
            sdfg = program.to_sdfg(simplify=True)
            return auto_optimize(sdfg, dace.DeviceType.CPU)
        else:
            return None

    def compile(sdfg, name):
        # A shape configuration of its own gets a name of its own: auto_optimize gives the temporary
        # between the two matrix products Persistent lifetime, so its size is fixed by
        # __dace_init_ and a second one is a second program, never a second init on this one.
        # The grid is NOT among those symbols any more -- see
        # test_pgemm_multiple_grids_one_compiled_object below.
        if sdfg is not None:
            sdfg.name = name
        return utils.distributed_compile(sdfg, commworld)

    sdfgs = [optimize(prog) for prog in (pdgemm, gemm, k2mm, k3mm)]
    base_names = [sdfg.name for sdfg in sdfgs] if rank == 0 else [None] * len(sdfgs)

    # Every (NPx, NPy) with NPx * NPy == size, tall to wide. Both extents divide size, so a problem
    # size that is a multiple of size splits evenly.
    for NPx, NPy in [(size // npy, npy) for npy in range(1, size + 1) if size % npy == 0]:

        cart_comm = commworld.Create_cart((NPx, NPy))
        i, j = cart_comm.Get_coords(rank)

        Mmult = 39
        Nmult = 57
        Kmult = 43
        Rmult = 32
        Smult = 67
        M, N, K, R, S = size * Mmult, size * Nmult, size * Kmult, size * Rmult, size * Smult

        for shapes in range(5):  # The sizes are permuted at the end of each iteration.

            if rank == 0:
                print(f"Testing PBLAS GEMM on a [{NPx}, {NPy}] grid with sizes ({M}, {N}, {K}, {R}, {S}).", flush=True)

            func, func1, func2, func3 = [
                compile(sd, f'{base}_{NPx}x{NPy}_{shapes}') for sd, base in zip(sdfgs, base_names)
            ]

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = A @ B

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()

            val = func(A=lA, B=lB, LMx=ti, LNy=tj, LKx=tki, LKy=tkj, GK=K, Px=NPx, Py=NPy)
            ref = C[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
            assert np.allclose(val, ref), mismatch(val, ref, rank, 'pdgemm')

            commworld.Barrier()

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = rng.random((M, N), dtype=np.float64)
            alpha = 1.5
            beta = 1.2
            C2 = alpha * A @ B + beta * C

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
            lC = C[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            func1(alpha=alpha, beta=beta, C=lC, A=lA, B=lB, LMx=ti, LNy=tj, LKx=tki, LKy=tkj, GK=K, Px=NPx, Py=NPy)
            ref = C2[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
            assert np.allclose(lC, ref), mismatch(lC, ref, rank, 'gemm')

            commworld.Barrier()

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = rng.random((N, R), dtype=np.float64)
            D = rng.random((M, R), dtype=np.float64)
            alpha = 1.5
            beta = 1.2
            D2 = alpha * A @ B @ C + beta * D

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            tji, tr = N // NPx, R // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
            lC = C[i * tji:(i + 1) * tji, j * tr:(j + 1) * tr].copy()
            lD = D[i * ti:(i + 1) * ti, j * tr:(j + 1) * tr].copy()

            func2(alpha=alpha,
                  beta=beta,
                  A=lA,
                  B=lB,
                  C=lC,
                  D=lD,
                  LMx=ti,
                  LNy=tj,
                  LKx=tki,
                  LKy=tkj,
                  LNx=tji,
                  LRy=tr,
                  GM=M,
                  GN=N,
                  GK=K,
                  GR=R,
                  Px=NPx,
                  Py=NPy)
            ref = D2[i * ti:(i + 1) * ti, j * tr:(j + 1) * tr]
            assert np.allclose(lD, ref), mismatch(lD, ref, rank, 'k2mm')

            commworld.Barrier()

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = rng.random((N, R), dtype=np.float64)
            D = rng.random((R, S), dtype=np.float64)
            alpha = 1.5
            beta = 1.2
            E = A @ B @ C @ D

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            tji, tri, trj, ts = N // NPx, R // NPx, R // NPy, S // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
            lC = C[i * tji:(i + 1) * tji, j * trj:(j + 1) * trj].copy()
            lD = D[i * tri:(i + 1) * tri, j * ts:(j + 1) * ts].copy()

            val = np.ndarray((ti, ts), dtype=np.float64)
            func3(A=lA,
                  B=lB,
                  C=lC,
                  D=lD,
                  E=val,
                  LMx=ti,
                  LNy=tj,
                  LKx=tki,
                  LKy=tkj,
                  LNx=tji,
                  LRx=tri,
                  LRy=trj,
                  LSy=ts,
                  GM=M,
                  GN=N,
                  GK=K,
                  GR=R,
                  GS=S,
                  Px=NPx,
                  Py=NPy)
            ref = E[i * ti:(i + 1) * ti, j * ts:(j + 1) * ts]
            assert np.allclose(val, ref), mismatch(val, ref, rank, 'k3mm')

            M, N, K, R, S = N, K, R, S, M


###############################################################################
# Regression test for the b_block_sizes "(name, range)" spelling.
#
# distr.MatMult accepts a_block_sizes/b_block_sizes as either a literal tuple of
# sizes, or a 2-tuple (array_name, range_string) pointing at an existing array
# that already holds the block sizes somewhere inside it. Nothing in-tree drives
# that second spelling through the @dace.program frontend (a pre-existing,
# separate frontend gap turns any string literal reaching this call into a
# StringLiteral, which Memlet.simple/SubsetProperty.from_string cannot consume -
# see the write-up in the accompanying report), so this test calls the
# replacement function _distr_matmult directly with plain Python strings,
# exactly as the (name, range) branch expects, and drives the resulting SDFG
# through the same PBLAS build/compare pipeline as test_pgemm above.


class _StubProgramVisitor:
    """Stands in for ProgramVisitor: _distr_matmult only calls add_temp_transient."""

    def __init__(self, sdfg):
        self.sdfg = sdfg

    def add_temp_transient(self, *args, output_index=None, **kwargs):
        kwargs['find_new_name'] = True
        return self.sdfg.add_transient('C', *args, **kwargs)


def _make_named_block_sizes_sdfg():
    sdfg = dace.SDFG('pgemm_named_block_sizes')
    state = sdfg.add_state()
    sdfg.add_array('A', (LMx, LKy), dace.float64)
    sdfg.add_array('B', (LKx, LNy), dace.float64)
    # Px, Py, GK only ever appear inside the Pgemm library node's symbolic m/n/k
    # properties (set from the "shape" argument below), never in a memlet or array
    # shape, so they need registering by hand for them to reach __dace_init_'s
    # signature - the @dace.program frontend normally does this automatically the
    # moment it visits a bare symbol name in an expression.
    for sym in (Px, Py, GK):
        sdfg.add_symbol(str(sym), sym.dtype)

    # A's block sizes: one block per process, i.e. the local shape itself.
    sdfg.add_array('a_bsizes', (2, ), dace.int32, transient=True)
    a_node = state.add_access('a_bsizes')
    a_tasklet = state.add_tasklet('_set_a_bsizes_', {}, {'__out'}, '__out[0] = LMx; __out[1] = LKy;')
    state.add_edge(a_tasklet, '__out', a_node, None, Memlet.from_array('a_bsizes', sdfg.arrays['a_bsizes']))

    # B's block sizes are packed behind a decoy pair: [0:2] must never be read (it
    # holds A's block sizes, a plausible-looking but wrong value for B), the real
    # block size for B lives at [2:4]. A correct implementation reads only [2:4];
    # the buggy branch falls back to the whole array and reads the decoy instead.
    sdfg.add_array('b_packed', (4, ), dace.int32, transient=True)
    b_node = state.add_access('b_packed')
    b_tasklet = state.add_tasklet('_set_b_packed_', {}, {'__out'},
                                  '__out[0] = LMx; __out[1] = LKy; __out[2] = LKx; __out[3] = LNy;')
    state.add_edge(b_tasklet, '__out', b_node, None, Memlet.from_array('b_packed', sdfg.arrays['b_packed']))

    pv = _StubProgramVisitor(sdfg)
    out_name = _distr_matmult(pv,
                              sdfg,
                              state,
                              'A',
                              'B', (LMx * Px, LNy * Py, GK),
                              a_block_sizes=('a_bsizes', '0:2'),
                              b_block_sizes=('b_packed', '2:4'))
    sdfg.arrays[out_name].transient = False

    for node in state.nodes():
        if type(node).__name__ == 'Pgemm':
            # Only the reference ScaLAPACK/OpenMPI build is installed on this machine.
            node.implementation = 'ReferenceOpenMPI'
    return sdfg


def test_distr_matmult_b_block_sizes_range_wiring():
    """Structural check: the Pgemm node's _b_block_sizes edge must read exactly the
    "2:4" range that was requested, not the whole 4-element b_packed array. This
    needs no MPI/compile and pins the memlet wiring directly, independent of
    whether ScaLAPACK happens to be installed."""
    sdfg = _make_named_block_sizes_sdfg()
    sdfg.validate()

    pgemm_node = None
    state = next(iter(sdfg.states()))
    for node in state.nodes():
        if type(node).__name__ == 'Pgemm':
            pgemm_node = node
    assert pgemm_node is not None

    a_edge = next(e for e in state.in_edges(pgemm_node) if e.dst_conn == '_a_block_sizes')
    b_edge = next(e for e in state.in_edges(pgemm_node) if e.dst_conn == '_b_block_sizes')

    assert a_edge.data.data == 'a_bsizes'
    assert str(a_edge.data.subset) == '0:2'
    assert b_edge.data.data == 'b_packed'
    assert str(b_edge.data.subset) == '2:4'


@pytest.mark.scalapack
def test_pgemm_named_block_sizes():

    from mpi4py import MPI

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if rank == 0:
        sdfg = _make_named_block_sizes_sdfg()
        sdfg.expand_library_nodes()
    else:
        sdfg = None
    # Multiples of size: the block sizes the node is handed ARE the local shape, so a grid that
    # left a remainder would hand ScaLAPACK a descriptor for a block the rank does not own.
    # At two ranks these are the (8, 12, 10) this regression test was first written against.
    M, N, K = size * 4, size * 6, size * 5
    A = rng.random((M, K), dtype=np.float64)
    B = rng.random((K, N), dtype=np.float64)
    C = A @ B

    func = utils.distributed_compile(sdfg, commworld)

    for NPx, NPy in [(size // npy, npy) for npy in range(1, size + 1) if size % npy == 0]:

        cart_comm = commworld.Create_cart((NPx, NPy))
        i, j = cart_comm.Get_coords(rank)

        ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
        lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
        lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
        lC = np.zeros((ti, tj), dtype=np.float64)

        func(A=lA, B=lB, C=lC, LMx=ti, LKy=tkj, LKx=tki, LNy=tj, GK=K, Px=NPx, Py=NPy)
        ref = C[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
        assert np.allclose(lC, ref), mismatch(lC, ref, rank, 'named block sizes')

        commworld.Barrier()


###############################################################################
# Two process grids, one compiled object.
#
# The grid used to be built in __dace_init_ out of the Px / Py symbols. A
# CompiledSDFG initializes once, so every grid after the first silently ran
# against the first grid's context, and the only way to get a second grid was a
# second build. Here the communicator arrives as a Fortran handle, CommF2c turns
# it into an MPI_Comm, BlacsGridInit builds the grid the runtime prows / pcols
# describe, and the context reaches Pgemm on its _context connector -- so every
# grid is just another set of arguments to the same compiled object.


def _make_comm_grid_sdfg():
    sdfg = dace.SDFG('pgemm_comm_grid')
    state = sdfg.add_state()
    sdfg.add_array('A', (LMx, LKy), dace.float64)
    sdfg.add_array('B', (LKx, LNy), dace.float64)
    sdfg.add_scalar('fcomm', dace.int32)
    sdfg.add_scalar('prows', dace.int32)
    sdfg.add_scalar('pcols', dace.int32)
    sdfg.add_scalar('usercomm', dace.dtypes.opaque('MPI_Comm'), transient=True)
    sdfg.add_scalar('context', dace.int32, transient=True)
    # The global sizes only ever appear inside the Pgemm node's symbolic m/n/k properties, never in
    # a memlet or an array shape, so they need registering by hand to reach the program signature.
    for sym in (GM, GN, GK):
        sdfg.add_symbol(str(sym), sym.dtype)

    f2c = CommF2c('_commf2c_')
    state.add_edge(state.add_read('fcomm'), None, f2c, '_fcomm', Memlet(data='fcomm', subset='0'))
    comm_node = state.add_access('usercomm')
    state.add_edge(f2c, '_comm', comm_node, None, Memlet(data='usercomm', subset='0'))

    gridinit = BlacsGridInit('_blacs_gridinit_')
    gridinit.add_in_connector('_comm', dace.dtypes.opaque('MPI_Comm'))
    state.add_edge(comm_node, None, gridinit, '_comm', Memlet(data='usercomm', subset='0'))
    state.add_edge(state.add_read('prows'), None, gridinit, '_prows', Memlet(data='prows', subset='0'))
    state.add_edge(state.add_read('pcols'), None, gridinit, '_pcols', Memlet(data='pcols', subset='0'))
    context_node = state.add_access('context')
    state.add_edge(gridinit, '_context', context_node, None, Memlet(data='context', subset='0'))

    out_name = _distr_matmult(_StubProgramVisitor(sdfg), sdfg, state, 'A', 'B', (GM, GN, GK))
    sdfg.arrays[out_name].transient = False
    pgemm = next(node for node in state.nodes() if isinstance(node, Pgemm))
    pgemm.add_in_connector('_context', dace.int32)
    state.add_edge(context_node, None, pgemm, '_context', Memlet(data='context', subset='0'))
    return sdfg


def test_pgemm_comm_grid_wiring():
    """Structural check: the context Pgemm reads is the one BlacsGridInit built out
    of the communicator CommF2c produced. Needs no MPI, and holds whether or not
    ScaLAPACK is installed."""
    sdfg = _make_comm_grid_sdfg()
    sdfg.validate()
    state = next(iter(sdfg.states()))

    pgemm = next(node for node in state.nodes() if isinstance(node, Pgemm))
    gridinit = next(node for node in state.nodes() if isinstance(node, BlacsGridInit))
    f2c = next(node for node in state.nodes() if isinstance(node, CommF2c))

    context_edge = next(e for e in state.in_edges(pgemm) if e.dst_conn == '_context')
    assert context_edge.data.data == 'context'
    assert next(e for e in state.out_edges(gridinit) if e.src_conn == '_context').data.data == 'context'

    comm_edge = next(e for e in state.in_edges(gridinit) if e.dst_conn == '_comm')
    assert comm_edge.data.data == 'usercomm'
    assert next(e for e in state.out_edges(f2c) if e.src_conn == '_comm').data.data == 'usercomm'

    # Nothing about the grid is a symbol, so nothing about it can reach __dace_init_.
    assert not ({'Px', 'Py'} & {str(s) for s in sdfg.free_symbols})


@pytest.mark.scalapack
def test_pgemm_multiple_grids_one_compiled_object():

    from mpi4py import MPI

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    sdfg = _make_comm_grid_sdfg() if rank == 0 else None
    func = utils.distributed_compile(sdfg, commworld)

    # Multiples of size, so every grid below splits the operands evenly.
    M, N, K = size * 4, size * 6, size * 5
    A = rng.random((M, K), dtype=np.float64)
    B = rng.random((K, N), dtype=np.float64)
    C = A @ B

    # The cartesian communicators are kept alive: a freed one hands its Fortran handle back for
    # reuse, and the handle is what identifies a grid that was already built.
    cart_comms = []
    for NPx, NPy in [(size // npy, npy) for npy in range(1, size + 1) if size % npy == 0]:

        cart_comm = commworld.Create_cart((NPx, NPy))
        cart_comms.append(cart_comm)
        i, j = cart_comm.Get_coords(rank)

        ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
        lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
        lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
        lC = np.zeros((ti, tj), dtype=np.float64)

        # prows / pcols keep the Py / Px order the column-major BLACS grid is built in.
        func(A=lA,
             B=lB,
             C=lC,
             fcomm=cart_comm.py2f(),
             prows=NPy,
             pcols=NPx,
             LMx=ti,
             LKy=tkj,
             LKx=tki,
             LNy=tj,
             GM=M,
             GN=N,
             GK=K)
        ref = C[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
        assert np.allclose(lC, ref), mismatch(lC, ref, rank, 'second grid')

        commworld.Barrier()


if __name__ == '__main__':

    test_pgemm()
