# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Broad numerical-correctness tests for the layout transformations across kernel patterns.

Covers build_relayout (the algebra->SDFG lowering) against numpy references, and the layout PASSES
(Permute, Block, Pad, Zip, block-aware tiling) on elementwise / stencil / multi-array patterns in
1D/2D/3D, each checked bit-exact.
"""
import copy
import itertools

import numpy
import pytest
import dace

from dace.libraries.layout.algebra import Permute, Block, Unblock, Pad as PadOp
from dace.libraries.layout.lowering import build_relayout
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.pad_dimensions import PadDimensions
from dace.transformation.layout.zip_arrays import ZipArrays
from dace.transformation.layout.block_aware_map_tiling import BlockAwareMapTiling

N = dace.symbol("N")


# ============================ build_relayout vs numpy ============================ #
_relayout_counter = [0]


def _relayout(shape_ints, ops, A):
    _relayout_counter[0] += 1
    sdfg = dace.SDFG(f"relayout_np_{_relayout_counter[0]}")  # unique name -> unique build dir
    sdfg.add_array("A_in", list(shape_ints), dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    build_relayout(sdfg, state, "A_in", "A_out", ops)
    sdfg.validate()
    out_shape = tuple(int(s) for s in sdfg.arrays["A_out"].shape)
    A_out = numpy.zeros(out_shape, dtype=numpy.float64)
    sdfg(A_in=A.copy(), A_out=A_out)
    return A_out


@pytest.mark.parametrize("perm", list(itertools.permutations(range(2))))
def test_relayout_permute_2d(perm):
    A = numpy.random.rand(6, 8)
    out = _relayout(A.shape, [Permute(perm)], A)
    assert numpy.array_equal(out, A.transpose(perm))


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_relayout_permute_3d(perm):
    A = numpy.random.rand(4, 6, 8)
    out = _relayout(A.shape, [Permute(perm)], A)
    assert numpy.array_equal(out, A.transpose(perm))


@pytest.mark.parametrize("factor", [4, 8, 16])
def test_relayout_block_1d(factor):
    n = 64
    A = numpy.random.rand(n)
    out = _relayout((n, ), [Block(0, factor)], A)
    assert numpy.array_equal(out, A.reshape(n // factor, factor))


def test_relayout_block_last_dim_2d():
    P, Q, b = 12, 32, 8
    A = numpy.random.rand(P, Q)
    out = _relayout((P, Q), [Block(1, b)], A)
    # block dim1: [P, Q/b, b]
    assert numpy.array_equal(out, A.reshape(P, Q // b, b))


def test_relayout_block_first_dim_2d():
    P, Q, b = 32, 12, 8
    A = numpy.random.rand(P, Q)
    out = _relayout((P, Q), [Block(0, b)], A)
    # block dim0: A_out[io, j, ii] = A[io*b+ii, j]
    assert numpy.array_equal(out, A.reshape(P // b, b, Q).transpose(0, 2, 1))


def test_relayout_permute_then_block():
    P, Q, b = 16, 32, 8
    A = numpy.random.rand(P, Q)
    # Permute (1,0) puts logical dim0 (P) at tuple pos 1; Block(0,b) then blocks the P dim.
    # Output digits [dim1=Q, dim0-outer=P/b, dim0-inner=b] => out[j, io, ii] = A[io*b+ii, j].
    out = _relayout((P, Q), [Permute((1, 0)), Block(0, b)], A)
    ref = A.T.reshape(Q, P // b, b)
    assert numpy.array_equal(out, ref)


@pytest.mark.parametrize("shape,ops", [
    ((64, ), [Block(0, 16), Unblock(0, 16)]),
    ((6, 8), [Permute((1, 0)), Permute((1, 0))]),
    ((4, 6, 8), [Permute((2, 0, 1)), Permute((1, 2, 0))]),
    ((4, 6, 8), [Permute((2, 1, 0)), Permute((2, 1, 0))]),
    ((64, ), [Block(0, 8), Block(0, 4), Unblock(0, 4), Unblock(0, 8)]),
])
def test_relayout_identity_sequences_are_copies(shape, ops):
    A = numpy.random.rand(*shape)
    out = _relayout(shape, ops, A)
    assert numpy.array_equal(out, A)


def test_relayout_pad_grows_and_copies_live_region():
    n = 10
    A = numpy.random.rand(n)
    out = _relayout((n, ), [PadOp(0, 6)], A)
    assert out.shape == (n + 6, )
    assert numpy.array_equal(out[:n], A)


# ============================ PermuteDimensions pass (add_permute_maps) ============================ #
def _permute_pass_kernel_check(prog, permute_map, shapes, seed=0):
    """Run a kernel with PermuteDimensions(add_permute_maps=True); inputs/outputs stay original."""
    original = prog.to_sdfg(simplify=False)
    original.simplify(skip=["ArrayElimination", "DeadDataflowElimination"])
    transformed = copy.deepcopy(original)
    transformed.name = original.name + "_perm"
    PermuteDimensions(permute_map=permute_map, add_permute_maps=True).apply_pass(sdfg=transformed, pipeline_results={})
    original.validate()
    transformed.validate()
    rng = numpy.random.default_rng(seed)
    args0 = {k: rng.random(s) for k, s in shapes.items()}
    args1 = {k: v.copy() for k, v in args0.items()}
    original(**args0, N=next(iter(shapes.values()))[0])
    transformed(**args1, N=next(iter(shapes.values()))[0])
    for k in shapes:
        assert numpy.allclose(args0[k], args1[k]), f"{k} mismatch"


@dace.program
def ew3d(A: dace.float64[N, N, N], B: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        C[i, j, k] = 2.0 * A[i, j, k] - B[i, j, k]


@pytest.mark.parametrize("perm", [[0, 2, 1], [2, 1, 0], [1, 2, 0]])
def test_permute_pass_elementwise_3d(perm):
    n = 6
    _permute_pass_kernel_check(ew3d, {"A": perm, "B": perm, "C": perm}, {
        "A": (n, n, n),
        "B": (n, n, n),
        "C": (n, n, n)
    })


# ============================ SplitDimensions (Block) pass ============================ #
@dace.program
def madd2d(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] - 0.25 * B[i, j]


@pytest.mark.parametrize("fa,fb", [(8, 8), (16, 4), (4, 16)])
def test_block_pass_matrix(fa, fb):
    original = madd2d.to_sdfg()
    sdfg = copy.deepcopy(original)
    sdfg.name = f"madd2d_blk_{fa}_{fb}"
    SplitDimensions(split_map={
        "A": ([True, False], [fa, 1]),
        "B": ([False, True], [1, fb]),
    }).apply_pass(sdfg, {})
    sdfg.validate()
    n = fa * fb * 2
    A = numpy.random.rand(n, n)
    B = numpy.random.rand(n, n)
    C0 = numpy.zeros((n, n))
    C1 = numpy.zeros((n, n))
    original(A=A.copy(), B=B.copy(), C=C0, N=n)
    A2 = A.reshape(n // fa, fa, n).transpose(0, 2, 1).copy()
    B2 = B.reshape(n, n // fb, fb).copy()
    sdfg(A=A2, B=B2, C=C1, N=n)
    assert numpy.allclose(C1, C0)


# ============================ PadDimensions pass ============================ #
@dace.program
def stencil1d(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N - 1] @ dace.ScheduleType.Sequential:
        B[i] = 0.25 * A[i - 1] + 0.5 * A[i] + 0.25 * A[i + 1]


@pytest.mark.parametrize("pad", [1, 5, 8])
def test_pad_pass_stencil_1d(pad):
    original = stencil1d.to_sdfg()
    sdfg = copy.deepcopy(original)
    sdfg.name = f"stencil1d_pad_{pad}"
    PadDimensions(pad_map={"A": [pad], "B": [pad]}).apply_pass(sdfg, {})
    sdfg.validate()
    n = 32
    A = numpy.random.rand(n + pad)
    B0 = numpy.zeros(n)
    B1 = numpy.zeros(n + pad)
    original(A=A[:n].copy(), B=B0, N=n)
    sdfg(A=A.copy(), B=B1, N=n)
    assert numpy.allclose(B1[:n], B0)


@dace.program
def ew2d_pad(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        B[i, j] = A[i, j] * A[i, j]


@pytest.mark.parametrize("pi,pj", [(0, 3), (2, 0), (4, 4)])
def test_pad_pass_elementwise_2d(pi, pj):
    original = ew2d_pad.to_sdfg()
    sdfg = copy.deepcopy(original)
    sdfg.name = f"ew2d_pad_{pi}_{pj}"
    PadDimensions(pad_map={"A": [pi, pj], "B": [pi, pj]}).apply_pass(sdfg, {})
    sdfg.validate()
    n = 8
    A = numpy.random.rand(n + pi, n + pj)
    B0 = numpy.zeros((n, n))
    B1 = numpy.zeros((n + pi, n + pj))
    original(A=A[:n, :n].copy(), B=B0, N=n)
    sdfg(A=A.copy(), B=B1, N=n)
    assert numpy.allclose(B1[:n, :n], B0)


# ============================ ZipArrays pass ============================ #
@dace.program
def zip3(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] + 2.0 * B[i] - 3.0 * D[i]


def test_zip_three_homogeneous_fields():
    original = zip3.to_sdfg()
    sdfg = copy.deepcopy(original)
    sdfg.name = "zip3_z"
    ZipArrays(zip_map={"Z": ["A", "B", "D"]}).apply_pass(sdfg, {})
    sdfg.validate()
    assert tuple(str(s) for s in sdfg.arrays["Z"].shape) == ("N", "3")
    n = 12
    A = numpy.random.rand(n)
    B = numpy.random.rand(n)
    D = numpy.random.rand(n)
    C0 = numpy.zeros(n)
    C1 = numpy.zeros(n)
    original(A=A.copy(), B=B.copy(), D=D.copy(), C=C0, N=n)
    Z = numpy.stack([A, B, D], axis=-1).copy()
    sdfg(Z=Z, C=C1, N=n)
    assert numpy.allclose(C1, C0)


@dace.program
def zip2d(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * B[i, j]


def test_zip_2d_homogeneous():
    original = zip2d.to_sdfg()
    sdfg = copy.deepcopy(original)
    sdfg.name = "zip2d_z"
    ZipArrays(zip_map={"Z": ["A", "B"]}).apply_pass(sdfg, {})
    sdfg.validate()
    assert tuple(str(s) for s in sdfg.arrays["Z"].shape) == ("N", "N", "2")
    n = 8
    A = numpy.random.rand(n, n)
    B = numpy.random.rand(n, n)
    C0 = numpy.zeros((n, n))
    C1 = numpy.zeros((n, n))
    original(A=A.copy(), B=B.copy(), C=C0, N=n)
    Z = numpy.stack([A, B], axis=-1).copy()
    sdfg(Z=Z, C=C1, N=n)
    assert numpy.allclose(C1, C0)


# ============================ BlockAwareMapTiling (schedule-only) ============================ #
@dace.program
def ew2d_tile(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 3.0 * A[i, j] + B[i, j]


@pytest.mark.parametrize("ti,tj", [(16, 16), (16, 4), (8, 8)])
def test_block_aware_tiling_is_schedule_only(ti, tj):
    # Tiling does not change data layout -> original inputs, bit-exact vs untiled.
    original = ew2d_tile.to_sdfg()
    sdfg = copy.deepcopy(original)
    sdfg.name = f"ew2d_tile_{ti}_{tj}"
    BlockAwareMapTiling(tile_sizes=(ti, tj), divides_evenly=True).apply_pass(sdfg, {})
    sdfg.validate()
    n = ti * tj
    A = numpy.random.rand(n, n)
    B = numpy.random.rand(n, n)
    C0 = numpy.zeros((n, n))
    C1 = numpy.zeros((n, n))
    original(A=A.copy(), B=B.copy(), C=C0, N=n)
    sdfg(A=A.copy(), B=B.copy(), C=C1, N=n)
    assert numpy.allclose(C1, C0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
