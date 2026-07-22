# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Combined Block + Permute numerical-verification sweep on a shared elementwise kernel.

Each case stacks the two layout primitives on the SAME array(s) of one kernel, in BOTH orders:

  * ``block_then_permute`` -- ``SplitDimensions`` blocks one logical dimension (the tile axis is
    appended LAST), the schedule is re-tiled, and then ``PermuteDimensions`` reorders the now
    ``ndim + 1`` PHYSICAL axes of the blocked descriptor.
  * ``permute_then_block`` -- ``PermuteDimensions`` reorders the ``ndim`` logical axes first, then
    ``SplitDimensions`` blocks one dimension of the (transparently permuted) array.

``PermuteDimensions(add_permute_maps=True)`` wraps every touched array with permute_in/out states,
so the permute is transparent at the external interface: after both transforms the external
descriptor of an array equals its Block-packed shape, whatever the permutation of the internal
physical axes. The run closure therefore derives the physical layout from
``sdfg.arrays[name].shape`` (asserting it matches the analytic Block-pack shape), packs the logical
input into that shape, runs, unpacks the blocked output, and checks bit-exactness against a numpy
oracle. The combined transform is applied to BOTH an input (A, packed) and the output (C, unpacked)
so each case exercises packing AND recovery.

The invariant asserted here is CORRECTNESS: every combined layout reproduces the elementwise oracle
to floating-point tolerance. Blocking uses divisible extents so every tile is perfect.
"""
import numpy
import pytest

import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout
from dace.transformation.layout.prepare import prepare_for_layout

M, N, K = (dace.symbol(s) for s in ("M", "N", "K"))

SIZES_2D = (12, 8)  # dim0 divisible by 2/3/4/6, dim1 by 2/4
SIZES_3D = (8, 12, 6)  # dim0 by 2/4, dim1 by 2/3/4/6, dim2 by 2/3


@dace.program
def elementwise_2d(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        C[i, j] = A[i, j] * 2.0 + B[i, j] + 1.0


@dace.program
def elementwise_3d(A: dace.float64[M, N, K], B: dace.float64[M, N, K], C: dace.float64[M, N, K]):
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        C[i, j, k] = A[i, j, k] * 3.0 + B[i, j, k] - 0.5


def block_pack(a: numpy.ndarray, blocked_dim: int, factor: int) -> numpy.ndarray:
    """Lay logical ``a`` out into ``SplitDimensions``' physical shape for blocking one dimension.

    Split dimension ``blocked_dim`` into ``(size // factor, factor)`` in place, then move that inner
    tile axis to the END (the append-last convention), matching the blocked descriptor exactly.
    """
    ndim = a.ndim
    expanded = []
    for d in range(ndim):
        if d == blocked_dim:
            expanded.append(a.shape[d] // factor)
            expanded.append(factor)
        else:
            expanded.append(a.shape[d])
    inner_axis = blocked_dim + 1
    order = [ax for ax in range(ndim + 1) if ax != inner_axis] + [inner_axis]
    return a.reshape(expanded).transpose(order).copy()


def block_unpack(p: numpy.ndarray, blocked_dim: int, factor: int, logical_shape) -> numpy.ndarray:
    """Inverse of :func:`block_pack`: move the trailing tile axis back next to its outer axis and
    reshape the physical ``p`` to ``logical_shape``."""
    ndim = len(logical_shape)
    inner_axis = blocked_dim + 1
    order = [ax for ax in range(ndim + 1) if ax != inner_axis] + [inner_axis]
    inverse = [int(ax) for ax in numpy.argsort(order)]
    return p.transpose(inverse).reshape(logical_shape).copy()


def apply_combined(sdfg: dace.SDFG, order: str, ndim: int, blocked_dim: int, factor: int, perm) -> None:
    """Apply the combined Block + Permute layout to A and C on ``sdfg`` in place."""
    masks = [d == blocked_dim for d in range(ndim)]
    factors = [factor if d == blocked_dim else 1 for d in range(ndim)]
    split_map = {name: (masks, factors) for name in ("A", "C")}
    permute_map = {name: list(perm) for name in ("A", "C")}
    if order == "block_then_permute":
        SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
        normalize_schedule_for_layout(sdfg)
        PermuteDimensions(permute_map=permute_map, add_permute_maps=True).apply_pass(sdfg, {})
    else:  # permute_then_block
        PermuteDimensions(permute_map=permute_map, add_permute_maps=True).apply_pass(sdfg, {})
        SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
        normalize_schedule_for_layout(sdfg)


def descriptor_shape(sdfg: dace.SDFG, name: str, symmap) -> tuple:
    return tuple(int(dace.symbolic.evaluate(s, symmap)) for s in sdfg.arrays[name].shape)


def run_combined(ndim: int, order: str, blocked_dim: int, factor: int, perm):
    program = elementwise_2d if ndim == 2 else elementwise_3d
    sizes = SIZES_2D if ndim == 2 else SIZES_3D

    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    apply_combined(sdfg, order, ndim, blocked_dim, factor, perm)

    symmap = dict(zip((M, N, K)[:ndim], sizes))
    rng = numpy.random.default_rng(0)
    a_logical = rng.random(sizes)
    b_logical = rng.random(sizes)
    if ndim == 2:
        oracle = a_logical * 2.0 + b_logical + 1.0
    else:
        oracle = a_logical * 3.0 + b_logical - 0.5

    # Derive the physical layout from the descriptor and lay the logical data out to match it.
    a_packed = block_pack(a_logical, blocked_dim, factor)
    assert a_packed.shape == descriptor_shape(sdfg, "A", symmap)
    c_physical = numpy.zeros(descriptor_shape(sdfg, "C", symmap))

    kwargs = {"A": a_packed, "B": b_logical.copy(), "C": c_physical}
    kwargs.update({name: sz for name, sz in zip(("M", "N", "K")[:ndim], sizes)})
    sdfg(**kwargs)

    got = block_unpack(c_physical, blocked_dim, factor, sizes)
    return got, oracle


# (ndim, order, blocked_dim, factor, perm) -- perm has length ndim (permute_then_block) or
# ndim + 1 (block_then_permute, the blocked descriptor's physical axes).
CASES = [
    # ---- 2D, block_then_permute (perm over the 3 physical axes) --------------------------------
    (2, "block_then_permute", 0, 2, [2, 0, 1]),
    (2, "block_then_permute", 0, 2, [1, 2, 0]),
    (2, "block_then_permute", 0, 4, [0, 2, 1]),
    (2, "block_then_permute", 0, 3, [2, 1, 0]),
    (2, "block_then_permute", 0, 4, [1, 0, 2]),
    (2, "block_then_permute", 1, 2, [0, 2, 1]),
    (2, "block_then_permute", 1, 4, [1, 0, 2]),
    (2, "block_then_permute", 1, 4, [2, 0, 1]),
    # ---- 2D, permute_then_block (perm over the 2 logical axes) ---------------------------------
    (2, "permute_then_block", 0, 2, [1, 0]),
    (2, "permute_then_block", 1, 4, [1, 0]),
    (2, "permute_then_block", 0, 4, [1, 0]),
    (2, "permute_then_block", 1, 2, [1, 0]),
    (2, "permute_then_block", 0, 3, [1, 0]),
    (2, "permute_then_block", 1, 4, [0, 1]),
    # ---- 3D, block_then_permute (perm over the 4 physical axes) --------------------------------
    (3, "block_then_permute", 0, 2, [3, 0, 1, 2]),
    (3, "block_then_permute", 0, 4, [2, 3, 0, 1]),
    (3, "block_then_permute", 1, 3, [0, 2, 3, 1]),
    (3, "block_then_permute", 1, 4, [3, 0, 1, 2]),
    (3, "block_then_permute", 2, 2, [3, 0, 1, 2]),
    (3, "block_then_permute", 2, 3, [1, 2, 0, 3]),
    # ---- 3D, permute_then_block (perm over the 3 logical axes) ---------------------------------
    (3, "permute_then_block", 0, 2, [1, 2, 0]),
    (3, "permute_then_block", 1, 3, [2, 0, 1]),
    (3, "permute_then_block", 2, 2, [2, 1, 0]),
    (3, "permute_then_block", 0, 4, [0, 2, 1]),
    (3, "permute_then_block", 1, 4, [1, 0, 2]),
    (3, "permute_then_block", 2, 3, [1, 2, 0]),
]


def case_id(case) -> str:
    ndim, order, blocked_dim, factor, perm = case
    tag = "bp" if order == "block_then_permute" else "pb"
    return "%dd-%s-d%d-f%d-p%s" % (ndim, tag, blocked_dim, factor, "".join(str(x) for x in perm))


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_block_permute_combined(case):
    ndim, order, blocked_dim, factor, perm = case
    got, oracle = run_combined(ndim, order, blocked_dim, factor, perm)
    assert numpy.allclose(got, oracle), "combined %s d%d f%d perm %s max err %s" % (order, blocked_dim, factor, perm,
                                                                                    numpy.max(numpy.abs(got - oracle)))


if __name__ == "__main__":
    for one_case in CASES:
        result, expected = run_combined(*one_case)
        assert numpy.allclose(result, expected), case_id(one_case)
        print("PASS", case_id(one_case))
    print("block/permute combined sweep: %d cases PASS" % len(CASES))