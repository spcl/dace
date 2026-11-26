# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement splitting orders the per-statement loops by dependences ISL decides exactly when the offset between two
accesses is not a constant (strided subscripts with different coefficients)."""

import contextlib
import io

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

L = dace.symbol("L", dtype=dace.int64, positive=True)


@dace.program
def strided_overlap(a: dace.float64[L], b: dace.float64[L]):
    # Over the first fifth: stores at 5 i and 3 i meet at multiples of 15, the 3 i store's iteration never earlier
    for i in range(L // 5):
        a[5 * i] = b[i] + 1.0
        a[3 * i] = b[i] * 2.0


def strided_overlap_reference(a, b):
    # Over the first fifth, sequentially: the oracle for the split
    for i in range(a.shape[0] // 5):
        a[5 * i] = b[i] + 1.0
        a[3 * i] = b[i] * 2.0


@dace.program
def mixed_direction(a: dace.float64[L], b: dace.float64[L]):
    # Over the first fifth: stores at 2 i and i + 3 meet in earlier, same and later iterations of each other
    for i in range(L // 5):
        a[2 * i] = b[i] + 1.0
        a[i + 3] = b[i] * 2.0


def mixed_direction_reference(a, b):
    # Over the first fifth, sequentially: the oracle for the refused loop
    for i in range(a.shape[0] // 5):
        a[2 * i] = b[i] + 1.0
        a[i + 3] = b[i] * 2.0


@dace.program
def guarded_strided(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L], d: dace.float64[L]):
    # Over the first half: one arm stores c[2 i + 1], the other reads c[i], stored only in an earlier iteration
    for i in range(L // 2):
        if b[i] < 0.0:
            c[2 * i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * b[i]


def guarded_strided_reference(a, b, c, d):
    # Over the first half, sequentially: the oracle for the split
    for i in range(a.shape[0] // 2):
        if b[i] < 0.0:
            c[2 * i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * b[i]


def canonical_loops(prog) -> tuple[dace.SDFG, list[str]]:
    sdfg = prog.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(sdfg, target="cpu")
    # Over every control-flow region: the loops canonicalization left sequential
    return sdfg, [r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def check_numerics(sdfg: dace.SDFG, reference, names: str = "ab") -> None:
    rng = np.random.default_rng(0)
    # Over the argument names: one random input each
    arrays = {name: rng.standard_normal(61) for name in names}
    # Over the inputs: the reference runs on its own copy
    expected = {name: value.copy() for name, value in arrays.items()}
    reference(**expected)
    sdfg(**arrays, L=61)
    # Over the arguments: every one must match the sequential reference
    for name in names:
        assert np.allclose(arrays[name], expected[name]), name


# Reproducer: canonicalize kept LLR loop_to_map_overlap_seq one sequential loop: its two stores to a[5 i] and a[3 i]
# meet at multiples of 15, always in the second store's iteration or later, and the constant-offset test refused them
def test_strided_overlap_splits_into_maps():
    sdfg, loops = canonical_loops(strided_overlap)
    assert not loops, loops
    # Over every node: the maps the two statement loops lifted to
    maps = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(maps) == 2, maps
    check_numerics(sdfg, strided_overlap_reference)


# Reproducer: the stores a[2 i] and a[i + 3] meet in earlier, same and later iterations of each other, so no order of
# the per-statement loops reproduces the fused loop and the loop stays sequential
def test_mixed_direction_stays_sequential():
    sdfg, loops = canonical_loops(mixed_direction)
    assert len(loops) == 1, loops
    check_numerics(sdfg, mixed_direction_reference)


# Reproducer: a guarded body storing c[2 i + 1] in one arm and reading c[i] in the other stayed one sequential loop: the
# offset between the two is not a constant, though every store hitting a read element runs in an earlier iteration
def test_guarded_strided_store_splits_store_first():
    sdfg, loops = canonical_loops(guarded_strided)
    assert not loops, loops
    check_numerics(sdfg, guarded_strided_reference, "abcd")


if __name__ == "__main__":
    test_strided_overlap_splits_into_maps()
    test_mixed_direction_stays_sequential()
    test_guarded_strided_store_splits_store_first()
