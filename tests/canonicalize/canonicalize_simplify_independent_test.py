# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The canonical form does not depend on whether the frontend simplified the program first."""

import numpy as np

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64, positive=True)
M = dace.symbol("M", dace.int64, positive=True)
H = dace.symbol("H", dace.int64, positive=True)


@dace.program
def reduce_into_slice(m: dace.float64[N, N], out: dace.float64[M]):
    out[0:N] = np.sum(m, axis=1)


@dace.program
def pair_sum(x: dace.float64[N], y: dace.float64[H]):
    y[:] = x[0 : 2 * H : 2] + x[1 : 2 * H : 2]


@dace.program
def nested_strided_slices(x: dace.float64[N], y: dace.float64[H]):
    for _ in range(3):
        pair_sum(x, y)
        x[0:H] = y * 0.5


def canonical_hashes(program, target: str) -> tuple[str, str]:
    hashes = []
    for simplify in (False, True):
        sdfg = program.to_sdfg(simplify=simplify)
        canonicalize(sdfg, target=target)
        hashes.append(sdfg.hash_sdfg())
    return hashes[0], hashes[1]


def test_reduction_into_a_slice_canonicalizes_alike_with_and_without_simplify():
    unsimplified, simplified = canonical_hashes(reduce_into_slice, "cpu")
    assert unsimplified == simplified


def test_strided_slices_of_a_nested_call_canonicalize_alike_with_and_without_simplify():
    for target in ("cpu", "gpu"):
        unsimplified, simplified = canonical_hashes(nested_strided_slices, target)
        assert unsimplified == simplified


if __name__ == "__main__":
    test_reduction_into_a_slice_canonicalizes_alike_with_and_without_simplify()
    test_strided_slices_of_a_nested_call_canonicalize_alike_with_and_without_simplify()
