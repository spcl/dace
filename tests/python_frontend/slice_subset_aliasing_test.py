# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Python frontend must not share a Subset object between two memlets.

When the same array slice (e.g. ``arr[i, k]``) is read in two sibling scopes,
the frontend's per-array access cache returns the *same* ``Range`` object on a
cache hit, and ``Memlet.simple`` stores a passed-in ``Subset`` by reference. As
a result two distinct read edges ended up pointing at one shared subset object.
That violates the SDFG invariant that each memlet owns its subset: any in-place
subset rewrite -- loop-iterator renaming, symbol replacement, offsetting, etc.,
all routine pass operations -- on one edge then silently corrupted the other.
"""
import numpy as np

import dace

N = dace.symbol('N')


@dace.program
def two_sibling_slice_reads(out: dace.float64[N, N], arr: dace.float64[N, N]):
    """Outer parallel ``i`` map whose body reads ``arr[i, k]`` in two sibling
    ``k`` loops. ``arr[i]`` is materialized as a row-slice temp and indexed by
    ``k`` in both loops -- the second read hits the access cache."""
    for i in dace.map[0:N]:
        for k in range(N):
            out[i, k] = arr[i, k]
        for k in range(N):
            out[i, k] += arr[i, k]


def test_frontend_memlets_do_not_share_subset_objects():
    """Every memlet's subset (and other_subset) must be a distinct object."""
    sdfg = two_sibling_slice_reads.to_sdfg(simplify=False)
    seen = {}
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for e in state.edges():
                if e.data is None:
                    continue
                for sub in (e.data.subset, e.data.other_subset):
                    if sub is None:
                        continue
                    here = (state.label, e.data.data, str(sub))
                    assert id(sub) not in seen, \
                        f'subset object shared by two memlets: {seen[id(sub)]} and {here}'
                    seen[id(sub)] = here


def test_sibling_slice_reads_value_preserving():
    """End-to-end: the kernel itself is correct (the shared subset was an
    object-identity bug, not a value bug, but this guards the shape)."""
    n = 8
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, n)).astype(np.float64)
    out = np.zeros((n, n))
    two_sibling_slice_reads(out=out, arr=arr, N=n)
    assert np.allclose(out, 2.0 * arr)


if __name__ == '__main__':
    test_frontend_memlets_do_not_share_subset_objects()
    test_sibling_slice_reads_value_preserving()
