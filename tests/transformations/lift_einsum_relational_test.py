# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LiftEinsum`` must refuse a map whose tasklet compares its operands instead of multiplying them.

CP2K's ``cp2k_grid_integrate`` gates a table with the broadcast mask ``zi <= si``: a map over two
index axes whose tasklet is ``__out = (__in1 <= __in2)``. That is an outer-product SHAPE, so every
structural check in ``can_be_applied`` passes, and the tasklet check then parsed the comparison to
a sympy ``LessThan`` and divided it by the operand product -- a ``TypeError`` the pattern matcher
swallowed into a warning on every canonicalize run of the kernel.
"""
import numpy as np

import dace
from dace.transformation.dataflow.lift_einsum import LiftEinsum

N = dace.symbol('N')


@dace.program
def broadcast_compare(a: dace.int64[N], b: dace.int64[N], out: dace.bool_[N, N]):
    out[:] = a[None, :] <= b[:, None]


def test_comparison_map_is_not_lifted():
    """The comparison map is refused outright (not by an exception) and still computes the mask."""
    sdfg = broadcast_compare.to_sdfg(simplify=True)
    # Surface a matcher exception instead of letting the pattern matcher log and skip it.
    with dace.config.set_temporary('optimizer', 'match_exception', value=True):
        lifted = sdfg.apply_transformations_repeated(LiftEinsum)
    assert lifted == 0, 'a comparison is no contraction'
    sdfg.validate()

    n = 7
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, n)
    b = rng.integers(0, 5, n)
    out = np.zeros((n, n), dtype=np.bool_)
    sdfg(a=a, b=b, out=out, N=n)
    assert np.array_equal(out, a[None, :] <= b[:, None])
