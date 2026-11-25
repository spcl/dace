# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC's ``IF (ZQXN(JN) < ZEPSEC)`` update of the one row ``ZQXN(NCLDQV)``, summed over the species ``JN``.

Every species iteration adds its own row to row ``4`` and zeroes its own row, under a data-dependent guard. Rows ``0..3``
never meet row ``4``, so the species loop is a ``+`` reduction into row ``4`` with the identity ``0`` where the guard is
false, and canonicalization should lift it.

What stops it today: the frontend orders the snapshot of ``ZQXN(4, JL)`` before the zeroing write of ``ZQXN(JN-1, JL)``
with an ordering edge, because ``JN - 1 != 4`` is not known inside the nested SDFG (it needs ``JN < 5`` from the loop
header). The edge keeps ``FoldScalarReadCopies`` from reading row ``4`` in place, so ``AugAssignToWCR`` never sees
``zq[4] = zq[4] + x``, and ``LoopToMap`` refuses the write to a row the loop counter does not fix.
"""

import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N")


@dace.program
def conditional_row_accumulation(zq: dace.float64[5, N]):
    for jn in range(1, 5):
        for jl in range(N):
            if zq[jn - 1, jl] < 1e-14:
                zq[4, jl] = zq[4, jl] + zq[jn - 1, jl]
                zq[jn - 1, jl] = 0.0


@pytest.mark.xfail(
    strict=True, reason="an ordering edge between rows 4 and JN-1 hides the accumulation from WCR conversion"
)
def test_a_conditional_accumulation_into_one_row_lifts_as_a_reduction_over_the_species():
    sdfg = conditional_row_accumulation.to_sdfg(simplify=True)

    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")

    assert not [
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)
    ]


if __name__ == "__main__":
    # test_a_conditional_accumulation_into_one_row_lifts_as_a_reduction_over_the_species() -- xfail until lifted
    pass
