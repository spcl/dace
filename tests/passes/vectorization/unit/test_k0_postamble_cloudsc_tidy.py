# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K=0 / ``widths=(1,)`` single-lane postamble exercised on the cloudsc
``tidy_branch`` 2-D guarded RMW pattern.

This pins the exact orchestrator knob combo the user reported (a 2D
``widths=(8, 8)`` main loop with a ``scalar_postamble`` + ``tile_k1``
remainder, ``branch_mode='merge'`` body, ``nest_map_bodies=True``,
``insert_copies=True``) so the K=0 postamble path stays exercised on
the cloudsc-characteristic guarded multi-statement RMW chain.

Three failure modes surfaced while the K=0 wiring landed:

1. ``EmitTileOps`` / ``PromoteNSDFGBodyToTiles`` rejected the
   ``__tile_k1_tail`` map as "no TileMaskGen in scope and not __tile_main"
   — fixed by treating ``__tile_k1_tail`` like the ``__tile_main``
   divisible interior (single in-bounds element per iteration, no mask
   needed).

2. The K-param count check at the descent ``len(n.map.params) < K``
   refused the tail (the tail is pinned at K=1 widths=(1,), independent
   of the orchestrator-level widths) — fixed by overriding ``map_K=1``
   for ``__tile_k1_tail`` maps.

3. The boundary connector classifier ``_box_classification`` rejected a
   constant ``[0]`` subset as ``BROADCAST_SYMBOL`` — at widths=(1,) a
   broadcast-to-1-lane IS a single-element load, so the classifier now
   accepts it and treats it as ``CONTIGUOUS`` at K=1.

The kernel below (``cloudsc_tidy_branch``) is the canonical guarded RMW
chain. The test only verifies that the orchestrator + descent run to
completion; the produced SDFG must validate (correctness is the
existing K=0 ``test_k0_remainder`` arm's responsibility).
"""

import pytest
pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import dace
import pytest

from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )

KLEV = dace.symbol("KLEV")
KLON = dace.symbol("KLON")
NCLV = dace.symbol("NCLV")

_PTSPHY = 50.0
_RLMIN = 1.0e-8
_RAMIN = 1.0e-8
_RALVDCP = 2.5008e6 / 1004.7
_RALSDCP = 2.8345e6 / 1004.7
_ZQTMST = 1.0 / _PTSPHY


@dace.program
def _cloudsc_tidy_branch(
    zqx_l: dace.float64[KLEV, KLON],
    zqx_i: dace.float64[KLEV, KLON],
    zqx_v: dace.float64[KLEV, KLON],
    za: dace.float64[KLEV, KLON],
    ptend_q: dace.float64[KLEV, KLON],
    ptend_t: dace.float64[KLEV, KLON],
):
    # cloudsc_bottom_lower.F90 "Tidy up very small cloud cover or total
    # cloud water" — guarded read-modify-write over several arrays (the
    # CLOUDSC-characteristic conditional accumulation pattern).
    for jk in range(KLEV):
        for jl in range(KLON):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < _RLMIN or za[jk, jl] < _RAMIN:
                zqadj_l = zqx_l[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALVDCP * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0
                zqadj_i = zqx_i[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_i
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALSDCP * zqadj_i
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_i[jk, jl]
                zqx_i[jk, jl] = 0.0
                za[jk, jl] = 0.0


def test_k0_postamble_runs_on_cloudsc_tidy_branch():
    """End-to-end: the K=0 ``tile_k1`` postamble on cloudsc tidy_branch.

    The orchestrator must:
      1. Split the 2-D main loop into a ``__tile_main`` interior + a
         ``__tile_k1_tail`` postamble (K=2 widths=(8,8) main, K=1
         widths=(1,) tail).
      2. Mark / mask / stride / promote / emit every region cleanly.
      3. Produce a validating SDFG.

    The body has a guarded multi-statement RMW chain (10+ writes inside
    the same ``if`` branch over ``ptend_q`` / ``ptend_t`` / ``zqx_*`` /
    ``za``), so the descent's broadcast-symbol relaxation, the
    no-mask-on-``__tile_k1_tail`` exception, and the K=1 override on
    the param-count check all fire.
    """
    sdfg = _cloudsc_tidy_branch.to_sdfg()
    sdfg.name = "k0_tidy_branch"
    sdfg.validate()

    VectorizeCPUMultiDim(
        widths=(8, 8),
        target_isa="SCALAR",
        remainder_strategy="scalar_postamble",
        branch_mode="merge",
        loop_to_map_permissive=False,
        nest_map_bodies=True,
        insert_copies=True,
        fuse_overlapping_loads=False,
        scalar_remainder_emit="tile_k1",
    ).apply_pass(sdfg, {})
    sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
