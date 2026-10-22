# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose only cross-iteration writes are ``X[s] = X[s] + r`` at a data-dependent slice maps once the updates
are WCR, and the result is the same whichever slices overlap (lavamd's box loop stayed sequential)."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize

NB, P, Q, NT = dace.symbol("NB"), dace.symbol("P"), dace.symbol("Q"), dace.symbol("NT")


@dace.program
def boxes(offsets: dace.int64[NB], vals: dace.float64[NB, P, Q], out: dace.float64[NT]):
    for b in range(NB):
        o = offsets[b]
        row = np.sum(vals[b, :, :], axis=1)
        out[o : o + P] = out[o : o + P] + row


def test_a_box_loop_accumulating_at_data_offsets_maps_and_matches():
    sdfg = boxes.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    assert not [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    rng = np.random.default_rng(0)
    nb, p = 9, 5
    offsets = np.array([0, 3, 3, 10, 12, 20, 21, 30, 35], dtype=np.int64)  # boxes 1 and 2 overlap
    vals = rng.random((nb, p, 4))
    out = rng.random(40)
    want = out.copy()
    for b in range(nb):
        want[offsets[b] : offsets[b] + p] += vals[b].sum(axis=1)
    sdfg(offsets=offsets, vals=vals, out=out, NB=nb, P=p, Q=4, NT=40)
    assert np.allclose(out, want)


if __name__ == "__main__":
    test_a_box_loop_accumulating_at_data_offsets_maps_and_matches()
