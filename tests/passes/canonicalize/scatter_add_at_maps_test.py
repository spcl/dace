# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""cp2k_trs4's ``np.add.at``: a row scatter through ``x0[i0]`` maps over ``i0`` with a sum WCR into the sink.

The row index is an assignment inside the ``t0`` / ``t1`` loops it does not vary with; hoisted out of them alone,
they map and the scatter is an in-place update of the sink, whose WCR leaves no read of the sink behind.
"""

import numpy as np

import dace
from dace import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

NNZ, BS = (dace.symbol(s) for s in ("NNZ", "BS"))


@dace.program
def add_at(sink: dace.float64[NNZ + 1, BS, BS], x0: dace.int64[3 * NNZ], v: dace.float64[3 * NNZ, BS, BS]):
    for i0 in range(3 * NNZ):
        for t0 in range(BS):
            for t1 in range(BS):
                sink[x0[i0], t0, t1] = sink[x0[i0], t0, t1] + v[i0, t0, t1]


def test_the_row_scatter_maps_with_a_sum_into_the_sink():
    sdfg = add_at.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not any(
        isinstance(r, LoopRegion) for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
    )
    writes = [
        e.data
        for n, st in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.AccessNode) and n.data == "sink"
        for e in st.in_edges(n)
    ]
    assert writes and all(m.wcr is not None for m in writes)

    rng = np.random.default_rng(0)
    n, bs = 50, 3
    sink = rng.random((n + 1, bs, bs))
    x0 = rng.integers(0, n + 1, 3 * n)
    v = rng.random((3 * n, bs, bs))
    ref = sink.copy()
    np.add.at(ref, x0, v)
    sdfg(sink=sink, x0=x0, v=v, NNZ=n, BS=bs)
    assert np.allclose(sink, ref)


if __name__ == "__main__":
    test_the_row_scatter_maps_with_a_sum_into_the_sink()
