# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.transformation.passes import FullMapFusion

N = dace.symbol("N")


@dace.program
def two_maps(a: dace.float64[N], b: dace.float64[N]):
    tmp = a + 1.0
    b[:] = tmp * 2.0


def test_full_map_fusion_runs_outside_a_pipeline():
    sdfg = two_maps.to_sdfg(simplify=True)
    assert FullMapFusion().apply_pass(sdfg, {}) is not None
    assert sum(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()) == 1
    a = np.random.rand(8)
    b = np.zeros(8)
    sdfg(a=a, b=b, N=8)
    assert np.allclose(b, (a + 1.0) * 2.0)


if __name__ == "__main__":
    test_full_map_fusion_runs_outside_a_pipeline()
