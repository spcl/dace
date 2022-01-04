# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.pattern_matching import match_patterns
from dace.transformation.dataflow import MapTiling
import numpy as np


@dace.program
def tile_twice_test(a: dace.float64[200]):
    a *= 2.0


def test():
    sdfg = tile_twice_test.to_sdfg()
    sdfg.coarsen_dataflow()
    sdfg.apply_transformations(MapTiling, options={'tile_sizes': (5, )})
    for i, match in enumerate(match_patterns(sdfg, MapTiling, states=[sdfg.node(0)])):
        if i == 0:  # Match the first map again
            match.tile_sizes = (4, )
            match.apply_pattern(sdfg)

    A = np.random.rand(200)
    expected = 2 * A

    sdfg(a=A)

    diff = np.linalg.norm(A - expected)
    print('Difference:', diff)
    assert diff <= 1e-8


if __name__ == "__main__":
    test()
