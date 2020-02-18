import dace
from dace.transformation.pattern_matching import match_pattern
from dace.transformation.dataflow import MapTiling
import numpy as np


@dace.program
def tile_twice_test(a: dace.float64[200]):
    a *= 2.0


if __name__ == '__main__':
    sdfg = tile_twice_test.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(MapTiling, options={'tile_sizes': (5, )})
    for i, match in enumerate(match_pattern(sdfg.nodes()[0], MapTiling, sdfg)):
        if i == 0:  # Match the first map again
            match.tile_sizes = (4, )
            match.apply_pattern(sdfg)

    A = np.random.rand(200)
    expected = 2 * A

    sdfg(a=A)

    diff = np.linalg.norm(A - expected)
    print('Difference:', diff)
    exit(1 if diff > 1e-8 else 0)
