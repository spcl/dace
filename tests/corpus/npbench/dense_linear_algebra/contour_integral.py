# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``contour_integral`` (dense_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128


def rng_complex(shape, rng, datatype):
    return (rng.random(shape, dtype=datatype) + rng.random(shape, dtype=datatype) * 1j)


# S preset is NR=50, NM=150 (NR != NM, so the reference takes the non-square
# ``np.linalg.solve`` branch that the dace kernel also uses). The harness caps ints
# to 16, which would collapse 50 and 150 to an equal 16 and wrongly trip the square
# ``np.linalg.inv`` branch -- so keep NR != NM with distinct sub-cap sizes.
SIZES = {'NR': 8, 'NM': 12, 'slab_per_bc': 2, 'num_int_pts': 32}
INPUT_ARGS = ('NR', 'NM', 'slab_per_bc', 'num_int_pts')
ARRAY_ARGS = ('Ham', 'int_pts', 'Y', 'P0', 'P1')
SCALARS = {}
OUTPUT_ARGS = ('P0', 'P1')

NR, NM, slab_per_bc = (dc.symbol(s, dtype=dc.int64) for s in ('NR', 'NM', 'slab_per_bc'))


def initialize(NR, NM, slab_per_bc, num_int_pts, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)
    Ham = rng_complex((slab_per_bc + 1, NR, NR), rng, datatype)
    int_pts = rng_complex((num_int_pts, ), rng, datatype)
    Y = rng_complex((NR, NM), rng, datatype)
    P0 = np.zeros((NR, NM), dtype=np.complex128)
    P1 = np.zeros((NR, NM), dtype=np.complex128)
    return (Ham, int_pts, Y, P0, P1)


def reference(NR, NM, slab_per_bc, Ham, int_pts, Y, P0, P1):
    for z in int_pts:
        Tz = np.zeros((NR, NR), dtype=np.complex128)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        if NR == NM:
            X = np.linalg.inv(Tz)
        else:
            X = np.linalg.solve(Tz, Y)
        if abs(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X


@dc.program
def kernel(Ham: dc_complex_float[slab_per_bc + 1, NR, NR], int_pts: dc_complex_float[32], Y: dc_complex_float[NR, NM]):
    P0 = np.zeros((NR, NM), dtype=dc_complex_float)
    P1 = np.zeros((NR, NM), dtype=dc_complex_float)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), dtype=dc_complex_float)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        X = np.linalg.solve(Tz, Y)
        if np.absolute(z) < 1.0:
            X[:] = -X
        P0 += X
        P1 += z * X
    return (P0, P1)


CORPUS = dict(name='contour_integral',
              dwarf='dense_linear_algebra',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
