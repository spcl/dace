# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``arc_distance`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'N': 100000}
INPUT_ARGS = ('N', )
ARRAY_ARGS = ('theta_1', 'phi_1', 'theta_2', 'phi_2', 'distance_matrix')
SCALARS = {}
OUTPUT_ARGS = ('distance_matrix', )

N = dc.symbol('N', dtype=dc.int64)


def initialize(N, datatype=np.float32):
    rng = np.random.default_rng(42)
    t0, p0, t1, p1 = (rng.random((N, )), rng.random((N, )), rng.random((N, )), rng.random((N, )))
    distance_matrix = np.zeros((N, ), dtype=datatype)
    return (t0.astype(datatype), p0.astype(datatype), t1.astype(datatype), p1.astype(datatype), distance_matrix)


def reference(theta_1, phi_1, theta_2, phi_2, distance_matrix):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    temp = np.sin((theta_2 - theta_1) / 2)**2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2)**2
    distance_matrix[:] = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))


@dc.program
def kernel(theta_1: dc_float[N], phi_1: dc_float[N], theta_2: dc_float[N], phi_2: dc_float[N]):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    temp = np.sin((theta_2 - theta_1) / 2)**2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2)**2
    distance_matrix = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
    return distance_matrix


CORPUS = dict(name='arc_distance',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
