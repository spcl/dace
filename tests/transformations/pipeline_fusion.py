# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np

# This test transforms a sequence of vector addition kernels created with the
# numpy frontend into to multiple, pipeline parallel kernels executed on the
# FPGA, by using pipeline fusion.

DTYPE = dace.float32
N = dace.symbol("N")


@dace.program
def add_four_vectors(v0: DTYPE[N], v1: DTYPE[N], v2: DTYPE[N], v3: DTYPE[N],
                     res: DTYPE[N]):
    res[:] = (v0[:] + v1[:] + v2[:] + v3[:]) / 2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    sdfg = add_four_vectors.to_sdfg()

    # Transform intermediate buffers into pipelines
    sdfg.apply_transformations_repeated(
        dace.transformation.dataflow.PipelineFusion)

    # Transform to run on the FPGA
    sdfg.apply_transformations_repeated(
        dace.transformation.interstate.FPGATransformState)

    v0 = 1 * np.ones((args.N, ), dtype=DTYPE.type)
    v1 = 2 * np.ones((args.N, ), dtype=DTYPE.type)
    v2 = 3 * np.ones((args.N, ), dtype=DTYPE.type)
    v3 = 4 * np.ones((args.N, ), dtype=DTYPE.type)
    res = np.zeros((args.N, ), dtype=DTYPE.type)

    sdfg(v0=v0, v1=v1, v2=v2, v3=v3, res=res, N=np.int32(args.N))

    if not all(res == 5):
        raise ValueError("Unexpected result.")
