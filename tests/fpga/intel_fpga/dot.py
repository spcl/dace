# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Dot product with WCR
# Used as simple test for WCR over scalar

#!/usr/bin/env python

import click
import dace
import numpy as np

from dace.transformation.dataflow import MapTiling
from dace.transformation.interstate import FPGATransformSDFG

N = dace.symbol("N")


@dace.program
def dot(A: dace.float32[N], B: dace.float32[N], out: dace.float32[1]):
    @dace.map
    def product(i: _[0:N]):
        a << A[i]
        b << B[i]
        o >> out(1, lambda x, y: x + y)
        o = a * b


@click.command()
@click.option("--n", type=int, default=64)
@click.option("--tile-first/--no-tile-first", default=False)
def cli(n, tile_first):
    N.set(n)
    A = dace.ndarray([N], dtype=dace.float32)
    B = dace.ndarray([N], dtype=dace.float32)
    out_AB = dace.scalar(dace.float32)

    print('Dot product %d' % (N.get()))

    A[:] = np.random.rand(N.get()).astype(dace.float32.type)
    B[:] = np.random.rand(N.get()).astype(dace.float32.type)
    out_AB[0] = dace.float32(0)

    sdfg = dot.to_sdfg()
    if tile_first:
        sdfg.apply_transformations(MapTiling)
        sdfg.apply_transformations(FPGATransformSDFG)
    else:
        sdfg.apply_transformations(FPGATransformSDFG)
        sdfg.apply_transformations(MapTiling)

    sdfg(A=A, B=B, out=out_AB, N=N)

    diff_ab = np.linalg.norm(np.dot(A, B) - out_AB) / float(N.get())
    print("Difference (A*B):", diff_ab)
    exit(0 if (diff_ab <= 1e-5) else 1)


if __name__ == "__main__":
    cli()
