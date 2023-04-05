import cupy as cp
import cupyx as cpx
import dace
import numpy as np

from cupyx.profiler import benchmark
from dace.transformation.auto.auto_optimize import auto_optimize


NBLOCKS = 65536
NPROMA = 1
KLEV = 137

@dace.program
def map_loop(inp: dace.float64[NBLOCKS, KLEV], out: dace.float64[NBLOCKS, KLEV]):
    tmp = dace.define_local([NBLOCKS, KLEV], dace.float64)
    for i in dace.map[0:NBLOCKS]:
        tmp[i, 0] = inp[i, 0]
        tmp[i, 1] = (inp[i, 0] + inp[i, 1]) / 2.0
        for j in range(2, KLEV):
            tmp[i, j] = (inp[i, j] + inp[i, j - 1] + inp[i, j - 2]) / 3.0
            out[i, j] = (tmp[i, j] + tmp[i, j - 1] + tmp[i, j - 2]) / 3.0


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    inp = rng.random((NBLOCKS, KLEV), dtype=np.float64)
    ref = np.empty((NBLOCKS, KLEV), dtype=np.float64)
    val = np.empty((NBLOCKS, KLEV), dtype=np.float64)

    map_loop.f(inp, ref)
    map_loop(inp=inp, out=val)

    assert np.allclose(ref, val)

    sdfg = map_loop.to_sdfg(simplify=True)
    sdfg.arrays['inp'].storage = dace.StorageType.GPU_Global
    sdfg.arrays['out'].storage = dace.StorageType.GPU_Global
    auto_optimize(sdfg, dace.DeviceType.GPU)

    inp_dev = cp.asarray(inp)
    ref_dev = cp.asarray(ref)
    val_dev = cp.empty_like(inp_dev)

    sdfg(inp=inp_dev, out=val_dev)

    assert cp.allclose(ref_dev, val_dev)

    # Benchmark
    csdfg = sdfg.compile()
    def func(*args):
        inp_dev, val_dev = args
        csdfg(inp=inp_dev, out=val_dev)
    print(benchmark(func, (inp_dev, val_dev), n_repeat=10, n_warmup=10))
