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


@dace.program
def map_loop2(inp: dace.float64[KLEV, NBLOCKS], out: dace.float64[KLEV, NBLOCKS]):
    tmp = dace.define_local([KLEV, NBLOCKS], dace.float64)
    for i in dace.map[0:NBLOCKS]:
        tmp[0, i] = inp[0, i]
        tmp[1, i] = (inp[0, i] + inp[1, i]) / 2.0
        for j in range(2, KLEV):
            tmp[j, i] = (inp[j, i] + inp[j - 1, i] + inp[j - 2, i]) / 3.0
            out[j, i] = (tmp[j, i] + tmp[j - 1, i] + tmp[j - 2, i]) / 3.0


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    inp = rng.random((NBLOCKS, KLEV), dtype=np.float64)
    ref = np.empty((NBLOCKS, KLEV), dtype=np.float64)
    val = np.empty((NBLOCKS, KLEV), dtype=np.float64)

    map_loop.f(inp, ref)
    map_loop(inp=inp, out=val)

    assert np.allclose(ref, val)

    inp2 = inp.T.copy()
    val2 = np.empty((KLEV, NBLOCKS), dtype=np.float64)
    map_loop2.f(inp2, val2)

    assert np.allclose(ref.T, val2)

    sdfg = map_loop.to_sdfg(simplify=True)
    sdfg.arrays['inp'].storage = dace.StorageType.GPU_Global
    sdfg.arrays['out'].storage = dace.StorageType.GPU_Global
    auto_optimize(sdfg, dace.DeviceType.GPU)

    inp_dev = cp.asarray(inp)
    ref_dev = cp.asarray(ref)
    val_dev = cp.empty_like(inp_dev)

    sdfg(inp=inp_dev, out=val_dev)

    assert cp.allclose(ref_dev, val_dev)

    sdfg2 = map_loop2.to_sdfg(simplify=True)
    sdfg2.arrays['inp'].storage = dace.StorageType.GPU_Global
    sdfg2.arrays['out'].storage = dace.StorageType.GPU_Global
    auto_optimize(sdfg2, dace.DeviceType.GPU)

    inp2_dev = cp.asarray(inp2)
    val2_dev = cp.empty_like(inp2_dev)

    sdfg2(inp=inp2_dev, out=val2_dev)

    assert cp.allclose(ref_dev.T, val2_dev)

    # Benchmark
    csdfg = sdfg.compile()
    def func(*args):
        inp_dev, val_dev = args
        csdfg(inp=inp_dev, out=val_dev)
    print(benchmark(func, (inp_dev, val_dev), n_repeat=10, n_warmup=10))

    csdfg = sdfg2.compile()
    print(benchmark(func, (inp2_dev, val2_dev), n_repeat=10, n_warmup=10))
