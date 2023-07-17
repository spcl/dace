# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np
import os

N = dc.symbol('N')


@dc.program
def nested_name_accesses(a: dc.float32, x: dc.float32[N, N, N], w: dc.float32[N, N]):
    out = np.ndarray(x.shape, x.dtype)
    for i in dc.map[0:N]:
        out[i] = a * x[i] @ w
    return out


def test_nested_name_accesses():
    N.set(10)
    a = np.random.rand(1).astype(np.float32)[0]
    x = np.random.rand(N.get(), N.get(), N.get()).astype(np.float32)
    w = np.random.rand(N.get(), N.get()).astype(np.float32)
    dc_out = nested_name_accesses(a, x, w)
    np_out = np.empty(x.shape, x.dtype)
    for i in range(N.get()):
        np_out[i] = a * x[i] @ w
    diff_norm = np.linalg.norm(dc_out - np_out)
    ref_norm = np.linalg.norm(np_out)
    rel_err = diff_norm / ref_norm
    assert rel_err < 1e-7


def test_nested_offset_access():
    @dc.program
    def nested_offset_access(inp: dc.float64[6, 5, 5]):
        out = np.zeros((5, 5, 5), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(1, 4):
                out[i, j, k] = 0.25 * (inp[i + 1, j, k + 1] + inp[i, j, k + 1])
        return out

    inp = np.reshape(np.arange(6 * 5 * 5, dtype=np.float64), (6, 5, 5)).copy()
    out = nested_offset_access(inp)
    ref = nested_offset_access.f(inp)
    assert np.allclose(out, ref)


def test_nested_offset_access_dappy():
    @dc.program
    def nested_offset_access(inp: dc.float64[6, 5, 5]):
        out = np.zeros((5, 5, 5), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(1, 4):
                with dc.tasklet():
                    in1 << inp[i + 1, j, k + 1]
                    in2 << inp[i, j, k + 1]
                    out1 >> out[i, j, k]
                    out1 = 0.25 * (in1 + in2)
        return out

    inp = np.reshape(np.arange(6 * 5 * 5, dtype=np.float64), (6, 5, 5)).copy()
    out = nested_offset_access(inp)
    ref = nested_offset_access.f(inp)
    assert np.allclose(out, ref)


def test_nested_multi_offset_access():
    @dc.program
    def nested_offset_access(inp: dc.float64[6, 5, 10]):
        out = np.zeros((5, 5, 10), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(1, 5):
                for l in range(4):
                    out[i, j, k + l] = 0.25 * (inp[i + 1, j, k + l + 1] + inp[i, j, k + l + 1])
        return out

    inp = np.reshape(np.arange(6 * 5 * 10, dtype=np.float64), (6, 5, 10)).copy()
    out = nested_offset_access(inp)
    ref = nested_offset_access.f(inp)
    assert np.allclose(out, ref)


def test_nested_multi_offset_access_dappy():
    @dc.program
    def nested_offset_access(inp: dc.float64[6, 5, 10]):
        out = np.zeros((5, 5, 10), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(1, 5):
                for l in range(4):
                    with dc.tasklet():
                        in1 << inp[i + 1, j, k + l + 1]
                        in2 << inp[i, j, k + l + 1]
                        out1 >> out[i, j, k + l]
                        out1 = 0.25 * (in1 + in2)
        return out

    inp = np.reshape(np.arange(6 * 5 * 10, dtype=np.float64), (6, 5, 10)).copy()
    out = nested_offset_access(inp)
    ref = nested_offset_access.f(inp)
    assert np.allclose(out, ref)


def test_nested_dec_offset_access():
    @dc.program
    def nested_offset_access(inp: dc.float64[6, 5, 5]):
        out = np.zeros((5, 5, 5), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(3, 0, -1):
                out[i, j, k] = 0.25 * (inp[i + 1, j, k + 1] + inp[i, j, k + 1])
        return out

    inp = np.reshape(np.arange(6 * 5 * 5, dtype=np.float64), (6, 5, 5)).copy()
    out = nested_offset_access(inp)
    ref = nested_offset_access.f(inp)
    assert np.allclose(out, ref)


def test_nested_dec_offset_access_dappy():
    @dc.program
    def nested_offset_access(inp: dc.float64[6, 5, 5]):
        out = np.zeros((5, 5, 5), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(3, 0, -1):
                with dc.tasklet():
                    in1 << inp[i + 1, j, k + 1]
                    in2 << inp[i, j, k + 1]
                    out1 >> out[i, j, k]
                    out1 = 0.25 * (in1 + in2)
        return out

    inp = np.reshape(np.arange(6 * 5 * 5, dtype=np.float64), (6, 5, 5)).copy()
    out = nested_offset_access(inp)
    ref = nested_offset_access.f(inp)
    assert np.allclose(out, ref)


def test_nested_offset_access_nested_dependency():
    @dc.program
    def nested_offset_access_nested_dep(inp: dc.float64[6, 5, 5]):
        out = np.zeros((5, 5, 5), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(1, 4):
                for l in range(k, 5):
                    out[i, j, k] = 0.25 * (inp[i + 1, j, l - k + 1] + inp[i, j, l - k + 1])
        return out

    inp = np.reshape(np.arange(6 * 5 * 5, dtype=np.float64), (6, 5, 5)).copy()
    last_value = os.environ.get('DACE_testing_serialization', '0')
    os.environ['DACE_testing_serialization'] = '0'
    with dc.config.set_temporary('testing', 'serialization', value=False):
        out = nested_offset_access_nested_dep(inp)
    os.environ['DACE_testing_serialization'] = last_value
    ref = nested_offset_access_nested_dep.f(inp)
    assert np.allclose(out, ref)


def test_nested_offset_access_nested_dependency_dappy():
    @dc.program
    def nested_offset_access_nested_dep(inp: dc.float64[6, 5, 10]):
        out = np.zeros((5, 5, 10), np.float64)
        for i, j in dc.map[0:5, 0:5]:
            out[i, j, 0] = 0.25 * (inp[i + 1, j, 1] + inp[i, j, 1])
            for k in range(1, 5):
                for l in range(k, 4):
                    with dc.tasklet():
                        in1 << inp[i + 1, j, k + l + 1]
                        in2 << inp[i, j, k + l + 1]
                        out1 >> out[i, j, k + l]
                        out1 = 0.25 * (in1 + in2)
        return out

    inp = np.reshape(np.arange(6 * 5 * 10, dtype=np.float64), (6, 5, 10)).copy()
    out = nested_offset_access_nested_dep(inp)
    ref = nested_offset_access_nested_dep.f(inp)
    assert np.allclose(out, ref)


def test_access_to_nested_transient():

    KLEV = 3
    KLON = 4
    NBLOCKS = 5

    @dc.program
    def small_wip(inp: dc.float64[KLEV+1, KLON, NBLOCKS], out: dc.float64[KLEV, KLON, NBLOCKS]):
        for jn in dc.map[0:NBLOCKS]:
            tmp = np.zeros([KLEV+1, KLON])
            for jl in range(KLON):
                for jk in range(KLEV):
                    tmp[jk, jl] = inp[jk, jl, jn] + inp[jk+1, jl, jn]

            for jl in range(KLON):
                for jk in range(KLEV):
                    out[jk, jl, jn] = tmp[jk, jl] + tmp[jk+1, jl]
    
    rng = np.random.default_rng(42)
    inp = rng.random((KLEV+1, KLON, NBLOCKS))
    ref = np.zeros((KLEV, KLON, NBLOCKS))
    val = np.zeros((KLEV, KLON, NBLOCKS))

    small_wip(inp, val)
    small_wip.f(inp, ref)

    assert np.allclose(val, ref)


def test_access_to_nested_transient_dappy():

    KLEV = 3
    KLON = 4
    NBLOCKS = 5

    @dc.program
    def small_wip_dappy(inp: dc.float64[KLEV+1, KLON, NBLOCKS], out: dc.float64[KLEV, KLON, NBLOCKS]):
        for jn in dc.map[0:NBLOCKS]:
            tmp = np.zeros([KLEV+1, KLON])
            for jl in range(KLON):
                for jk in range(KLEV):
                    with dc.tasklet():
                        in1 << inp[jk, jl, jn]
                        in2 << inp[jk+1, jl, jn]
                        out1 >> tmp[jk, jl]
                        out1 = in1 + in2

            for jl in range(KLON):
                for jk in range(KLEV):
                    with dc.tasklet():
                        in1 << tmp[jk, jl]
                        in2 << tmp[jk+1, jl]
                        out1 >> out[jk, jl, jn]
                        out1 = in1 + in2
    
    rng = np.random.default_rng(42)
    inp = rng.random((KLEV+1, KLON, NBLOCKS))
    ref = np.zeros((KLEV, KLON, NBLOCKS))
    val = np.zeros((KLEV, KLON, NBLOCKS))

    small_wip_dappy(inp, val)
    small_wip_dappy.f(inp, ref)

    assert np.allclose(val, ref)


if __name__ == "__main__":
    test_nested_name_accesses()
    test_nested_offset_access()
    test_nested_offset_access_dappy()
    test_nested_multi_offset_access()
    test_nested_multi_offset_access_dappy()
    test_nested_dec_offset_access()
    test_nested_dec_offset_access_dappy()
    test_nested_offset_access_nested_dependency()
    test_nested_offset_access_nested_dependency_dappy()
    test_access_to_nested_transient()
    test_access_to_nested_transient_dappy()
