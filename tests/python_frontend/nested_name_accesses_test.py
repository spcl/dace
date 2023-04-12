# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np

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
    assert (rel_err < 1e-7)


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
    assert (np.allclose(out, ref))


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
    assert (np.allclose(out, ref))


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
    assert (np.allclose(out, ref))


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
    assert (np.allclose(out, ref))


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
    assert (np.allclose(out, ref))


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
    assert (np.allclose(out, ref))


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
    with dc.config.set_temporary('testing', 'serialization', value=False):
        out = nested_offset_access_nested_dep(inp)
    ref = nested_offset_access_nested_dep.f(inp)
    assert (np.allclose(out, ref))


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
    assert (np.allclose(out, ref))



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
