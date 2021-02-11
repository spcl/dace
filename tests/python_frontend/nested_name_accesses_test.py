# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np


N = dc.symbol('N')


@dc.program
def nested_name_accesses(a: dc.float32,
                         x: dc.float32[N, N, N],
                         w: dc.float32[N, N]):
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
    assert(rel_err < 1e-7)


if __name__ == "__main__":
    test_nested_name_accesses()