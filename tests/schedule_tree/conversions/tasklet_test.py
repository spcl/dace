# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree, as_sdfg


def test_simple_tasklet():

    @dace.program
    def simple_tasklet(A: dace.float32[3, 3]):
        ret = dace.float32(0)
        with dace.tasklet:
            c << A[1, 1]
            n << A[0, 1]
            s << A[2, 1]
            w << A[1, 0]
            e << A[1, 2]
            out =  (c + n + s + w + e) / 5
            out >> ret
        return ret
    
    sdfg_pre = simple_tasklet.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((3, 3), dtype=np.float32)
    ref = (A[1, 1] + A[0, 1] + A[2, 1] + A[1, 0] + A[1, 2]) / 5

    val_pre = sdfg_pre(A=A)[0]
    val_post = sdfg_post(A=A)[0]

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


def test_multiple_outputs_tasklet():

    @dace.program
    def multiple_outputs_tasklet(A: dace.float32[3, 3]):
        ret = np.empty((2,), dtype=np.float32)
        with dace.tasklet:
            c << A[1, 1]
            n << A[0, 1]
            s << A[2, 1]
            w << A[1, 0]
            e << A[1, 2]
            out0 =  (c + n + s + w + e) / 5
            out1 = c / 2 + (n + s + w + e) / 2
            out0 >> ret[0]
            out1 >> ret[1]
        return ret
    
    sdfg_pre = multiple_outputs_tasklet.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((3, 3), dtype=np.float32)
    ref = np.empty((2,), dtype=np.float32)
    ref[0] = (A[1, 1] + A[0, 1] + A[2, 1] + A[1, 0] + A[1, 2]) / 5
    ref[1] = A[1, 1] / 2 + (A[0, 1] + A[2, 1] + A[1, 0] + A[1, 2]) / 2

    val_pre = sdfg_pre(A=A)
    val_post = sdfg_post(A=A)

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


def test_simple_wcr_tasklet():

    @dace.program
    def simple_wcr_tasklet(A: dace.float32[3, 3]):
        ret = dace.float32(2)
        with dace.tasklet:
            c << A[1, 1]
            n << A[0, 1]
            s << A[2, 1]
            w << A[1, 0]
            e << A[1, 2]
            out =  (c + n + s + w + e) / 5
            out >> ret(1, lambda x, y: x + y)
        return ret
    
    sdfg_pre = simple_wcr_tasklet.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((3, 3), dtype=np.float32)
    ref = 2 + (A[1, 1] + A[0, 1] + A[2, 1] + A[1, 0] + A[1, 2]) / 5

    val_pre = sdfg_pre(A=A)[0]
    val_post = sdfg_post(A=A)[0]

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


if __name__ == "__main__":
    # test_simple_tasklet()
    # test_multiple_outputs_tasklet()
    test_simple_wcr_tasklet()
