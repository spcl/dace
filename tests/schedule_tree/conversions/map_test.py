# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests conversion of Map scopes from SDFG to ScheduleTree and back. """
import dace
import numpy as np
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree, as_sdfg


def test_simple_map():
    """ Tests a Map Scope with a single (non-WCR) output. """

    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def simple_map(A: dace.float32[M, N]):
        B = np.zeros((M, N), dtype=A.dtype)
        for i, j in dace.map[1:M-1, 1:N-1]:
            with dace.tasklet:
                c << A[i, j]
                n << A[i-1, j]
                s << A[i+1, j]
                w << A[i, j-1]
                e << A[i, j+1]
                out =  (c + n + s + w + e) / 5
                out >> B[i, j]
        return B
    
    sdfg_pre = simple_map.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    print(tree.as_string())
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((20, 20), dtype=np.float32)
    ref = np.zeros_like(A)
    for i, j in dace.map[1:19, 1:19]:
        ref[i, j] = (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 5

    val_pre = sdfg_pre(A=A, M=20, N=20)
    val_post = sdfg_post(A=A, M=20, N=20)

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


def test_multiple_outputs_map():
    """ Tests a Map Scope with multiple (non-WCR) outputs. """

    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def multiple_outputs_map(A: dace.float32[M, N]):
        B = np.zeros((2, M, N), dtype=A.dtype)
        for i, j in dace.map[1:M-1, 1:N-1]:
            with dace.tasklet:
                c << A[i, j]
                n << A[i-1, j]
                s << A[i+1, j]
                w << A[i, j-1]
                e << A[i, j+1]
                out0 =  (c + n + s + w + e) / 5
                out1 = c / 2 + (n + s + w + e) / 2
                out0 >> B[0, i, j]
                out1 >> B[1, i, j]
        return B
    
    sdfg_pre = multiple_outputs_map.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((20, 20), dtype=np.float32)
    ref = np.zeros_like(A, shape=(2, 20, 20))
    for i, j in dace.map[1:19, 1:19]:
        ref[0, i, j] = (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 5
        ref[1, i, j] = A[i, j] / 2 + (A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 2

    val_pre = sdfg_pre(A=A, M=20, N=20)
    val_post = sdfg_post(A=A, M=20, N=20)

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


def test_simple_wcr_map():
    """ Tests a Map Scope with a single WCR output. """

    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def simple_wcr_map(A: dace.float32[M, N]):
        ret = dace.float32(0)
        for i, j in dace.map[1:M-1, 1:N-1]:
            with dace.tasklet:
                c << A[i, j]
                n << A[i-1, j]
                s << A[i+1, j]
                w << A[i, j-1]
                e << A[i, j+1]
                out =  (c + n + s + w + e) / 5
                out >> ret(1, lambda x, y: x + y)
        return ret
    
    sdfg_pre = simple_wcr_map.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((20, 20), dtype=np.float32)
    ref = np.float32(0)
    for i, j in dace.map[1:19, 1:19]:
        ref += (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 5

    val_pre = sdfg_pre(A=A, M=20, N=20)[0]
    val_post = sdfg_post(A=A, M=20, N=20)[0]

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


def test_simple_wcr_map2():
    """ Tests a Map Scope with a single WCR output. The output is also (fake) input with WCR. """

    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def simple_wcr_map2(A: dace.float32[M, N]):
        ret = dace.float32(0)
        for i, j in dace.map[1:M-1, 1:N-1]:
            with dace.tasklet:
                c << A[i, j]
                n << A[i-1, j]
                s << A[i+1, j]
                w << A[i, j-1]
                e << A[i, j+1]
                inp << ret(1, lambda x, y: x + y)
                out =  (c + n + s + w + e) / 5
                out >> ret(1, lambda x, y: x + y)
        return ret
    
    sdfg_pre = simple_wcr_map2.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((20, 20), dtype=np.float32)
    ref = np.float32(0)
    for i, j in dace.map[1:19, 1:19]:
        ref += (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 5

    val_pre = sdfg_pre(A=A, M=20, N=20)[0]
    val_post = sdfg_post(A=A, M=20, N=20)[0]

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


def test_multiple_outputs_mixed_map():
    """ Tests a Map Scope with multiple (WCR and non-WCR) outputs. """

    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def multiple_outputs_map(A: dace.float32[M, N]):
        B = np.zeros((M, N), dtype=A.dtype)
        ret = np.float32(0)
        for i, j in dace.map[1:M-1, 1:N-1]:
            with dace.tasklet:
                c << A[i, j]
                n << A[i-1, j]
                s << A[i+1, j]
                w << A[i, j-1]
                e << A[i, j+1]
                out0 =  (c + n + s + w + e) / 5
                out1 = out0
                out0 >> B[i, j]
                out1 >> ret(1, lambda x, y: x + y)
        return B, ret
    
    sdfg_pre = multiple_outputs_map.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((20, 20), dtype=np.float32)
    ref0 = np.zeros_like(A, shape=(20, 20))
    ref1 = np.float32(0)
    for i, j in dace.map[1:19, 1:19]:
        ref0[i, j] = (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 5
        ref1 += ref0[i, j]

    val_pre = sdfg_pre(A=A, M=20, N=20)
    val_post = sdfg_post(A=A, M=20, N=20)

    assert np.allclose(val_pre[0], ref0)
    assert np.allclose(val_pre[1], ref1)
    assert np.allclose(val_post[0], ref0)
    assert np.allclose(val_post[1], ref1)


# NOTE: This fails due to input connector appearing to be written (issue with Views)
def test_nested_simple_map():
    """ Tests a nested Map Scope with a single (non-WCR) output. """

    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def nested_simple_map(A: dace.float32[M, N]):
        B = np.zeros((M, N), dtype=A.dtype)
        for i, j in dace.map[1:M-2:2, 1:N-2:2]:
            inA = A[i-1:i+3, j-1:j+3]
            for k, l in dace.map[0:2, 0:2]:
                with dace.tasklet:
                    c << inA[k+1, l+1]
                    n << inA[k, l+1]
                    s << inA[k+2, l+1]
                    w << inA[k+1, l]
                    e << inA[k+1, l+2]
                    out =  (c + n + s + w + e) / 5
                    out >> B[i+k, j+l]
        return B
    
    sdfg_pre = nested_simple_map.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    sdfg_post = as_sdfg(tree)

    rng = np.random.default_rng(42)
    A = rng.random((20, 20), dtype=np.float32)
    ref = np.zeros_like(A)
    for i, j in dace.map[1:19, 1:19]:
        ref[i, j] = (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 5

    val_pre = sdfg_pre(A=A, M=20, N=20)
    val_post = sdfg_post(A=A, M=20, N=20)

    assert np.allclose(val_pre, ref)
    assert np.allclose(val_post, ref)


if __name__ == "__main__":
    test_simple_map()
    test_multiple_outputs_map()
    test_simple_wcr_map()
    test_simple_wcr_map2()
    test_multiple_outputs_mixed_map()
    test_nested_simple_map()
