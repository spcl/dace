import numpy as np
import dace
import pytest
import torch
from dace.transformation.dataflow import MapFusion

from dace.transformation.onnx import TaskletFusion, TaskletFission
from dace.transformation.onnx.tasklet_fusion import MergeTaskletReads
from torch import nn
from torch.nn import functional as F


def test_basic():

    @dace.program
    def test_basic_tf(A: dace.float32[5, 5]):
        B = A + 1
        return B * 2

    sdfg: dace.SDFG = test_basic_tf.to_sdfg()

    assert sdfg.apply_transformations(MapFusion) == 1
    assert sdfg.apply_transformations(TaskletFusion) == 1

    result = np.ones((5, 5), dtype=np.float32)
    sdfg.save("log_sdfgs/test_basic_tf_after.sdfg")
    result = sdfg(result)
    assert np.allclose(result, 2 * (np.ones_like(result) + 1))


def test_same_name():

    @dace.program
    def test_same_name(A: dace.float32[5, 5]):
        B = A + 1
        C = A * 3
        return B + C

    sdfg: dace.SDFG = test_same_name.to_sdfg()

    assert sdfg.apply_transformations_repeated(MapFusion) == 2
    assert sdfg.apply_transformations_repeated(TaskletFusion) == 2

    result = np.empty((5, 5), dtype=np.float32)
    A = np.ones_like(result)
    result = sdfg(A=A)
    assert np.allclose(result, A + 1 + A * 3)


def test_same_name_diff_memlet():

    @dace.program
    def test_same_name_diff_memlet(A: dace.float32[5, 5], B: dace.float32[5,
                                                                          5]):
        D = A + 1
        C = B * 3
        return D + C

    sdfg: dace.SDFG = test_same_name_diff_memlet.to_sdfg()

    assert sdfg.apply_transformations_repeated(MapFusion) == 2
    assert sdfg.apply_transformations_repeated(TaskletFusion) == 2

    result = np.empty((5, 5), dtype=np.float32)
    A = np.ones_like(result)
    B = np.ones_like(result) * 2
    result = sdfg(A=A, B=B)
    assert np.allclose(result, A + 1 + B * 3)


def test_tasklet_fission_dependent_statements():

    @dace.program
    def test_basic_tf(A: dace.float32, D: dace.float32):

        B = dace.define_local_scalar(dace.float32)
        C = dace.define_local([1], dace.float32)
        with dace.tasklet:
            a << A[0]
            d << D[0]
            b >> B[0]
            c >> C[0]

            b = d + 1
            c = a * 3 + b

        C += B
        return C

    sdfg: dace.SDFG = test_basic_tf.to_sdfg()

    assert sdfg.apply_transformations(TaskletFission) == 0


def test_tasklet_fission_useless_statement():

    @dace.program
    def test_basic_tf(A: dace.float32, D: dace.float32):

        B = dace.define_local_scalar(dace.float32)
        C = dace.define_local([1], dace.float32)
        with dace.tasklet:
            a << A[0]
            d << D[0]
            b >> B[0]
            c >> C[0]

            x = 42
            b = d + 1
            c = a * 3

        C += B
        return C

    sdfg: dace.SDFG = test_basic_tf.to_sdfg()

    assert sdfg.apply_transformations(TaskletFission) == 1

    result = np.empty((1, ), dtype=np.float32)
    result = sdfg(A=1, D=2)
    assert result[0] == 6


def test_tasklet_fission():

    @dace.program
    def test_basic_tf(A: dace.float32, D: dace.float32):

        B = dace.define_local_scalar(dace.float32)
        C = dace.define_local([1], dace.float32)
        with dace.tasklet:
            a << A[0]
            d << D[0]
            b >> B[0]
            c >> C[0]

            b = d + 1
            c = a * 3

        C += B
        return C

    sdfg: dace.SDFG = test_basic_tf.to_sdfg()

    assert sdfg.apply_transformations(TaskletFission) == 1

    result = np.empty((1, ), dtype=np.float32)
    result = sdfg(A=1, D=2)
    assert result[0] == 6


def test_tasklet_fusion_multiline():

    @dace.program
    def test_tf_multiline(A: dace.float32):

        D = dace.define_local([1], dace.float32)
        C = dace.define_local([1], dace.float32)
        B = A + 1
        with dace.tasklet:
            b << B[0]
            c >> C[0]
            d >> D[0]

            d = b * 3
            c = b + 3

        return C + D

    sdfg: dace.SDFG = test_tf_multiline.to_sdfg()

    assert sdfg.apply_transformations(TaskletFusion) == 1

    result = np.empty((1, ), dtype=np.float32)
    result = sdfg(A=1)
    assert result[0] == 11


@pytest.mark.skip(reason="Sigmoid expansion has changed, needs update")
@pytest.mark.pure
def test_silu():

    @dace.module
    class silu(nn.Module):

        def forward(self, x):
            return F.silu(x)

    m = silu()

    def fuse(m):
        sdfg = m.sdfg
        assert sdfg.apply_transformations(MapFusion) == 1
        assert sdfg.apply_transformations(TaskletFusion) == 1
        assert sdfg.apply_transformations(MergeTaskletReads) == 1

    m.append_post_onnx_hook("fuse", fuse)
    m(torch.rand(3, 3))


if __name__ == "__main__":
    test_basic()
    test_same_name()
    test_same_name_diff_memlet()
    test_tasklet_fission_dependent_statements()
    test_tasklet_fission_useless_statement()
    test_tasklet_fission()
    test_tasklet_fusion_multiline()
    test_silu()
