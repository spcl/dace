# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace import dtypes
from dace.transformation.dataflow import TaskletFusion, MapFusion
import pytest

datatype = dace.float32
np_datatype = np.float32
M = 10
N = 2 * M

@dace.program
def map_with_tasklets(A: datatype[N], B: datatype[M]):
    C = np.zeros_like(B)
    for i in dace.map[0:M]:
        a = A[i] + B[i]
        b = a * A[2 * i]
        c = a + B[i]
        d = c / b
        C[i] = a * d
    return C


def _make_sdfg(language: str, with_data: bool = False):
    lang = dtypes.Language.Python if language == 'Python' else dtypes.Language.CPP
    endl = '\n' if language == 'Python' else ';\n'

    sdfg = dace.SDFG(f'map_with_tasklets')
    sdfg.add_array('A', (N, ), datatype)
    sdfg.add_array('B', (M, ), datatype)
    sdfg.add_array('C', (M, ), datatype)
    state = sdfg.add_state(is_start_state=True)
    A = state.add_read('A')
    B = state.add_read('B')
    C = state.add_write('C')
    me, mx = state.add_map('Map', {'i': '0:' + str(M)})
    inputs = {
        '__inp1': datatype,
        '__inp2': datatype,
    }
    outputs = {
        '__out': datatype,
    }
    ta = state.add_tasklet(
        'a', inputs, {
            '__out1': datatype,
            '__out2': datatype,
            '__out3': datatype,
        },
        f'__out1 = __inp1 + __inp2{endl}__out2 = __out1{endl}__out3 = __out1{endl}',
        lang
    )
    tb = state.add_tasklet('b', inputs, outputs, f'__out = __inp1 * __inp2{endl}', lang)
    tc = state.add_tasklet('c', inputs, outputs, f'__out = __inp1 + __inp2{endl}', lang)
    td = state.add_tasklet('d', inputs, outputs, f'__out = __inp1 / __inp2{endl}', lang)
    te = state.add_tasklet('e', inputs, outputs, f'__out = __inp1 * __inp2{endl}', lang)
    state.add_memlet_path(A, me, ta, memlet=dace.Memlet('A[i]'), dst_conn='__inp1')
    state.add_memlet_path(B, me, ta, memlet=dace.Memlet('B[i]'), dst_conn='__inp2')
    state.add_memlet_path(A, me, tb, memlet=dace.Memlet('A[2*i]'), dst_conn='__inp2')
    state.add_memlet_path(B, me, tc, memlet=dace.Memlet('B[i]'), dst_conn='__inp2')
    if with_data:
        sdfg.add_array('tmp1', (1,), datatype, dtypes.StorageType.Default, None, True)
        sdfg.add_array('tmp2', (1,), datatype, dtypes.StorageType.Default, None, True)
        sdfg.add_array('tmp3', (1,), datatype, dtypes.StorageType.Default, None, True)
        sdfg.add_array('tmp4', (1,), datatype, dtypes.StorageType.Default, None, True)
        sdfg.add_array('tmp5', (1,), datatype, dtypes.StorageType.Default, None, True)
        sdfg.add_array('tmp6', (1,), datatype, dtypes.StorageType.Default, None, True)
        atemp1 = state.add_access('tmp1')
        atemp2 = state.add_access('tmp2')
        atemp3 = state.add_access('tmp3')
        atemp4 = state.add_access('tmp4')
        atemp5 = state.add_access('tmp5')
        atemp6 = state.add_access('tmp6')
        state.add_edge(ta, '__out1', atemp1, None, dace.Memlet('tmp1[0]'))
        state.add_edge(atemp1, None, tb, '__inp1', dace.Memlet('tmp1[0]'))
        state.add_edge(ta, '__out2', atemp2, None, dace.Memlet('tmp2[0]'))
        state.add_edge(atemp2, None, tc, '__inp1', dace.Memlet('tmp2[0]'))
        state.add_edge(tb, '__out', atemp3, None, dace.Memlet('tmp3[0]'))
        state.add_edge(atemp3, None, td, '__inp2', dace.Memlet('tmp3[0]'))
        state.add_edge(tc, '__out', atemp4, None, dace.Memlet('tmp4[0]'))
        state.add_edge(atemp4, None, td, '__inp1', dace.Memlet('tmp4[0]'))
        state.add_edge(ta, '__out3', atemp5, None, dace.Memlet('tmp5[0]'))
        state.add_edge(atemp5, None, te, '__inp1', dace.Memlet('tmp5[0]'))
        state.add_edge(td, '__out', atemp6, None, dace.Memlet('tmp6[0]'))
        state.add_edge(atemp6, None, te, '__inp2', dace.Memlet('tmp6[0]'))
    else:
        state.add_edge(ta, '__out1', tb, '__inp1', dace.Memlet())
        state.add_edge(ta, '__out2', tc, '__inp1', dace.Memlet())
        state.add_edge(tb, '__out', td, '__inp2', dace.Memlet())
        state.add_edge(tc, '__out', td, '__inp1', dace.Memlet())
        state.add_edge(ta, '__out3', te, '__inp1', dace.Memlet())
        state.add_edge(td, '__out', te, '__inp2', dace.Memlet())
    state.add_memlet_path(te, mx, C, memlet=dace.Memlet('C[i]'), src_conn='__out')

    return sdfg


def test_basic():
    @dace.program
    def test_basic_tf(A: datatype[5, 5]):
        B = A + 1
        return B * 2
    
    sdfg = test_basic_tf.to_sdfg(simplify=True)

    num_map_fusions = sdfg.apply_transformations(MapFusion)
    assert (num_map_fusions == 1)
    num_tasklet_fusions = sdfg.apply_transformations(TaskletFusion)
    assert (num_tasklet_fusions == 1)

    A = np.ones((5, 5), dtype=np_datatype)
    result = sdfg(A=A)
    assert np.allclose(result, 2 * (A + 1))


def test_same_name():
    @dace.program
    def test_same_name(A: datatype[5, 5]):
        B = A + 1
        C = A * 3
        return B + C

    sdfg = test_same_name.to_sdfg(simplify=True)

    num_map_fusions = sdfg.apply_transformations_repeated(MapFusion)
    assert (num_map_fusions == 2)
    num_tasklet_fusions = sdfg.apply_transformations_repeated(TaskletFusion)
    assert (num_tasklet_fusions == 2)

    A = np.ones((5, 5), dtype=np_datatype)
    result = sdfg(A=A)
    assert np.allclose(result, (A + 1) + (A * 3))


def test_same_name_different_memlet():
    @dace.program
    def test_same_name_different_memlet(A: datatype[5, 5], B: datatype[5, 5]):
        C = B * 3
        D = A + 1
        return D + C

    sdfg = test_same_name_different_memlet.to_sdfg(simplify=True)

    num_map_fusions = sdfg.apply_transformations_repeated(MapFusion)
    assert (num_map_fusions == 2)
    num_tasklet_fusions = sdfg.apply_transformations_repeated(TaskletFusion)
    assert (num_tasklet_fusions == 2)

    A = np.ones((5, 5), dtype=np_datatype)
    B = np.ones((5, 5), dtype=np_datatype) * 2
    result = sdfg(A=A, B=B)
    assert np.allclose(result, (A + 1) + (B * 3))


def test_tasklet_fusion_multiline():
    @dace.program
    def test_tasklet_fusion_multiline(A: datatype):
        B = A + 1
        C = dace.define_local([1], datatype)
        D = dace.define_local([1], datatype)

        with dace.tasklet:
            b << B[0]
            c >> C[0]
            d >> D[0]
            d = b * 3
            c = b + 3

        return C + D

    sdfg = test_tasklet_fusion_multiline.to_sdfg(simplify=True)

    num_tasklet_fusions = sdfg.apply_transformations(TaskletFusion)
    assert (num_tasklet_fusions == 1)

    result = sdfg(A=1)
    assert (result[0] == 11)


def test_map_param():
    @dace.program
    def map_uses_param(A: dace.float32[10], B: dace.float32[10], C: dace.float32[10]):
        for i in dace.map[0:10]:
            a = i - A[i]
            b = B[i] * i
            C[i] = a + b

    sdfg = map_uses_param.to_sdfg(simplify=True)

    num_tasklet_fusions = sdfg.apply_transformations(TaskletFusion)
    assert (num_tasklet_fusions == 1)

    A = np.zeros([10], dtype=np.float32)
    B = np.ones([10], dtype=np.float32)
    C = np.empty([10], dtype=np.float32)
    sdfg(A=A, B=B, C=C)

    ref = np.array(range(0, 10, 1)) * 2.0
    assert (C == ref).all()


@pytest.mark.parametrize('with_data', [pytest.param(True), pytest.param(False)])
@pytest.mark.parametrize('language', [pytest.param('CPP'), pytest.param('Python')])
def test_map_with_tasklets(language: str, with_data: bool):
    sdfg = _make_sdfg(language, with_data)
    sdfg.compile()
    sdfg.simplify()
    num = sdfg.apply_transformations_repeated(TaskletFusion)
    assert num == 3
    func = sdfg.compile()
    A = np.arange(1, N + 1, dtype=np_datatype)
    B = np.arange(1, M + 1, dtype=np_datatype)
    C = np.zeros((M, ), dtype=np_datatype)
    func(A=A, B=B, C=C)
    ref = map_with_tasklets.f(A, B)
    assert (np.allclose(C, ref))


if __name__ == '__main__':
    test_basic()
    test_same_name()
    test_same_name_different_memlet()
    test_tasklet_fusion_multiline()
    test_map_param()
    test_map_with_tasklets(language='Python', with_data=False)
    test_map_with_tasklets(language='Python', with_data=True)
    test_map_with_tasklets(language='CPP', with_data=False)
    test_map_with_tasklets(language='CPP', with_data=True)
