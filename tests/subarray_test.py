# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np

W = dc.symbol('W')


@dc.program
def subarray(A, B):
    @dc.map(_[0:W])
    def subarrays(i):
        a << A[:, i, i, i]
        a2 << A(1)[i, i, i, :]
        b >> B[i, :, i, i]
        b[i] = a[i] + a2[i]


def test():
    W.set(3)

    A = dc.ndarray([W, W, W, W])
    B = dc.ndarray([W, W, W, W])

    A[:] = np.mgrid[0:W.get(), 0:W.get(), 0:W.get()]
    for i in range(W.get()):
        A[i, :] += 10 * (i + 1)
    B[:] = dc.float32(0.0)

    subarray(A, B, W=W)


def test_strides_propagation_to_tasklet():
    N = 2
    vec_stride_W, vec_stride_d0 = (
        dc.symbol(s) for s in ['vec_stride_W', 'vec_stride_d0']
    )
    mat_stride_W, mat_stride_d0, mat_stride_d1 = (
        dc.symbol(s) for s in ['mat_stride_W', 'mat_stride_d0', 'mat_stride_d1']
    )

    def build_nsdfg():
        sdfg = dc.SDFG('vec_to_mat')
        sdfg.add_symbol('_w', dc.int32)
        sdfg.add_array(
            'vec_field', (W, N), dc.float64,
            strides=(vec_stride_W, vec_stride_d0)
        )
        sdfg.add_array(
            'mat_field', (1, N, N), dc.float64,
            strides=(mat_stride_W, mat_stride_d0, mat_stride_d1)
        )
        state = sdfg.add_state()
        tasklet = state.add_tasklet(
            'vec_to_mat', {'_vec'}, {'_mat'}, f"\
for i in range({N}):\
    _mat[0, i, i] = _vec[0, i] + dace.float64(1)\
        ")
        state.add_edge(
            state.add_access('vec_field'), None,
            tasklet, '_vec',
            dc.Memlet.simple('vec_field', f"_w, 0:{N}")
        )
        state.add_edge(
            tasklet, '_mat',
            state.add_access('mat_field'), None,
            dc.Memlet.from_array('mat_field', sdfg.arrays['mat_field'])
        )
        return sdfg

    sdfg = dc.SDFG('test')
    sdfg.add_array(
        'vec_field', (W, N), dc.float64,
        strides=(vec_stride_W, vec_stride_d0)
    )
    sdfg.add_array(
        'mat_field', (W, N, N), dc.float64,
        strides=(mat_stride_W, mat_stride_d0, mat_stride_d1)
    )
    state = sdfg.add_state()
    me, mx = state.add_map('map_W', dict(i='0:W'))
    nsdfg_node = state.add_nested_sdfg(
        build_nsdfg(),
        sdfg,
        inputs={'vec_field'},
        outputs={'mat_field'},
        symbol_mapping={
            '_w': 'i',
        }
    )
    state.add_memlet_path(
        state.add_access('vec_field'),
        me,
        nsdfg_node,
        dst_conn='vec_field',
        memlet=dc.Memlet.simple('vec_field', f"i,0:{N}")
    )
    state.add_memlet_path(
        nsdfg_node,
        mx,
        state.add_access('mat_field'),
        src_conn='mat_field',
        memlet=dc.Memlet.simple('mat_field', f"i,0:{N},0:{N}")
    )

    W.set(3)
    A = np.random.rand(W.get(), N)
    B = np.random.rand(W.get(), N, N)

    ref = np.ndarray(B.shape)
    for i in range(ref.shape[0]):
        ref[i,:,:] = np.diag(A[i,:] + 1) + np.rot90(np.diag(np.diag(np.rot90(B[i,:,:], axes=(1,0)))))

    A_stride_0, A_stride_1 = (s / A.itemsize for s in A.strides)
    B_stride_0, B_stride_1, B_stride_2 = (s / B.itemsize for s in B.strides)
    stride_symbols = {
        'vec_stride_W': A_stride_0,
        'vec_stride_d0': A_stride_1,
        'mat_stride_W': B_stride_0,
        'mat_stride_d0': B_stride_1,
        'mat_stride_d1': B_stride_2,
    }

    sdfg(vec_field=A, mat_field=B, W=W, **stride_symbols)
    np.allclose(ref, B)


if __name__ == "__main__":
    test()
    test_strides_propagation_to_tasklet()
