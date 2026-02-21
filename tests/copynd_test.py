# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=[pytest.mark.gpu])])
def test(device: str):
    N = dace.symbol('N')
    n = 20

    sdfg = dace.SDFG(f'copynd_{device}')
    state = sdfg.add_state()

    arrays = []

    # Copy 1: sub-2d array to sub-2d array
    sdfg.add_array('A_' + str(len(arrays)), [N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [5, 7], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-2], '5:10, N-7:N'))

    # Copy 2: 1d subset of a 4d array to a 1d array
    sdfg.add_array('A_' + str(len(arrays)), [N, N, N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [N - 1], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-2], '4,1,1:N,2'))

    # Copy 3: 5d array to 5d array
    sdfg.add_array('A_' + str(len(arrays)), [2, 3, 4, 5, 6], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [2, 3, 4, 5, 6], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-2], '0:2,0:3,0:4,0:5,0:6'))

    # Copy 4: contiguous 1d subset of a 4d array to a 1d array
    sdfg.add_array('A_' + str(len(arrays)), [N, N, N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [N - 1], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-2], '4,1,2,1:N'))

    # Copy 5: 1d array to a 1d subset of a 4d array
    sdfg.add_array('A_' + str(len(arrays)), [N - 2], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [N, N, N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-1], '4,1:N-1,1,2'))

    # Copy 6: 4d array to a contiguous 1d subset of a 4d array
    sdfg.add_array('A_' + str(len(arrays)), [N - 2], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [N, N, N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-1], '4,1,2,1:N-1'))

    # Copy 7: True 4d copy (4d subarray to 4d array)
    sdfg.add_array('A_' + str(len(arrays)), [N, N, N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [N - 5, N - 4, 3, N - 2], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-2],
                                                                                 '5:N,2:N-2,N-10:N-7,1:N-1'))

    # Copy 8: 4d array with a stride to a 1d array
    sdfg.add_array('A_' + str(len(arrays)), [N, N, N, N], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [N / 2 - 1], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None, dace.memlet.Memlet.simple(arrays[-2], '4,1,2,1:N-1:2'))

    # Copy 9: 2d array to a 3d array with an offset
    sdfg.add_array('A_' + str(len(arrays)), [40, 40], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [3, 40, 40], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None,
                   dace.memlet.Memlet.simple(arrays[-2], '20:40, 10:30', other_subset_str='2, 10:30, 20:40'))

    # Copy 10: Copying from 2d array to another 2d array a 1d slice
    # The src stride is 1, while the dst stride is N
    sdfg.add_array('A_' + str(len(arrays)), [40, 40], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [40, 40], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None,
                   dace.memlet.Memlet.simple(arrays[-2], '20, 10:30', other_subset_str='10:30, 10'))

    # Copy 11: Copying from 2d array to another 2d array a 1d slice
    # The src stride is N, while the dst stride is 1
    sdfg.add_array('A_' + str(len(arrays)), [40, 40], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [40, 40], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None,
                   dace.memlet.Memlet.simple(arrays[-2], '10:30, 20', other_subset_str='10, 10:30'))

    # Copy 12: (from resnet)
    # padded[:, 1:H+1, 1:W+1, :] = conv2d(input, conv1)
    # output shape is (N, H + 2,W + 2, C2)
    # input shape is (N, H , W, C2)
    sdfg.add_array('A_' + str(len(arrays)), [5, 6, 7, 8], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    sdfg.add_array('A_' + str(len(arrays)), [5, 8, 9, 8], dace.float32)
    arrays.append(state.add_access('A_' + str(len(arrays))))
    state.add_edge(arrays[-2], None, arrays[-1], None,
                   dace.memlet.Memlet.simple(arrays[-2], '0:5, 0:6, 0:7 , 0:8', other_subset_str='0:5, 1:7, 1:8, 0:8'))

    array_data = [
        np.random.rand(*[dace.symbolic.evaluate(s, {N: n})
                         for s in a.desc(sdfg).shape]).astype(a.desc(sdfg).dtype.type) for a in arrays
    ]

    args = {anode.label: adata for anode, adata in zip(arrays, array_data)}
    args['N'] = n
    if device == "gpu":
        sdfg.apply_gpu_transformations()
    sdfg(**args)

    N = n

    diffs = [
        np.linalg.norm(array_data[1] - array_data[0][5:10, N - 7:N]) / 5.0 * 7.0,
        np.linalg.norm(array_data[3] - array_data[2][4, 1, 1:, 2]) / (N - 1),
        np.linalg.norm(array_data[5] - array_data[4]) / 2.0 * 3 * 4 * 5 * 6,
        np.linalg.norm(array_data[7] - array_data[6][4, 1, 2, 1:]) / (N - 1),
        np.linalg.norm(array_data[9][4, 1:N - 1, 1, 2] - array_data[8]) / (N - 2),
        np.linalg.norm(array_data[11][4, 1, 2, 1:N - 1] - array_data[10]) / (N - 2),
        np.linalg.norm(array_data[13] - array_data[12][5:N, 2:N - 2, N - 10:N - 7, 1:N - 1]) / ((N - 5) * (N - 4) * 3 *
                                                                                                (N - 2)),
        np.linalg.norm(array_data[15] - array_data[14][4, 1, 2, 1:(N - 1):2]) / (N / 2 - 1),
        np.linalg.norm(array_data[17][2, 10:30, 20:40] - array_data[16][20:40, 10:30]) / 400,
        np.linalg.norm(array_data[19][10:30, 10] - array_data[18][20, 10:30]) / 20,
        np.linalg.norm(array_data[21][10, 10:30] - array_data[20][10:30, 20]) / 20,
        np.linalg.norm(array_data[23][:, 1:-1, 1:-1, :] - array_data[22]) / 1680
    ]

    assert all([diff < 1e-7 for diff in diffs])


if __name__ == "__main__":
    test("cpu")
    test("gpu")
