# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import dace
import numpy as np

csrmatrix = dace.struct(
    'csr',  # CSR Matrix definition type
    rows=dace.int32,
    cols=dace.int32,
    nnz=dace.int32,
    data=(dace.pointer(dace.float32), 'nnz'),
    rowsp1=dace.int32,
    rowptr=(dace.pointer(dace.int32), 'rowsp1'),
    colind=(dace.pointer(dace.int32), 'nnz'))

sdfg = dace.SDFG('addone')
state = sdfg.add_state()
sdfg.add_array('sparsemats_in', [5], dtype=csrmatrix)
sdfg.add_array('sparsemats_out', [5], dtype=csrmatrix)

ome, omx = state.add_map('matrices', dict(i='0:5'))
tasklet = state.add_tasklet('addone', {'mat_in'}, {'mat_out': dace.pointer(csrmatrix)},
                            '''
for (int j = 0; j < mat_in.nnz; ++j) {
    mat_out->data[j] = mat_in.data[j] + 1.0f;
}
''',
                            language=dace.Language.CPP)
matr = state.add_read('sparsemats_in')
matw = state.add_write('sparsemats_out')
state.add_memlet_path(matr, ome, tasklet, dst_conn='mat_in', memlet=dace.Memlet.simple('sparsemats_in', 'i'))
# state.add_nedge(tasklet, omx, dace.Memlet())
state.add_memlet_path(tasklet, omx, matw, src_conn='mat_out', memlet=dace.Memlet.simple('sparsemats_out', 'i'))


def toptr(arr):
    return arr.__array_interface__['data'][0]


def test():
    func = sdfg.compile()
    inp = np.ndarray([5], dtype=np.dtype(csrmatrix.as_ctypes()))
    out = np.ndarray([5], dtype=np.dtype(csrmatrix.as_ctypes()))
    in_data = []
    out_data = []
    refdata = []
    for i in range(5):
        in_data.append(np.array(list(range(i + 1))).astype(np.float32))
        out_data.append(np.array(list(range(i + 1))).astype(np.float32))
        refdata.append(np.array(list(range(i + 1))).astype(np.float32) + 1)

        inp[i]['nnz'] = i + 1
        inp[i]['data'] = toptr(in_data[-1])
        inp[i]['rowsp1'] = len(range(i + 1)) + 1
        inp[i]['rowptr'] = toptr(np.array(list(range(i + 1))).astype(np.int32))
        inp[i]['colind'] = toptr(np.array(list(range(i + 1))).astype(np.int32))

        out[i]['nnz'] = i + 1
        out[i]['data'] = toptr(out_data[-1])
        out[i]['rowsp1'] = len(range(i + 1)) + 1
        out[i]['rowptr'] = toptr(np.array(list(range(i + 1))).astype(np.int32))
        out[i]['colind'] = toptr(np.array(list(range(i + 1))).astype(np.int32))

    func(sparsemats_in=inp, sparsemats_out=out)
    diff = 0.0
    for i in range(5):
        diff += np.linalg.norm(out_data[i] - refdata[i])

    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
