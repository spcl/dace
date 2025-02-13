# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def tlarray(A: dace.int32[128]):
    tmp = dace.ndarray([128], dace.int32, storage=dace.StorageType.CPU_ThreadLocal)

    for i in dace.map[0:128]:
        with dace.tasklet:
            # Assuming OpenMP is used
            t = omp_get_thread_num()
            t >> tmp[i]

    for i in dace.map[0:128]:
        with dace.tasklet:
            t << tmp[i]
            o >> A[i]
            # If tmp is thread-local, will be distributed across thread IDs
            o = t


def test_threadlocal():
    A = np.ndarray([128], dtype=np.int32)
    A[:] = -1

    # Add OpenMP include
    sdfg = tlarray.to_sdfg()
    sdfg.set_global_code('#include <omp.h>')

    sdfg(A=A)
    assert np.all(A >= 0)
    print('OK. Detected threads:', np.max(A) + 1)


if __name__ == '__main__':
    test_threadlocal()
