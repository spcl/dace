""" Tries to allocate a symbolically-sized array on a register and makes
    sure that it is allocated on the heap instead.
"""
import dace
import numpy as np
import warnings

N = dace.symbol('N')

sdfg = dace.SDFG('vla_test')
sdfg.add_array('A', [N], dace.float32)
sdfg.add_transient('tmp', [N], dace.float32, storage=dace.StorageType.Register)
sdfg.add_array('B', [N], dace.float32)
state = sdfg.add_state()
A = state.add_read('A')
tmp = state.add_access('tmp')
B = state.add_write('B')

state.add_nedge(A, tmp, dace.Memlet.simple('A', '0:N'))
state.add_nedge(tmp, B, dace.Memlet.simple('tmp', '0:N'))

if __name__ == '__main__':
    N.set(12)
    A = np.random.rand(12).astype(np.float32)
    B = np.random.rand(12).astype(np.float32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sdfg(A=A, B=B, N=N)

        if not w:
            print('FAIL: No warnings caught')
            exit(2)
        if not any('Variable-length' in str(warn.message) for warn in w):
            print('FAIL: No VLA warnings caught')
            exit(3)

    diff = np.linalg.norm(A - B)
    print('Difference:', diff)
    exit(0 if diff < 1e-5 else 1)
