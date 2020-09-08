import unittest
import dace
import numpy as np
from dace.transformation.interstate import InlineSDFG
from dace.transformation.dataflow import SplitMapTiling

M, N, K = (dace.symbol(s) for s in ['M', 'N', 'K'])


@dace.program
def mm_kernel(A: dace.float64[M, K], B: dace.float64[K, N],
              C: dace.float64[M, N]):
    C[:] = A @ B


class SplitTilingTests(unittest.TestCase):
    def test_gemm(self):
        sdfg = mm_kernel.to_sdfg()
        sdfg.expand_library_nodes()
        sdfg.apply_transformations([InlineSDFG, SplitMapTiling],
                                   options=[{}, {'tile_sizes': [8]}])
        
        M.set(31), N.set(15), K.set(23)

        A = np.random.rand(M.get(), K.get()).astype(np.float64)
        B = np.random.rand(K.get(), N.get()).astype(np.float64)
        C = np.random.rand(M.get(), N.get()).astype(np.float64)

        origC = np.zeros([M.get(), N.get()], dtype=np.float64)
        origC[:] = C

        sdfg(A=A, B=B, C=C, M=M, N=N, K=K)
        realC = A @ B

        diff = np.linalg.norm(C - realC) / (M.get() * N.get())
        self.assertTrue(diff <= 1e-15)


if __name__ == '__main__':
    unittest.main()
