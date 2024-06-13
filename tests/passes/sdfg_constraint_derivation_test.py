# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.passes.analysis import DeriveSDFGConstraints


def test_infer_data_dim_constraints_nomax():
    N = dace.symbol('N')

    @dace.program
    def matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C[i, j] += A[i, k] * B[k, j]

    sdfg = matmul.to_sdfg()

    derive_pass = DeriveSDFGConstraints()
    _, inv, _ = derive_pass.apply_pass(sdfg, {})

    assert 'N' in inv
    assert 'N > 0' in inv['N']


def test_infer_data_dim_constraints_withmax():
    N = dace.symbol('N')

    @dace.program
    def matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C[i, j] += A[i, k] * B[k, j]

    sdfg = matmul.to_sdfg()

    derive_pass = DeriveSDFGConstraints()
    derive_pass.assume_max_data_size = 128
    _, inv, _ = derive_pass.apply_pass(sdfg, {})

    assert 'N' in inv
    assert 'N > 0' in inv['N']
    assert 'N <= 128' in inv['N']


if __name__ == "__main__":
    test_infer_data_dim_constraints_nomax()
    test_infer_data_dim_constraints_withmax()
