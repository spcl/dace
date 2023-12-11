# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace

def test_broken_symbol_pre_condition():
    N = dace.symbol('N')

    @dace.program
    def simple_matmul(A: dace.float64[N, N], B: dace.float64[N, N]):
        return A @ B

    sdfg = simple_matmul.to_sdfg(simplify=True)

    sdfg.pre_conditions['N'] = 'N > 30'
