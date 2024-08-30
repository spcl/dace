# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import dace
from dace.sdfg.state import FunctionCallRegion

def test_function_call():
    N = dace.symbol("N")
    def func(A: dace.float64[N]):
        return 5 * A + 10
    @dace.program
    def prog(I: dace.float64[N]):
        return func(I)
    prog.use_experimental_cfg_blocks = True
    sdfg = prog.to_sdfg()
    call_region: FunctionCallRegion = sdfg.nodes()[1]
    assert call_region.arguments == {'A': 'I'}
    assert sdfg(np.array([+1], dtype=np.float64), N=1) == 15
    assert sdfg(np.array([-1], dtype=np.float64), N=1) == 5

def test_function_call_with_args():
    N = dace.symbol("N")
    def func(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        return A * B + C
    @dace.program
    def prog(E: dace.float64[N], F: dace.float64[N], G: dace.float64[N]):
        func(A=E, B=F, C=G)
        func(A=G, B=E, C=E)
    prog.use_experimental_cfg_blocks = True
    E = np.array([1])
    F = np.array([2])
    G = np.array([3])
    sdfg = prog.to_sdfg(E=E, F=F, G=G, N=1)
    call1: FunctionCallRegion = sdfg.nodes()[1]
    call2: FunctionCallRegion = sdfg.nodes()[2]
    assert call1.arguments == {'A': 'E', 'B': 'F', 'C': 'G'}
    assert call2.arguments == {'A': 'G', 'B': 'E', 'C': 'E'}

def test_function_call_with_transients():
    N = dace.symbol("N")
    def func(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        return A * B + C
    @dace.program
    def prog():
        func(A=np.array([1]), B=np.array([2]), C=np.array([3]))
        func(A=np.array([3]), B=np.array([1]), C=np.array([1]))
    prog.use_experimental_cfg_blocks = True
    sdfg = prog.to_sdfg(N=1)
    call1: FunctionCallRegion = sdfg.nodes()[1]
    call2: FunctionCallRegion = sdfg.nodes()[2]
    assert call1.arguments == {'A': '__tmp0', 'B': '__tmp1', 'C': '__tmp2'}
    assert call2.arguments == {'A': '__tmp4', 'B': '__tmp5', 'C': '__tmp6'}

if __name__ == "__main__":
    test_function_call()
    test_function_call_with_args()
    test_function_call_with_transients()