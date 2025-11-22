# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from dace.transformation.helpers import replace_sdfg_dtypes


def test_simple_array_type_change():
    N = dace.symbol('N')
    
    @dace.program
    def simple(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0
    
    sdfg = simple.to_sdfg()
    
    assert sdfg.arrays['a'].dtype == dace.float64
    assert sdfg.arrays['b'].dtype == dace.float64
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert count >= 2
    assert sdfg.arrays['a'].dtype == dace.float32
    assert sdfg.arrays['b'].dtype == dace.float32
    
    A = np.random.rand(10).astype(np.float32)
    B = np.zeros(10, dtype=np.float32)
    sdfg(a=A, b=B, N=10)
    assert np.allclose(B, A * 2.0)


def test_multiple_arrays():
    @dace.program
    def multi_array(a: dace.float64[10], b: dace.float64[10], c: dace.float64[10]):
        c[:] = a[:] + b[:]
    
    sdfg = multi_array.to_sdfg()
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert sdfg.arrays['a'].dtype == dace.float32
    assert sdfg.arrays['b'].dtype == dace.float32
    assert sdfg.arrays['c'].dtype == dace.float32
    assert count == 3


def test_mixed_types():
    @dace.program
    def mixed_types(a: dace.float64[10], b: dace.int32[10], c: dace.float64[10]):
        for i in range(10):
            c[i] = a[i] * 2.0 + dace.float64(b[i])
    
    sdfg = mixed_types.to_sdfg()
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert sdfg.arrays['a'].dtype == dace.float32
    assert sdfg.arrays['c'].dtype == dace.float32
    
    # Verify int32 unchanged
    assert sdfg.arrays['b'].dtype == dace.int32
    assert count >= 2


def test_nested_sdfg():
    @dace.program
    def inner(x: dace.float64[10]):
        return x * 2.0
    
    @dace.program
    def outer(a: dace.float64[10], b: dace.float64[10]):
        b[:] = inner(a)
    
    sdfg = outer.to_sdfg()
    sdfg.simplify()
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert sdfg.arrays['a'].dtype == dace.float32
    assert sdfg.arrays['b'].dtype == dace.float32
    assert count > 0


def test_symbols():
    sdfg = dace.SDFG('test_symbols')
    sdfg.add_symbol('x', dace.float64)
    sdfg.add_symbol('y', dace.int32)
    sdfg.add_array('A', [10], dace.float64)
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert sdfg.symbols['x'] == dace.float32
    assert sdfg.symbols['y'] == dace.int32
    assert sdfg.arrays['A'].dtype == dace.float32
    assert count == 2


def test_connectors():
    sdfg = dace.SDFG('test_connectors')
    state = sdfg.add_state()
    
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    
    tasklet = state.add_tasklet('compute', {'in1'}, {'out1'}, 'out1 = in1 * 2.0')
    
    tasklet.in_connectors = {'in1': dace.float64}
    tasklet.out_connectors = {'out1': dace.float64}
    
    A = state.add_access('A')
    B = state.add_access('B')
    
    state.add_edge(A, None, tasklet, 'in1', dace.Memlet('A[0:10]'))
    state.add_edge(tasklet, 'out1', B, None, dace.Memlet('B[0:10]'))
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert tasklet.in_connectors['in1'] == dace.float32
    assert tasklet.out_connectors['out1'] == dace.float32
    assert count == 4  # 2 for the tasklet, 2 for the edges


def test_pointer_types():
    sdfg = dace.SDFG('test_pointers')
    state = sdfg.add_state()
    
    tasklet = state.add_tasklet('compute', {'ptr_in'}, {'ptr_out'}, 'ptr_out[0] = ptr_in[0] * 2.0')
    tasklet.in_connectors = {'ptr_in': dace.pointer(dace.float64)}
    tasklet.out_connectors = {'ptr_out': dace.pointer(dace.float64)}
    
    count = replace_sdfg_dtypes(sdfg, dace.float64, dace.float32)
    
    assert tasklet.in_connectors['ptr_in'].base_type == dace.float32
    assert tasklet.out_connectors['ptr_out'].base_type == dace.float32
    assert count == 2


def test_no_changes():
    @dace.program
    def no_match(a: dace.float64[10], b: dace.float64[10]):
        b[:] = a[:] * 2.0
    
    sdfg = no_match.to_sdfg()
    
    count = replace_sdfg_dtypes(sdfg, dace.int64, dace.int32)
    
    assert count == 0
    assert sdfg.arrays['a'].dtype == dace.float64
    assert sdfg.arrays['b'].dtype == dace.float64


def test_int_type_change():
    @dace.program
    def int_prog(a: dace.int32[10], b: dace.int32[10]):
        b[:] = a[:] * 2
    
    sdfg = int_prog.to_sdfg()
    
    count = replace_sdfg_dtypes(sdfg, dace.int32, dace.int64)
    
    assert count == 2
    assert sdfg.arrays['a'].dtype == dace.int64
    assert sdfg.arrays['b'].dtype == dace.int64
    
    A = np.arange(10, dtype=np.int64)
    B = np.zeros(10, dtype=np.int64)
    sdfg(a=A, b=B)
    assert np.array_equal(B, A * 2)


if __name__ == '__main__':
    pytest.main([__file__])

