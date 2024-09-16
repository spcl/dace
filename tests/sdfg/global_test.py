import ctypes
import dace
import numpy as np


def test_global_transient():
    sdfg = dace.SDFG('tester')
    sdfg.add_scalar('myglobal', dace.int32, lifetime=dace.AllocationLifetime.Global,
                    transient=True)
    sdfg.add_array('out', [1], dace.int32)
    state = sdfg.add_state()
    t = state.add_tasklet('save', {'inp'}, {'outp'}, 'outp = inp')
    r = state.add_access('out')
    w = state.add_access('myglobal')
    state.add_edge(r, None, t, 'inp', dace.Memlet('out[0]'))
    state.add_edge(t, 'outp', w, None, dace.Memlet('myglobal[0]'))
    
    out = np.random.randint(1, 5, size=(1,)).astype(np.int32)
    csdfg = sdfg.compile()
    csdfg(out=out)

    # Get global
    sym = csdfg._lib.get_raw_symbol('myglobal')
    ptr = ctypes.cast(sym, ctypes.POINTER(ctypes.c_int))
    assert ptr.contents == out[0]


def test_global_global():
    sdfg = dace.SDFG('tester')
    sdfg.add_scalar('errno', dace.int32, lifetime=dace.AllocationLifetime.Global)
    sdfg.add_array('out', [1], dace.int32)
    state = sdfg.add_state()
    t = state.add_tasklet('save', {'inp'}, {'outp'}, 'outp = inp')
    r = state.add_access('errno')
    w = state.add_access('out')
    state.add_edge(r, None, t, 'inp', dace.Memlet('errno[0]'))
    state.add_edge(t, 'outp', w, None, dace.Memlet('out[0]'))
    
    out = np.random.randint(1, 5, size=(1,)).astype(np.int32)
    sdfg(out=out)
    assert np.allclose(out, 0)


if __name__ == '__main__':
    test_global_transient()
    test_global_global()
