# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A symbolically-sized register array is emitted as a stack variable-length array. """

import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64)


def scratch_program(name: str):

    @dace.program
    def scratch(a: dace.float64[N], b: dace.float64[N]):
        tmp = dace.define_local([N], dace.float64, storage=dace.StorageType.Register)
        for i in dace.map[0:N]:
            tmp[i] = a[i] * 2.0
        for i in dace.map[0:N]:
            b[i] = tmp[i] + 1.0

    sdfg = scratch.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def test_symbolic_register_array_stays_on_the_stack():
    """The declaration carries the symbolic extent, and nothing is newed or deleted for it."""
    sdfg = scratch_program('vla_stack')
    code = sdfg.generate_code()[0].clean_code
    assert 'double tmp[N];' in code, code
    assert 'new double' not in code
    assert 'delete[] tmp' not in code

    a = np.random.rand(32)
    b = np.zeros(32)
    sdfg(a=a, b=b, N=32)
    assert np.allclose(b, a * 2.0 + 1.0)


def test_constant_sized_register_array_is_unchanged():
    """A statically-sized register array keeps its alignment and brace initializer."""
    sdfg = dace.SDFG('vla_constant')
    sdfg.add_array('a', (16, ), dace.float64)
    sdfg.add_array('b', (16, ), dace.float64)
    sdfg.add_transient('tmp', (16, ), dace.float64, storage=dace.StorageType.Register)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('copy', {'i': '0:16'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp + 1.0', {'o': dace.Memlet('tmp[i]')},
                             external_edges=True)
    state2 = sdfg.add_state_after(state, 'out')
    state2.add_mapped_tasklet('back', {'i': '0:16'}, {'inp': dace.Memlet('tmp[i]')},
                              'o = inp', {'o': dace.Memlet('b[i]')},
                              external_edges=True)
    sdfg.validate()

    code = sdfg.generate_code()[0].clean_code
    assert 'DACE_ALIGN(64)' in code
    assert 'new double' not in code

    a = np.random.rand(16)
    b = np.zeros(16)
    sdfg(a=a, b=b)
    assert np.allclose(b, a + 1.0)


def test_global_lifetime_keeps_the_heap():
    """A VLA dies with its block, but a Global array is declared outside it -- codegen emits the
    program-level pointer for it, so the local declaration must not shadow that."""
    sdfg = dace.SDFG('vla_global_lifetime')
    sdfg.add_array('a', (N, ), dace.float64)
    sdfg.add_array('b', (N, ), dace.float64)
    sdfg.add_transient('tmp', (N, ),
                       dace.float64,
                       storage=dace.StorageType.Register,
                       lifetime=dace.AllocationLifetime.Global)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('copy', {'i': '0:N'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp + 1.0', {'o': dace.Memlet('tmp[i]')},
                             external_edges=True)
    state2 = sdfg.add_state_after(state, 'out')
    state2.add_mapped_tasklet('back', {'i': '0:N'}, {'inp': dace.Memlet('tmp[i]')},
                              'o = inp', {'o': dace.Memlet('b[i]')},
                              external_edges=True)
    sdfg.validate()

    code = sdfg.generate_code()[0].clean_code
    assert 'double tmp[N];' not in code, code
    assert 'new' in code

    a = np.random.rand(8)
    b = np.zeros(8)
    sdfg(a=a, b=b, N=8)
    assert np.allclose(b, a + 1.0)


def split_declaration_sdfg():
    """An SDFG whose register transient is sized by a symbol an interstate edge assigns."""
    K = dace.symbol('K', dtype=dace.int64)
    sdfg = dace.SDFG('vla_split_declaration')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_array('a', (N, ), dace.float64)
    sdfg.add_array('b', (N, ), dace.float64)
    sdfg.add_transient('tmp', (K, ), dace.float64, storage=dace.StorageType.Register)
    init = sdfg.add_state('init', is_start_block=True)
    first = sdfg.add_state('first')
    second = sdfg.add_state('second')
    sdfg.add_edge(init, first, dace.InterstateEdge(assignments={'K': 'N'}))
    sdfg.add_edge(first, second, dace.InterstateEdge())
    first.add_mapped_tasklet('copy', {'i': '0:K'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp + 1.0', {'o': dace.Memlet('tmp[i]')},
                             external_edges=True)
    second.add_mapped_tasklet('back', {'i': '0:K'}, {'inp': dace.Memlet('tmp[i]')},
                              'o = inp', {'o': dace.Memlet('b[i]')},
                              external_edges=True)
    sdfg.validate()
    return sdfg


def test_split_declaration_keeps_the_heap():
    """A size that only an interstate edge defines is declared in one block and allocated in
    another, and a VLA cannot bridge that split."""
    sdfg = split_declaration_sdfg()

    code = sdfg.generate_code()[0].clean_code
    assert 'double tmp[K];' not in code, code

    a = np.random.rand(8)
    b = np.zeros(8)
    sdfg(a=a, b=b, N=8)
    assert np.allclose(b, a + 1.0)


if __name__ == '__main__':
    test_symbolic_register_array_stays_on_the_stack()
    test_constant_sized_register_array_is_unchanged()
    test_global_lifetime_keeps_the_heap()
    test_split_declaration_keeps_the_heap()
