# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests promotion of program-wide-sized transients to Persistent allocation. """
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.persistent_transients import MakeTransientsPersistent

N = dace.symbol('N')


def loop_with_transient(name: str, shape, symbols=(), lifetime=dace.AllocationLifetime.Scope) -> dace.SDFG:
    """``for i in 0:N: buf[0:4] = a[i]; out[i] = buf[0]`` with ``buf`` shaped by ``shape``.

    The buffer is written and read inside the loop body, so a Scope allocation is a malloc and a
    free on every iteration -- the case the promotion exists for.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    for sym in symbols:
        sdfg.add_symbol(sym, dace.int64)
    sdfg.add_transient('buf', shape, dace.float64, storage=dace.StorageType.CPU_Heap, lifetime=lifetime)

    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    writer = body.add_tasklet('w', {'x'}, {'y'}, 'y = x * 2.0')
    body.add_edge(body.add_read('a'), None, writer, 'x', dace.Memlet('a[i]'))
    access = body.add_access('buf')
    body.add_edge(writer, 'y', access, None, dace.Memlet('buf[0]'))
    reader = body.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    body.add_edge(access, None, reader, 'x', dace.Memlet('buf[0]'))
    body.add_edge(reader, 'y', body.add_write('out'), None, dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def allocates_in_init(sdfg: dace.SDFG, name: str) -> bool:
    """Whether ``name``'s allocation sits in the generated ``__dace_init_<name>`` function."""
    code = sdfg.generate_code()[0].clean_code
    head = code.index(f'__dace_init_{sdfg.name}(')
    init = code[head:code.index(f'__dace_exit_{sdfg.name}(', head)]
    return any(f'{name} = new ' in line for line in init.splitlines())


def test_free_symbol_sized_transient_is_promoted():
    """The shape reads only N, an argument of the SDFG, so one allocation serves the whole run."""
    sdfg = loop_with_transient('persist_free_symbol', [N])
    assert not allocates_in_init(sdfg, 'buf'), 'the malloc must start out inside the loop'

    assert MakeTransientsPersistent().apply_pass(sdfg, {})[sdfg.cfg_id] == {'buf'}
    assert sdfg.arrays['buf'].lifetime == dace.AllocationLifetime.Persistent
    assert allocates_in_init(sdfg, 'buf')

    a = np.arange(8.0)
    out = np.zeros(8)
    sdfg(a=a, out=out, N=8)
    assert np.allclose(out, a * 2.0)


def test_the_auto_optimize_helper_delegates():
    """``make_transients_persistent`` constructs the pass with a keyword, which a Property-only class
    does not accept -- so this is the only caller shape that exercises the explicit ``__init__``."""
    from dace.transformation.auto.auto_optimize import make_transients_persistent

    sdfg = loop_with_transient('persist_helper', [N])
    assert make_transients_persistent(sdfg, dace.DeviceType.CPU)[sdfg.cfg_id] == {'buf'}
    assert sdfg.arrays['buf'].lifetime == dace.AllocationLifetime.Persistent


def test_constant_sized_transient_is_promoted():
    """A compile-time constant size qualifies trivially -- it reads no symbol at all."""
    sdfg = loop_with_transient('persist_constant', [4])
    assert MakeTransientsPersistent().apply_pass(sdfg, {})[sdfg.cfg_id] == {'buf'}
    assert sdfg.arrays['buf'].lifetime == dace.AllocationLifetime.Persistent


def test_reassigned_size_symbol_is_not_promoted():
    """``M`` is free at the top level, but the loop's back edge reassigns it every iteration, so the
    init-time size would be whatever it held on entry."""
    sdfg = loop_with_transient('persist_reassigned', ['M'], symbols=('M', ))
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    loop.add_edge(loop.nodes()[0], loop.add_state('bump'), dace.InterstateEdge(assignments={'M': 'M + 1'}))

    assert MakeTransientsPersistent().apply_pass(sdfg, {}) is None
    assert sdfg.arrays['buf'].lifetime == dace.AllocationLifetime.Scope
    assert not allocates_in_init(sdfg, 'buf')


def test_loop_variable_sized_nested_transient_is_not_promoted():
    """The nested SDFG's ``K`` is free THERE, but symbol_mapping binds it to the enclosing loop
    variable, so the size changes on every execution of the allocation site."""
    sdfg = dace.SDFG('persist_nested_loop_var')
    sdfg.add_array('out', [N], dace.float64)

    nest = dace.SDFG('nest')
    nest.add_symbol('K', dace.int64)
    nest.add_array('o', [1], dace.float64)
    nest.add_transient('buf', ['K'], dace.float64, storage=dace.StorageType.CPU_Heap)
    nstate = nest.add_state('n', is_start_block=True)
    writer = nstate.add_tasklet('w', {}, {'y'}, 'y = 1.0')
    access = nstate.add_access('buf')
    nstate.add_edge(writer, 'y', access, None, dace.Memlet('buf[0]'))
    reader = nstate.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    nstate.add_edge(access, None, reader, 'x', dace.Memlet('buf[0]'))
    nstate.add_edge(reader, 'y', nstate.add_write('o'), None, dace.Memlet('o[0]'))

    loop = LoopRegion('loop', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    node = body.add_nested_sdfg(nest, inputs=set(), outputs={'o'}, symbol_mapping={'K': 'i'})
    body.add_edge(node, 'o', body.add_write('out'), None, dace.Memlet('out[i]'))
    sdfg.validate()

    assert MakeTransientsPersistent().apply_pass(sdfg, {}) is None
    assert nest.arrays['buf'].lifetime == dace.AllocationLifetime.Scope


def test_stronger_lifetimes_are_not_demoted():
    for lifetime in (dace.AllocationLifetime.Global, dace.AllocationLifetime.External):
        sdfg = loop_with_transient(f'persist_keep_{lifetime.name}', [N], lifetime=lifetime)
        assert MakeTransientsPersistent().apply_pass(sdfg, {}) is None
        assert sdfg.arrays['buf'].lifetime == lifetime


if __name__ == '__main__':
    test_free_symbol_sized_transient_is_promoted()
    test_the_auto_optimize_helper_delegates()
    test_constant_sized_transient_is_promoted()
    test_reassigned_size_symbol_is_not_promoted()
    test_loop_variable_sized_nested_transient_is_not_promoted()
    test_stronger_lifetimes_are_not_demoted()
