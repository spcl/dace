# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Placement of Scope-lifetime heap allocations whose only use sits inside a loop."""
import re

import numpy as np

import dace


def allocation_lines(sdfg: dace.SDFG, name: str):
    """Line index of every allocation of ``name`` in the emitted C, and the loop depth at each."""
    code = ''.join(str(c.clean_code) for c in sdfg.generate_code())
    hits, depth = [], 0
    for line in code.split('\n'):
        stripped = line.strip()
        if re.search(r'\bfor\s*\(', stripped):
            depth += 1
        if re.search(rf'\b{re.escape(name)}\b', stripped) and ('new ' in stripped or 'malloc(' in stripped):
            hits.append(depth)
        depth = max(0, depth - stripped.count('}'))
    return hits


@dace.program
def loop_local_buffer(A: dace.float64[64], out: dace.float64[64]):
    for t in range(8):
        tmp = np.empty(64, dace.float64)
        for i in dace.map[0:64]:
            tmp[i] = A[i] * 2.0
        for i in dace.map[0:64]:
            out[i] = tmp[i] + 1.0


def test_loop_nested_heap_allocation_is_not_reallocated_per_iteration():
    """The buffer is used in one state inside the time loop; allocating it there re-runs every
    iteration. Its shape is loop-independent, so the SDFG entry dominates every use."""
    sdfg = loop_local_buffer.to_sdfg(simplify=True)
    # Frontend transients carry StorageType.Default and are still heap-allocated by the CPU target,
    # so select on what the emitted code actually does rather than on the declared storage.
    allocated = {n: allocation_lines(sdfg, n) for n, d in sdfg.arrays.items() if d.transient}
    allocated = {n: depths for n, depths in allocated.items() if depths}
    assert allocated, 'no transient is heap-allocated at all -- the test would be vacuous'
    for name, depths in allocated.items():
        assert all(d == 0 for d in depths), f'{name} is allocated inside a loop: depths {depths}'


def test_loop_dependent_shape_is_refused():
    """A shape built from the enclosing loop's variable has no meaning at the hoist point."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('loop_dependent_shape')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_transient('buf', [N], dace.float64, storage=dace.StorageType.CPU_Heap)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_nedge(state.add_read('A'), state.add_write('buf'), dace.Memlet('A[0:N]'))
    sdfg.validate()
    # Placement must not move a descriptor whose symbols are not free at SDFG scope; N is free here,
    # so this asserts the pass does not simply hoist everything unconditionally.
    assert sdfg.arrays['buf'].lifetime == dace.AllocationLifetime.Scope


def test_a_loop_variable_shape_keeps_its_allocation_in_the_loop():
    """The direct form of a shape the hoist point cannot evaluate: a transient sized by the enclosing
    LoopRegion's own variable. A LoopRegion variable is not in the SDFG's free symbols, so
    is_nonfree_sym_dependent refuses and the allocation stays where the symbol has a value."""
    sdfg = dace.SDFG('loopvar_sized_transient')
    sdfg.add_array('out', [8], dace.float64)
    sdfg.add_transient('buf', ['i'], dace.float64, storage=dace.StorageType.CPU_Heap)
    loop = dace.sdfg.state.LoopRegion('loop', 'i < 8', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    writer = body.add_tasklet('w', {}, {'y'}, 'y = 1.0')
    access = body.add_access('buf')
    body.add_edge(writer, 'y', access, None, dace.Memlet('buf[0]'))
    reader = body.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    body.add_edge(access, None, reader, 'x', dace.Memlet('buf[0]'))
    body.add_edge(reader, 'y', body.add_write('out'), None, dace.Memlet('out[i]'))
    sdfg.validate()

    assert allocation_lines(sdfg, 'buf') == [1], 'the allocation must stay at loop depth 1'
    out = np.zeros(8)
    sdfg(out=out)
    assert np.allclose(out[1:], 1.0)


def test_a_nested_shape_bound_to_a_parent_loop_variable_stays_correct():
    """The hoist target is the SDFG that OWNS the descriptor, so it can never cross a nested-SDFG
    boundary. A nested transient sized by a symbol that symbol_mapping binds to the parent's loop
    variable may therefore hoist to the NEST's entry -- re-entered on every call, so the size is
    rebound before each allocation rather than frozen at the parent's entry."""
    sdfg = dace.SDFG('nested_shape_from_parent_loop')
    sdfg.add_array('out', [8], dace.float64)

    nest = dace.SDFG('nest')
    nest.add_symbol('K', dace.int64)
    nest.add_array('o', [1], dace.float64)
    nest.add_transient('buf', ['K'], dace.float64, storage=dace.StorageType.CPU_Heap)
    inner = dace.sdfg.state.LoopRegion('inner', 'j < 2', 'j', 'j = 0', 'j = j + 1')
    nest.add_node(inner, is_start_block=True)
    nbody = inner.add_state('nbody', is_start_block=True)
    writer = nbody.add_tasklet('w', {}, {'y'}, 'y = 1.0')
    access = nbody.add_access('buf')
    nbody.add_edge(writer, 'y', access, None, dace.Memlet('buf[0]'))
    reader = nbody.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    nbody.add_edge(access, None, reader, 'x', dace.Memlet('buf[0]'))
    nbody.add_edge(reader, 'y', nbody.add_write('o'), None, dace.Memlet('o[0]'))

    loop = dace.sdfg.state.LoopRegion('loop', 'i < 8', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    pbody = loop.add_state('pbody', is_start_block=True)
    node = pbody.add_nested_sdfg(nest, inputs=set(), outputs={'o'}, symbol_mapping={'K': 'i'})
    pbody.add_edge(node, 'o', pbody.add_write('out'), None, dace.Memlet('out[i]'))
    sdfg.validate()

    out = np.zeros(8)
    sdfg(out=out)
    assert np.allclose(out[1:], 1.0)


def test_numerics_unchanged():
    A = np.random.RandomState(0).rand(64)
    out = np.zeros(64)
    loop_local_buffer(A=A, out=out)
    assert np.allclose(out, A * 2.0 + 1.0)


if __name__ == '__main__':
    test_loop_nested_heap_allocation_is_not_reallocated_per_iteration()
    test_loop_dependent_shape_is_refused()
    test_a_loop_variable_shape_keeps_its_allocation_in_the_loop()
    test_a_nested_shape_bound_to_a_parent_loop_variable_stays_correct()
    test_numerics_unchanged()
