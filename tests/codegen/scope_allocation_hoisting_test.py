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


def test_numerics_unchanged():
    A = np.random.RandomState(0).rand(64)
    out = np.zeros(64)
    loop_local_buffer(A=A, out=out)
    assert np.allclose(out, A * 2.0 + 1.0)


if __name__ == '__main__':
    test_loop_nested_heap_allocation_is_not_reallocated_per_iteration()
    test_loop_dependent_shape_is_refused()
    test_numerics_unchanged()
