# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pool pre-grow in the experimental CUDA codegen.

When the SDFG uses the memory pool (``cudaMallocAsync`` transients), the experimental codegen
emits, once in ``__dace_init``, a single ``cudaMallocAsync`` + ``cudaFreeAsync`` sized to the
working set so that no per-invocation allocation triggers (synchronous) pool growth. The size is the
sum of the pool transients whose sizes depend only on symbols known at init (global free symbols /
constants); the retain threshold keeps the freed reservation mapped.

Codegen-text only -- no GPU / nvcc required.
"""
import re

import dace
from dace import dtypes

GG = dtypes.StorageType.GPU_Global
GD = dtypes.ScheduleType.GPU_Device


def _experimental_code(sdfg: dace.SDFG) -> str:
    old = dace.Config.get('compiler', 'cuda', 'implementation')
    try:
        dace.Config.set('compiler', 'cuda', 'implementation', value='experimental')
        return '\n'.join(o.code for o in sdfg.generate_code())
    finally:
        dace.Config.set('compiler', 'cuda', 'implementation', value=old)


def _gpu_copy(state, src, dst, rng='0:N'):
    """A GPU_Device map copying ``src[i] -> dst[i]`` over ``rng``."""
    e, x = state.add_map(f'm_{src}_{dst}', dict(i=rng), schedule=GD)
    t = state.add_tasklet(f't_{src}_{dst}', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(state.add_access(src), e, t, dst_conn='a', memlet=dace.Memlet(f'{src}[i]'))
    state.add_memlet_path(t, x, state.add_access(dst), src_conn='b', memlet=dace.Memlet(f'{dst}[i]'))


def _prewarm_bytes_expr(code: str):
    """The byte-size argument of the pre-grow ``cudaMallocAsync``, or ``None`` if not emitted."""
    m = re.search(r'cudaMallocAsync\(&__dace_pool_prewarm,\s*(.+?),\s*__state', code)
    return m.group(1).strip() if m else None


def test_prewarm_emitted_for_symbolic_pool_transient():
    """One pool transient sized by an init symbol -> pre-grow of ``(N) * sizeof(double)``."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('pw_one')
    sdfg.add_array('A', (N, ), dace.float64, storage=GG)
    sdfg.add_array('B', (N, ), dace.float64, storage=GG)
    sdfg.add_transient('tmp', (N, ), dace.float64, storage=GG)
    sdfg.arrays['tmp'].pool = True
    st = sdfg.add_state('m')
    _gpu_copy(st, 'A', 'tmp')
    _gpu_copy(st, 'tmp', 'B')

    expr = _prewarm_bytes_expr(_experimental_code(sdfg))
    assert expr is not None, 'no pre-grow emitted'
    assert expr == '(N) * sizeof(double)', expr


def test_prewarm_sums_multiple_pool_transients():
    """Two pool transients -> the pre-grow size is the sum of both."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('pw_two')
    sdfg.add_array('A', (N, ), dace.float64, storage=GG)
    sdfg.add_array('B', (N, ), dace.float64, storage=GG)
    for nm in ('t0', 't1'):
        sdfg.add_transient(nm, (N, ), dace.float64, storage=GG)
        sdfg.arrays[nm].pool = True
    st = sdfg.add_state('m')
    _gpu_copy(st, 'A', 't0')
    _gpu_copy(st, 't0', 't1')
    _gpu_copy(st, 't1', 'B')

    expr = _prewarm_bytes_expr(_experimental_code(sdfg))
    assert expr is not None
    # Sum of the two identical terms (order not guaranteed; both terms must be present).
    assert expr.count('(N) * sizeof(double)') == 2 and '+' in expr, expr


def test_no_prewarm_without_pool_arrays():
    """No pool transient -> the pool is not used and no pre-grow is emitted."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('pw_none')
    sdfg.add_array('A', (N, ), dace.float64, storage=GG)
    sdfg.add_array('B', (N, ), dace.float64, storage=GG)
    st = sdfg.add_state('m')
    _gpu_copy(st, 'A', 'B')

    assert '__dace_pool_prewarm' not in _experimental_code(sdfg)


def test_prewarm_pins_non_init_symbol_to_one():
    """A pool transient sized by a symbol *defined inside* the SDFG (not known at init) is still
    pre-reserved, with that symbol pinned to 1 in the size expression."""
    N = dace.symbol('N')
    M = dace.symbol('M')
    sdfg = dace.SDFG('pw_pin')
    sdfg.add_array('A', (N, ), dace.float64, storage=GG)
    sdfg.add_array('B', (N, ), dace.float64, storage=GG)
    sdfg.add_transient('tmp', (M, ), dace.float64, storage=GG)  # size uses internally-defined M
    sdfg.arrays['tmp'].pool = True

    s0 = sdfg.add_state('s0', is_start_block=True)
    _gpu_copy(s0, 'A', 'B')
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'M': 'N'}))  # M defined here, not at init
    # Use tmp so it is a real pool allocation (write a constant over 0:M).
    e, x = s1.add_map('mw', dict(i='0:M'), schedule=GD)
    t = s1.add_tasklet('w', set(), {'b'}, 'b = 1.0')
    s1.add_edge(e, None, t, None, dace.Memlet())
    s1.add_memlet_path(t, x, s1.add_access('tmp'), src_conn='b', memlet=dace.Memlet('tmp[i]'))

    expr = _prewarm_bytes_expr(_experimental_code(sdfg))
    assert expr is not None, 'tmp must still be pre-reserved (M pinned to 1)'
    assert 'M' not in expr, f'M should be pinned to 1, not appear in {expr!r}'
    assert expr == '(1) * sizeof(double)', expr


if __name__ == '__main__':
    test_prewarm_emitted_for_symbolic_pool_transient()
    test_prewarm_sums_multiple_pool_transients()
    test_no_prewarm_without_pool_arrays()
    test_prewarm_pins_non_init_symbol_to_one()
    print('ok')
