# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LiftMapReductionToReduce`` must not size its staging buffer by an enclosing map's param.

The lift stages per-iteration products in a transient sized ``(trip,)``, added to the state's SDFG.
A descriptor may only be sized by symbols in scope where it LIVES, and a map param is not -- it is
bound per iteration, inside the map. A triangular inner reduction (polybench trmm's
``for k in range(i + 1, M)`` under a ``for i`` map) would allocate ``_red_buf[M - i - 1]``; once the
owning nested SDFG is inlined, ``i`` becomes a free symbol of the whole program, lands in the
compiled signature, and the call fails with ``Missing program argument "i"``.

The enclosing map usually sits one nested-SDFG boundary OUT, feeding its param in under
``symbol_mapping`` -- so the guard has to follow the scope tree across that boundary, which is what
these tests pin.
"""
import copy

import dace
import pytest
from dace.transformation.passes.vectorization.lift_map_reduction import _trip_depends_on_enclosing_map

M = 32


def _nested_reduction(inner_lower: str):
    """``for i: <nsdfg>{ for k in [<inner_lower>, M): acc += A[k] }`` -- the reduction map is
    top-level inside the body nest, and the enclosing ``i`` map is one boundary out."""
    sdfg = dace.SDFG('outer')
    sdfg.add_array('A', [M], dace.float64)
    sdfg.add_array('out', [M], dace.float64)
    st = sdfg.add_state('main', is_start_block=True)

    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    inner.add_array('A', [M], dace.float64)
    inner.add_array('acc', [1], dace.float64)
    ist = inner.add_state('reduce', is_start_block=True)
    me, mx = ist.add_map('red', dict(k=f'{inner_lower}:M'))
    t = ist.add_tasklet('prod', {'a'}, {'o'}, 'o = a')
    ist.add_memlet_path(ist.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[k]'))
    ist.add_memlet_path(t,
                        mx,
                        ist.add_access('acc'),
                        src_conn='o',
                        memlet=dace.Memlet('acc[0]', wcr='lambda x, y: x + y'))

    ome, omx = st.add_map('col', dict(i=f'0:{M}'))
    ns = st.add_nested_sdfg(inner, {'A'}, {'acc'}, symbol_mapping={'i': 'i', 'M': str(M)})
    st.add_memlet_path(st.add_access('A'), ome, ns, dst_conn='A', memlet=dace.Memlet(f'A[0:{M}]'))
    st.add_memlet_path(ns, omx, st.add_access('out'), src_conn='acc', memlet=dace.Memlet('out[i]'))
    return ist, me


def test_trip_sized_by_an_outer_map_param_is_refused():
    """``k in [i + 1, M)``: the trip ``M - i - 1`` carries the outer map's ``i``."""
    state, map_entry = _nested_reduction('i + 1')
    assert _trip_depends_on_enclosing_map(state, map_entry, dace.symbolic.pystr_to_symbolic('M - i - 1')) is True


def test_globally_sized_trip_is_allowed():
    """``k in [0, M)``: the trip is a program symbol, so the buffer is legally sized."""
    state, map_entry = _nested_reduction('0')
    assert _trip_depends_on_enclosing_map(state, map_entry, dace.symbolic.pystr_to_symbolic('M')) is False


def test_trmm_has_no_map_param_in_its_signature():
    """End-to-end: trmm compiles and is correct under both vectorizing pipelines.

    Skipped if the corpus harness is unavailable in this environment.
    """
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    for config in ('canon+vec', 'parallelize+vec'):
        base, checker = mp.CORPORA['poly'][1]('trmm')
        sd = copy.deepcopy(base)
        mp.apply_config(sd, config, mp.cpu_params(4))
        assert 'i' not in {str(s) for s in sd.free_symbols}, 'a map param must not escape into the signature'
        fin = finalize_for_target(copy.deepcopy(sd), 'cpu')
        fin.name = 'trmm_' + config.replace('+', '_')
        assert bool(checker(fin)), f'trmm must be correct under {config}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
