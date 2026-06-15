# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime-trip guard tests for :class:`MarkTileDims`.

The K-dim vectorization pipeline assumes ``trip >= W`` on every tiled inner
dim. Statically too-small trips are soft-skipped at compile time (the map
is left for sequential codegen). Symbolic trips are assumed to satisfy
the rule; a ``__builtin_trap`` guard tasklet planted at the SDFG entry
catches a runtime violation.

These tests cover three contracts:

1. Symbolic trip + plain inner dim -> guard state + tasklet is planted.
2. Static trip == W -> no guard needed (the assumption holds).
3. Symbolic trip violated at runtime -> the compiled binary aborts.
"""
import subprocess
import textwrap

import dace
import pytest

from dace.transformation.passes.vectorization.mark_tile_dims import (MarkTileDims, _RUNTIME_GUARD_STATE_LABEL)


def _build_inner_map_sdfg(name: str, trip):
    """Build a single-state SDFG with one innermost map iterating ``i`` over
    ``[0, trip)``. ``trip`` may be a Python int OR a :class:`dace.symbol`."""
    sdfg = dace.SDFG(name)
    if isinstance(trip, int):
        ub = trip - 1
        sdfg.add_array('a', [trip], dtype=dace.float64, transient=False)
    else:
        N = trip
        ub = N - 1
        sdfg.add_symbol(str(N), dace.int64)
        sdfg.add_array('a', [N], dtype=dace.float64, transient=False)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet(
        name='kern',
        map_ranges={'i': dace.subsets.Range([(0, ub, 1)])},
        inputs={},
        code='_out = 0.0',
        outputs={'_out': dace.memlet.Memlet('a[i]')},
        external_edges=True,
    )
    return sdfg


def test_mark_tile_dims_plants_runtime_guard_for_symbolic_trip():
    """Symbolic trip ``N``: a ``_tile_runtime_check`` state with a single
    zero-connector ``__builtin_trap`` tasklet is prepended to the SDFG."""
    N = dace.symbol('N')
    sdfg = _build_inner_map_sdfg('symbolic_trip', N)
    pre_states = list(sdfg.nodes())

    res = MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    assert res is not None, "MarkTileDims should classify the symbolic-trip map"

    guard_states = [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label == _RUNTIME_GUARD_STATE_LABEL]
    assert len(guard_states) == 1, f'expected 1 guard state, got {len(guard_states)}'
    assert guard_states[0] not in pre_states, 'guard state must be newly created'
    assert sdfg.start_block is guard_states[0], 'guard state must be the SDFG start block'

    guards = [n for n in guard_states[0].nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(guards) == 1
    assert '__builtin_trap' in guards[0].code.as_string
    assert 'N' in guards[0].code.as_string
    assert '>= 8' in guards[0].code.as_string


def test_mark_tile_dims_no_guard_for_static_trip_at_or_above_width():
    """Static trip == W: assumption holds at compile time, no guard needed."""
    sdfg = _build_inner_map_sdfg('static_trip_eq_w', 8)
    res = MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    assert res is not None
    guard_states = [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label == _RUNTIME_GUARD_STATE_LABEL]
    assert not guard_states, 'no guard state should be planted when trip >= W is static'


def test_mark_tile_dims_specs_static_trip_below_width_for_masked_tail():
    """Static trip < W: under the unified "no-mask interior + w-mask remainder"
    model the dim is STILL tiled -- it lowers to a single masked remainder tile
    (empty interior + ``mask l < trip``), so :class:`MarkTileDims` records a spec
    for it (it is NOT soft-skipped). A static trip needs NO runtime guard (the
    trip is known at compile time; the mask handles the short dim).

    Regression contract: the prior design soft-skipped ``trip < W`` and returned
    ``None``; the masked-tail model removed that skip (every inner-tiled dim is
    spec'd) -- so a trip-4 / W-8 kernel vectorises correctly end-to-end via the
    masked tail rather than being dropped to sequential codegen."""
    sdfg = _build_inner_map_sdfg('static_trip_below_w', 4)
    res = MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    assert res is not None, "static trip < W must be spec'd (masked-tail), not soft-skipped"
    specs = list(res.values())
    assert len(specs) == 1, f"expected one spec for the single inner map, got {len(specs)}"
    assert specs[0].iter_vars == ('i', ) and specs[0].widths == (8, )
    guard_states = [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label == _RUNTIME_GUARD_STATE_LABEL]
    assert not guard_states, 'a STATIC trip < W needs no runtime guard (mask handles it at compile time)'


def test_mark_tile_dims_runtime_guard_traps_on_violating_symbol():
    """Symbolic trip + run with ``N < W`` -> the planted ``__builtin_trap``
    fires, the subprocess aborts with a non-zero exit status. Subprocess
    isolation prevents the trap from killing the test runner."""
    src = textwrap.dedent('''
        import sys
        sys.path.insert(0, '/home/primrose/Work/dace')
        import numpy as np
        import dace
        from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims

        N = dace.symbol('N')
        sdfg = dace.SDFG('rt_trap')
        sdfg.add_symbol('N', dace.int64)
        sdfg.add_array('a', [N], dtype=dace.float64, transient=False)
        state = sdfg.add_state('main', is_start_block=True)
        state.add_mapped_tasklet(
            name='kern',
            map_ranges={'i': dace.subsets.Range([(0, N - 1, 1)])},
            inputs={},
            code='_out = 0.0',
            outputs={'_out': dace.memlet.Memlet('a[i]')},
            external_edges=True,
        )
        assert MarkTileDims(widths=(8, )).apply_pass(sdfg, {}) is not None
        sdfg(a=np.zeros(4), N=4)  # N=4 < W=8 -> trap
    ''')
    res = subprocess.run(
        ['/home/primrose/.pyenv/versions/py13/bin/python', '-c', src],
        capture_output=True,
        timeout=120,
    )
    assert res.returncode != 0, ('runtime trip guard did not trap on N=4 < W=8 '
                                 f'(stdout={res.stdout!r}, stderr={res.stderr[-400:]!r})')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
