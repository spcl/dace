# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for make_explicit() — converting transient arrays to AllocationLifetime.Explicit
with alloc/free annotations on interstate edges.

Test coverage:
  - Single-state array: alloc on incoming edge, free on outgoing edge
  - Multi-state array (SDFG-scope): alloc on incoming edge of first-use state,
    free on outgoing edge of last-use state
  - Start-state array (no incoming edges): thin predecessor state inserted
  - Sink-state array (no outgoing edges): thin successor state inserted
  - Multiple arrays sharing the same first/last states batch onto the same edges
  - Lifetime is set to Explicit
  - Error cases: non-existent array, non-transient array
  - Code generation: new[]/delete[] appear on the correct edges in the output C++
  - Validation: SDFG validates after make_explicit
"""

import re

import pytest
import dace
from dace import dtypes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.libraries.allocation import make_explicit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_sdfg(array_name='buf', shape=(10,), dtype=dace.float64):
    """Three-state SDFG:  init --[e0]--> use --[e1]--> done
    The 'use' state contains one access node for *array_name*.
    """
    sdfg = dace.SDFG('simple')
    sdfg.add_array(array_name, shape, dtype, transient=True)

    init = sdfg.add_state('init')
    use  = sdfg.add_state('use')
    done = sdfg.add_state('done')

    e0 = sdfg.add_edge(init, use,  InterstateEdge())
    e1 = sdfg.add_edge(use,  done, InterstateEdge())

    # Minimal tasklet that writes into buf so the access node is present
    t = use.add_tasklet('fill', {}, {'out'}, 'out = 1.0')
    use.add_edge(t, 'out', use.add_write(array_name), None,
                 dace.Memlet(f'{array_name}[0]'))
    return sdfg, init, use, done, e0, e1


def _get_cpp(sdfg):
    """Return the generated CPU C++ as a string."""
    codes = sdfg.generate_code()
    return codes[0].clean_code


# ---------------------------------------------------------------------------
# Lifetime tests
# ---------------------------------------------------------------------------

def test_lifetime_set_to_explicit():
    sdfg, *_ = _make_simple_sdfg()
    make_explicit(sdfg, ['buf'])
    assert sdfg.arrays['buf'].lifetime is dtypes.AllocationLifetime.Explicit


def test_error_nonexistent_array():
    sdfg, *_ = _make_simple_sdfg()
    with pytest.raises(ValueError, match='not found'):
        make_explicit(sdfg, ['does_not_exist'])


def test_error_non_transient():
    sdfg, *_ = _make_simple_sdfg()
    sdfg.arrays['buf'].transient = False
    with pytest.raises(ValueError, match='not a transient'):
        make_explicit(sdfg, ['buf'])


# ---------------------------------------------------------------------------
# Edge annotation tests (single-state array)
# ---------------------------------------------------------------------------

def test_alloc_on_incoming_edge():
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    make_explicit(sdfg, ['buf'])
    # 'buf' is first (and only) used in 'use'; its alloc must be on e0 (init→use)
    assert 'buf' in e0.data.alloc
    assert 'buf' not in e1.data.alloc


def test_free_on_outgoing_edge():
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    make_explicit(sdfg, ['buf'])
    # 'buf' is last used in 'use'; its free must be on e1 (use→done)
    assert 'buf' in e1.data.free
    assert 'buf' not in e0.data.free


def test_no_duplicate_on_double_call():
    """Calling make_explicit twice must not add the name twice to the same edge."""
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    make_explicit(sdfg, ['buf'])
    make_explicit(sdfg, ['buf'])
    assert e0.data.alloc.count('buf') == 1
    assert e1.data.free.count('buf') == 1


# ---------------------------------------------------------------------------
# Edge annotation tests (multi-state array)
# ---------------------------------------------------------------------------

def test_multistate_alloc_on_first_use_edge():
    """Array used in both 'use1' and 'use2':  init→use1→use2→done
    Alloc must be on init→use1, free on use2→done.
    """
    sdfg = dace.SDFG('multi')
    sdfg.add_array('buf', (10,), dace.float64, transient=True)

    init = sdfg.add_state('init')
    use1 = sdfg.add_state('use1')
    use2 = sdfg.add_state('use2')
    done = sdfg.add_state('done')

    e0 = sdfg.add_edge(init, use1, InterstateEdge())
    e1 = sdfg.add_edge(use1, use2, InterstateEdge())
    e2 = sdfg.add_edge(use2, done, InterstateEdge())

    for st in (use1, use2):
        t = st.add_tasklet('t', {}, {'out'}, 'out = 0.0')
        st.add_edge(t, 'out', st.add_write('buf'), None, dace.Memlet('buf[0]'))

    make_explicit(sdfg, ['buf'])

    assert 'buf' in e0.data.alloc   # before first use
    assert 'buf' not in e1.data.alloc
    assert 'buf' in e2.data.free    # after last use
    assert 'buf' not in e1.data.free


# ---------------------------------------------------------------------------
# Start-state / sink-state edge cases
# ---------------------------------------------------------------------------

def test_start_state_inserts_predecessor():
    """Array first used in the SDFG start state (no incoming edges).
    make_explicit must insert a thin predecessor state.
    """
    sdfg = dace.SDFG('startstate')
    sdfg.add_array('buf', (5,), dace.float32, transient=True)

    start = sdfg.add_state('start', is_start_block=True)
    end   = sdfg.add_state('end')
    sdfg.add_edge(start, end, InterstateEdge())

    t = start.add_tasklet('t', {}, {'out'}, 'out = 0.0')
    start.add_edge(t, 'out', start.add_write('buf'), None, dace.Memlet('buf[0]'))

    make_explicit(sdfg, ['buf'])

    # A new predecessor state must have been inserted before 'start'
    preds = sdfg.in_edges(start)
    assert len(preds) == 1
    alloc_edge = preds[0]
    assert 'buf' in alloc_edge.data.alloc


def test_sink_state_inserts_successor():
    """Array last used in a sink state (no outgoing edges).
    make_explicit must insert a thin successor state.
    """
    sdfg = dace.SDFG('sinkstate')
    sdfg.add_array('buf', (5,), dace.float32, transient=True)

    init = sdfg.add_state('init', is_start_block=True)
    sink = sdfg.add_state('sink')
    sdfg.add_edge(init, sink, InterstateEdge())

    t = sink.add_tasklet('t', {}, {'out'}, 'out = 0.0')
    sink.add_edge(t, 'out', sink.add_write('buf'), None, dace.Memlet('buf[0]'))

    make_explicit(sdfg, ['buf'])

    succs = sdfg.out_edges(sink)
    assert len(succs) == 1
    free_edge = succs[0]
    assert 'buf' in free_edge.data.free


# ---------------------------------------------------------------------------
# Loop-scoped arrays: semantics preservation
# ---------------------------------------------------------------------------

def _make_loop_sdfg(lifetime=dtypes.AllocationLifetime.Scope):
    """SDFG with a single LoopRegion; 'buf' is used only inside the loop."""
    sdfg = dace.SDFG('loop_scoped')
    sdfg.add_array('buf', (10,), dace.float32, transient=True, lifetime=lifetime)
    sdfg.add_array('out', (10,), dace.float32, transient=False)

    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)

    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('compute', {}, {'a'}, 'a = 42.0')
    body.add_edge(t, 'a', body.add_access('buf'), None, dace.Memlet('buf[i]'))
    body.add_edge(body.add_access('buf'), None, body.add_access('out'), None,
                  dace.Memlet('out[i]'))

    exit_state = sdfg.add_state('exit')
    sdfg.add_edge(loop, exit_state, dace.InterstateEdge())
    return sdfg, loop


def test_loop_scope_alloc_inside_loop():
    """Scope-lifetime array used only in a loop: alloc/free must go on edges
    *inside* the LoopRegion, not on the SDFG-level edges surrounding the loop.
    """
    sdfg, loop = _make_loop_sdfg(dtypes.AllocationLifetime.Scope)
    make_explicit(sdfg, ['buf'])

    # No alloc/free on the SDFG-level edges
    for edge in sdfg.edges():
        assert 'buf' not in edge.data.alloc, "alloc must not appear on SDFG-level edge"
        assert 'buf' not in edge.data.free,  "free must not appear on SDFG-level edge"

    # alloc/free must be on edges inside the LoopRegion
    loop_alloc = any('buf' in e.data.alloc for e in loop.edges())
    loop_free  = any('buf' in e.data.free  for e in loop.edges())
    assert loop_alloc, "alloc must be on an edge inside the LoopRegion"
    assert loop_free,  "free must be on an edge inside the LoopRegion"


def test_loop_sdfg_lifetime_alloc_outside_loop():
    """SDFG-lifetime array used only in a loop: alloc/free must be placed at
    the SDFG level (hoisted), since the declared scope is the whole SDFG.
    """
    sdfg, loop = _make_loop_sdfg(dtypes.AllocationLifetime.SDFG)
    make_explicit(sdfg, ['buf'])

    sdfg_alloc = any('buf' in e.data.alloc for e in sdfg.edges())
    sdfg_free  = any('buf' in e.data.free  for e in sdfg.edges())
    assert sdfg_alloc, "SDFG-lifetime alloc must be on a SDFG-level edge"
    assert sdfg_free,  "SDFG-lifetime free must be on a SDFG-level edge"


def test_loop_scope_validates():
    sdfg, _ = _make_loop_sdfg(dtypes.AllocationLifetime.Scope)
    make_explicit(sdfg, ['buf'])
    sdfg.validate()


# ---------------------------------------------------------------------------
# Multiple arrays batch onto the same edges
# ---------------------------------------------------------------------------

def test_multiple_arrays_same_edges():
    """Two arrays with the same first/last state end up on the same edges."""
    sdfg = dace.SDFG('multi_arr')
    sdfg.add_array('a', (4,), dace.float64, transient=True)
    sdfg.add_array('b', (4,), dace.float64, transient=True)

    init = sdfg.add_state('init')
    use  = sdfg.add_state('use')
    done = sdfg.add_state('done')
    e0 = sdfg.add_edge(init, use,  InterstateEdge())
    e1 = sdfg.add_edge(use,  done, InterstateEdge())

    for arr in ('a', 'b'):
        t = use.add_tasklet(f't_{arr}', {}, {'out'}, 'out = 0.0')
        use.add_edge(t, 'out', use.add_write(arr), None, dace.Memlet(f'{arr}[0]'))

    make_explicit(sdfg, ['a', 'b'])

    assert 'a' in e0.data.alloc and 'b' in e0.data.alloc
    assert 'a' in e1.data.free  and 'b' in e1.data.free


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def test_codegen_new_in_edge_code():
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg('buf', (10,), dace.float64)
    make_explicit(sdfg, ['buf'])
    cpp = _get_cpp(sdfg)
    assert 'new double' in cpp, "Expected heap allocation in generated C++"


def test_codegen_delete_in_edge_code():
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg('buf', (10,), dace.float64)
    make_explicit(sdfg, ['buf'])
    cpp = _get_cpp(sdfg)
    assert 'delete[]' in cpp, "Expected deallocation in generated C++"


def test_codegen_no_auto_alloc():
    """With Explicit lifetime, the normal auto new[] at function scope must not appear."""
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg('buf', (10,), dace.float64)
    make_explicit(sdfg, ['buf'])
    cpp = _get_cpp(sdfg)
    # The auto-alloc path would declare `buf` as a local variable and assign it
    # with `new` at function scope. With Explicit lifetime the pointer lives in
    # the state struct, so no function-local `double *buf = new ...` should appear.
    assert 'double *buf' not in cpp, "Auto-alloc declaration should not appear"


def test_codegen_state_struct_declaration():
    """Array pointer must be declared inside the state struct."""
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg('buf', (10,), dace.float64)
    make_explicit(sdfg, ['buf'])
    cpp = _get_cpp(sdfg)
    # State struct contains __<cfg_id>_buf
    assert '__0_buf' in cpp or f'__{sdfg.cfg_id}_buf' in cpp


def test_codegen_symbolic_shape():
    """Symbolic shape N must be emitted as the symbol name, not a literal."""
    sdfg = dace.SDFG('sym_shape')
    N = dace.symbol('N')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('buf', (N,), dace.float64, transient=True)

    init = sdfg.add_state('init')
    use  = sdfg.add_state('use')
    done = sdfg.add_state('done')
    sdfg.add_edge(init, use,  InterstateEdge())
    sdfg.add_edge(use,  done, InterstateEdge())

    t = use.add_tasklet('t', {}, {'out'}, 'out = 1.0')
    use.add_edge(t, 'out', use.add_write('buf'), None, dace.Memlet('buf[0]'))

    make_explicit(sdfg, ['buf'])
    cpp = _get_cpp(sdfg)
    assert 'new double' in cpp
    assert '[N]' in cpp or 'N]' in cpp, "Symbolic size N must appear in allocation"


def test_codegen_alloc_before_free_ordering():
    """new[] must appear before delete[] in the generated output."""
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg('buf', (10,), dace.float64)
    make_explicit(sdfg, ['buf'])
    cpp = _get_cpp(sdfg)
    alloc_pos = cpp.find('new double')
    free_pos  = cpp.find('delete[]')
    assert alloc_pos != -1 and free_pos != -1
    assert alloc_pos < free_pos, "new[] must precede delete[]"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_sdfg_validates_after_make_explicit():
    sdfg, *_ = _make_simple_sdfg()
    make_explicit(sdfg, ['buf'])
    sdfg.validate()  # must not raise


def test_validation_rejects_non_explicit_on_alloc_edge():
    """Manually annotating an edge with a non-Explicit array must fail validation."""
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    # Do NOT call make_explicit — lifetime stays Scope
    e0.data.alloc.append('buf')
    with pytest.raises(Exception):
        sdfg.validate()


def test_validation_rejects_non_explicit_on_free_edge():
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    e1.data.free.append('buf')
    with pytest.raises(Exception):
        sdfg.validate()


def test_validation_rejects_nonexistent_array_on_edge():
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit
    e0.data.alloc.append('DOES_NOT_EXIST')
    with pytest.raises(Exception):
        sdfg.validate()


# ---------------------------------------------------------------------------
# Size-1 array limitation
# ---------------------------------------------------------------------------

def test_error_size_1_array():
    """Arrays with exactly one element must raise ValueError."""
    sdfg = dace.SDFG('size1')
    sdfg.add_array('scalar', (1,), dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    use  = sdfg.add_state('use')
    done = sdfg.add_state('done')
    sdfg.add_edge(init, use,  InterstateEdge())
    sdfg.add_edge(use,  done, InterstateEdge())
    t = use.add_tasklet('t', {}, {'out'}, 'out = 1.0')
    use.add_edge(t, 'out', use.add_write('scalar'), None, dace.Memlet('scalar[0]'))
    with pytest.raises(ValueError, match='total_size=1'):
        make_explicit(sdfg, ['scalar'])


def test_unused_array_lifetime_set_no_edges_annotated():
    """Array with no access nodes anywhere: lifetime becomes Explicit but no
    edges are annotated (the all_states list is empty, so we continue)."""
    sdfg, init, use, done, e0, e1 = _make_simple_sdfg()
    sdfg.add_array('unused', (10,), dace.float64, transient=True)
    make_explicit(sdfg, ['unused'])
    assert sdfg.arrays['unused'].lifetime is dtypes.AllocationLifetime.Explicit
    all_allocs = [n for e in sdfg.edges() for n in e.data.alloc]
    all_frees  = [n for e in sdfg.edges() for n in e.data.free]
    assert 'unused' not in all_allocs
    assert 'unused' not in all_frees


if __name__ == '__main__':
    print("Running basic make_explicit tests...")
    test_lifetime_set_to_explicit()
    test_alloc_on_incoming_edge()
    test_free_on_outgoing_edge()
    test_codegen_new_in_edge_code()
    test_codegen_delete_in_edge_code()
    test_codegen_no_auto_alloc()
    test_sdfg_validates_after_make_explicit()
    print("Basic tests PASSED")
