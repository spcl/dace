# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for hoist_alloc_out_of_loop() — lifting explicit allocations out of
LoopRegions so that memory is allocated once before the loop instead of on every
iteration.

Test coverage:
  - Basic hoist: alloc/free move from inside loop to parent edges
  - Loop is SDFG start (no incoming edges): thin predecessor inserted
  - Loop is SDFG sink (no outgoing edges): thin successor inserted
  - Multiple arrays hoisted in one call
  - Nested loop: alloc on inner-loop edge hoisted to outer parent CFG
  - TypeError when a non-LoopRegion is passed (Map, plain state, etc.)
  - ValueError when array is not found in the SDFG
  - ValueError when array is not transient
  - ValueError when array has no alloc annotation inside the loop
  - Alloc without matching free inside: only alloc is moved, no free added
  - SDFG validates after hoist
  - Code generation: new[] before the loop, delete[] after the loop
"""

import re

import pytest
import dace
from dace import dtypes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.libraries.allocation import make_explicit, hoist_alloc_out_of_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop_sdfg(array_name='buf', shape=(10,), dtype=dace.float64):
    """Build an SDFG of the form:

        init --[e_before]--> [LoopRegion] --[e_after]--> done

    Inside the LoopRegion:
        body_start --[e_body]--> body_end
           (body_end loops back via the region's implicit back-edge)

    The body_start state contains a write access node for *array_name*.
    Returns (sdfg, loop, e_before, e_after, e_body).
    """
    sdfg = dace.SDFG('loop_hoist_test')
    sdfg.add_array(array_name, shape, dtype, transient=True)

    init = sdfg.add_state('init')
    loop = LoopRegion('myloop',
                      condition_expr='i < 10',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    done = sdfg.add_state('done')

    e_before = sdfg.add_edge(init, loop, InterstateEdge())
    e_after  = sdfg.add_edge(loop, done, InterstateEdge())

    # Loop body: two states with an edge inside the loop
    body_start = loop.add_state('body_start', is_start_block=True)
    body_end   = loop.add_state('body_end')
    e_body     = loop.add_edge(body_start, body_end, InterstateEdge())

    # Access node so the array is "used" inside the loop
    t = body_start.add_tasklet('fill', {}, {'out'}, 'out = 1.0')
    body_start.add_edge(t, 'out', body_start.add_write(array_name), None,
                        dace.Memlet(f'{array_name}[0]'))

    return sdfg, loop, e_before, e_after, e_body


def _annotate_inside(loop, e_body, name):
    """Manually place alloc and free annotations on the body edge and set
    the array's lifetime to Explicit so hoist does not call make_explicit."""
    node = loop
    while not isinstance(node, dace.SDFG):
        node = node.parent_graph
    node.arrays[name].lifetime = dtypes.AllocationLifetime.Explicit
    e_body.data.alloc.append(name)
    e_body.data.free.append(name)


def _get_cpp(sdfg):
    """Return the generated CPU C++ as a string."""
    codes = sdfg.generate_code()
    return codes[0].clean_code


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

def test_alloc_moved_to_before_loop():
    """alloc on a body edge is removed and added to the incoming loop edge."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    _annotate_inside(loop, e_body, 'buf')

    hoist_alloc_out_of_loop(loop, ['buf'])

    assert 'buf' not in e_body.data.alloc
    assert 'buf' in e_before.data.alloc


def test_free_moved_to_after_loop():
    """free on a body edge is removed and added to the outgoing loop edge."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    _annotate_inside(loop, e_body, 'buf')

    hoist_alloc_out_of_loop(loop, ['buf'])

    assert 'buf' not in e_body.data.free
    assert 'buf' in e_after.data.free


def test_alloc_only_no_free_inside():
    """If alloc is inside but free is not, hoist alloc only; no free is added."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit
    e_body.data.alloc.append('buf')
    # No free annotation inside

    hoist_alloc_out_of_loop(loop, ['buf'])

    assert 'buf' in e_before.data.alloc
    assert 'buf' not in e_after.data.free   # nothing added


def test_loop_is_sdfg_start_thin_predecessor_inserted():
    """When the loop has no incoming edges, a thin predecessor state is inserted."""
    sdfg = dace.SDFG('start_loop')
    sdfg.add_array('buf', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)

    loop = LoopRegion('myloop', condition_expr='i < 5', loop_var='i',
                      initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    done = sdfg.add_state('done')
    e_after = sdfg.add_edge(loop, done, InterstateEdge())

    body = loop.add_state('body', is_start_block=True)
    e_body = loop.add_edge(body, loop.add_state('body_end'), InterstateEdge())
    e_body.data.alloc.append('buf')

    hoist_alloc_out_of_loop(loop, ['buf'])

    in_edges = sdfg.in_edges(loop)
    assert len(in_edges) == 1
    assert 'buf' in in_edges[0].data.alloc


def test_loop_is_sdfg_sink_thin_successor_inserted():
    """When the loop has no outgoing edges, a thin successor state is inserted."""
    sdfg = dace.SDFG('sink_loop')
    sdfg.add_array('buf', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)

    init = sdfg.add_state('init')
    loop = LoopRegion('myloop', condition_expr='i < 5', loop_var='i',
                      initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    e_before = sdfg.add_edge(init, loop, InterstateEdge())

    body = loop.add_state('body', is_start_block=True)
    e_body = loop.add_edge(body, loop.add_state('body_end'), InterstateEdge())
    e_body.data.alloc.append('buf')
    e_body.data.free.append('buf')

    hoist_alloc_out_of_loop(loop, ['buf'])

    out_edges = sdfg.out_edges(loop)
    assert len(out_edges) == 1
    assert 'buf' in out_edges[0].data.free


def test_multiple_arrays_hoisted():
    """Multiple arrays can be hoisted in a single call."""
    sdfg = dace.SDFG('multi')
    sdfg.add_array('dx', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)
    sdfg.add_array('dy', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)

    init = sdfg.add_state('init')
    loop = LoopRegion('myloop', condition_expr='i < 10', loop_var='i',
                      initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    done = sdfg.add_state('done')
    e_before = sdfg.add_edge(init, loop, InterstateEdge())
    e_after  = sdfg.add_edge(loop, done, InterstateEdge())

    body = loop.add_state('body', is_start_block=True)
    e_body = loop.add_edge(body, loop.add_state('body_end'), InterstateEdge())
    e_body.data.alloc = ['dx', 'dy']
    e_body.data.free  = ['dx', 'dy']

    hoist_alloc_out_of_loop(loop, ['dx', 'dy'])

    assert 'dx' in e_before.data.alloc
    assert 'dy' in e_before.data.alloc
    assert 'dx' in e_after.data.free
    assert 'dy' in e_after.data.free
    assert e_body.data.alloc == []
    assert e_body.data.free  == []


# ---------------------------------------------------------------------------
# Nested loops
# ---------------------------------------------------------------------------

def test_nested_loop_inner_alloc_hoisted_to_outer_parent():
    """Alloc on an edge inside an inner loop is hoisted out of the outer loop
    when hoist_alloc_out_of_loop is called on the outer loop."""
    sdfg = dace.SDFG('nested')
    sdfg.add_array('buf', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)

    init  = sdfg.add_state('init')
    outer = LoopRegion('outer', condition_expr='i < 3', loop_var='i',
                       initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(outer)
    done  = sdfg.add_state('done')
    e_before = sdfg.add_edge(init,  outer, InterstateEdge())
    e_after  = sdfg.add_edge(outer, done,  InterstateEdge())

    inner = LoopRegion('inner', condition_expr='j < 3', loop_var='j',
                       initialize_expr='j = 0', update_expr='j = j + 1',
                       sdfg=sdfg)
    outer.add_node(inner, is_start_block=True)

    inner_body = inner.add_state('inner_body', is_start_block=True)
    e_inner = inner.add_edge(inner_body,
                              inner.add_state('inner_end'), InterstateEdge())
    e_inner.data.alloc.append('buf')
    e_inner.data.free.append('buf')

    # Hoist out of the OUTER loop — should reach into nested region
    hoist_alloc_out_of_loop(outer, ['buf'])

    assert 'buf' not in e_inner.data.alloc
    assert 'buf' in e_before.data.alloc
    assert 'buf' in e_after.data.free


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_type_error_for_non_loop_region():
    """Passing anything other than a LoopRegion raises TypeError."""
    sdfg = dace.SDFG('err')
    state = sdfg.add_state('s')
    with pytest.raises(TypeError, match="LoopRegion"):
        hoist_alloc_out_of_loop(state, ['buf'])


def test_type_error_message_mentions_maps():
    """The TypeError message specifically explains that Maps are parallel."""
    sdfg = dace.SDFG('err')
    # MapEntry is not a LoopRegion — simulate by passing a random object
    class FakeMap:
        pass
    with pytest.raises(TypeError, match="Map"):
        hoist_alloc_out_of_loop(FakeMap(), ['buf'])


def test_value_error_unknown_array():
    sdfg = dace.SDFG('err')
    loop = LoopRegion('l', condition_expr='True', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    with pytest.raises(ValueError, match="not in the SDFG"):
        hoist_alloc_out_of_loop(loop, ['nonexistent'])


def test_value_error_non_transient_array():
    sdfg = dace.SDFG('err')
    sdfg.add_array('inp', [10], dace.float64, transient=False)
    loop = LoopRegion('l', condition_expr='True', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    with pytest.raises(ValueError, match="not a transient"):
        hoist_alloc_out_of_loop(loop, ['inp'])


def test_explicit_no_alloc_inside_loop_is_skipped():
    """An Explicit-lifetime array with no alloc annotation inside the loop
    (e.g. allocated outside the loop) is silently skipped — no error raised."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit
    # Alloc is on the edge before the loop, not inside — nothing to hoist
    e_before.data.alloc.append('buf')
    e_after.data.free.append('buf')

    hoist_alloc_out_of_loop(loop, ['buf'])  # must not raise

    # Edges unchanged
    assert 'buf' in e_before.data.alloc
    assert 'buf' in e_after.data.free


# ---------------------------------------------------------------------------
# Auto make_explicit + hoist
# ---------------------------------------------------------------------------

def test_non_explicit_array_is_made_explicit_and_hoisted():
    """An array with Scope lifetime (not yet Explicit) is automatically
    converted to Explicit and hoisted out of the loop in one call."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    # buf has default Scope lifetime — no alloc/free annotations yet

    hoist_alloc_out_of_loop(loop, ['buf'])

    assert sdfg.arrays['buf'].lifetime is dtypes.AllocationLifetime.Explicit
    assert 'buf' in e_before.data.alloc, "alloc must be on the edge before the loop"
    assert 'buf' in e_after.data.free,   "free must be on the edge after the loop"
    # No alloc/free must remain inside the loop
    for e in loop.edges():
        assert 'buf' not in e.data.alloc
        assert 'buf' not in e.data.free


def test_non_explicit_array_validates_after_hoist():
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    hoist_alloc_out_of_loop(loop, ['buf'])
    sdfg.validate()


def test_non_explicit_sdfg_lifetime_array_skipped():
    """An array with SDFG lifetime is placed at SDFG level by make_explicit —
    no alloc inside the loop, so hoist silently skips it."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.SDFG

    hoist_alloc_out_of_loop(loop, ['buf'])  # must not raise

    # make_explicit placed alloc/free at SDFG level already
    assert sdfg.arrays['buf'].lifetime is dtypes.AllocationLifetime.Explicit
    sdfg_alloc = any('buf' in e.data.alloc for e in sdfg.edges())
    assert sdfg_alloc, "SDFG-lifetime array should be allocated at SDFG level"


# ---------------------------------------------------------------------------
# SDFG validation
# ---------------------------------------------------------------------------

def test_sdfg_validates_after_hoist():
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    _annotate_inside(loop, e_body, 'buf')
    sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit

    hoist_alloc_out_of_loop(loop, ['buf'])
    sdfg.validate()


# ---------------------------------------------------------------------------
# Code-generation smoke test
# ---------------------------------------------------------------------------

def test_codegen_new_before_loop_delete_after():
    """After hoisting, new[] must appear before the loop body and delete[] after."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    _annotate_inside(loop, e_body, 'buf')
    sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit

    hoist_alloc_out_of_loop(loop, ['buf'])

    cpp = _get_cpp(sdfg)

    # new[] must appear before the loop keyword
    new_pos    = cpp.find('new double')
    loop_pos   = cpp.find('for (')
    delete_pos = cpp.find('delete[]')

    assert new_pos    != -1, "new[] not found in generated C++"
    assert loop_pos   != -1, "for-loop not found in generated C++"
    assert delete_pos != -1, "delete[] not found in generated C++"

    assert new_pos < loop_pos,    "new[] should appear before the loop"
    assert loop_pos < delete_pos, "delete[] should appear after the loop"


# ---------------------------------------------------------------------------
# Multiple incoming / outgoing edges
# ---------------------------------------------------------------------------

def test_multiple_incoming_edges_alloc_on_all():
    """When the loop has two incoming edges (from a conditional branch), alloc
    is added to every incoming edge so the array is allocated on both paths."""
    sdfg = dace.SDFG('multi_in')
    sdfg.add_array('buf', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)

    init     = sdfg.add_state('init', is_start_block=True)
    true_br  = sdfg.add_state('true_br')
    false_br = sdfg.add_state('false_br')
    loop     = LoopRegion('myloop', 'i < 10', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    done     = sdfg.add_state('done')

    sdfg.add_edge(init, true_br,  InterstateEdge(condition='1 == 1'))
    sdfg.add_edge(init, false_br, InterstateEdge(condition='1 == 0'))
    e1 = sdfg.add_edge(true_br,  loop, InterstateEdge())
    e2 = sdfg.add_edge(false_br, loop, InterstateEdge())
    sdfg.add_edge(loop, done, InterstateEdge())

    body   = loop.add_state('body', is_start_block=True)
    e_body = loop.add_edge(body, loop.add_state('body_end'), InterstateEdge())
    e_body.data.alloc.append('buf')
    e_body.data.free.append('buf')

    hoist_alloc_out_of_loop(loop, ['buf'])

    assert 'buf' in e1.data.alloc, "alloc must appear on the true-branch incoming edge"
    assert 'buf' in e2.data.alloc, "alloc must appear on the false-branch incoming edge"
    assert 'buf' not in e_body.data.alloc


def test_multiple_outgoing_edges_free_on_all():
    """When the loop has two outgoing edges, free is added to every outgoing
    edge so the array is released on every exit path."""
    sdfg = dace.SDFG('multi_out')
    sdfg.add_array('buf', [10], dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)

    init  = sdfg.add_state('init', is_start_block=True)
    loop  = LoopRegion('myloop', 'i < 10', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    done1 = sdfg.add_state('done1')
    done2 = sdfg.add_state('done2')

    sdfg.add_edge(init, loop, InterstateEdge())
    e_out1 = sdfg.add_edge(loop, done1, InterstateEdge(condition='1 == 1'))
    e_out2 = sdfg.add_edge(loop, done2, InterstateEdge(condition='1 == 0'))

    body   = loop.add_state('body', is_start_block=True)
    e_body = loop.add_edge(body, loop.add_state('body_end'), InterstateEdge())
    e_body.data.alloc.append('buf')
    e_body.data.free.append('buf')

    hoist_alloc_out_of_loop(loop, ['buf'])

    assert 'buf' in e_out1.data.free, "free must appear on first outgoing edge"
    assert 'buf' in e_out2.data.free, "free must appear on second outgoing edge"
    assert 'buf' not in e_body.data.free


# ---------------------------------------------------------------------------
# Constructed example: quantitative hoisting check
# ---------------------------------------------------------------------------

def test_hoist_moves_alloc_out_of_loop_body_completely():
    """Constructed example: an array allocated on every loop body edge has
    all internal alloc/free annotations removed after hoisting, and gains
    exactly one alloc annotation on the pre-loop edge."""
    sdfg, loop, e_before, e_after, e_body = _make_loop_sdfg()
    _annotate_inside(loop, e_body, 'buf')

    # Sanity: one alloc inside the loop, none outside
    inner_allocs_before = sum(1 for e in loop.edges() if 'buf' in e.data.alloc)
    outer_allocs_before = sum(1 for e in sdfg.edges() if 'buf' in e.data.alloc)
    assert inner_allocs_before == 1
    assert outer_allocs_before == 0

    hoist_alloc_out_of_loop(loop, ['buf'])

    # After hoist: no alloc/free remain inside the loop
    inner_allocs_after = sum(1 for e in loop.edges() if 'buf' in e.data.alloc)
    inner_frees_after  = sum(1 for e in loop.edges() if 'buf' in e.data.free)
    assert inner_allocs_after == 0, "no alloc must remain inside the loop body"
    assert inner_frees_after  == 0, "no free must remain inside the loop body"

    # Exactly one alloc on the pre-loop edge and one free on the post-loop edge
    outer_allocs_after = sum(1 for e in sdfg.edges() if 'buf' in e.data.alloc)
    outer_frees_after  = sum(1 for e in sdfg.edges() if 'buf' in e.data.free)
    assert outer_allocs_after == 1
    assert outer_frees_after  == 1
