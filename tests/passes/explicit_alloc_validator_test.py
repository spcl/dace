# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for _validate_explicit_allocation_balance — alloc/free balance
checks for explicit allocation annotations on interstate edges.
"""
import pytest
import warnings
import dace
from dace import dtypes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock
from dace.sdfg.validation import (
    InvalidSDFGError,
    InvalidSDFGInterstateEdgeError,
    _validate_explicit_allocation_balance,
)


def _explicit_array(sdfg: SDFG, name: str, shape=(4,)):
    sdfg.add_array(name, shape, dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)


def test_validator_is_callable_on_empty_sdfg():
    sdfg = SDFG('empty')
    sdfg.add_state('only')
    _validate_explicit_allocation_balance(sdfg)   # must not raise


def _sdfg_with_use_and(opt_free: bool, opt_alloc: bool):
    """3-state SDFG with an Explicit array 'tmp1' used in the middle state.
    Conditionally attach alloc to edge0 and/or free to edge1."""
    sdfg = SDFG('b1')
    _explicit_array(sdfg, 'tmp1')
    s0 = sdfg.add_state('init'); s1 = sdfg.add_state('use'); s2 = sdfg.add_state('done')
    e0 = sdfg.add_edge(s0, s1, InterstateEdge())
    e1 = sdfg.add_edge(s1, s2, InterstateEdge())
    s1.add_access('tmp1')     # ensure 'tmp1' is "used"
    if opt_alloc: e0.data.alloc = ['tmp1']
    if opt_free:  e1.data.free  = ['tmp1']
    return sdfg


def test_b1_missing_alloc_raises():
    sdfg = _sdfg_with_use_and(opt_free=True, opt_alloc=False)
    with pytest.raises(InvalidSDFGError, match="tmp1"):
        _validate_explicit_allocation_balance(sdfg)


def test_b2_missing_free_raises():
    sdfg = _sdfg_with_use_and(opt_free=False, opt_alloc=True)
    with pytest.raises(InvalidSDFGError, match="tmp1"):
        _validate_explicit_allocation_balance(sdfg)


def test_b3_unused_explicit_warns():
    sdfg = SDFG('b3')
    _explicit_array(sdfg, 'scratch')           # declared Explicit, never used
    sdfg.add_state('only')
    with pytest.warns(UserWarning, match="scratch"):
        _validate_explicit_allocation_balance(sdfg)


def test_b4_duplicate_in_alloc_raises():
    sdfg = _sdfg_with_use_and(opt_free=True, opt_alloc=False)
    # Put tmp1 in alloc twice on the init->use edge
    init, use = list(sdfg.nodes())[0], list(sdfg.nodes())[1]
    e = sdfg.edges_between(init, use)[0]
    e.data.alloc = ['tmp1', 'tmp1']
    with pytest.raises(InvalidSDFGInterstateEdgeError, match="duplicate"):
        _validate_explicit_allocation_balance(sdfg)


def test_b4_duplicate_in_free_raises():
    sdfg = _sdfg_with_use_and(opt_free=False, opt_alloc=True)
    init, use = list(sdfg.nodes())[0], list(sdfg.nodes())[1]
    use_done = sdfg.edges_between(use, list(sdfg.nodes())[2])[0]
    use_done.data.free = ['tmp1', 'tmp1']
    with pytest.raises(InvalidSDFGInterstateEdgeError, match="duplicate"):
        _validate_explicit_allocation_balance(sdfg)


def test_b5_alloc_free_overlap_raises():
    sdfg = _sdfg_with_use_and(opt_free=False, opt_alloc=False)
    init, use = list(sdfg.nodes())[0], list(sdfg.nodes())[1]
    e = sdfg.edges_between(init, use)[0]
    e.data.alloc = ['tmp1']
    e.data.free  = ['tmp1']
    with pytest.raises(InvalidSDFGInterstateEdgeError, match="alloc and free"):
        _validate_explicit_allocation_balance(sdfg)


def test_happy_path_linear():
    sdfg = _sdfg_with_use_and(opt_free=True, opt_alloc=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')           # warnings → errors
        _validate_explicit_allocation_balance(sdfg)


def test_happy_path_loop_region():
    """Explicit array 'acc' allocated on the loop-internal header edge,
    freed on the loop-internal exit edge. If _all_interstate_edges did NOT
    recurse into LoopRegion, this would trip B1/B2."""
    sdfg = SDFG('b7')
    _explicit_array(sdfg, 'acc')
    pre = sdfg.add_state('pre')
    post = sdfg.add_state('post')

    loop = LoopRegion('L', condition_expr='i < 10',
                      loop_var='i', initialize_expr='i = 0',
                      update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, InterstateEdge())
    sdfg.add_edge(loop, post, InterstateEdge())

    body_a = loop.add_state('body_a', is_start_block=True)
    body_b = loop.add_state('body_b')
    body_c = loop.add_state('body_c')
    ea = loop.add_edge(body_a, body_b, InterstateEdge(assignments={}))
    eb = loop.add_edge(body_b, body_c, InterstateEdge(assignments={}))
    ea.data.alloc = ['acc']
    eb.data.free  = ['acc']
    body_b.add_access('acc')

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _validate_explicit_allocation_balance(sdfg)


def test_b4_inside_loop_region_error_renders():
    """Regression: nested-region edge ids must be resolvable by the exception's
    __str__ path. Before the owner-arg fix, str(exc) would report the wrong edge
    (a top-level SDFG edge) rather than the nested ba->bb edge where the
    violation actually lives."""
    sdfg = SDFG('nested_b4')
    _explicit_array(sdfg, 'nx')
    pre = sdfg.add_state('pre'); post = sdfg.add_state('post')
    loop = LoopRegion('L', condition_expr='i < 2', loop_var='i',
                      initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, InterstateEdge())
    sdfg.add_edge(loop, post, InterstateEdge())
    ba = loop.add_state('ba', is_start_block=True)
    bb = loop.add_state('bb')
    e = loop.add_edge(ba, bb, InterstateEdge(assignments={}))
    e.data.alloc = ['nx', 'nx']   # B4 violation on nested edge
    bb.add_access('nx')
    # The fix guarantees both: (a) the error is raised, and (b) str() identifies
    # the correct nested edge (ba->bb), not a top-level edge (pre->L).
    with pytest.raises(InvalidSDFGInterstateEdgeError, match="duplicate") as excinfo:
        _validate_explicit_allocation_balance(sdfg)
    rendered = str(excinfo.value)   # must not raise
    assert 'ba' in rendered and 'bb' in rendered, (
        f"str(exc) should reference the nested ba->bb edge, got: {rendered!r}"
    )


def test_happy_path_conditional_block():
    """alloc on pre-branch edge, free on post-branch edge, use inside one
    branch. Top-level scan already sees pre/post edges, so this test mostly
    confirms that a ConditionalBlock doesn't somehow swallow top-level edges."""
    sdfg = SDFG('b8')
    _explicit_array(sdfg, 'tmp')

    pre = sdfg.add_state('pre')
    post = sdfg.add_state('post')
    cond = ConditionalBlock('C')
    sdfg.add_node(cond)

    true_body = ControlFlowRegion('true_body', sdfg=sdfg)
    t_use = true_body.add_state('t_use', is_start_block=True)
    t_use.add_access('tmp')
    cond.add_branch(CodeBlock('True'), true_body)

    e_pre  = sdfg.add_edge(pre,  cond, InterstateEdge())
    e_post = sdfg.add_edge(cond, post, InterstateEdge())
    e_pre.data.alloc  = ['tmp']
    e_post.data.free  = ['tmp']

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _validate_explicit_allocation_balance(sdfg)
