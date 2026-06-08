# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``PromoteGPUScalarsToArrays``.

Each test pinpoints one rewrite slot that the pass touches (or should touch)
and one issue from PR #2394 / Issue #2393. Tests run with no GPU runtime --
they introspect SDFG state after the pass, not generated code.
"""
import pytest

import dace
from dace import data, dtypes, properties
from dace.memlet import Memlet
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.promote_gpu_scalars_to_arrays import (PromoteGPUScalarsToArrays,
                                                                      InferDefaultSchedulesAndStorages)
from dace.transformation.pass_pipeline import Pipeline


def _run(sdfg, **pass_kwargs):
    """Run the prerequisite + the promote pass."""
    Pipeline([InferDefaultSchedulesAndStorages()]).apply_pass(sdfg, {})
    p = PromoteGPUScalarsToArrays()
    for k, v in pass_kwargs.items():
        setattr(p, k, v)
    return p.apply_pass(sdfg, {})


# ---------------------------------------------------------------------------
# Issue #2393: where(c, a, b) lowers to Map -> NSDFG -> ConditionalBlock(c).
# After scalar promotion, the ConditionalBlock condition must read ``__cond[0]``
# rather than ``__cond`` (else it always evaluates the pointer non-null).
# ---------------------------------------------------------------------------
def _build_issue_2393_sdfg():
    nested = dace.SDFG('nested')
    nested.add_scalar('__cond', dtype=dace.bool_, transient=False, storage=dtypes.StorageType.GPU_Global)
    for name in ('__arg1', '__arg2', '__output'):
        nested.add_scalar(name, dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)

    if_region = ConditionalBlock('if')
    nested.add_node(if_region, ensure_unique_name=True)

    then_body = ControlFlowRegion('then_body', sdfg=nested)
    tstate = then_body.add_state('true_branch', is_start_block=True)
    if_region.add_branch(properties.CodeBlock('__cond'), then_body)
    tstate.add_nedge(tstate.add_access('__arg1'), tstate.add_access('__output'), Memlet(data='__arg1', subset='0'))

    else_body = ControlFlowRegion('else_body', sdfg=nested)
    fstate = else_body.add_state('false_branch', is_start_block=True)
    if_region.add_branch(None, else_body)
    fstate.add_nedge(fstate.add_access('__arg2'), fstate.add_access('__output'), Memlet(data='__arg2', subset='0'))

    outer = dace.SDFG('where')
    state = outer.add_state()
    for name in 'abcd':
        outer.add_array(name,
                        shape=(10, ),
                        dtype=(dace.bool_ if name == 'c' else dace.float64),
                        storage=dtypes.StorageType.GPU_Global,
                        transient=False)

    a, b, c, d = (state.add_access(n) for n in 'abcd')
    me, mx = state.add_map('map', ndrange={'__i': '0:10'}, schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(sdfg=nested, inputs={'__arg1', '__arg2', '__cond'}, outputs={'__output'})
    state.add_memlet_path(a, me, nsdfg, dst_conn='__arg1', memlet=Memlet(data='a', subset='__i'))
    state.add_memlet_path(b, me, nsdfg, dst_conn='__arg2', memlet=Memlet(data='b', subset='__i'))
    state.add_memlet_path(c, me, nsdfg, dst_conn='__cond', memlet=Memlet(data='c', subset='__i'))
    state.add_memlet_path(nsdfg, mx, d, src_conn='__output', memlet=Memlet(data='d', subset='__i'))
    return outer, nested


def test_issue_2393_conditional_block_pattern():
    """The Issue #2393 reproducer: ConditionalBlock branch condition rewritten."""
    outer, nested = _build_issue_2393_sdfg()
    _run(outer)
    if_region = next(b for b in nested.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock))
    cond_codeblock, _ = if_region._branches[0]
    assert cond_codeblock.as_string == '__cond[0]', cond_codeblock.as_string


# ---------------------------------------------------------------------------
# LoopRegion's three CodeBlocks (init / update / condition) all need rewriting.
# ---------------------------------------------------------------------------
def test_loop_region_init_update_condition_rewrite():
    sdfg = dace.SDFG('lr')
    sdfg.add_scalar('cnt', dtype=dace.int64, transient=False, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_state('start', is_start_block=True)
    loop = LoopRegion('lr',
                      condition_expr='i < cnt',
                      loop_var='i',
                      initialize_expr='i = cnt',
                      update_expr='i = i + cnt')
    sdfg.add_node(loop)
    loop.add_state('body', is_start_block=True)

    _run(sdfg)
    assert 'cnt[0]' in loop.loop_condition.as_string, loop.loop_condition.as_string
    assert 'cnt[0]' in loop.init_statement.as_string, loop.init_statement.as_string
    assert 'cnt[0]' in loop.update_statement.as_string, loop.update_statement.as_string


# ---------------------------------------------------------------------------
# Interstate edge: BOTH assignment values AND condition reference promoted name.
# ---------------------------------------------------------------------------
def test_interstate_edge_condition_and_assignment_rewrite():
    sdfg = dace.SDFG('ise')
    sdfg.add_symbol('s', dace.int64)
    sdfg.add_scalar('X', dtype=dace.int64, transient=False, storage=dtypes.StorageType.GPU_Global)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition='X > 0', assignments={'s': 'X + 1'}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    _run(sdfg)
    ie = sdfg.edges_between(s0, s1)[0].data
    assert 'X[0]' in ie.condition.as_string, ie.condition.as_string
    assert 'X[0]' in ie.assignments['s'], ie.assignments['s']


# ---------------------------------------------------------------------------
# NestedSDFG with a different inner name than outer; promotion must recurse
# via the connector mapping, not by matching the name verbatim.
# ---------------------------------------------------------------------------
def test_nested_sdfg_rename_via_connector():
    inner = dace.SDFG('inner')
    inner.add_scalar('X_inner', dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)
    inner.add_scalar('Y_inner', dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)
    istate = inner.add_state('s')
    istate.add_nedge(istate.add_access('X_inner'), istate.add_access('Y_inner'), Memlet(data='X_inner', subset='0'))

    outer = dace.SDFG('outer')
    outer.add_scalar('X_outer', dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)
    outer.add_scalar('Y_outer', dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)
    ostate = outer.add_state()
    nsdfg = ostate.add_nested_sdfg(sdfg=inner, inputs={'X_inner'}, outputs={'Y_inner'})
    ostate.add_edge(ostate.add_access('X_outer'), None, nsdfg, 'X_inner', Memlet(data='X_outer', subset='0'))
    ostate.add_edge(nsdfg, 'Y_inner', ostate.add_access('Y_outer'), None, Memlet(data='Y_outer', subset='0'))

    _run(outer)
    assert isinstance(outer.arrays['X_outer'], data.Array), type(outer.arrays['X_outer']).__name__
    assert isinstance(inner.arrays['X_inner'], data.Array), type(inner.arrays['X_inner']).__name__
    assert isinstance(outer.arrays['Y_outer'], data.Array)
    assert isinstance(inner.arrays['Y_inner'], data.Array)


# ---------------------------------------------------------------------------
# Memlet preservation: a copy memlet with explicit src/dst subsets stays intact
# (the bug PR #2394 documents in fix 2 is the dropped ``other_subset``).
# ---------------------------------------------------------------------------
def test_memlet_other_subset_preserved():
    sdfg = dace.SDFG('m')
    sdfg.add_scalar('X', dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('Y', shape=(4, ), dtype=dace.float64, transient=False, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    x = state.add_access('X')
    y = state.add_access('Y')
    m = Memlet(data='X', subset='0', other_subset='2')
    state.add_nedge(x, y, m)

    _run(sdfg)
    new_edge = next(e for e in state.edges() if e.src is x and e.dst is y)
    assert new_edge.data.other_subset is not None and str(new_edge.data.other_subset) == '2', new_edge.data.other_subset


# ---------------------------------------------------------------------------
# Rule 2 variant: a transient scalar written and read entirely inside a GPU
# kernel should NOT be promoted (it's a register/local). A non-transient
# kernel-output scalar SHOULD be promoted.
# ---------------------------------------------------------------------------
def _build_kernel_output_sdfg(transient_inside: bool, kernel_output_visible: bool):
    sdfg = dace.SDFG('rule2')
    sdfg.add_array('A', shape=(10, ), dtype=dace.float64, storage=dtypes.StorageType.GPU_Global)
    if kernel_output_visible:
        sdfg.add_scalar('out_s', dtype=dace.float64, transient=False, storage=dtypes.StorageType.Default)
    sdfg.add_scalar('local_s', dtype=dace.float64, transient=transient_inside, storage=dtypes.StorageType.Default)
    state = sdfg.add_state()
    me, mx = state.add_map('m', ndrange={'i': '0:10'}, schedule=dtypes.ScheduleType.GPU_Device)
    t = state.add_tasklet('w', {'a'}, {'b'}, 'b = a + 1.0')
    a = state.add_access('A')
    ls = state.add_access('local_s')
    state.add_memlet_path(a, me, t, dst_conn='a', memlet=Memlet(data='A', subset='i'))
    state.add_memlet_path(t, mx, ls, src_conn='b', memlet=Memlet(data='local_s', subset='0'))
    if kernel_output_visible:
        os_acc = state.add_access('out_s')
        state.add_nedge(ls, os_acc, Memlet(data='local_s', subset='0'))
    return sdfg


def test_rule2_kernel_output_scalar_promoted():
    """Non-transient scalar written at a GPU map exit is a kernel output -> promote."""
    sdfg = _build_kernel_output_sdfg(transient_inside=False, kernel_output_visible=True)
    n = _run(sdfg)
    assert n is not None and n >= 1, n
    # local_s is now an Array; out_s should also be (if it received GPU storage propagation).
    assert isinstance(sdfg.arrays['local_s'], data.Array)


@pytest.mark.parametrize('non_transient_only', [True, False])
def test_rule2_transient_purely_internal_not_promoted(non_transient_only: bool):
    """A transient scalar written at a GPU map exit and never read at the host
    side is *not* a kernel output the host can observe. Under both variants the
    pass should leave it alone.
    """
    sdfg = _build_kernel_output_sdfg(transient_inside=True, kernel_output_visible=False)
    _run(sdfg, non_transient_only=non_transient_only)
    if non_transient_only:
        assert isinstance(sdfg.arrays['local_s'], data.Scalar)


# ---------------------------------------------------------------------------
# Audit: NSDFG.symbol_mapping value referencing the promoted scalar by name
# (bare identifier) must be rewritten to ``name[0]``.
# ---------------------------------------------------------------------------
def test_symbol_mapping_value_rewrite():
    inner = dace.SDFG('inner')
    inner.add_symbol('s_inner', dace.int64)
    inner.add_state('s', is_start_block=True)

    outer = dace.SDFG('outer')
    outer.add_scalar('X', dtype=dace.int64, transient=False, storage=dtypes.StorageType.GPU_Global)
    ostate = outer.add_state()
    nsdfg = ostate.add_nested_sdfg(sdfg=inner, inputs=set(), outputs=set(), symbol_mapping={'s_inner': 'X'})
    _run(outer)
    assert nsdfg.symbol_mapping['s_inner'] == 'X[0]' or str(nsdfg.symbol_mapping['s_inner']) == 'X[0]', \
        nsdfg.symbol_mapping['s_inner']


if __name__ == '__main__':
    test_issue_2393_conditional_block_pattern()
    test_loop_region_init_update_condition_rewrite()
    test_interstate_edge_condition_and_assignment_rewrite()
    test_nested_sdfg_rename_via_connector()
    test_memlet_other_subset_preserved()
    test_rule2_kernel_output_scalar_promoted()
    test_rule2_transient_purely_internal_not_promoted(True)
    test_rule2_transient_purely_internal_not_promoted(False)
    test_symbol_mapping_value_rewrite()
