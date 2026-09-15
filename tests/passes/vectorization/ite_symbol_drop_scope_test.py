# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A guard symbol consumed inside a NESTED SDFG must survive ``SameWriteSetIfElseToITECFG``.

``_drop_interstate_symbol`` deletes a symbol's interstate assignments through
``all_control_flow_regions(recursive=True)``, which descends into nested SDFGs. Its
``_symbol_has_external_consumer`` gate therefore has to look at least as deep: while two of its
four scans stopped at the top level, a top-level rewrite deleted the assignment that a nested
``ConditionalBlock`` was still testing, and the nested guard came out as an unbound free symbol --
"Missing symbols on nested SDFG" the moment anything validated (CloudSC, ``zsolqa_index_58_3``).
"""
import dace
import pytest
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (SameWriteSetIfElseToITECFG,
                                                                                        _symbol_has_external_consumer)

SYM = 'guard_sym'


def build_nested_consumer(kind: str) -> dace.SDFG:
    """An outer SDFG assigning ``SYM`` on an interstate edge, consumed only inside a nested SDFG.

    :param kind: ``'condition'`` puts the consumer in a nested ``ConditionalBlock`` guard,
        ``'tasklet'`` puts it in a nested tasklet's code. Both were invisible to the shallow scan.
    :returns: the outer SDFG.
    """
    outer = dace.SDFG(f'outer_{kind}')
    outer.add_array('A', [8], dace.float64)
    outer.add_symbol(SYM, dace.float64)

    entry = outer.add_state('entry', is_start_block=True)
    body = outer.add_state('body')
    # The sole definition of the guard symbol. This is what the drop deletes.
    outer.add_edge(entry, body, dace.InterstateEdge(assignments={SYM: 'A[0]'}))

    # The nested SDFG carries its OWN definition of the symbol and does NOT bind it in
    # ``symbol_mapping`` -- the shape ``LoopToMap`` leaves behind, which deliberately keeps a
    # loop-internal interstate-assignment target off the node's mapping. A binding would itself
    # register as a consumer, so a fixture that had one could not expose the shallow scan.
    inner = dace.SDFG(f'inner_{kind}')
    inner.add_array('A', [8], dace.float64)
    inner.add_symbol(SYM, dace.float64)
    inner_entry = inner.add_state('inner_entry', is_start_block=True)
    if kind == 'condition':
        cb = ConditionalBlock('nested_guard', sdfg=inner)
        inner.add_node(cb)
        branch = dace.sdfg.state.ControlFlowRegion('taken', sdfg=inner)
        branch.add_state('taken_body', is_start_block=True)
        cb.add_branch(CodeBlock(f'{SYM} < 0.0'), branch)
        inner.add_edge(inner_entry, cb, dace.InterstateEdge(assignments={SYM: 'A[0]'}))
    else:
        inner_state = inner.add_state('inner_body')
        inner.add_edge(inner_entry, inner_state, dace.InterstateEdge(assignments={SYM: 'A[0]'}))
        tasklet = inner_state.add_tasklet('use_guard', {}, {'_o'}, f'_o = {SYM} * 2.0')
        inner_state.add_edge(tasklet, '_o', inner_state.add_write('A'), None, dace.Memlet('A[0]'))

    nsdfg = body.add_nested_sdfg(inner, {'A'}, {'A'}, symbol_mapping={})
    body.add_edge(body.add_read('A'), None, nsdfg, 'A', dace.Memlet('A[0:8]'))
    body.add_edge(nsdfg, 'A', body.add_write('A'), None, dace.Memlet('A[0:8]'))
    return outer


@pytest.mark.parametrize('kind', ['condition', 'tasklet'])
def test_nested_consumer_is_visible(kind):
    """The gate reports the nested consumer, so the caller never reaches the deletion."""
    outer = build_nested_consumer(kind)
    assert _symbol_has_external_consumer(outer, SYM, None) is True


@pytest.mark.parametrize('kind', ['condition', 'tasklet'])
def test_nested_consumer_keeps_the_definition(kind):
    """The drop is a no-op while a nested consumer remains: assignment, symbol and mapping stay.

    Checked on the graph rather than only on the gate's return value -- the gate is what the drop
    consults, but it is the surviving assignment that keeps the nested guard bound.
    """
    outer = build_nested_consumer(kind)
    # Only the OUTER edge is handed to the drop, exactly as the rewrite collects it -- the nested
    # definition must survive because the drop was never asked about it, and the nested guard
    # must keep the definition it reads.
    outer_edges = [e for e in outer.edges() if SYM in e.data.assignments]
    assert len(outer_edges) == 1, 'fixture should carry one top-level definition of the guard symbol'

    SameWriteSetIfElseToITECFG()._drop_interstate_symbol(outer, SYM, outer_edges)

    surviving = [
        e for cfg in outer.all_control_flow_regions(recursive=True) for e in cfg.edges() if SYM in e.data.assignments
    ]
    assert len(surviving) == 2, (f'{SYM} lost a definition while a nested consumer still reads it; '
                                 f'{len(surviving)} of 2 left')
    assert SYM in outer.symbols


def test_unconsumed_symbol_is_still_dropped():
    """The deeper scan must not freeze every symbol: one nothing reads is still removed."""
    outer = build_nested_consumer('tasklet')
    outer.add_symbol('dead_sym', dace.float64)
    edge = next(e for e in outer.edges() if SYM in e.data.assignments)
    edge.data.assignments['dead_sym'] = 'A[1]'

    SameWriteSetIfElseToITECFG()._drop_interstate_symbol(outer, 'dead_sym', [edge])

    assert 'dead_sym' not in edge.data.assignments
    assert 'dead_sym' not in outer.symbols


@pytest.mark.parametrize('kind', ['condition', 'tasklet'])
def test_the_public_pass_leaves_a_validating_sdfg(kind):
    """The shipped symptom, through the public entry point: "Missing symbols on nested SDFG".

    The tests above drive ``_drop_interstate_symbol`` and its gate directly, so a deletion that
    unbinds the nested guard shows up only as a count -- never as the validation failure the bug
    was actually reported as.
    """
    outer = build_nested_consumer(kind)

    SameWriteSetIfElseToITECFG().apply_pass(outer, {})

    outer.validate()


if __name__ == '__main__':
    test_nested_consumer_is_visible('condition')
    test_nested_consumer_is_visible('tasklet')
    test_nested_consumer_keeps_the_definition('condition')
    test_nested_consumer_keeps_the_definition('tasklet')
    test_unconsumed_symbol_is_still_dropped()
    test_the_public_pass_leaves_a_validating_sdfg('condition')
    test_the_public_pass_leaves_a_validating_sdfg('tasklet')
