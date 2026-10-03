# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lift that REFUSES must leave the guard symbol's interstate definition alone.

``SameWriteSetIfElseToITECFG`` deleted the definition as soon as it had read the assignment, before
the checks that can still refuse the lift. On a refusal the caller keeps the guard as free-symbol
TEXT in the ITE tasklet (``_o = ITE(<cond_sym>, _t, _e)``), so the deletion left that text naming a
symbol nothing defines any more -- "Missing symbols on nested SDFG" on the enclosing loop body
(CloudSC: ``zqxfg_index_36``, ``zcovpclr_index_3``, ``__tmp234``).

A compound guard has the same hole one level up: it lifts its names one at a time, and a LATER
name's refusal aborts the whole compound after the earlier names already committed their deletions.
"""
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG

# ``i`` is not a registered SDFG symbol, so the gather index cannot be promoted to an interstate
# symbol and the lift refuses -- the refusal that sits AFTER the old deletion site.
GATHER = 'w[idx[i]] > 0.0'
PLAIN = 'w[0] > 0.0'


def build_guard_sdfg(assignments: dict):
    """An SDFG whose one interstate edge assigns every ``symbol: rhs`` in ``assignments``."""
    sdfg = dace.SDFG('refused_lift')
    sdfg.add_array('w', [8], dace.float64)
    sdfg.add_array('idx', [8], dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)
    body = sdfg.add_state('body')
    for sym in assignments:
        sdfg.add_symbol(sym, dace.bool_)
    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments=dict(assignments)))
    return sdfg, body


def test_refused_scalar_lift_keeps_the_definition():
    """The un-promotable gather is refused, and the assignment the ITE text still names survives."""
    sdfg, body = build_guard_sdfg({'gather_sym': GATHER})

    assert SameWriteSetIfElseToITECFG()._lift_interstate_cond_to_tasklet(sdfg, body, 'gather_sym', '0:8') is None

    assignments = sdfg.edges()[0].data.assignments
    assert assignments.get('gather_sym') == GATHER, 'refused lift deleted the guard it did not replace'
    assert 'gather_sym' in sdfg.symbols


def test_refused_compound_keeps_every_component_definition():
    """``a_plain`` lifts, ``z_gather`` refuses: the compound aborts, so neither definition may go.

    Names are lifted in sorted order, so the liftable one commits first and is exactly the one an
    unbuffered deletion loses.
    """
    sdfg, body = build_guard_sdfg({'a_plain': PLAIN, 'z_gather': GATHER})

    assert SameWriteSetIfElseToITECFG()._lift_compound_cond_to_tasklet(sdfg, body, '(a_plain or z_gather)',
                                                                       '0:8') is None

    assignments = sdfg.edges()[0].data.assignments
    assert 'a_plain' in assignments, 'the component that lifted lost its definition when a later one refused'
    assert 'z_gather' in assignments


def test_committed_compound_still_drops_its_components():
    """Buffering must not freeze the deletions: a compound where every name lifts still prunes."""
    sdfg, body = build_guard_sdfg({'a_plain': PLAIN, 'b_plain': 'w[1] > 0.0'})

    assert SameWriteSetIfElseToITECFG()._lift_compound_cond_to_tasklet(sdfg, body, '(a_plain or b_plain)',
                                                                       '0:8') is not None

    assignments = sdfg.edges()[0].data.assignments
    assert 'a_plain' not in assignments and 'b_plain' not in assignments, \
        f'a committed compound left {sorted(assignments)} behind'
    assert 'a_plain' not in sdfg.symbols and 'b_plain' not in sdfg.symbols


def build_refused_if_else_in_a_loop() -> dace.SDFG:
    """``for i: if gather_sym: w[0] = 1.0 else: w[0] = 2.0``, the guard defined on the loop's edge.

    The gather index is the LOOP variable, which validation has in scope but ``sdfg.symbols`` does
    not -- the shape a CloudSC loop body arrives in, and exactly the one the promotion refuses.
    :func:`build_guard_sdfg` cannot serve here: its bare ``i`` is in no scope at all, so the SDFG
    it hands back does not validate whatever the pass does to it.

    :returns: the SDFG.
    """
    sdfg = dace.SDFG('refused_lift_in_loop')
    sdfg.add_array('w', [8], dace.float64)
    sdfg.add_array('idx', [8], dace.int64)
    sdfg.add_symbol('gather_sym', dace.bool_)
    loop = LoopRegion('sweep', 'i < 8', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    entry = loop.add_state('entry', is_start_block=True)
    cond = ConditionalBlock('same_write_set', sdfg=sdfg, parent=loop)
    loop.add_node(cond)
    loop.add_edge(entry, cond, dace.InterstateEdge(assignments={'gather_sym': GATHER}))
    for label, value, branch_cond in (('then', 1.0, CodeBlock('gather_sym')), ('otherwise', 2.0, None)):
        arm = ControlFlowRegion(label, sdfg=sdfg)
        arm_state = arm.add_state(f'{label}_body', is_start_block=True)
        tasklet = arm_state.add_tasklet(f'{label}_t', {}, {'_o'}, f'_o = {value}')
        arm_state.add_edge(tasklet, '_o', arm_state.add_write('w'), None, dace.Memlet('w[0]'))
        cond.add_branch(branch_cond, arm)
    return sdfg


def test_the_public_pass_leaves_a_validating_sdfg_when_the_lift_refuses():
    """The shipped symptom, through the public entry point: "Missing symbols on nested SDFG".

    The three tests above drive the private lift helpers and read the assignment dict back, so
    none of them ever reaches the state the bug was reported from -- a whole-SDFG rewrite
    followed by a validation.
    """
    sdfg = build_refused_if_else_in_a_loop()
    sdfg.validate()

    assert SameWriteSetIfElseToITECFG().apply_pass(sdfg, {}) == 1, 'the fixture must reach the rewrite'

    sdfg.validate()
    defined = [e for e in sdfg.all_interstate_edges() if 'gather_sym' in e.data.assignments]
    assert defined, 'the refused lift deleted the only definition of the guard the ITE text still names'


if __name__ == '__main__':
    test_refused_scalar_lift_keeps_the_definition()
    test_refused_compound_keeps_every_component_definition()
    test_committed_compound_still_drops_its_components()
    test_the_public_pass_leaves_a_validating_sdfg_when_the_lift_refuses()
