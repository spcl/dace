# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression: ``ControlGraphView.replace`` must reach every symbol reference a region holds.

``ControlGraphView.replace`` (``dace/sdfg/state.py``) used to hand-walk ``nodes()`` and ``edges()``::

    def replace(self, name, new_name):
        for n in self.nodes():
            n.replace(name, new_name)
        for e in self.edges():
            e.data.replace(name, new_name)

A control-flow region holds symbol references that are reachable from NEITHER collection:

* a ``ConditionalBlock``'s branch CONDITIONS live in ``_branches``, not in ``nodes()``;
* a nested ``LoopRegion``'s ``init_statement`` / ``loop_condition`` / ``update_statement`` are its own
  properties, not nodes or edges.

Both classes override ``replace_dict`` to reach those; NEITHER overrides ``replace``. So the hand-walk
left four references naming the old symbol. That is a silent miscompile whenever the name later goes
away: ``TrivialLoopElimination`` substituted its eliminated iterator this way, and the leftovers dangled
the instant ``UniqueLoopIterators`` renamed the iterator, surfacing as
``SDFG.arglist() -> KeyError: '_loop_it_1'`` on polybench nussinov (see
``tests/corpus/nussinov_canonicalize_test.py``, which documents the same root from the caller's side).

The fix routes ``replace`` through ``replace_dict``, which every subclass already overrides correctly.

``replace_keys=False`` is load-bearing and is asserted below: ``LoopRegion.replace_dict`` defaults it to
True, which renames ``loop_variable``. Callers of ``replace`` substitute a VALUE, not a name --
``loop_overwrite_elimination`` passes the loop's last-iteration expression -- so renaming a loop variable
to ``N - 1`` would be nonsense.
"""
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def _region_with_hidden_references():
    """A region whose only ``i`` references are the ones ``nodes()``/``edges()`` cannot see.

    Deliberately holds NO tasklet or memlet naming ``i``: every reference here is hidden from the
    hand-walk, so a passing assert cannot be produced by the parts ``replace`` already handled.
    """
    sdfg = dace.SDFG('hidden_refs')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_symbol('i', dace.int64)

    region = ControlFlowRegion('region', sdfg=sdfg)
    sdfg.add_node(region, is_start_block=True)

    # Hidden reference 1: a branch condition, stored in ConditionalBlock._branches.
    cond = ConditionalBlock('cblock', sdfg=sdfg, parent=region)
    region.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion('branch_body', sdfg=sdfg)
    cond.add_branch(CodeBlock('i < 10'), branch)
    branch.add_state('bstate', is_start_block=True)

    # Hidden references 2-4: a nested loop's own init / condition / update.
    inner = LoopRegion('inner', 'j < i', 'j', 'j = i', 'j = j + i', sdfg=sdfg)
    region.add_node(inner)
    inner.add_state('istate', is_start_block=True)

    return sdfg, region, cond, inner


def _references(cond: ConditionalBlock, inner: LoopRegion):
    return {
        'branch_condition': cond.branches[0][0].as_string,
        'inner_init': inner.init_statement.as_string,
        'inner_condition': inner.loop_condition.as_string,
        'inner_update': inner.update_statement.as_string,
    }


def test_replace_reaches_conditional_branches_and_nested_loop_headers():
    """The core regression: all four hidden references must be substituted.

    Before the fix every one of these still read ``i``; the assert names which survived so a partial
    regression is diagnosable rather than just "not equal".
    """
    _, region, cond, inner = _region_with_hidden_references()

    region.replace('i', 'RENAMED')

    stale = {site: text for site, text in _references(cond, inner).items() if 'RENAMED' not in text}
    assert not stale, f"ControlGraphView.replace left references naming the old symbol: {stale}"


def test_replace_renames_interstate_edge_assignment_keys():
    """``replace`` is a RENAME: an assignment ``i = ...`` must become ``new_name = ...``.

    The hand-walk got this right by accident -- it called ``InterstateEdge.replace``, whose
    ``replace_keys`` defaults to True. Routing through ``ControlGraphView.replace_dict`` does NOT: that
    method's own default is False, and it forwards the flag to edges. Passing ``replace_keys=False``
    here therefore silently stopped renaming assignment keys, leaving the edge defining a symbol nobody
    reads (``otf_map_fusion`` renames ``s`` -> ``s_`` through this exact path). This pins the flag.
    """
    sdfg = dace.SDFG('iedge_keys')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_symbol('q', dace.int64)

    region = ControlFlowRegion('region', sdfg=sdfg)
    sdfg.add_node(region, is_start_block=True)
    first = region.add_state('first', is_start_block=True)
    second = region.add_state('second')
    # ``i`` appears as an assignment KEY and inside another assignment's VALUE.
    region.add_edge(first, second, dace.InterstateEdge(assignments={'i': 'q + 1', 'q': 'i * 2'}))

    region.replace('i', 'RENAMED')

    assignments = dict(list(region.edges())[0].data.assignments)
    assert 'RENAMED' in assignments, f'assignment key was not renamed: {assignments}'
    assert 'i' not in assignments, f'stale assignment key survived the rename: {assignments}'
    assert 'RENAMED' in assignments['q'], f'assignment value was not renamed: {assignments}'


def test_replace_renames_a_loops_own_loop_variable():
    """A rename reaches ``loop_variable`` too -- it is the symbol's defining occurrence.

    The hand-walk never touched it (it never looked at properties), which is the same blind spot that
    left branch conditions and nested loop headers stale. Renaming references while leaving the loop
    still declaring the OLD name would split one symbol into two.

    Callers that substitute a VALUE rather than rename must not use ``replace`` at all -- they call
    ``replace_dict(..., replace_keys=False)``; ``LoopOverwriteElimination`` is the live example.
    """
    sdfg = dace.SDFG('loopvar')
    sdfg.add_symbol('i', dace.int64)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state('body', is_start_block=True)

    loop.replace('i', 'RENAMED')

    assert loop.loop_variable == 'RENAMED', \
        f'replace left the loop declaring the old name: {loop.loop_variable!r}'
    assert 'RENAMED' in loop.loop_condition.as_string


def test_replace_is_a_noop_when_name_is_unused():
    """Over-reach guard: a symbol the region never names must leave every reference untouched."""
    _, region, cond, inner = _region_with_hidden_references()
    before = _references(cond, inner)

    region.replace('completely_unrelated', 'SOMETHING')

    assert _references(cond, inner) == before, "replace perturbed references for a symbol it never names"


if __name__ == '__main__':
    pytest.main([__file__, '-x', '-q'])
