# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SymbolPropagation must not let a one-sided binding escape a control-flow join.

Shape (cloudsc ``ZFAC``, cloudsc.F90:2354)::

    if cond > 0:  fac = 1.0        # interstate assignment inside the then-region
    else:         fac = src[0]     # interstate assignment inside the else-region
    out[0] = fac                   # use AFTER the join

``fac = src[0]`` contains an array access, so ``_get_in_syms``'s array-access filter drops the
key from the else branch's table rather than marking it live. ``_combine_syms`` is the meet at
the join, so if it only iterates the second table's keys, a key present on one side and absent
on the other survives with the present side's value -- ``fac`` leaves the ConditionalBlock as the
constant ``1.0`` and the else path is silently miscompiled.
"""
import copy

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.symbol_propagation import SymbolPropagation


def build_branch_join_sdfg(else_rhs: str = 'src[0]') -> dace.SDFG:
    """Two branches binding the same symbol, then a use after the join."""
    sdfg = dace.SDFG('symprop_branch_join')
    sdfg.add_array('src', [1], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_symbol('fac', dace.float64)
    sdfg.add_symbol('other', dace.float64)
    sdfg.add_symbol('cond', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    cb = ConditionalBlock('cb')
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())

    then_region = ControlFlowRegion('then_region', sdfg=sdfg)
    t0 = then_region.add_state('t0', is_start_block=True)
    t1 = then_region.add_state('t1')
    then_region.add_edge(t0, t1, dace.InterstateEdge(assignments={'fac': '1.0'}))
    cb.add_branch(CodeBlock('cond > 0'), then_region)

    else_region = ControlFlowRegion('else_region', sdfg=sdfg)
    e0 = else_region.add_state('e0', is_start_block=True)
    e1 = else_region.add_state('e1')
    else_region.add_edge(e0, e1, dace.InterstateEdge(assignments={'fac': else_rhs}))
    cb.add_branch(None, else_region)

    use = sdfg.add_state('use')
    sdfg.add_edge(cb, use, dace.InterstateEdge())
    tasklet = use.add_tasklet('emit', {}, {'o'}, 'o = fac')
    use.add_edge(tasklet, 'o', use.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


def run_else_path(sdfg: dace.SDFG, tag: str) -> float:
    """Take the else branch (``cond = 0``) and return what was written to ``out``."""
    graph = copy.deepcopy(sdfg)
    graph.name = f'symprop_{tag}'
    src = np.array([7.0], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    graph(src=src, out=out, cond=np.int32(0))
    return float(out[0])


def tasklet_sources(sdfg: dace.SDFG):
    return [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]


def test_untransformed_else_path_reads_the_array():
    """Baseline: without any pass the else path yields src[0]."""
    assert run_else_path(build_branch_join_sdfg(), 'baseline') == 7.0


def test_one_sided_binding_does_not_escape_the_join():
    """The regression: the then-branch's literal must not be folded into the post-join use."""
    sdfg = build_branch_join_sdfg()
    SymbolPropagation().apply_pass(sdfg, {})
    assert tasklet_sources(sdfg) == ['o = fac'], tasklet_sources(sdfg)
    assert run_else_path(sdfg, 'symprop') == 7.0


def test_disagreeing_plain_symbols_are_still_invalidated():
    """Control: when the else RHS has no array access its key survives the filters, so the
    existing disagreement branch of the meet already marks the symbol live. This must keep
    working -- the fix must not be the only thing doing the invalidation."""
    sdfg = build_branch_join_sdfg(else_rhs='other')
    SymbolPropagation().apply_pass(sdfg, {})
    assert tasklet_sources(sdfg) == ['o = fac'], tasklet_sources(sdfg)


def test_agreeing_branches_still_propagate():
    """Both branches bind the same literal, so folding it is correct and must still happen --
    otherwise the fix would have disabled propagation across joins entirely."""
    sdfg = build_branch_join_sdfg(else_rhs='1.0')
    SymbolPropagation().apply_pass(sdfg, {})
    assert tasklet_sources(sdfg) == ['o = 1.0'], tasklet_sources(sdfg)


if __name__ == '__main__':
    test_untransformed_else_path_reads_the_array()
    test_one_sided_binding_does_not_escape_the_join()
    test_disagreeing_plain_symbols_are_still_invalidated()
    test_agreeing_branches_still_propagate()
