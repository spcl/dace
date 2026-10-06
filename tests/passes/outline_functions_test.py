# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests planning and applying the outlining of control flow blocks into code generator function regions. """

import numpy as np

import dace
from dace.sdfg.state import CodeGeneratorFunctionRegion, LoopRegion
from dace.transformation.passes.outlining import (CostModel, OutlineFunctions, OutliningPlanner,
                                                  assign_translation_units, maximal_chains)

N = dace.symbol('N')


def _loop(sdfg_or_region, name: str, src: str, dst: str, expr: str, var: str = 'i') -> LoopRegion:
    """ Adds a loop computing ``dst[var] = expr(src[var])`` to an SDFG or region (unconnected). """
    loop = LoopRegion(name, f'{var} < N', var, f'{var} = 0', f'{var} = {var} + 1')
    sdfg_or_region.add_node(loop, is_start_block=sdfg_or_region.number_of_nodes() == 0)
    body = loop.add_state(f'{name}_body', is_start_block=True)
    t = body.add_tasklet(f'{name}_compute', {'a'}, {'b'}, f'b = {expr}')
    body.add_edge(body.add_read(src), None, t, 'a', dace.Memlet(f'{src}[{var}]'))
    body.add_edge(t, 'b', body.add_write(dst), None, dace.Memlet(f'{dst}[{var}]'))
    return loop


def _chain_of_loops(name: str, count: int) -> dace.SDFG:
    """ ``count`` loops in sequence, each adding one to ``A`` in place. """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    previous = None
    for k in range(count):
        loop = _loop(sdfg, f'loop_{k}', 'A', 'A', 'a + 1')
        if previous is not None:
            sdfg.add_edge(previous, loop, dace.InterstateEdge())
        previous = loop
    return sdfg


def test_cost_model():
    sdfg = _chain_of_loops('outline_cost_model', 3)
    costs = CostModel()
    loop = next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))
    # A loop costs its header and latch on top of its body (one state with one statement)
    assert costs.block(loop).basic_blocks == 3 + 1
    assert costs.block(loop).statements == 1
    assert costs.block(sdfg).basic_blocks == 3 * 4


def test_maximal_chains():
    sdfg = _chain_of_loops('outline_chains', 3)
    chains = maximal_chains(sdfg)
    assert len(chains) == 1 and len(chains[0]) == 3

    # A conditional edge ends a chain
    sdfg = _chain_of_loops('outline_chains_cond', 3)
    first, second = [e for e in sdfg.edges()]
    first.data.condition = dace.properties.CodeBlock('N > 1')
    chains = maximal_chains(sdfg)
    assert sorted(len(c) for c in chains) == [1, 2]


def _scalar_chain() -> dace.SDFG:
    """ Four loops; a scalar ``s`` is written by the first state and read by the third block. """
    sdfg = dace.SDFG('outline_scalar_chain')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    t = init.add_tasklet('set', {}, {'o'}, 'o = 2')
    init.add_edge(t, 'o', init.add_write('s'), None, dace.Memlet('s'))
    loop0 = _loop(sdfg, 'loop_0', 'A', 'A', 'a + 1')
    loop1 = LoopRegion('loop_1', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop1)
    body = loop1.add_state('loop_1_body', is_start_block=True)
    t = body.add_tasklet('scale', {'a', 'f'}, {'b'}, 'b = a * f')
    body.add_edge(body.add_read('A'), None, t, 'a', dace.Memlet('A[i]'))
    body.add_edge(body.add_read('s'), None, t, 'f', dace.Memlet('s'))
    body.add_edge(t, 'b', body.add_write('A'), None, dace.Memlet('A[i]'))
    loop2 = _loop(sdfg, 'loop_2', 'A', 'A', 'a - 1')
    sdfg.add_edge(init, loop0, dace.InterstateEdge())
    sdfg.add_edge(loop0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, loop2, dace.InterstateEdge())
    return sdfg


def test_cut_weights():
    sdfg = _scalar_chain()
    chain = maximal_chains(sdfg)[0]
    weights = OutliningPlanner(max_basic_blocks=100).cut_weights(chain)
    # The scalar is live across the first two cuts, not across the last one
    assert weights == [4.0, 4.0, 0.0]


def test_cut_weights_symbol():
    """ A symbol assigned before a cut and read after it weighs on that cut only. """
    sdfg = _chain_of_loops('outline_symbol_weights', 3)
    first_edge = sdfg.out_edges(maximal_chains(sdfg)[0][0])[0]
    second_loop = first_edge.dst
    first_edge.data.assignments['k'] = '3'
    # The second loop reads ``k``
    second_loop.nodes()[0].nodes()[1].code = dace.properties.CodeBlock('b = a + k')
    weights = OutliningPlanner(max_basic_blocks=100).cut_weights(maximal_chains(sdfg)[0])
    # The assignment stays with the caller when cutting at its edge (cut 0), but is inside a function otherwise
    assert weights[0] == 0.0


def test_planner_avoids_live_scalar():
    sdfg = _scalar_chain()
    costs = CostModel()
    chain = maximal_chains(sdfg)[0]
    sizes = [costs.block(b).basic_blocks for b in chain]
    # A budget that fits the first three blocks but not all four: the only free cut is the last one
    budget = sum(sizes[:3])
    plans = OutliningPlanner(max_basic_blocks=budget).plan(sdfg)
    assert [[b.label for b in p.blocks] for p in plans] == [[b.label for b in chain[:3]], [chain[3].label]]


def test_planner_divides_large_loop_body():
    sdfg = dace.SDFG('outline_large_body')
    sdfg.add_array('A', [N, N], dace.float64)
    outer = LoopRegion('outer', 'j < N', 'j', 'j = 0', 'j = j + 1')
    sdfg.add_node(outer, is_start_block=True)
    previous = None
    for k in range(4):
        loop = LoopRegion(f'inner_{k}', 'i < N', 'i', 'i = 0', 'i = i + 1')
        outer.add_node(loop, is_start_block=(k == 0))
        body = loop.add_state(f'inner_{k}_body', is_start_block=True)
        t = body.add_tasklet(f'inner_{k}_compute', {'a'}, {'b'}, 'b = a + 1')
        body.add_edge(body.add_read('A'), None, t, 'a', dace.Memlet('A[j, i]'))
        body.add_edge(t, 'b', body.add_write('A'), None, dace.Memlet('A[j, i]'))
        if previous is not None:
            outer.add_edge(previous, loop, dace.InterstateEdge())
        previous = loop
    inner_size = CostModel().block(previous).basic_blocks
    plans = OutliningPlanner(max_basic_blocks=2 * inner_size).plan(sdfg)
    # The outer loop is too large: its body is divided into two functions of two loops each
    assert len(plans) == 2
    assert all(len(p.blocks) == 2 and p.blocks[0].parent_graph is outer for p in plans)


def test_translation_units_balanced():
    sdfg = _chain_of_loops('outline_units', 8)
    size = CostModel().block(next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))).basic_blocks
    plans = OutliningPlanner(max_basic_blocks=size).plan(sdfg)
    assert len(plans) == 8
    assign_translation_units(plans, 3)
    units = [p.translation_unit for p in plans]
    assert sorted(units.count(u) for u in set(units)) == [2, 3, 3]


def test_statement_budget():
    """ The statement budget limits functions as well as the basic block budget. """
    sdfg = _chain_of_loops('outline_statement_budget', 6)
    plans = OutliningPlanner(max_basic_blocks=10**6, max_statements=2).plan(sdfg)
    assert [len(p.blocks) for p in plans] == [2, 2, 2]


def test_translation_units_by_size():
    """ Small programs use fewer translation units, since each one parses the common preamble. """
    sdfg = _chain_of_loops('outline_units_by_size', 8)
    plans = OutliningPlanner(max_basic_blocks=10**6, max_statements=1).plan(sdfg)
    assign_translation_units(plans, 8, min_unit_statements=3)
    assert len({p.translation_unit for p in plans}) == 2


def test_planner_keeps_opaque_code_in_caller():
    """ A block calling opaque code (e.g., a callback) stays in its caller, and the functions form around it. """
    sdfg = _chain_of_loops('outline_opaque', 4)
    chain = maximal_chains(sdfg)[0]
    opaque = sdfg.add_state_after(chain[1], 'opaque')
    opaque.add_tasklet('external', {}, {}, 'external_call()', language=dace.Language.CPP, side_effects=True)
    plans = OutliningPlanner(max_basic_blocks=10**6, max_statements=10**6).plan_region(sdfg)
    assert [[b.label for b in p.blocks] for p in plans] == [['loop_0', 'loop_1'], ['loop_2', 'loop_3']]


def test_outline_functions_pass():
    sdfg = _chain_of_loops('outline_functions_pass', 6)
    size = CostModel().block(next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))).basic_blocks
    result = OutlineFunctions(max_basic_blocks=2 * size, min_basic_blocks=0, translation_units=2,
                              min_unit_statements=0).apply_pass(sdfg, {})
    assert result == 3
    regions = [b for b in sdfg.nodes() if isinstance(b, CodeGeneratorFunctionRegion)]
    assert len(regions) == 3
    assert {r.translation_unit for r in regions} == {'unit_0', 'unit_1'}
    sdfg.validate()

    sources = [o for o in sdfg.generate_code() if o.language == 'cpp' and o.linkable]
    assert len(sources) == 3  # The frame code and two units
    A = np.random.rand(10)
    ref = A + 6
    sdfg(A=A, N=10)
    assert np.allclose(A, ref)


def test_outline_functions_in_codegen():
    """ With the configuration entry set, code generation outlines the functions by itself. """
    sdfg = _chain_of_loops('outline_functions_codegen', 6)
    size = CostModel().block(next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))).basic_blocks
    with dace.config.set_temporary('compiler', 'outlining', 'enabled', value=True), \
            dace.config.set_temporary('compiler', 'outlining', 'max_basic_blocks', value=2 * size), \
            dace.config.set_temporary('compiler', 'outlining', 'min_basic_blocks', value=0), \
            dace.config.set_temporary('compiler', 'outlining', 'translation_units', value=2), \
            dace.config.set_temporary('compiler', 'outlining', 'min_unit_statements', value=0):
        sources = [o for o in sdfg.generate_code() if o.language == 'cpp' and o.linkable]
        assert len(sources) == 3
        A = np.random.rand(10)
        ref = A + 6
        sdfg(A=A, N=10)
    assert np.allclose(A, ref)


def test_outline_functions_small_sdfg_unchanged():
    sdfg = _chain_of_loops('outline_functions_small', 3)
    assert OutlineFunctions().apply_pass(sdfg, {}) is None
    assert not any(isinstance(b, CodeGeneratorFunctionRegion) for b in sdfg.nodes())


if __name__ == '__main__':
    test_cost_model()
    test_maximal_chains()
    test_cut_weights()
    test_cut_weights_symbol()
    test_planner_avoids_live_scalar()
    test_planner_divides_large_loop_body()
    test_translation_units_balanced()
    test_statement_budget()
    test_translation_units_by_size()
    test_planner_keeps_opaque_code_in_caller()
    test_outline_functions_pass()
    test_outline_functions_in_codegen()
    test_outline_functions_small_sdfg_unchanged()
