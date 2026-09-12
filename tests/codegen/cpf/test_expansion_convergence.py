# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's expand-and-reselect loop stops on a node that expands into itself, and on nothing else.

The loop consumes ONE library node per state per round, so a state holding many of them needs many
rounds, and every one of those rounds is progress. Refusing on a round count derived from the
population that is being drained cannot tell the two apart: the count rises while the population
falls, so the two meet part way through any state that started with more nodes than the tolerance,
and a graph that was converging is reported as a node that expands into itself. What separates the
cases is whether the census of library nodes ever CHANGES, which is what these tests hold.
"""
import numpy as np
import pytest

import dace
from dace import nodes
from dace.codegen.cpf import MAX_EXPANSION_STALLED_ROUNDS, force_renderable_expansions, render

from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone

#: Fills in one state, comfortably past the stall tolerance so the old round budget is exceeded.
FILL_COUNT = MAX_EXPANSION_STALLED_ROUNDS + 8
#: Elements per filled row. Small enough that every fill stays a single call.
ROW = 4


def many_fills_sdfg() -> dace.SDFG:
    """One state holding :data:`FILL_COUNT` independent fills, one per row of ``out``.

    Built directly rather than through ``@dace.program`` because the frontend fuses adjacent
    constant assignments into one map, and the defect needs the nodes to stay separate and in the
    SAME state -- the loop's budget was per state, not per SDFG.
    """
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    sdfg = dace.SDFG('many_fills')
    sdfg.add_array('out', [FILL_COUNT, ROW], dace.float64)
    state = sdfg.add_state('fills')
    write = state.add_write('out')
    for row in range(FILL_COUNT):
        fill = FillLibraryNode(f'fill_{row}', value=float(row))
        state.add_node(fill)
        state.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, write, None, dace.Memlet(f'out[{row}, 0:{ROW}]'))
    return sdfg


def test_a_state_holding_more_library_nodes_than_the_tolerance_is_expanded_not_counted_out():
    """Every fill is expanded, and the loop does not mistake a long drain for a cycle."""
    sdfg = many_fills_sdfg()
    force_renderable_expansions(sdfg)
    left = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.LibraryNode)]
    assert not left, f'{len(left)} library nodes survived expansion: {sorted(type(n).__name__ for n in left)}'


def test_a_state_holding_more_library_nodes_than_the_tolerance_renders_and_runs():
    """The same graph renders to a standalone unit that writes every row."""
    sdfg = many_fills_sdfg()
    rendering = render(sdfg, language='c++')
    assert_standalone(rendering.code, sdfg.name)
    out = np.zeros((FILL_COUNT, ROW), dtype=np.float64)
    call_standalone(build_standalone(rendering.code, sdfg.name), sdfg, {'out': out})
    expected = np.repeat(np.arange(FILL_COUNT, dtype=np.float64)[:, None], ROW, axis=1)
    assert np.array_equal(out, expected), f'fills did not reach every row:\n{out}'


def self_reproducing_expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
    """A fill expansion that lowers into another fill, which is the shape the loop must refuse."""
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    edge = next(e for e in parent_state.out_edges(node) if e.src_conn == FillLibraryNode.OUTPUT_CONNECTOR_NAME)
    outer = parent_state.sdfg.arrays[edge.data.data]

    sdfg = dace.SDFG(f'{node.label}_sdfg')
    sdfg.add_array(FillLibraryNode.OUTPUT_CONNECTOR_NAME, [ROW], outer.dtype, outer.storage)
    state = sdfg.add_state(f'{node.label}_state')
    inner = FillLibraryNode(f'{node.label}_again', value=node.value)
    state.add_node(inner)
    state.add_edge(inner, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write(FillLibraryNode.OUTPUT_CONNECTOR_NAME),
                   None, dace.Memlet(f'{FillLibraryNode.OUTPUT_CONNECTOR_NAME}[0:{ROW}]'))
    return sdfg


def test_a_library_node_that_expands_into_itself_is_reported_by_what_was_observed(monkeypatch):
    """The loop refuses, and the message reports the census it saw rather than asserting a cause.

    The refusal this replaced named "a library node appears to expand into itself" on every graph it
    gave up on, including graphs that were converging, and that sent readers hunting a cycle that did
    not exist. So the message must carry the stalled census and its length, and must not state a
    cause the loop has not established.

    The cycle is injected into an EXISTING node's expansion rather than registered as a new library
    node class: a class declared here would stay in ``LibraryNode.__subclasses__()`` for the rest of
    the session and reach every other test that walks it.
    """
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    sdfg = dace.SDFG('self_expanding')
    sdfg.add_array('out', [ROW], dace.float64)
    state = sdfg.add_state('cycle')
    node = FillLibraryNode('cycler', value=1.0)
    state.add_node(node)
    state.add_edge(node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write('out'), None,
                   dace.Memlet(f'out[0:{ROW}]'))

    for implementation in FillLibraryNode.implementations.values():
        monkeypatch.setattr(implementation, 'expansion', staticmethod(self_reproducing_expansion), raising=False)

    with pytest.raises(NotImplementedError) as caught:
        force_renderable_expansions(sdfg)
    message = str(caught.value)
    assert 'FillLibraryNode x1' in message, f'the refusal does not report the census it saw: {message}'
    assert f'{MAX_EXPANSION_STALLED_ROUNDS + 1} consecutive rounds' in message, (
        f'the refusal does not say how long the census stood: {message}')
    assert 'appears to expand into itself' not in message, (
        f'the refusal asserts a cause it has not established: {message}')


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
