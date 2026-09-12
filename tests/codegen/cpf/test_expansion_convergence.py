# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's expand-and-reselect loop stops on a node that expands into itself, and on nothing else.

The loop consumes ONE library node per state per round, so a state holding many of them needs many
rounds, and every one of those rounds is progress. Refusing on a round count derived from the
population that is being drained cannot tell the two apart: the count rises while the population
falls, so the two meet part way through any state that started with more nodes than the tolerance,
and a graph that was converging is reported as a node that expands into itself. What separates the
cases is whether the census of library nodes ever CHANGES, which is what these tests hold.
"""
import copy
import json
import os
import subprocess
import sys
from typing import Dict

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


def render_many_fills() -> Dict[str, str]:
    """CPF's C and C++ text, plus the pre-CPF codegen text, for one fresh build of the fills SDFG.

    All three go through :func:`force_renderable_expansions` on ``many_fills_sdfg`` (24
    independent fills in one state), which is what used to pick the node to expand next by
    ``node.guid`` -- a fresh ``uuid4()`` -- so the fill order, and with it the emitted
    comments/loop order, changed on every process.
    """
    from dace.codegen import codegen as dace_codegen
    from dace.codegen.cpf import cpf_lowering, dialect_for, frame_object, prepare

    cpp = render(many_fills_sdfg(), language='c++').code
    c = render(many_fills_sdfg(), language='c').code

    raw = copy.deepcopy(many_fills_sdfg())
    with cpf_lowering.dialect_scope(dialect_for('c++')):
        prepare(raw)
        objects = dace_codegen.generate_code(raw)
    codegen_cpp = frame_object(objects, raw.name).clean_code
    return {'cpf_c': c, 'cpf_cpp': cpp, 'codegen_cpp': codegen_cpp}


def test_the_same_sdfg_renders_byte_identically_across_fresh_processes():
    """The same SDFG rendered in two fresh, ``PYTHONHASHSEED=0`` interpreters must come out
    byte-identical: node GUIDs are fresh ``uuid4()`` values (dace/sdfg/graph.py), so a hash-seed
    pin alone does not make :func:`force_renderable_expansions` deterministic if it still picks
    the next node to expand by GUID -- it must pick by graph order instead.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(dace.__file__)))
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=repo_root)
    runs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, __file__, '--render-worker'],
                              cwd=repo_root,
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=300)
        assert proc.returncode == 0, f'render worker failed:\n{proc.stderr}'
        runs.append(json.loads(proc.stdout))
    first, second = runs
    for key in first:
        assert first[key] == second[key], (
            f'{key}: CPF output differs between two PYTHONHASHSEED=0 renders of the same SDFG in fresh '
            'processes -- expansion order is not deterministic')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--render-worker':
        print(json.dumps(render_many_fills()))
        sys.exit(0)
    pytest.main([__file__, '-q'])
