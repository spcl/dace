# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Copy shapes CPF renders only once ``InsertExplicitCopies`` gives them a node of their own.

A copy edge left implicit reaches ``cpu.py`` as ``dace::CopyND``, which a self-contained unit
cannot hold, so the render refuses it. Each kernel here is a copy shape the pass once declined. It
is rendered in both dialects, built in an empty directory, and its numbers are checked against
numpy and against the runtime build of the same SDFG.
"""
import copy
from typing import Dict

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render as render_sdfg
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

from tests.codegen.cpf.conftest import assert_matches, assert_standalone, build_standalone, call_standalone

LANGUAGES = ('c', 'c++')


def named(sdfg: dace.SDFG, language: str) -> dace.SDFG:
    """``sdfg`` renamed per dialect, so the two builds never share an entry point or a cache folder."""
    sdfg.name = f"{sdfg.name}_{'c' if language == 'c' else 'cpp'}"
    return sdfg


def lifted_copies(sdfg: dace.SDFG) -> int:
    """Number of copy nodes ``InsertExplicitCopies`` inserts into a copy of ``sdfg``."""
    lowered = copy.deepcopy(sdfg)
    InsertExplicitCopies().apply_pass(lowered, {})
    lowered.validate()
    return sum(1 for node, _ in lowered.all_nodes_recursive() if isinstance(node, CopyLibraryNode))


def render_and_run(sdfg: dace.SDFG, arguments: Dict[str, np.ndarray],
                   language: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Render ``sdfg`` in ``language`` and run the rendering and the runtime build on separate copies.

    :param sdfg: the SDFG to render and run.
    :param arguments: name -> array for every argument; never written.
    :param language: the CPF dialect.
    :returns: ``{'cpf': ..., 'runtime': ...}``, each the arrays after its run.
    """
    rendering = render_sdfg(sdfg, language=language)
    assert 'CopyND' not in rendering.code, 'a copy reached dace::CopyND'
    assert_standalone(rendering.code, sdfg.name, language=language)
    cpf_arguments = {name: np.copy(value) for name, value in arguments.items()}
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, cpf_arguments)
    runtime_arguments = {name: np.copy(value) for name, value in arguments.items()}
    sdfg(**runtime_arguments)
    return {'cpf': cpf_arguments, 'runtime': runtime_arguments}


def converting_stage_in_sdfg() -> dace.SDFG:
    """Each map iteration stages a ``float32`` row into a ``float64`` transient and sums two of its
    elements: the stage-in copy converts."""
    sdfg = dace.SDFG('converting_stage_in')
    sdfg.add_array('A', [8, 4], dace.float32)
    sdfg.add_array('out', [8], dace.float64)
    sdfg.add_transient('row', [4], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': '0:8'})
    row = state.add_access('row')
    state.add_memlet_path(state.add_read('A'), entry, row, memlet=dace.Memlet('A[i, 0:4]'))
    tasklet = state.add_tasklet('ends', {'r'}, {'o'}, 'o = r[0] + r[3]')
    state.add_edge(row, None, tasklet, 'r', dace.Memlet('row[0:4]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('out'), src_conn='o', memlet=dace.Memlet('out[i]'))
    return sdfg


def overlapping_copies_sdfg() -> dace.SDFG:
    """``out[0:8] = a`` then ``out[4:12] = b``, as two copy edges into one access node."""
    sdfg = dace.SDFG('overlapping_copies')
    sdfg.add_array('a', [8], dace.float64)
    sdfg.add_array('b', [8], dace.float64)
    sdfg.add_array('out', [12], dace.float64)
    state = sdfg.add_state()
    out = state.add_write('out')
    state.add_edge(state.add_read('a'), None, out, None, dace.Memlet('out[0:8]'))
    state.add_edge(state.add_read('b'), None, out, None, dace.Memlet('out[4:12]'))
    return sdfg


def superseded_copy_sdfg() -> dace.SDFG:
    """The npbench ``vadv`` shape: ``dcol[:] = rhs`` then ``dcol[:] = 2 * rhs``, where the second
    write is a map into the same access node and nothing but emission order puts it last."""
    sdfg = dace.SDFG('superseded_copy')
    sdfg.add_array('rhs', [16], dace.float64)
    sdfg.add_array('dcol', [16], dace.float64)
    state = sdfg.add_state()
    rhs, dcol = state.add_read('rhs'), state.add_write('dcol')
    state.add_edge(rhs, None, dcol, None, dace.Memlet('dcol[0:16]'))
    state.add_mapped_tasklet('supersede', {'i': '0:16'}, {'x': dace.Memlet('rhs[i]')},
                             'y = 2.0 * x', {'y': dace.Memlet('dcol[i]')},
                             input_nodes={'rhs': rhs},
                             output_nodes={'dcol': dcol},
                             external_edges=True)
    return sdfg


@pytest.mark.parametrize('language', LANGUAGES)
def test_a_converting_stage_in_copy_renders_as_an_explicit_cast(language):
    """A copy into a map whose inner transient has another dtype is a cast the copy node can carry."""
    sdfg = named(converting_stage_in_sdfg(), language)
    assert lifted_copies(sdfg) == 1, 'the converting stage-in copy was not lowered to a copy node'

    A = np.random.default_rng(7).random((8, 4), dtype=np.float32)
    runs = render_and_run(sdfg, {'A': A, 'out': np.zeros(8)}, language)
    expected = {'out': A[:, 0].astype(np.float64) + A[:, 3].astype(np.float64)}
    assert_matches(expected, {'out': runs['cpf']['out']}, sdfg.name)
    assert_matches(expected, {'out': runs['runtime']['out']}, f'{sdfg.name}/runtime')


@pytest.mark.parametrize('language', LANGUAGES)
def test_overlapping_copies_into_one_node_keep_their_write_order(language):
    """The later copy wins the shared region, as in numpy: ``out[4:8]`` holds ``b``, not ``a``."""
    sdfg = named(overlapping_copies_sdfg(), language)
    assert lifted_copies(sdfg) == 2, 'the competing copies were not both lowered to copy nodes'

    a, b = np.arange(8, dtype=np.float64) + 1.0, np.arange(8, dtype=np.float64) + 101.0
    runs = render_and_run(sdfg, {'a': a, 'b': b, 'out': np.zeros(12)}, language)
    expected = np.zeros(12)
    expected[0:8] = a
    expected[4:12] = b
    assert_matches({'out': expected}, {'out': runs['cpf']['out']}, sdfg.name)
    assert_matches({'out': expected}, {'out': runs['runtime']['out']}, f'{sdfg.name}/runtime')


@pytest.mark.parametrize('language', LANGUAGES)
def test_a_copy_superseded_by_a_later_map_stays_superseded(language):
    """Flipping the two writes leaves ``dcol == rhs``; the program's order leaves ``dcol == 2 * rhs``."""
    sdfg = named(superseded_copy_sdfg(), language)
    assert lifted_copies(sdfg) == 1, 'the superseded copy was not lowered to a copy node'

    rhs = np.arange(16, dtype=np.float64) + 1.0
    runs = render_and_run(sdfg, {'rhs': rhs, 'dcol': np.zeros(16)}, language)
    assert_matches({'dcol': 2.0 * rhs}, {'dcol': runs['cpf']['dcol']}, sdfg.name)
    assert_matches({'dcol': 2.0 * rhs}, {'dcol': runs['runtime']['dcol']}, f'{sdfg.name}/runtime')
