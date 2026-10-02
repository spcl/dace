# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the InlineTaskletConnectors pass. """
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.inline_tasklet_connectors import InlineTaskletConnectors

N, M = dace.symbol('N'), dace.symbol('M')


def tasklets(sdfg):
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, nodes.Tasklet)]


def test_elementwise_inlined_and_valid():

    @dace.program
    def ew(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
        C[:] = A + B

    sdfg = ew.to_sdfg(simplify=True)
    assert InlineTaskletConnectors().apply_pass(sdfg, {})
    tasklet = tasklets(sdfg)[0]
    body = tasklet.code.as_string
    assert '__in1' not in body and '__in2' not in body and '__out' not in body
    assert 'A[' in body and 'B[' in body and 'C[' in body
    assert {'A', 'B', 'C'} <= set(tasklet.ignored_symbols)
    sdfg.validate()


def test_stencil_keeps_the_offset_of_each_read():

    @dace.program
    def stencil(A: dace.float64[N], B: dace.float64[N]):
        B[1:N - 1] = A[0:N - 2] + A[2:N]

    sdfg = stencil.to_sdfg(simplify=True)
    InlineTaskletConnectors().apply_pass(sdfg, {})
    body = tasklets(sdfg)[0].code.as_string.replace(' ', '')
    assert 'A[__i0]' in body or 'A[(__i0)]' in body
    assert '__i0+2' in body
    sdfg.validate()


def test_a_second_application_changes_nothing():

    @dace.program
    def ew(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        C[:] = A + B

    sdfg = ew.to_sdfg(simplify=True)
    assert InlineTaskletConnectors().apply_pass(sdfg, {})
    assert InlineTaskletConnectors().apply_pass(sdfg, {}) is None


def test_a_write_conflict_output_keeps_its_connector():
    sdfg = dace.SDFG('conflict')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('s', [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('acc', {'a'}, {'o'}, 'o = a')
    state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet,
                          exit_node,
                          state.add_write('s'),
                          src_conn='o',
                          memlet=dace.Memlet('s[0]', wcr='lambda x, y: x + y'))
    InlineTaskletConnectors().apply_pass(sdfg, {})
    sdfg.validate()
    body = tasklet.code.as_string
    assert 'o' in body.split('=')[0] and 'A[i]' in body, body


def inout_sdfg(read: str, written: str) -> dace.SDFG:
    """One tasklet, ``x = x + 1``, whose input and output connector share a name."""
    sdfg = dace.SDFG('inout')
    sdfg.add_array('A', [4], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('inc', {'x'}, {'x'}, 'x = x + 1')
    state.add_edge(state.add_read('A'), None, tasklet, 'x', dace.Memlet(f'A[{read}]'))
    state.add_edge(tasklet, 'x', state.add_write('A'), None, dace.Memlet(f'A[{written}]'))
    return sdfg


INOUT_CASES = [('1', '1', True), ('1', '2', False)]


@pytest.mark.parametrize('read, written, inlined', INOUT_CASES)
def test_an_inout_connector_is_inlined_only_when_both_sides_are_the_same_element(read, written, inlined):
    sdfg = inout_sdfg(read, written)
    InlineTaskletConnectors().apply_pass(sdfg, {})
    body = tasklets(sdfg)[0].code.as_string
    assert ('A[1]' in body) is inlined, body


if __name__ == '__main__':
    test_elementwise_inlined_and_valid()
    test_stencil_keeps_the_offset_of_each_read()
    test_a_second_application_changes_nothing()
    test_a_write_conflict_output_keeps_its_connector()
    for case in INOUT_CASES:
        test_an_inout_connector_is_inlined_only_when_both_sides_are_the_same_element(*case)
