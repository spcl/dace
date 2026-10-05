# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import os
import tempfile
import numpy as np

import dace as dp
from dace.sdfg import SDFG, dealias
from dace.memlet import Memlet


def test():
    # Externals (parameters, symbols)
    N = dp.symbol('N')

    @dp.program
    def sdfg_internal(input: dp.float32, output: dp.float32[1]):

        @dp.tasklet
        def init():
            inp << input
            out >> output
            out = inp

        for k in range(4):

            @dp.tasklet
            def do():
                inp << input
                oin << output
                out >> output
                out = oin * inp

    # Construct SDFG
    mysdfg = SDFG('outer_sdfg')
    mysdfg.add_array('A', [N, N], dp.float32)
    mysdfg.add_array('B', [N, N], dp.float32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')

    map_entry, map_exit = state.add_map('elements', [('i', '0:N'), ('j', '0:N')])
    nsdfg = state.add_nested_sdfg(sdfg_internal.to_sdfg(), {'input'}, {'output'})

    # Add edges
    state.add_memlet_path(A, map_entry, nsdfg, dst_conn='input', memlet=Memlet.simple(A, 'i,j'))
    state.add_memlet_path(nsdfg, map_exit, B, src_conn='output', memlet=Memlet.simple(B, 'i,j'))

    dealias.integrate_nested_sdfg(nsdfg.sdfg)

    N = 64

    input = dp.ndarray([N, N], dp.float32)
    output = dp.ndarray([N, N], dp.float32)
    input[:] = np.random.rand(N, N).astype(dp.float32.type)
    output[:] = dp.float32(0)

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N * N)
    assert diff <= 1e-5


def test_external_nsdfg():
    N = dp.symbol('N')
    i = dp.symbol('i')
    j = dp.symbol('j')

    @dp.program
    def sdfg_internal(input: dp.float32[N, N], output: dp.float32[N, N]):

        @dp.tasklet
        def init():
            inp << input[i, j]
            out >> output[i, j]
            out = inp

        for k in range(4):

            @dp.tasklet
            def do():
                inp << input[i, j]
                oin << output[i, j]
                out >> output[i, j]
                out = oin * inp

    # Construct SDFG
    mysdfg = SDFG('outer_sdfg')
    mysdfg.add_array('A', [N, N], dp.float32)
    mysdfg.add_array('B', [N, N], dp.float32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')

    map_entry, map_exit = state.add_map('elements', [('i', '0:N'), ('j', '0:N')])
    internal = sdfg_internal.to_sdfg()
    fd, filename = tempfile.mkstemp(suffix='.sdfg')
    internal.save(filename)
    nsdfg = state.add_nested_sdfg(None, {'input'}, {'output'},
                                  name='sdfg_internal',
                                  external_path=filename,
                                  symbol_mapping={
                                      'N': N,
                                      'i': i,
                                      'j': j
                                  })

    # Add edges
    state.add_memlet_path(A, map_entry, nsdfg, dst_conn='input', memlet=Memlet.simple(A, 'i,j'))
    state.add_memlet_path(nsdfg, map_exit, B, src_conn='output', memlet=Memlet.simple(B, 'i,j'))

    N = 64

    input = dp.ndarray([N, N], dp.float32)
    output = dp.ndarray([N, N], dp.float32)
    input[:] = np.random.rand(N, N).astype(dp.float32.type)
    output[:] = dp.float32(0)

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N * N)
    assert diff <= 1e-5

    os.close(fd)


def _make_meta_read_sdfg():
    """Nested SDFG whose branch condition is the only reader of one of its connectors."""
    sdfg = SDFG('meta_read_tester')
    _, a_desc = sdfg.add_array('A', [4], dp.float64)
    _, b_desc = sdfg.add_array('B', [4], dp.float64)
    _, cond_desc = sdfg.add_array('COND', [4], dp.bool_)
    _, out_desc = sdfg.add_array('OUT', [4], dp.float64)

    nsdfg = SDFG('nested')
    nsdfg.add_scalar('a', a_desc.dtype)
    nsdfg.add_scalar('b', b_desc.dtype)
    nsdfg.add_scalar('cond', cond_desc.dtype)
    nsdfg.add_scalar('out', out_desc.dtype)

    if_region = dp.sdfg.state.ConditionalBlock('if')
    nsdfg.add_node(if_region)
    entry_state = nsdfg.add_state('entry', is_start_block=True)
    nsdfg.add_edge(entry_state, if_region, dp.InterstateEdge())

    then_body = dp.sdfg.state.ControlFlowRegion('then_body', sdfg=nsdfg)
    a_state = then_body.add_state('true_branch', is_start_block=True)
    if_region.add_branch(dp.sdfg.state.CodeBlock('cond'), then_body)
    a_state.add_nedge(a_state.add_access('a'), a_state.add_access('out'), Memlet('out'))

    else_body = dp.sdfg.state.ControlFlowRegion('else_body', sdfg=nsdfg)
    b_state = else_body.add_state('false_branch', is_start_block=True)
    if_region.add_branch(dp.sdfg.state.CodeBlock('not (cond)'), else_body)
    b_state.add_nedge(b_state.add_access('b'), b_state.add_access('out'), Memlet('out'))

    state = sdfg.add_state()
    node = state.add_nested_sdfg(nsdfg, inputs={'a', 'b', 'cond'}, outputs={'out'})
    me, mx = state.add_map('map', dict(i='0:4'))
    state.add_memlet_path(state.add_access('A'), me, node, dst_conn='a', memlet=Memlet('A[i]'))
    state.add_memlet_path(state.add_access('B'), me, node, dst_conn='b', memlet=Memlet('B[i]'))
    state.add_memlet_path(state.add_access('COND'), me, node, dst_conn='cond', memlet=Memlet('COND[i]'))
    state.add_memlet_path(node, mx, state.add_access('OUT'), src_conn='out', memlet=Memlet('OUT[i]'))
    return sdfg, node


def test_integrate_meta_only_read():
    """A connector read only by meta code must be redirected to the parent container.

    Integration expresses narrowing with a ``View``, which only takes effect through the ``views``
    edge of an access node. A container read solely by a branch condition never gets one, so before
    the redirection the generated code did not even compile (``'cond' was not declared in this
    scope``).
    """
    sdfg, node = _make_meta_read_sdfg()
    node.integrate_into_parent()

    # The dangling view is gone and the condition now addresses the parent container
    assert 'cond' not in node.sdfg.arrays
    conditions = [c.as_string for cfg in node.sdfg.all_control_flow_regions() for c in cfg.get_meta_codeblocks()]
    assert any('COND[i]' in c for c in conditions), conditions

    sdfg.validate()

    A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    B = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    COND = np.array([True, False, True, False])
    OUT = np.zeros(4, dtype=np.float64)
    sdfg(A=A, B=B, COND=COND, OUT=OUT)
    assert np.allclose(OUT, np.where(COND, A, B))


def _iota_nsdfg() -> SDFG:
    """Nested SDFG writing ``a[k] = k`` for ``k`` in ``0:N``, with ``a`` of shape ``[N]``."""
    N = dp.symbol('N')
    nsdfg = SDFG('iota')
    nsdfg.add_symbol('N', dp.int64)
    nsdfg.add_array('a', [N], dp.float64)
    nstate = nsdfg.add_state()
    nstate.add_mapped_tasklet('iota', dict(k='0:N'), {}, 'o = k', dict(o=Memlet('a[k]')), external_edges=True)
    return nsdfg


def _iota_parent(name: str, nsdfg, external_path=None):
    """Parent writing its array ``A`` of shape ``[M]`` through ``_iota_nsdfg``, with the mapping ``N -> M``."""
    M = dp.symbol('M')
    sdfg = SDFG(name)
    sdfg.add_symbol('M', dp.int64)
    sdfg.add_array('A', [M], dp.float64)
    state = sdfg.add_state()
    node = state.add_nested_sdfg(nsdfg, {}, {'a'}, name='iota', symbol_mapping={'N': M}, external_path=external_path)
    state.add_edge(node, 'a', state.add_write('A'), None, Memlet('A[0:M]'))
    return sdfg, node


def test_symbol_mapping_folded_in_memory_and_external():
    """
    A symbol mapped to a symbol of the parent is folded into the nested SDFG when it is integrated, the same way for
    a nested SDFG given in memory and one loaded from a file.
    """
    sdfg, node = _iota_parent('iota_in_memory', _iota_nsdfg())
    assert str(node.symbol_mapping['N']) == 'M'  # Stored as given
    node.integrate_into_parent()

    fd, filename = tempfile.mkstemp(suffix='.sdfg')
    os.close(fd)
    _iota_nsdfg().save(filename)
    ext_sdfg, ext_node = _iota_parent('iota_external', None, external_path=filename)
    assert str(ext_node.symbol_mapping['N']) == 'M'
    list(ext_sdfg.all_control_flow_regions(recursive=True, load_ext=True))  # Loads and integrates the file
    os.remove(filename)

    for n in (node, ext_node):
        assert {k: str(v) for k, v in n.symbol_mapping.items()} == {'M': 'M'}
        assert 'N' not in n.sdfg.symbols
        assert str(n.sdfg.arrays['a'].shape[0]) == 'M'

    for s in (sdfg, ext_sdfg):
        A = np.zeros(5)
        s(A=A, M=5)
        assert np.allclose(A, np.arange(5))


if __name__ == "__main__":
    test()
    test_external_nsdfg()
    test_symbol_mapping_folded_in_memory_and_external()
    test_integrate_meta_only_read()
