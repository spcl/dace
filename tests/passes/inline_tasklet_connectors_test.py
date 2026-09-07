# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the InlineTaskletConnectors pass. """
import numpy as np
import dace
from dace.sdfg import nodes as dnodes
from dace.transformation.passes.inline_tasklet_connectors import InlineTaskletConnectors

N, M = dace.symbol('N'), dace.symbol('M')


def _tasklets(sdfg):
    return [n for st in sdfg.states() for n in st.nodes() if isinstance(n, dnodes.Tasklet)]


def test_elementwise_inlined_and_valid():

    @dace.program
    def ew(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
        C[:] = A + B

    sdfg = ew.to_sdfg(simplify=True)
    changed = InlineTaskletConnectors().apply_pass(sdfg, {})
    assert changed  # at least one tasklet rewritten
    tk = _tasklets(sdfg)[0]
    body = tk.code.as_string
    # Connectors are gone from the body; arrays are referenced directly.
    assert '__in1' not in body and '__in2' not in body and '__out' not in body
    assert 'A[' in body and 'B[' in body and 'C[' in body
    # Array names are excluded from the tasklet's free symbols.
    assert {'A', 'B', 'C'} <= set(tk.ignored_symbols)
    sdfg.validate()


def test_stencil_offsets():

    @dace.program
    def stencil(A: dace.float64[N], B: dace.float64[N]):
        B[1:N - 1] = A[0:N - 2] + A[2:N]

    sdfg = stencil.to_sdfg(simplify=True)
    InlineTaskletConnectors().apply_pass(sdfg, {})
    body = _tasklets(sdfg)[0].code.as_string
    # The two distinct reads of A keep their distinct per-element offsets.
    assert 'A[__i0]' in body.replace(' ', '') or 'A[(__i0)]' in body.replace(' ', '')
    assert '__i0+2' in body.replace(' ', '')
    sdfg.validate()


def test_idempotent():

    @dace.program
    def ew(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        C[:] = A + B

    sdfg = ew.to_sdfg(simplify=True)
    first = InlineTaskletConnectors().apply_pass(sdfg, {})
    assert first
    second = InlineTaskletConnectors().apply_pass(sdfg, {})
    assert second is None  # nothing left to inline


def test_wcr_output_not_inlined():
    # A reduction's WCR output connector must be preserved (it goes through the
    # atomic resolve path), while its non-WCR input may be inlined.
    @dace.program
    def red(A: dace.float64[N], s: dace.float64[1]):
        s[0] = np.sum(A)

    sdfg = red.to_sdfg(simplify=True)
    InlineTaskletConnectors().apply_pass(sdfg, {})
    sdfg.validate()
    # Find the reducing tasklet: it must still have an out-connector referenced
    # in its body (the WCR output was not inlined).
    for st in sdfg.states():
        for n in st.nodes():
            if isinstance(n, dnodes.Tasklet):
                for e in st.out_edges(n):
                    if e.data.wcr is not None and e.src_conn:
                        assert e.src_conn in n.code.as_string


def test_shadowed_connector_not_inlined():

    # A lambda parameter sharing a connector's name denotes the parameter inside the lambda,
    # so that connector must keep its classic copy-in/out lowering.
    @dace.program
    def lamb(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a >> A[i]
                b << B[i]
                c << C[i]
                f = lambda a, b: a + b
                a = f(b, c)

    sdfg = lamb.to_sdfg(simplify=True)
    InlineTaskletConnectors().apply_pass(sdfg, {})
    tk = _tasklets(sdfg)[0]
    body = tk.code.as_string
    assert 'A[' not in body and 'B[' not in body
    assert 'a' in body and 'b' in body
    # The unshadowed connector is still inlined.
    assert 'C[' in body and 'c' not in body
    assert not ({'A', 'B'} & set(tk.ignored_symbols))
    assert 'C' in set(tk.ignored_symbols)
    sdfg.validate()


def _keyword_named_input_sdfg():
    """A writer and a reader of a length-1 transient, with the input array named ``in``."""
    sdfg = dace.SDFG('keyword_named_input')
    state = sdfg.add_state()
    sdfg.add_array('in', [1], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_transient('A', [1], dace.float64)
    read, mid, write = state.add_access('in'), state.add_access('A'), state.add_access('out')
    writer = state.add_tasklet('comp1', {'x'}, {'a'}, 'a = x + 1')
    reader = state.add_tasklet('comp2', {'a'}, {'y'}, 'y = a * 2')
    state.add_edge(read, None, writer, 'x', dace.Memlet('in[0]'))
    state.add_edge(writer, 'a', mid, None, dace.Memlet('A[0]'))
    state.add_edge(mid, None, reader, 'a', dace.Memlet('A[0]'))
    state.add_edge(reader, 'y', write, None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


def test_a_container_named_like_a_python_keyword_is_not_inlined():
    """``in`` is not a name the rewritten body could carry.

    The body is unparsed and reparsed, so inlining ``in[0]`` into it comes back a SyntaxError. The
    transient beside it stays inlinable -- only the connector reading ``in`` keeps classic form.
    """
    sdfg = _keyword_named_input_sdfg()
    InlineTaskletConnectors().apply_pass(sdfg, {})
    bodies = {tk.label: tk.code.as_string for tk in _tasklets(sdfg)}
    assert 'in[' not in bodies['comp1'], bodies['comp1']
    assert 'x' in bodies['comp1'], 'the connector reading `in` must stay classic'
    assert 'A' in bodies['comp1'], 'the transient is still inlinable'
    sdfg.validate()


def test_the_writer_and_the_reader_of_a_transient_agree():
    """A container is inlined in its reader only if its WRITER is inlined too.

    The reader names the array directly and relies on the writer's inlining to DECLARE it, so
    inlining one without the other emits a name nothing declares -- the generated code then does
    not compile. Checked on the codegen text, because that is where the declaration has to appear.
    """
    from dace.codegen import codegen

    sdfg = _keyword_named_input_sdfg()
    code = [obj for obj in codegen.generate_code(sdfg) if obj.language == 'cpp'][0].clean_code
    body = code.split('_internal(')[1].split('DACE_EXPORTED')[0]
    assert 'A' in body, body
    declares = [line for line in body.splitlines() if 'A' in line and ('double A' in line or 'A[' in line)]
    assert declares, f'the transient A is used but never declared:\n{body}'


def test_the_keyword_named_program_still_computes():
    sdfg = _keyword_named_input_sdfg()
    out = np.array([0.0])
    sdfg(**{'in': np.array([3.0]), 'out': out})
    assert out[0] == 8.0, out


if __name__ == '__main__':
    test_elementwise_inlined_and_valid()
    test_stencil_offsets()
    test_idempotent()
    test_wcr_output_not_inlined()
    test_shadowed_connector_not_inlined()
    test_a_container_named_like_a_python_keyword_is_not_inlined()
    test_the_writer_and_the_reader_of_a_transient_agree()
    test_the_keyword_named_program_still_computes()
    print('ok')
