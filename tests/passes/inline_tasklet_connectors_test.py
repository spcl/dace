# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the InlineTaskletConnectors pass. """
import numpy as np
import pytest
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


if __name__ == '__main__':
    test_elementwise_inlined_and_valid()
    test_stencil_offsets()
    test_idempotent()
    test_wcr_output_not_inlined()
    print('ok')
