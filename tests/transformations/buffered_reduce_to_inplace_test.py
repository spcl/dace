# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``BufferedReduceToInplace`` — pinning the contract that a
``Map -> buffer -> Reduce`` pattern collapses into a sequential Map that
accumulates into the destination directly, with the buffer + Reduce
node removed and *no* WCR edges introduced.
"""
import ctypes

import dace
import numpy as np

from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg import nodes as nd
from dace.transformation.dataflow.buffered_reduce_to_inplace import BufferedReduceToInplace

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass


def _build_sum_sdfg(n: int) -> dace.SDFG:
    """``out = sum(a * 2)`` via Map(buffer write) -> Reduce."""
    sdfg = dace.SDFG(f"sum_buffered_{n}")
    sdfg.add_array("a", [n], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("buf", [n], dace.float64)
    state = sdfg.add_state("compute")

    a_in = state.add_read("a")
    buf_w = state.add_access("buf")
    out_w = state.add_access("out")

    state.add_mapped_tasklet(
        "doubler",
        dict(i=f"0:{n}"),
        {"_in": dace.Memlet("a[i]")},
        "_out = _in * 2.0",
        {"_out": dace.Memlet("buf[i]")},
        external_edges=True,
        input_nodes={"a": a_in},
        output_nodes={"buf": buf_w},
    )

    red = state.add_reduce("lambda a, b: a + b", None, 0.0)
    state.add_edge(buf_w, None, red, None, dace.Memlet(f"buf[0:{n}]"))
    state.add_edge(red, None, out_w, None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg


def _has_wcr_edges(sdfg: dace.SDFG) -> bool:
    """True iff any memlet in the SDFG carries a WCR lambda."""
    for state in sdfg.states():
        for e in state.edges():
            if e.data is not None and getattr(e.data, "wcr", None) is not None:
                return True
    return False


def test_pattern_matches():
    """The transform's pattern should match a freshly built Map+Reduce SDFG."""
    sdfg = _build_sum_sdfg(8)
    assert sdfg.apply_transformations(BufferedReduceToInplace) == 1


def test_buffer_and_reduce_eliminated():
    """After applying, the buffer descriptor and the Reduce node are gone
    and the resulting SDFG has zero WCR edges."""
    sdfg = _build_sum_sdfg(8)
    sdfg.apply_transformations(BufferedReduceToInplace)
    state = next(iter(sdfg.states()))
    assert "buf" not in sdfg.arrays, "buffer descriptor should be removed"
    assert not any(isinstance(n, Reduce) for n in state.nodes()), "Reduce node should be removed"
    assert not _has_wcr_edges(sdfg), "transform must not introduce any WCR edges"


def test_numerical_sum():
    """End-to-end: sum reduction with real values matches numpy."""
    n = 32
    sdfg = _build_sum_sdfg(n)
    sdfg.apply_transformations(BufferedReduceToInplace)
    sdfg.validate()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n, dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    sdfg(a=a, out=out)
    np.testing.assert_allclose(out[0], (a * 2.0).sum())


def test_numerical_max():
    """End-to-end: max reduction (function-form WCR) matches numpy."""
    n = 24
    sdfg = dace.SDFG(f"max_buffered_{n}")
    sdfg.add_array("a", [n], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("buf", [n], dace.float64)
    state = sdfg.add_state("compute")
    state.add_mapped_tasklet(
        "ident",
        dict(i=f"0:{n}"),
        {"_in": dace.Memlet("a[i]")},
        "_out = _in",
        {"_out": dace.Memlet("buf[i]")},
        external_edges=True,
    )
    buf_node = next(n for n in state.nodes() if isinstance(n, nd.AccessNode) and n.data == "buf")
    out_w = state.add_access("out")
    red = state.add_reduce("lambda a, b: max(a, b)", None, np.finfo(np.float64).min)
    state.add_edge(buf_node, None, red, None, dace.Memlet(f"buf[0:{n}]"))
    state.add_edge(red, None, out_w, None, dace.Memlet("out[0]"))
    sdfg.validate()

    sdfg.apply_transformations(BufferedReduceToInplace)
    sdfg.validate()
    assert not _has_wcr_edges(sdfg)
    rng = np.random.default_rng(1)
    a = rng.standard_normal(n, dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    sdfg(a=a, out=out)
    np.testing.assert_allclose(out[0], a.max())


def test_does_not_match_without_identity():
    """Without an identity on the Reduce, the transform must refuse to
    fire — there is no safe value to seed the destination with."""
    n = 4
    sdfg = dace.SDFG("sum_no_identity")
    sdfg.add_array("a", [n], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("buf", [n], dace.float64)
    state = sdfg.add_state("compute")
    state.add_mapped_tasklet(
        "ident",
        dict(i=f"0:{n}"),
        {"_in": dace.Memlet("a[i]")},
        "_out = _in",
        {"_out": dace.Memlet("buf[i]")},
        external_edges=True,
    )
    buf_node = next(n for n in state.nodes() if isinstance(n, nd.AccessNode) and n.data == "buf")
    out_w = state.add_access("out")
    red = state.add_reduce("lambda a, b: a + b", None, identity=None)
    state.add_edge(buf_node, None, red, None, dace.Memlet(f"buf[0:{n}]"))
    state.add_edge(red, None, out_w, None, dace.Memlet("out[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(BufferedReduceToInplace) == 0
