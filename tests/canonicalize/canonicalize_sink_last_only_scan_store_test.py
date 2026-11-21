# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`SinkLastOnlyScanStore`: a scan read only at its last element loses its dead stores and keeps
its summation order."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.sink_last_only_scan_store import SinkLastOnlyScanStore

N = dace.symbol("N")


def _apply(sdfg) -> int:
    res = Pipeline([SinkLastOnlyScanStore()]).apply_pass(sdfg, {})
    return (res or {}).get("SinkLastOnlyScanStore", 0) or 0


def _num_loops(sdfg) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))


@dace.program
def _two_scans_read_at_last(a: dace.float64[N], b: dace.float64[N], out: dace.float64[2]):
    scan_aa = np.zeros(N, dtype=np.float64)
    scan_ab = np.zeros(N, dtype=np.float64)
    acc_aa = 0.0
    acc_ab = 0.0
    for i in range(N):
        acc_aa = acc_aa + a[i] * a[i]
        acc_ab = acc_ab + a[i] * b[i]
        scan_aa[i] = acc_aa
        scan_ab[i] = acc_ab
    out[0] = scan_aa[N - 1]
    out[1] = scan_ab[N - 1]


@dace.program
def _scan_read_at_middle(a: dace.float64[N], out: dace.float64[2]):
    scan = np.zeros(N, dtype=np.float64)
    acc = 0.0
    for i in range(N):
        acc = acc + a[i]
        scan[i] = acc
    out[0] = scan[N - 1]
    out[1] = scan[N // 2]


@dace.program
def _scan_read_in_loop(a: dace.float64[N], out: dace.float64[N]):
    scan = np.zeros(N, dtype=np.float64)
    acc = 0.0
    for i in range(N):
        acc = acc + a[i]
        scan[i] = acc
        out[i] = scan[i] * 2.0
    out[0] = scan[N - 1]


@dace.program
def _acc_written_after_store(a: dace.float64[N], out: dace.float64[1]):
    scan = np.zeros(N, dtype=np.float64)
    acc = 0.0
    for i in range(N):
        acc = acc + a[i]
        scan[i] = acc
        acc = acc * 0.5
    out[0] = scan[N - 1]


def test_scans_read_at_last_keep_one_store_and_the_sequential_order():
    sdfg = _two_scans_read_at_last.to_sdfg(simplify=True)
    assert _apply(sdfg) == 2
    sdfg.validate()
    sdfg = _two_scans_read_at_last.to_sdfg(simplify=True)
    canonicalize(sdfg)
    assert _num_loops(sdfg) == 1
    assert not any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())
    a, b, out = np.random.rand(1000), np.random.rand(1000), np.zeros(2)
    sdfg(a=a, b=b, out=out, N=1000)
    # Bit-identical to the sequential cumsum: the summation order is kept
    assert out[0] == np.cumsum(a * a)[-1] and out[1] == np.cumsum(a * b)[-1]


def test_scan_read_at_another_index_is_refused():
    assert _apply(_scan_read_at_middle.to_sdfg(simplify=True)) == 0


def test_scan_read_in_the_loop_is_refused():
    assert _apply(_scan_read_in_loop.to_sdfg(simplify=True)) == 0


def test_accumulator_written_after_the_store_is_refused():
    assert _apply(_acc_written_after_store.to_sdfg(simplify=True)) == 0


def _store_named_by_its_source() -> dace.SDFG:
    """``scan[i] = acc`` with the memlet naming ``acc``, so it carries no subset of ``scan``."""
    sdfg = dace.SDFG("sink_last_store_named_by_source")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_array("scan", [N], dace.float64, transient=True)
    sdfg.add_array("acc", [1], dace.float64, transient=True)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("add", {"x", "s"}, {"o"}, "o = s + x")
    acc_in, acc = body.add_read("acc"), body.add_access("acc")
    body.add_edge(body.add_read("a"), None, tasklet, "x", dace.Memlet("a[i]"))
    body.add_edge(acc_in, None, tasklet, "s", dace.Memlet("acc[0]"))
    body.add_edge(tasklet, "o", acc, None, dace.Memlet("acc[0]"))
    body.add_nedge(acc, body.add_write("scan"), dace.Memlet(data="acc", subset="0"))
    tail = sdfg.add_state_after(loop, "tail")
    tail.add_nedge(tail.add_read("scan"), tail.add_write("out"), dace.Memlet("scan[N - 1]", other_subset="0"))
    return sdfg


def test_store_memlet_naming_its_source_is_refused():
    sdfg = _store_named_by_its_source()
    sdfg.validate()
    body = next(s for s in sdfg.states() if s.label == "body")
    store = next(e for e in body.edges() if isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == "scan")
    assert store.data.get_dst_subset(store, body) is None
    assert _apply(sdfg) == 0


if __name__ == "__main__":
    test_scans_read_at_last_keep_one_store_and_the_sequential_order()
    test_scan_read_at_another_index_is_refused()
    test_scan_read_in_the_loop_is_refused()
    test_accumulator_written_after_the_store_is_refused()
    test_store_memlet_naming_its_source_is_refused()
