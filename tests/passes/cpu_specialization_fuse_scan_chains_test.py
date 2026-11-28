# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FuseScanChains: a producer map feeding a Scan becomes one tiled scan tasklet with the producer inlined."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.scan import (
    INIT_CONNECTOR_NAME,
    INPUT_CONNECTOR_NAME,
    OUTPUT_CONNECTOR_NAME,
    Scan,
    ScanOp,
)
from dace.transformation.passes.cpu_specialization.fuse_scan_chains import FuseScanChains

N = dace.symbol("N")


def chain_sdfg(name: str, op: ScanOp, seeded: bool, buffer_readers: int = 1) -> dace.SDFG:
    """``b[1:N] = scan(c[i]*d[i] + c[i]*e[i] for i in 1:N)``, seeded with ``b[0]`` when ``seeded``; the producer is a
    map of two tasklets through a scalar temporary writing ``tmp[i - 1]``, as LoopToScan leaves it."""
    sdfg = dace.SDFG(name)
    for array in "cdeb":
        sdfg.add_array(array, [N], dace.float64)
    sdfg.add_array("tmp", [N - 1], dace.float64, transient=True)
    sdfg.add_scalar("prod", dace.float64, transient=True)
    state = sdfg.add_state("body")
    entry, exit_node = state.add_map("produce", {"i": "1:N"}, schedule=dtypes.ScheduleType.CPU_Multicore)
    mult = state.add_tasklet("mult", {"x", "y"}, {"z"}, "z = x * y")
    add = state.add_tasklet("add", {"p", "x", "w"}, {"z"}, "z = p + x * w")
    c, d, e = state.add_read("c"), state.add_read("d"), state.add_read("e")
    state.add_memlet_path(c, entry, mult, dst_conn="x", memlet=dace.Memlet("c[i]"))
    state.add_memlet_path(d, entry, mult, dst_conn="y", memlet=dace.Memlet("d[i]"))
    prod = state.add_access("prod")
    state.add_edge(mult, "z", prod, None, dace.Memlet("prod[0]"))
    state.add_edge(prod, None, add, "p", dace.Memlet("prod[0]"))
    state.add_memlet_path(c, entry, add, dst_conn="x", memlet=dace.Memlet("c[i]"))
    state.add_memlet_path(e, entry, add, dst_conn="w", memlet=dace.Memlet("e[i]"))
    tmp = state.add_access("tmp")
    state.add_memlet_path(add, exit_node, tmp, src_conn="z", memlet=dace.Memlet("tmp[i - 1]"))
    scan = Scan("scan", op=op)
    state.add_node(scan)
    state.add_edge(tmp, None, scan, INPUT_CONNECTOR_NAME, dace.Memlet("tmp[0:N - 1]"))
    if seeded:
        scan.add_in_connector(INIT_CONNECTOR_NAME)
        state.add_edge(state.add_read("b"), None, scan, INIT_CONNECTOR_NAME, dace.Memlet("b[0]"))
    state.add_edge(scan, OUTPUT_CONNECTOR_NAME, state.add_write("b"), None, dace.Memlet("b[1:N]"))
    for k in range(buffer_readers - 1):
        sdfg.add_array(f"copy{k}", [N - 1], dace.float64)
        state.add_nedge(tmp, state.add_write(f"copy{k}"), dace.Memlet("tmp[0:N - 1]"))
    sdfg.validate()
    return sdfg


def reference(c: np.ndarray, d: np.ndarray, e: np.ndarray, b0: float, op: ScanOp) -> np.ndarray:
    values = c[1:] * d[1:] + c[1:] * e[1:]
    out = np.empty_like(c)
    out[0] = b0
    fold = {ScanOp.SUM: np.add, ScanOp.PRODUCT: np.multiply, ScanOp.MIN: np.minimum, ScanOp.MAX: np.maximum}[op]
    out[1:] = fold.accumulate(np.concatenate([[b0], values]))[1:]
    return out


@pytest.mark.parametrize("op", [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
def test_a_seeded_producer_chain_fuses_into_one_scan_over_several_tiles(op: ScanOp):
    sdfg = chain_sdfg(f"fuse_{op.value}", op, seeded=True)
    assert FuseScanChains().apply_pass(sdfg, {}) == 1
    assert not any(isinstance(n, Scan) for n, _ in sdfg.all_nodes_recursive())
    assert "tmp" not in sdfg.arrays
    sdfg.validate()
    n = 3 * 65536 * 4 + 17  # several tiles of four 512 KB blocks
    rng = np.random.default_rng(1)
    c, d, e = (rng.uniform(0.5, 1.0, n) for _ in range(3))
    if op is ScanOp.PRODUCT:
        c, d, e = np.full(n, 0.5), np.ones(n), np.full(n, 1.0)  # values of exactly 1: the product stays finite
    b = np.zeros(n)
    b[0] = 1.5
    sdfg(c=c, d=d, e=e, b=b, N=n)
    np.testing.assert_allclose(b, reference(c, d, e, 1.5, op), rtol=1e-9)


def test_an_unseeded_min_scan_without_identity_is_left_alone():
    sdfg = chain_sdfg("fuse_min_unseeded", ScanOp.MIN, seeded=False)
    assert FuseScanChains().apply_pass(sdfg, {}) is None
    assert any(isinstance(n, Scan) for n, _ in sdfg.all_nodes_recursive())


def test_a_buffer_with_a_second_reader_is_left_alone():
    sdfg = chain_sdfg("fuse_two_readers", ScanOp.SUM, seeded=True, buffer_readers=2)
    assert FuseScanChains().apply_pass(sdfg, {}) is None
    assert "tmp" in sdfg.arrays


def position_sdfg(name: str) -> dace.SDFG:
    """``b = exclusive_scan(2 * i for i in 0:N)``: the producer reads no array, only its map parameter."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("b", [N], dace.int64)
    sdfg.add_array("tmp", [N], dace.int64, transient=True)
    state = sdfg.add_state("body")
    state.add_mapped_tasklet(
        "produce",
        {"i": "0:N"},
        {},
        "z = 2 * i",
        {"z": dace.Memlet("tmp[i]")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    tmp = next(n for n in state.data_nodes() if n.data == "tmp")
    scan = Scan("scan", op=ScanOp.SUM, exclusive=True, identity=0)
    state.add_node(scan)
    state.add_edge(tmp, None, scan, INPUT_CONNECTOR_NAME, dace.Memlet("tmp[0:N]"))
    state.add_edge(scan, OUTPUT_CONNECTOR_NAME, state.add_write("b"), None, dace.Memlet("b[0:N]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("n", [5, 3 * 65536 * 4 + 17])
def test_a_producer_that_reads_only_its_position_scans_the_position(n: int):
    """The map parameter read as a value was pasted unbound, and a producer reading no array divided by zero bytes."""
    sdfg = position_sdfg(f"fuse_position_{n}")
    assert FuseScanChains().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    b = np.zeros(n, np.int64)
    sdfg(b=b, N=n)
    values = 2 * np.arange(n)
    assert np.array_equal(b, np.cumsum(values) - values)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
