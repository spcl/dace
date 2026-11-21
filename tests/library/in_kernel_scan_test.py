# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``Scan`` inside a GPU kernel lowers to the thread-block collective in every shape it has one for.

One kernel iteration per row, each row long enough (``N`` = 1000, four chunks of 256 lanes) that the carry
crosses chunk boundaries: exclusive, seeded, multi-chain, affine, strided and column scans, each checked
against numpy after asserting that the block collective -- not the one-thread ``pure`` loop -- is what ran.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg import nodes

ROWS, LENGTH = 5, 1000
R = dace.symbol("R", dtype=dace.int64)
N = dace.symbol("N", dtype=dace.int64)


def scan_kernel(name: str, scan: Scan, inputs: dict[str, str], outputs: dict[str, str], arrays: dict[str, list]):
    """A ``GPU_Device`` map over rows ``i`` holding ``scan``, wired to global ``float64`` arrays."""
    sdfg = dace.SDFG(name)
    for array, shape in arrays.items():
        sdfg.add_array(array, shape, dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:R"}, schedule=dtypes.ScheduleType.GPU_Device)
    state.add_node(scan)
    for conn, subset in inputs.items():
        if conn not in scan.in_connectors:
            scan.add_in_connector(conn)
        state.add_memlet_path(
            state.add_read(subset.split("[")[0]), entry, scan, dst_conn=conn, memlet=dace.Memlet(subset)
        )
    for conn, subset in outputs.items():
        state.add_memlet_path(
            scan, exit_node, state.add_write(subset.split("[")[0]), src_conn=conn, memlet=dace.Memlet(subset)
        )
    return sdfg


def block_code(sdfg: dace.SDFG) -> str:
    """Expand by default and return the scan's code, asserting it is the block collective."""
    sdfg.expand_library_nodes()
    code = "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    assert "block_scan<" in code or "block_affine_scan<" in code, f"not the block collective:\n{code}"
    return code


def run(sdfg: dace.SDFG, **arrays: np.ndarray) -> dict[str, np.ndarray]:
    import cupy

    device = {name: cupy.asarray(value) for name, value in arrays.items()}
    sdfg(**device, R=ROWS, N=LENGTH)
    return {name: value.get() for name, value in device.items()}


def data(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).random((ROWS, LENGTH))


@pytest.mark.gpu
def test_exclusive_sum():
    sdfg = scan_kernel(
        "excl_sum",
        Scan("scan", exclusive=True),
        {"_scan_in": "a[i, 0:N]"},
        {"_scan_out": "b[i, 0:N]"},
        {"a": [R, N], "b": [R, N]},
    )
    assert ", true>" in block_code(sdfg)
    a = data()
    out = run(sdfg, a=a, b=np.zeros_like(a))
    expected = np.concatenate([np.zeros((ROWS, 1)), np.cumsum(a, axis=1)[:, :-1]], axis=1)
    assert np.allclose(out["b"], expected)


@pytest.mark.gpu
def test_exclusive_min_with_an_identity():
    sdfg = scan_kernel(
        "excl_min",
        Scan("scan", op=ScanOp.MIN, exclusive=True, identity=2.0),
        {"_scan_in": "a[i, 0:N]"},
        {"_scan_out": "b[i, 0:N]"},
        {"a": [R, N], "b": [R, N]},
    )
    block_code(sdfg)
    a = data() + 0.5
    out = run(sdfg, a=a, b=np.zeros_like(a))
    expected = np.minimum.accumulate(np.concatenate([np.full((ROWS, 1), 2.0), a[:, :-1]], axis=1), axis=1)
    assert np.array_equal(out["b"], expected)


@pytest.mark.gpu
@pytest.mark.parametrize("op", [ScanOp.SUM, ScanOp.MAX])
def test_seeded_inclusive(op):
    sdfg = scan_kernel(
        f"seeded_{op.value}",
        Scan("scan", op=op),
        {"_scan_in": "a[i, 0:N]", "_scan_init": "s[i]"},
        {"_scan_out": "b[i, 0:N]"},
        {"a": [R, N], "b": [R, N], "s": [R]},
    )
    assert "__b_scan_init[0]" in block_code(sdfg)
    a, s = data(), np.linspace(-1.0, 3.0, ROWS)
    out = run(sdfg, a=a, b=np.zeros_like(a), s=s)
    if op is ScanOp.SUM:
        assert np.allclose(out["b"], s[:, None] + np.cumsum(a, axis=1))
    else:
        assert np.array_equal(out["b"], np.maximum(s[:, None], np.maximum.accumulate(a, axis=1)))


@pytest.mark.gpu
def test_unseeded_min():
    sdfg = scan_kernel(
        "min",
        Scan("scan", op=ScanOp.MIN),
        {"_scan_in": "a[i, 0:N]"},
        {"_scan_out": "b[i, 0:N]"},
        {"a": [R, N], "b": [R, N]},
    )
    block_code(sdfg)
    a = data()
    out = run(sdfg, a=a, b=np.zeros_like(a))
    assert np.array_equal(out["b"], np.minimum.accumulate(a, axis=1))


@pytest.mark.gpu
def test_two_chains():
    sdfg = scan_kernel(
        "chains",
        Scan("scan", chains=2),
        {"_scan_in": "a[i, 0:N]", "_scan_in_1": "c[i, 0:N]"},
        {"_scan_out": "b[i, 0:N]", "_scan_out_1": "d[i, 0:N]"},
        {"a": [R, N], "b": [R, N], "c": [R, N], "d": [R, N]},
    )
    assert block_code(sdfg).count("block_scan<") == 2
    a, c = data(1), data(2)
    out = run(sdfg, a=a, b=np.zeros_like(a), c=c, d=np.zeros_like(c))
    assert np.allclose(out["b"], np.cumsum(a, axis=1))
    assert np.allclose(out["d"], np.cumsum(c, axis=1))


@pytest.mark.gpu
@pytest.mark.parametrize("seeded", [False, True])
def test_affine(seeded):
    inputs = {"_scan_in": "d[i, 0:N]", "_scan_coef": "c[i, 0:N]"}
    if seeded:
        inputs["_scan_init"] = "s[i]"
    sdfg = scan_kernel(
        f"affine_{seeded}",
        Scan("scan", op=ScanOp.AFFINE),
        inputs,
        {"_scan_out": "b[i, 0:N]"},
        {"c": [R, N], "d": [R, N], "b": [R, N], "s": [R]},
    )
    assert "block_affine_scan<" in block_code(sdfg)
    c, d = 0.5 + 0.5 * data(4), data(5)
    s = np.linspace(1.0, 2.0, ROWS) if seeded else np.zeros(ROWS)
    out = run(sdfg, c=c, d=d, b=np.zeros_like(d), s=s)
    expected, carry = np.empty_like(d), s.copy()
    for k in range(LENGTH):
        carry = c[:, k] * carry + d[:, k]
        expected[:, k] = carry
    assert np.allclose(out["b"], expected)


@pytest.mark.gpu
def test_strided_sum():
    """``out[k] = out[k - 3] + in[k]``: three residue classes, never ``N`` elements three apart."""
    scan = Scan("scan")
    scan.stride = 3
    sdfg = scan_kernel(
        "strided", scan, {"_scan_in": "a[i, 0:N]"}, {"_scan_out": "b[i, 0:N]"}, {"a": [R, N], "b": [R, N]}
    )
    block_code(sdfg)
    a = data()
    out = run(sdfg, a=a, b=np.zeros_like(a))
    expected = np.empty_like(a)
    for r in range(3):
        expected[:, r::3] = np.cumsum(a[:, r::3], axis=1)
    assert np.allclose(out["b"], expected)


@pytest.mark.gpu
def test_column_into_row():
    """Each row ``i`` scans column ``i`` of ``a``, which is ``R`` elements apart in memory."""
    sdfg = scan_kernel(
        "column", Scan("scan"), {"_scan_in": "a[0:N, i]"}, {"_scan_out": "b[i, 0:N]"}, {"a": [N, R], "b": [R, N]}
    )
    block_code(sdfg)
    a = np.ascontiguousarray(data().T)
    out = run(sdfg, a=a, b=np.zeros((ROWS, LENGTH)))
    assert np.allclose(out["b"], np.cumsum(a, axis=0).T)


if __name__ == "__main__":
    test_exclusive_sum()
    test_exclusive_min_with_an_identity()
    for op in (ScanOp.SUM, ScanOp.MAX):
        test_seeded_inclusive(op)
    test_unseeded_min()
    test_two_chains()
    for seeded in (False, True):
        test_affine(seeded)
    test_strided_sum()
    test_column_into_row()
