# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The device scans that batch many independent scans into one call: the residue classes of a strided scan
and the rows of a segmented one.

The strided scan with FEW residue classes (``a[i] = a[i-K] + x[i]`` with a symbolic ``K`` of 1..8) used to
give each class ONE block, so ``K`` blocks walked the whole array (tsvc ``scan_strided_sym`` at 1.9e8
elements: 697 ms on an MI300 against 188 ms on 16 CPU cores). Every long scan is now cut into chunks and
scanned in three phases, and CPF spells the entry points for itself instead of refusing the unit over the
``dace::`` name. The segmented scan reuses the same phases with one scan per row.
"""

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from dace.libraries.standard.nodes.scan import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, Scan, ScanOp

K = dace.symbol("K", dtype=dace.int64)


def strided_device_scan(name: str, length: int, op: ScanOp, rows: int = 1) -> dace.SDFG:
    """One ``Scan`` with the symbolic stride ``K`` over device memory at host level, as ``LoopToScan`` leaves it;
    with ``rows`` it is instead a unit-stride scan of that many consecutive rows."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("K", dace.int64)
    for array in ("x", "a"):
        sdfg.add_array(array, [length], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Scan("scan", op=op, exclusive=False)
    if rows == 1:
        node.stride = K
    else:
        node.segments = rows
    node.implementation = "CUDA"
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f"x[0:{length}]"))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write("a"), None, dace.Memlet(f"a[0:{length}]"))
    sdfg.validate()
    return sdfg


def residue_class_oracle(values: np.ndarray, stride: int, combine) -> np.ndarray:
    out = values.copy()
    for index in range(stride, len(out)):
        out[index] = combine(out[index - stride], values[index])
    return out


def test_cpf_renders_the_strided_device_scan_self_contained():
    """CPF defines the strided entry point the expansion calls, so the unit renders instead of raising
    "a DaCe runtime symbol ('dace::')", and the definition it carries is the chunked one."""
    rendering = cpf.render(strided_device_scan("cpf_hip_strided_scan", 1000, ScanOp.SUM), language="hip")
    code = rendering.code + rendering.device_code
    assert "dace::" not in code, "the unit still names the DaCe runtime"
    assert "strided_inclusive_sum<double>(" in code, "the wrapper must call the strided entry point"
    assert "__global__ void cpf_chunk_totals_kernel" in code, "few classes must take the chunked path"
    assert "get_scratch<ScanTag>" in code, "the chunk totals live in the unit's own scratch pool"


@pytest.mark.gpu
@pytest.mark.parametrize("stride", [1, 3, 8])
@pytest.mark.parametrize("op,combine", [(ScanOp.SUM, np.add), (ScanOp.MAX, np.maximum)], ids=["sum", "max"])
def test_a_long_strided_device_scan_matches_the_residue_class_oracle(stride: int, op: ScanOp, combine):
    """Every class spans several 4096-element chunks plus a ragged tail, so a wrong carry across a chunk
    boundary, or a class past its own end, shows up as a wrong element."""
    import cupy

    length = 3 * 4096 * stride + 5
    values = np.random.default_rng(20261006).uniform(-1.0, 1.0, size=length)
    device_out = cupy.zeros(length, dtype=np.float64)
    sdfg = strided_device_scan(f"strided_chunked_{op.value}_s{stride}", length, op)
    sdfg(x=cupy.asarray(values), a=device_out, K=stride)
    np.testing.assert_allclose(
        cupy.asnumpy(device_out), residue_class_oracle(values, stride, combine), rtol=1e-9, atol=1e-9
    )


@pytest.mark.gpu
@pytest.mark.parametrize("rows,length", [(5, 3 * 4096 + 7), (300, 100)], ids=["long_rows", "short_rows"])
def test_a_segmented_device_scan_restarts_at_every_row(rows: int, length: int):
    """Rows longer than a chunk exercise the carry across chunks of one row; short ones, one block per row.
    Either way no carry may cross from one row into the next."""
    import cupy

    values = np.random.default_rng(20261006).uniform(-1.0, 1.0, size=rows * length)
    device_out = cupy.zeros(rows * length, dtype=np.float64)
    sdfg = strided_device_scan(f"segmented_{rows}x{length}", rows * length, ScanOp.SUM, rows=rows)
    sdfg(x=cupy.asarray(values), a=device_out)
    np.testing.assert_allclose(
        cupy.asnumpy(device_out), np.cumsum(values.reshape(rows, length), axis=1).ravel(), rtol=1e-9, atol=1e-9
    )


if __name__ == "__main__":
    test_cpf_renders_the_strided_device_scan_self_contained()
    for stride in (1, 3, 8):
        test_a_long_strided_device_scan_matches_the_residue_class_oracle(stride, ScanOp.SUM, np.add)
        test_a_long_strided_device_scan_matches_the_residue_class_oracle(stride, ScanOp.MAX, np.maximum)
    test_a_segmented_device_scan_restarts_at_every_row(5, 3 * 4096 + 7)
    test_a_segmented_device_scan_restarts_at_every_row(300, 100)
