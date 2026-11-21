# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF renders a scan with carry distance ``S`` as a blocked scan over contiguous chunks, for any ``S``."""

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.libraries.standard.nodes.scan import (
    INIT_CONNECTOR_NAME,
    INPUT_CONNECTOR_NAME,
    OUTPUT_CONNECTOR_NAME,
    Scan,
    ScanOp,
)
from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone, tolerance_for

N = dace.symbol("N", dace.int64)
S = dace.symbol("S", dace.int64)
UFUNCS = {ScanOp.SUM: np.add, ScanOp.PRODUCT: np.multiply, ScanOp.MIN: np.minimum, ScanOp.MAX: np.maximum}
#: Distances 1 to 3, and one wider than a chunk; lengths below a team, at one element, past whole chunks and rounds.
DISTANCES = (1, 2, 3, 5000)
LENGTHS = (1, 3, 4096 * 3 + 5, 4096 * 40 + 7)


def strided_scan_sdfg(name: str, op: ScanOp, dtype: dace.typeclass, seeded: bool) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    # the stride is a node property, which the SDFG's free symbols do not see
    sdfg.add_symbol("S", dace.int64)
    sdfg.add_array("arr_in", [N], dtype)
    sdfg.add_array("arr_out", [N], dtype)
    state = sdfg.add_state("scan")
    node = Scan("Scan", op=op, exclusive=False)
    node.stride = S
    node.implementation = "CPU"
    state.add_node(node)
    state.add_edge(state.add_read("arr_in"), None, node, INPUT_CONNECTOR_NAME, dace.Memlet("arr_in[0:N]"))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write("arr_out"), None, dace.Memlet("arr_out[0:N]"))
    if seeded:
        sdfg.add_array("seeds", [S], dtype)
        node.add_in_connector(INIT_CONNECTOR_NAME)
        state.add_edge(state.add_read("seeds"), None, node, INIT_CONNECTOR_NAME, dace.Memlet("seeds[0:S]"))
    return sdfg


def sequential_scan(op: ScanOp, values: np.ndarray, distance: int, seeds: np.ndarray | None) -> np.ndarray:
    result = np.empty_like(values)
    for k in range(min(distance, values.size)):
        chain = values[k::distance]
        start = chain if seeds is None else np.concatenate(([seeds[k]], chain))
        result[k::distance] = UFUNCS[op].accumulate(start)[start.size - chain.size :]
    return result


def inputs(op: ScanOp, dtype: np.dtype, size: int, rng: np.random.Generator) -> np.ndarray:
    if np.dtype(dtype).kind in "iu":
        return rng.integers(-1000, 1000, size).astype(dtype)
    if op is ScanOp.PRODUCT:
        return rng.uniform(0.999, 1.001, size).astype(dtype)
    return rng.uniform(-1.0, 1.0, size).astype(dtype)


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize(
    "op, dtype, seeded",
    [
        (ScanOp.SUM, dace.float64, True),
        (ScanOp.SUM, dace.int64, False),
        (ScanOp.PRODUCT, dace.float64, False),
        (ScanOp.MIN, dace.float64, True),
        (ScanOp.MAX, dace.int32, False),
    ],
)
def test_a_strided_scan_runs_blocked_over_chunks_for_every_distance(
    op: ScanOp, dtype: dace.typeclass, seeded: bool, language: str
) -> None:
    name = f"cpf_blocked_{op.value}_{dtype.to_string()}_{'seeded' if seeded else 'plain'}_{language.replace('+', 'p')}"
    sdfg = strided_scan_sdfg(name, op, dtype, seeded)
    rendering = render(sdfg, language=language)
    assert "pass 2: rescan each chunk" in rendering.code, rendering.code
    assert "cpf_r < cpf_s" not in rendering.code, rendering.code
    assert_standalone(rendering.code, name, language=language)
    sut = build_standalone(rendering.code, name, language=language)
    np_dtype = dtype.as_numpy_dtype()
    rtol, atol = tolerance_for(np_dtype)
    rng = np.random.default_rng(7)
    for distance in DISTANCES:
        for size in LENGTHS:
            values = inputs(op, np_dtype, size, rng)
            seeds = inputs(op, np_dtype, distance, rng) if seeded else None
            out = np.zeros_like(values)
            arguments = {"arr_in": values, "arr_out": out, "N": size, "S": distance}
            if seeds is not None:
                arguments["seeds"] = seeds
            call_standalone(sut, sdfg, arguments)
            np.testing.assert_allclose(
                out,
                sequential_scan(op, values, distance, seeds),
                rtol=rtol,
                atol=atol,
                err_msg=f"S={distance} N={size}",
            )


if __name__ == "__main__":
    for case in [
        (ScanOp.SUM, dace.float64, True),
        (ScanOp.SUM, dace.int64, False),
        (ScanOp.PRODUCT, dace.float64, False),
        (ScanOp.MIN, dace.float64, True),
        (ScanOp.MAX, dace.int32, False),
    ]:
        for language in ("c", "c++"):
            test_a_strided_scan_runs_blocked_over_chunks_for_every_distance(*case, language)
