# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``i < n and a[i] != 0`` reads ``a[i]`` only when ``i < n`` holds (gromacs_nbnxm): the read at the end of the
list would be out of bounds."""

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N")


@dace.program
def scan_to_mask(excl: dace.int64[N], start: dace.int64, end: dace.int64, out: dace.int64[1]):
    k = start + 0
    while k < end and excl[k] != 65535:
        k = k + 1
    out[0] = k


def scan_reference(excl: np.ndarray, start: int, end: int) -> int:
    k = start
    while k < end and excl[k] != 65535:
        k += 1
    return k


def test_the_right_operand_of_and_is_read_only_when_the_left_holds():
    sdfg = scan_to_mask.to_sdfg(simplify=False)
    reads = [state for state in sdfg.states() if any(node.data == "excl" for node in state.data_nodes())]
    assert reads
    for state in reads:
        block = state.parent_graph
        while block is not None and not isinstance(block, ConditionalBlock):
            block = block.parent_graph
        assert block is not None, f"{state.label} reads excl outside the guard of the bound check"
    sdfg = scan_to_mask.to_sdfg(simplify=True)
    sdfg.name = "cpf_scan_to_mask"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    result = render(sdfg, language="c")
    library = build_standalone(result.code, sdfg.name, language="c")
    excl = np.array([3, 5, 7, 65535, 1, 2], dtype=np.int64)
    for start, end in ((0, 6), (4, 6), (6, 6), (0, 2)):
        out = np.zeros(1, dtype=np.int64)
        # The list ends at ``end``: a read of excl[end] is out of bounds, so the view is cut there.
        view = np.ascontiguousarray(excl[:end])
        call_standalone(library, result.sdfg, {"excl": view, "start": start, "end": end, "out": out, "N": end})
        assert out[0] == scan_reference(excl[:end], start, end), (start, end, out[0])


if __name__ == "__main__":
    test_the_right_operand_of_and_is_read_only_when_the_left_holds()
