# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Detection of the staged buffers of a loop, and multiple buffering through the frontend and code generation."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.multiple_buffering import find_staged_buffers
from tests.passes.multiple_buffering.lower_multiple_buffering_test import TILE, expected_add, tiled_add, tiled_gemm


def _loop(sdfg: dace.SDFG) -> LoopRegion:
    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert len(loops) == 1
    return loops[0]


def test_finds_both_tiles():
    sdfg = tiled_gemm("staged_gemm", 2)
    staged, rejected = find_staged_buffers(_loop(sdfg), sdfg)
    assert sorted(buffer.name for buffer in staged) == ["a_tile", "b_tile"]
    assert {copy.src.data for buffer in staged for copy in buffer.copies} == {"A", "B"}
    # The output is written by the computation, not by a copy
    assert "C" in rejected


def test_names_restrict_candidates():
    sdfg = tiled_gemm("staged_gemm_names", 2)
    staged, rejected = find_staged_buffers(_loop(sdfg), sdfg, {"b_tile", "C"})
    assert [buffer.name for buffer in staged] == ["b_tile"]
    assert set(rejected) == {"C"}


def test_copy_using_body_symbol_is_rejected():
    """A copy subset that uses a symbol assigned in the body cannot be moved ahead of that assignment."""
    sdfg = tiled_add("staged_body_symbol", 2)
    loop = _loop(sdfg)
    (copy_edge,) = [e for e in loop.start_block.edges() if e.data.data == "A"]
    copy_edge.data.subset = dace.subsets.Range.from_string(f"u:u+{TILE}")
    first = loop.add_state_before(loop.start_block, "assign", is_start_block=True)
    loop.edges_between(first, loop.out_edges(first)[0].dst)[0].data.assignments["u"] = "t"
    staged, rejected = find_staged_buffers(loop, sdfg)
    assert not staged
    assert "assigns" in rejected["buf"]


def test_codegen_lowers_pipeline_depth():
    """Compiling an SDFG with a pipelined loop lowers it without calling the pass."""
    sdfg = tiled_add("staged_codegen", 3, 28)
    code = sdfg.generate_code()[0].clean_code
    assert "% 3" in code
    A = np.random.rand(28)
    B = np.zeros(28)
    sdfg(A=A, B=B)
    assert np.allclose(B, expected_add(A, 0, TILE))


def test_python_frontend():
    N = dace.symbol("N")

    @dace.program
    def tiled(A: dace.float64[N], B: dace.float64[N]):
        # A staging buffer in another storage than its source, which simplification keeps
        buf = dace.define_local([TILE], dace.float64, storage=dace.StorageType.Register)
        for t in range(0, N, TILE):
            buf[:] = A[t : t + TILE]
            B[t : t + TILE] = 2 * buf + 1

    sdfg = tiled.to_sdfg(simplify=True)
    loop = _loop(sdfg)
    staged, rejected = find_staged_buffers(loop, loop.sdfg)
    assert [buffer.name for buffer in staged] == ["buf"], rejected
    loop.pipeline_depth = 3
    assert "% 3" in sdfg.generate_code()[0].clean_code
    A = np.random.rand(24)
    B = np.zeros(24)
    sdfg(A=A, B=B, N=24)
    assert np.allclose(B, 2 * A + 1)


if __name__ == "__main__":
    test_finds_both_tiles()
    test_names_restrict_candidates()
    test_copy_using_body_symbol_is_rejected()
    test_codegen_lowers_pipeline_depth()
    test_python_frontend()
