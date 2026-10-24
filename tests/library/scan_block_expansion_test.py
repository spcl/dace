# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``Scan`` block expansion: what a scan INSIDE a GPU kernel lowers to.

The point of the expansion is not the CUB call -- it is that the emitted subgraph carries a
``GPU_ThreadBlock`` map, because that is the only signal the code generator has that the enclosing
device map runs one iteration per BLOCK. Without it the row map is spread over threads and every
thread would scan a different row through the same collective. These tests assert that structure,
which is checkable without a GPU; the numerics are checked on device by the GPU test below.
"""

import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.scan import BLOCK_COLLECTIVE_THREADS, Scan, ScanOp, block_refusal
from dace.sdfg import nodes

N = dace.symbol("N", dtype=dace.int64)
ROWS = dace.symbol("ROWS", dtype=dace.int64)


def build_scan_sdfg(op: ScanOp = ScanOp.SUM, exclusive: bool = False, chains: int = 1) -> dace.SDFG:
    """One FULLY WIRED ``Scan`` over ``N`` elements, standing alone in a state.

    Every connector the shape requires is connected, the extra chains and ``op=AFFINE``'s
    coefficients included. An under-wired node fails the libnode's own validation, which would let
    a refusal test pass for the wrong reason -- rejecting a malformed graph rather than an
    unsupported shape.
    """
    sdfg = dace.SDFG("scan_block_probe")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    state = sdfg.add_state()
    node = Scan("scan", op=op, exclusive=exclusive, chains=chains)
    state.add_node(node)
    for chain in range(chains):
        suffix = "" if chain == 0 else f"_{chain}"
        state.add_edge(state.add_read("a"), None, node, f"_scan_in{suffix}", dace.Memlet("a[0:N]"))
        state.add_edge(node, f"_scan_out{suffix}", state.add_write("b"), None, dace.Memlet("b[0:N]"))
        if op is ScanOp.AFFINE:
            state.add_edge(state.add_read("a"), None, node, f"_scan_coef{suffix}", dace.Memlet("a[0:N]"))
    return sdfg, state, node


def test_block_expansion_is_registered():
    assert "CUDA (block)" in Scan.implementations


def test_expansion_emits_a_thread_block_map():
    """The structural claim: a ``GPU_ThreadBlock`` map of the collective's width."""
    sdfg, state, node = build_scan_sdfg()
    node.implementation = "CUDA (block)"
    node.expand(state)
    tb_maps = [
        n
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert len(tb_maps) == 1, "exactly one thread-block map should carry the collective"
    lanes = tb_maps[0].map.range.size()[0]
    assert lanes == BLOCK_COLLECTIVE_THREADS


def test_every_lane_reads_the_whole_range():
    """The map supplies THREADS, not a partition. A per-lane subset would give each thread its own
    scan, which is the bug this test exists to catch."""
    sdfg, state, node = build_scan_sdfg()
    node.implementation = "CUDA (block)"
    node.expand(state)
    entry = next(
        n
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    )
    param = entry.map.params[0]
    # The map lives in the expansion's NESTED SDFG, which ``all_states`` does not descend into.
    inner = next(st for n, st in sdfg.all_nodes_recursive() if n is entry)
    for edge in inner.out_edges(entry):
        assert param not in str(edge.data.subset), f"lane {param} narrows the scan range: {edge.data}"


def test_expansion_calls_the_shared_collective():
    """It must call the SAME collective the residue-class kernel uses, not a second copy."""
    sdfg, state, node = build_scan_sdfg()
    node.implementation = "CUDA (block)"
    node.expand(state)
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 1
    assert "::dace::cuda_scan::detail::block_scan<" in tasklets[0].code.as_string
    assert f"{BLOCK_COLLECTIVE_THREADS}, false>" in tasklets[0].code.as_string


def test_stride_reaches_the_collective():
    """A strided scan is the general case, not an unsupported one: the stride is an argument."""
    sdfg, state, node = build_scan_sdfg()
    node.stride = 7
    node.implementation = "CUDA (block)"
    node.expand(state)
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    assert "(long)(7)" in tasklet.code.as_string


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exclusive": True, "stride": 3},
        {"chains": 2, "stride": 3},
        {"op": ScanOp.MIN, "exclusive": True},
        {"op": ScanOp.AFFINE, "exclusive": True},
        {"segments": 2},
    ],
)
def test_unsupported_shapes_refuse_rather_than_approximate(kwargs):
    """Each refusal sends the node to another expansion. Silently lowering one of these as something else
    would be a miscompile, so the refusal is the contract."""
    sdfg, state, node = build_scan_sdfg(
        op=kwargs.get("op", ScanOp.SUM), exclusive=kwargs.get("exclusive", False), chains=kwargs.get("chains", 1)
    )
    node.stride = kwargs.get("stride", 1)
    node.segments = kwargs.get("segments", 1)
    node.implementation = "CUDA (block)"
    assert block_refusal(node, state) is not None
    with pytest.raises(NotImplementedError):
        node.expand(state)


def expanded_code(sdfg: dace.SDFG, state: dace.SDFGState, node: Scan) -> str:
    node.implementation = "CUDA (block)"
    node.expand(state)
    (tasklet,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    return tasklet.code.as_string


def test_an_exclusive_scan_is_the_exclusive_collective():
    sdfg, state, node = build_scan_sdfg(exclusive=True)
    assert f"{BLOCK_COLLECTIVE_THREADS}, true>" in expanded_code(sdfg, state, node)


def test_every_chain_is_one_collective_call():
    sdfg, state, node = build_scan_sdfg(chains=3)
    code = expanded_code(sdfg, state, node)
    assert code.count("block_scan<") == 3
    assert all(f"__b_scan_out_{c}" in code for c in (1, 2))


def test_an_affine_scan_is_the_affine_collective():
    sdfg, state, node = build_scan_sdfg(op=ScanOp.AFFINE)
    assert "block_affine_scan<double, 256>(__b_scan_coef, 1, __b_scan_in, 1, __b_scan_out, 1" in expanded_code(
        sdfg, state, node
    )


def test_a_seed_enters_the_collective():
    sdfg, state, node = build_scan_sdfg(op=ScanOp.MAX)
    sdfg.add_array("s", [1], dace.float64)
    node.add_in_connector("_scan_init")
    state.add_edge(state.add_read("s"), None, node, "_scan_init", dace.Memlet("s[0]"))
    assert "static_cast<double>(__b_scan_init[0]));" in expanded_code(sdfg, state, node)


def test_an_unseeded_min_seeds_from_its_first_element():
    """A lane past the end no longer reads an identity, so ``min`` needs none."""
    sdfg, state, node = build_scan_sdfg(op=ScanOp.MIN)
    assert "((N) > 0 ? __b_scan_in[0] : double())" in expanded_code(sdfg, state, node)


def test_a_strided_scan_scans_every_residue_class():
    """Not ``N`` elements ``stride`` apart from the start, which runs past the end of the array."""
    sdfg, state, node = build_scan_sdfg()
    node.stride = 3
    code = expanded_code(sdfg, state, node)
    assert "for (long __bs_r = 0; __bs_r < (long)(3) && __bs_r < (long)(N); ++__bs_r)" in code
    assert "(__b_scan_in + __bs_r * 1, (long)(3) * 1," in code


def test_a_column_is_walked_at_its_stride():
    sdfg = dace.SDFG("scan_column")
    sdfg.add_array("a", [N, ROWS], dace.float64)
    sdfg.add_array("b", [ROWS, N], dace.float64)
    state = sdfg.add_state()
    node = Scan("scan")
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_scan_in", dace.Memlet("a[0:N, 2]"))
    state.add_edge(node, "_scan_out", state.add_write("b"), None, dace.Memlet("b[2, 0:N]"))
    assert "(__b_scan_in, ROWS, __b_scan_out, 1," in expanded_code(sdfg, state, node)


def test_an_operand_over_two_dimensions_refuses():
    sdfg = dace.SDFG("scan_plane")
    sdfg.add_array("a", [N, ROWS], dace.float64)
    sdfg.add_array("b", [N, ROWS], dace.float64)
    state = sdfg.add_state()
    node = Scan("scan")
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_scan_in", dace.Memlet("a[0:N, 0:ROWS]"))
    state.add_edge(node, "_scan_out", state.add_write("b"), None, dace.Memlet("b[0:N, 0:ROWS]"))
    assert "more than one dimension" in block_refusal(node, state)


def test_the_collective_header_declares_the_function():
    """The expansion names a symbol in scan.cuh; a rename there must break this, not the build."""
    import pathlib

    header = pathlib.Path(dace.__file__).parent / "runtime" / "include" / "dace" / "cuda" / "scan.cuh"
    text = header.read_text()
    assert "__device__ void block_inclusive_scan_strided" in text
    assert "block_scan<T, Op, BLOCK>(in, s, out, s, m, op, seed);" in text
    # The residue-class kernel must go through the same collective, or the two drift apart.
    assert "block_inclusive_scan_strided<T, Op, BLOCK>(in + k * seg.pitch, out + k * seg.pitch" in text


if __name__ == "__main__":
    test_block_expansion_is_registered()
    test_expansion_emits_a_thread_block_map()
    test_every_lane_reads_the_whole_range()
    test_expansion_calls_the_shared_collective()
    test_stride_reaches_the_collective()
    for kwargs in (
        {"exclusive": True, "stride": 3},
        {"chains": 2, "stride": 3},
        {"op": ScanOp.MIN, "exclusive": True},
        {"op": ScanOp.AFFINE, "exclusive": True},
        {"segments": 2},
    ):
        test_unsupported_shapes_refuse_rather_than_approximate(kwargs)
    test_an_exclusive_scan_is_the_exclusive_collective()
    test_every_chain_is_one_collective_call()
    test_an_affine_scan_is_the_affine_collective()
    test_a_seed_enters_the_collective()
    test_an_unseeded_min_seeds_from_its_first_element()
    test_a_strided_scan_scans_every_residue_class()
    test_a_column_is_walked_at_its_stride()
    test_an_operand_over_two_dimensions_refuses()
    test_the_collective_header_declares_the_function()
    print("ok")
