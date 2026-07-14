# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the GPU-only ``branched_tail`` remainder strategy.

``branched_tail`` fuses the vectorized main-tiled map and its scalar remainder map into ONE
GPU kernel whose body is an ``if(full-tile)/else(scalar-tail)`` ``ConditionalBlock`` -- instead
of two separate ``GPU_Device`` kernels (the ``masked_tail`` / ``scalar_postamble`` remainder on a
non-divisible extent would emit two kernels). The vectorized tile ops survive UNCHANGED inside
the ``if`` branch (no arithmetic-select flatten).

Covered:
  * default off -> the pipeline is byte-identical (no ``FuseBranchedTailRemainder``, no
    ``ConditionalBlock``, single strided ``assume_even`` map);
  * the strategy is refused on CPU;
  * the provably-non-divisible extent (1022 @ W=8) takes the fused path instead of RAISING the
    ``assume_even`` error;
  * the fused SDFG is one map + a conditional with vectorized tile ops in the ``if`` branch and a
    scalar loop in the ``else``;
  * the emitted ``.cu`` has exactly ONE kernel with a clean (split-residue-free) bound + condition;
  * bit-exact vs the NumPy fp16 oracle on the device (elementwise + neighbour-read stencil, W=4/8).

GPU-executing tests fork before any CUDA use so a device fault cannot crash the pytest parent.
"""
import os
import shutil
import traceback

import pytest

import dace
from dace.dtypes import DeviceType
from dace.transformation.interstate import LoopToMap
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeMultiDim

N = dace.symbol("N")


@dace.program
def _add16(A: dace.float16[N], B: dace.float16[N], C: dace.float16[N]):
    for i in dace.map[1:N - 1]:
        C[i] = A[i] + B[i]


@dace.program
def _neighbor16(A: dace.float16[N], D: dace.float16[N]):
    # A neighbour-read "stencil" shape (boundary lanes read A[i-1] / A[i+1]) but a SINGLE fp16 add,
    # so the NumPy oracle is unambiguous and the comparison is exactly bit-exact.
    for i in dace.map[1:N - 1]:
        D[i] = A[i - 1] + A[i + 1]


@dace.program
def _add16_literal(A: dace.float16[1024], B: dace.float16[1024], C: dace.float16[1024]):
    # Literal interior extent 1022 (== indices 1..1022), PROVABLY not a multiple of 8.
    for i in dace.map[1:1023]:
        C[i] = A[i] + B[i]


def _prep(prog):
    """@dace.program -> simplify + LoopToMap + simplify -> GPU-offload (the caller-side recipe)."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    return sdfg


def _top_maps(sdfg):
    return [
        n for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)
        and isinstance(g, dace.SDFGState) and g.sdfg is sdfg and g.scope_dict()[n] is None
    ]


def _conditionals(sdfg):
    return [
        b for sd in sdfg.all_sdfgs_recursive() for b in sd.all_control_flow_blocks() if isinstance(b, ConditionalBlock)
    ]


def _run_in_fork(work) -> int:
    """Run ``work()`` (GPU compile + execute + assert) in a child; fork BEFORE any CUDA use so a
    device-side fault cannot crash the pytest parent. Returns the child's exit status (0 == pass)."""
    pid = os.fork()
    if pid == 0:  # child
        code = 0
        try:
            work()
        except BaseException:  # noqa: BLE001 - report + non-zero exit, never raise past the fork
            traceback.print_exc()
            code = 1
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    return status


# --------------------------------------------------------------------------------------------------
# Structural / gating tests (no GPU device needed)
# --------------------------------------------------------------------------------------------------
def test_branched_tail_refused_on_cpu():
    """The strategy is GPU-only: a CPU config raises ``NotImplementedError`` at construction."""
    with pytest.raises(NotImplementedError, match="branched_tail.*GPU-only"):
        VectorizeMultiDim(VectorizeConfig(widths=(8, ), device=DeviceType.CPU, remainder_strategy="branched_tail"))


def test_assume_even_opts_out_of_branched_tail():
    """Explicit ``assume_even=True`` opts out of the branched_tail machinery (it is the GPU K=1
    default): the pipeline carries NO fuse pass and emits the single strided map with no
    ``ConditionalBlock``. (Without it, the K=1 GPU default is branched_tail -- see
    :func:`test_default_gpu_k1_is_branched_tail`.)"""
    from dace.transformation.passes.vectorization.fuse_branched_tail_remainder import FuseBranchedTailRemainder

    even = VectorizeGPU(VectorizeConfig(widths=(2, ), assume_even=True))
    assert not any(isinstance(p, FuseBranchedTailRemainder) for p in even.passes), \
        "assume_even pipeline must NOT contain the branched-tail fuse pass"

    sdfg = _prep(_add16)
    even.apply_pass(sdfg, {})
    assert len(_top_maps(sdfg)) == 1, "assume_even keeps a single map"
    assert not _conditionals(sdfg), "assume_even must emit no ConditionalBlock"


def test_default_gpu_k1_is_branched_tail():
    """The GPU K=1 default (widths=(W,), no explicit strategy) IS branched_tail: the fuse pass is in
    the pipeline so a non-divisible extent works out of the box (one kernel, scalar remainder loop)."""
    from dace.transformation.passes.vectorization.fuse_branched_tail_remainder import FuseBranchedTailRemainder

    default = VectorizeGPU(VectorizeConfig(widths=(2, )))
    assert any(isinstance(p, FuseBranchedTailRemainder) for p in default.passes), \
        "GPU K=1 default must be branched_tail (fuse pass present)"


def test_branched_tail_provably_nondivisible_does_not_raise():
    """The provably-non-divisible extent (1022 @ W=8) RAISES under explicit ``assume_even`` but takes
    the fused path under ``branched_tail`` (the GPU K=1 default) -- one fused map over the ORIGINAL
    [1:1023) range."""
    with pytest.raises(ValueError, match="provably not a multiple of tile width 8"):
        VectorizeGPU(VectorizeConfig(widths=(8, ), assume_even=True)).apply_pass(_prep(_add16_literal), {})

    sdfg = _prep(_add16_literal)
    VectorizeGPU(VectorizeConfig(widths=(8, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
    sdfg.validate()
    maps = _top_maps(sdfg)
    assert len(maps) == 1, f"branched_tail must fuse to one map; got {len(maps)}"
    lb, ub, step = maps[0].map.range.ranges[-1]
    assert (str(lb), str(ub), str(step)) == ("1", "1022", "8"), \
        f"fused map must iterate the original element range strided by W; got {(str(lb), str(ub), str(step))}"


def test_branched_tail_structure_if_vector_else_scalar():
    """The fused body is ONE ``ConditionalBlock``: the ``if`` (full-tile) branch holds the vectorized
    tile ops (TileLoad/TileBinop/TileStore) and NO scalar tasklet; the ``else`` branch holds a
    Sequential loop with the scalar tasklet."""
    sdfg = _prep(_add16)
    VectorizeGPU(VectorizeConfig(widths=(8, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
    sdfg.validate()
    assert len(_top_maps(sdfg)) == 1
    conds = _conditionals(sdfg)
    assert len(conds) == 1, f"expected exactly one ConditionalBlock; got {len(conds)}"
    cb = conds[0]
    assert [c is not None for c, _ in cb.branches] == [True, False], "expected an if-branch + a bare else"

    (if_cond, if_region), (_, else_region) = cb.branches

    def _node_types(region):
        return [type(n).__name__ for st in region.all_states() for n in st.nodes()]

    def _all_nodes_recursive(region):
        for st in region.all_states():
            yield from st.all_nodes_recursive()

    if_kinds = {type(n).__name__ for n, _ in _all_nodes_recursive(if_region)}
    assert "TileBinop" in if_kinds and "TileLoad" in if_kinds and "TileStore" in if_kinds, \
        f"vectorized tile ops must survive intact in the if-branch; got {sorted(if_kinds)}"
    assert "Tasklet" not in _node_types(if_region), "the if-branch must be tile ops, not a scalar tasklet"

    assert "MapEntry" in _node_types(else_region), "the else-branch must wrap a scalar loop (Sequential map)"
    else_kinds = {type(n).__name__ for n, _ in _all_nodes_recursive(else_region)}
    assert "Tasklet" in else_kinds, "the else-branch must run the scalar (tasklet) body"
    assert not (else_kinds & {"TileBinop", "TileLoad", "TileStore"}), "the else-branch must stay scalar"

    # The if-condition + else-loop bound are clean expressions in N/W (no int_floor split residue).
    assert "int_floor" not in if_cond.as_string, f"if-condition carries a split residue: {if_cond.as_string}"


def test_branched_tail_emits_single_kernel():
    """The emitted device (``.cu``) TU contains exactly ONE ``__global__`` kernel for the fused
    region -- the two-kernel remainder is folded into one -- with a split-residue-free bound."""
    sdfg = _prep(_add16)
    VectorizeGPU(VectorizeConfig(widths=(8, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    cu = "\n".join(c.clean_code for c in sdfg.generate_code() if c.language == "cu")
    assert cu.count("__global__ void") == 1, "branched_tail must emit exactly ONE kernel for the tiled region"
    # The vectorized tile contract survives inside the single kernel.
    assert "dace::tileops::tile_binop<dace::float16, 8" in cu
    assert "dace::tileops::tile_load<dace::float16, 8" in cu


# --------------------------------------------------------------------------------------------------
# GPU-executing numeric tests (forked; bit-exact vs the NumPy fp16 oracle)
# --------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("width", [8, 4])
def test_branched_tail_elementwise_bitexact(width):
    """C = A + B over interior [1:N-1], N=1024 (extent 1022, non-divisible): one fused kernel,
    bit-exact fp16 result including the partial tail lanes."""

    def work():
        import numpy as np
        import cupy
        sdfg = _prep(_add16)
        sdfg.name = f"bt_add16_w{width}"
        VectorizeGPU(VectorizeConfig(widths=(width, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        cu = "\n".join(c.clean_code for c in sdfg.generate_code() if c.language == "cu")
        assert cu.count("__global__ void") == 1
        shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
        csr = sdfg.compile()
        n = 1024
        rng = np.random.default_rng(width)
        A = rng.random(n).astype(np.float16)
        B = rng.random(n).astype(np.float16)
        C = np.zeros(n, np.float16)
        csr(A=cupy.asarray(A), B=cupy.asarray(B), C=(dC := cupy.asarray(C)), N=n)
        got = cupy.asnumpy(dC)
        exp = np.zeros(n, np.float16)
        exp[1:n - 1] = A[1:n - 1] + B[1:n - 1]
        assert np.array_equal(got.view(np.uint16), exp.view(np.uint16)), "not bit-exact vs numpy fp16"

    assert _run_in_fork(work) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("width", [8, 4])
def test_branched_tail_neighbor_stencil_bitexact(width):
    """D[i] = A[i-1] + A[i+1] over interior [1:N-1] (boundary lanes read neighbours): one fused
    kernel, bit-exact fp16 including the tail."""

    def work():
        import numpy as np
        import cupy
        sdfg = _prep(_neighbor16)
        sdfg.name = f"bt_neighbor16_w{width}"
        VectorizeGPU(VectorizeConfig(widths=(width, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        cu = "\n".join(c.clean_code for c in sdfg.generate_code() if c.language == "cu")
        assert cu.count("__global__ void") == 1
        shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
        csr = sdfg.compile()
        n = 1024
        rng = np.random.default_rng(100 + width)
        A = rng.random(n).astype(np.float16)
        D = np.zeros(n, np.float16)
        csr(A=cupy.asarray(A), D=(dD := cupy.asarray(D)), N=n)
        got = cupy.asnumpy(dD)
        exp = np.zeros(n, np.float16)
        exp[1:n - 1] = A[0:n - 2] + A[2:n]
        assert np.array_equal(got.view(np.uint16), exp.view(np.uint16)), "not bit-exact vs numpy fp16"

    assert _run_in_fork(work) == 0


if __name__ == "__main__":
    test_branched_tail_refused_on_cpu()
    test_assume_even_opts_out_of_branched_tail()
    test_default_gpu_k1_is_branched_tail()
    test_branched_tail_provably_nondivisible_does_not_raise()
    test_branched_tail_structure_if_vector_else_scalar()
    test_branched_tail_emits_single_kernel()
    test_branched_tail_elementwise_bitexact(8)
    test_branched_tail_elementwise_bitexact(4)
    test_branched_tail_neighbor_stencil_bitexact(8)
    test_branched_tail_neighbor_stencil_bitexact(4)
    print("all branched_tail tests passed")
