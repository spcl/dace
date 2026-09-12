# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widened fp16 loads whose base is NOT on a vector boundary (the +-1 stencil neighbour).

``c13a93c97`` widened a CUDA fp16 tile copy to ``half2`` only where the memlet offset is provably
divisible by the vector width. A stencil defeats that: with an even tile base ``A[i-1]`` and
``A[i+1]`` land on ODD elements, a 32-bit load from an odd ``__half`` is INVALID on NVIDIA (not
merely slow), and the gate correctly declined -- leaving heat3d reading every element with its own
``LDG.E.U16``.

The residue is nevertheless a compile-time constant whenever the strides are, so the load can read
the ALIGNED window below the access and cut its elements out of the register pair with ``PRMT``.
Neighbouring accesses in a row round DOWN to the same words, so the loads CSE and a 7-point stencil
reads 10 aligned words instead of 20 halves.

Covered:
  * the odd-offset load emits the shifted template argument, the even-offset one the plain aligned
    argument, and an odd-offset STORE stays scalar (widening it would clobber the neighbours);
  * a symbolic row stride widens only where a host-side ``stride % 2`` abort backs it, and
    declines again the moment that guard is removed;
  * a transient over the stencil's INTERIOR, strided by ``N - 2`` rather than ``N``, widens too;
  * the emitted SASS for a heat3d kernel has NO ``LDG.E.U16`` and does not grow;
  * bit-exact vs the NumPy fp16 oracle, boundary iterations included.

GPU-executing tests run in a fresh interpreter so a device fault cannot crash the pytest parent.
"""
import collections
import glob
import os
import re
import subprocess
import sys

import pytest

import dace
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import STRIDE_GUARD_PREFIX
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

N = dace.symbol("N")
tsteps = dace.symbol("tsteps")

# Constant so the row stride is a literal and the parity needs no runtime promise at all; the
# symbolic-stride counterpart buys the same parity with a guard (see
# ``test_symbolic_stride_widens_only_under_its_guard``).
NC = 128
H = dace.float16
# Module-level so the frontend folds them as literals; ``C8`` written inline inside the
# program becomes a host constant the kernel would have to copy in.
C8 = dace.float16(0.125)
C2 = dace.float16(2.0)


@dace.program
def _stencil3_2d(A: H[NC, NC], C: H[NC, NC]):
    # The inner map starts at 1 and is tiled by 2, so A[i, j] lands on an ODD element while its
    # +-1 neighbours land on even ones -- both residues in one kernel.
    for i, j in dace.map[1:NC - 1, 1:NC - 1]:
        C[i, j] = A[i, j - 1] + A[i, j] + A[i, j + 1]


@dace.program
def _stencil3_2d_symbolic(A: H[N, N], C: H[N, N]):
    for i, j in dace.map[1:N - 1, 1:N - 1]:
        C[i, j] = A[i, j - 1] + A[i, j] + A[i, j + 1]


@dace.program
def _heat3d(A: H[NC, NC, NC], B: H[NC, NC, NC]):
    for t in range(1, tsteps):
        B[1:-1, 1:-1,
          1:-1] = (C8 * (A[2:, 1:-1, 1:-1] - C2 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + C8 *
                   (A[1:-1, 2:, 1:-1] - C2 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + C8 *
                   (A[1:-1, 1:-1, 2:] - C2 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (C8 * (B[2:, 1:-1, 1:-1] - C2 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + C8 *
                   (B[1:-1, 2:, 1:-1] - C2 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + C8 *
                   (B[1:-1, 1:-1, 2:] - C2 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


@dace.program
def _heat3d_symbolic(A: H[N, N, N], B: H[N, N, N]):
    for t in range(1, tsteps):
        B[1:-1, 1:-1,
          1:-1] = (C8 * (A[2:, 1:-1, 1:-1] - C2 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + C8 *
                   (A[1:-1, 2:, 1:-1] - C2 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + C8 *
                   (A[1:-1, 1:-1, 2:] - C2 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (C8 * (B[2:, 1:-1, 1:-1] - C2 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + C8 *
                   (B[1:-1, 2:, 1:-1] - C2 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + C8 *
                   (B[1:-1, 1:-1, 2:] - C2 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def _heat3d_oracle(A, B, steps):
    """The same expression in the same fp16 op order on the host."""
    import numpy as np
    A, B = A.copy(), B.copy()
    o, tw = np.float16(0.125), np.float16(2.0)
    for _ in range(1, steps):
        B[1:-1, 1:-1,
          1:-1] = (o * (A[2:, 1:-1, 1:-1] - tw * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + o *
                   (A[1:-1, 2:, 1:-1] - tw * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + o *
                   (A[1:-1, 1:-1, 2:] - tw * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (o * (B[2:, 1:-1, 1:-1] - tw * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + o *
                   (B[1:-1, 2:, 1:-1] - tw * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + o *
                   (B[1:-1, 1:-1, 2:] - tw * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])
    return A, B


def _vectorized(prog, name=None, assume_even=False, fuse=False):
    """@dace.program -> simplify -> GPU offload -> the half2 GPU vectorizer.

    ``fuse`` adds the LoopToMap + MapFusion pair a real optimization recipe runs first. It is not
    cosmetic: fusing the stencil's two statements materializes the shared subexpressions as
    transients over the INTERIOR, whose row stride is ``N - 2`` rather than ``N``, and those are
    the accesses the alignment proof used to refuse.
    """
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.simplify()
    if fuse:
        sdfg.apply_transformations_repeated([LoopToMap])
        sdfg.apply_transformations_repeated([MapFusion])
        sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(2, ), assume_even=assume_even)).apply_pass(sdfg, {})
    if name:
        sdfg.name = name
    return sdfg


def _device_code(sdfg) -> str:
    return "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")


def _run_isolated(body: str) -> int:
    """Run the module-level ``body`` function in a FRESH interpreter; 0 == pass.

    A fork is not enough: an earlier test in the same session may already have initialized CUDA in
    the parent, and the inherited context is unusable in the child (``cudaErrorInitializationError``).
    A new interpreter starts from a clean device state, and still keeps a device fault out of the
    pytest parent. The parent's ``sys.path`` is handed down so the child imports the same dace.
    """
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    return subprocess.run([sys.executable, __file__, body], env=env).returncode


def _sass_ops(build_folder) -> collections.Counter:
    """SASS mnemonic histogram of every device kernel in the built shared object."""
    ops = collections.Counter()
    for lib in glob.glob(os.path.join(build_folder, "**", "*.so"), recursive=True):
        text = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True).stdout
        for op in re.findall(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9._]*)", text, re.M):
            ops[op] += 1
    return ops


# --------------------------------------------------------------------------------------------------
# Structural: which template argument each residue produces (no GPU device needed)
# --------------------------------------------------------------------------------------------------
def test_odd_neighbour_load_is_shifted_even_is_plain():
    """``A[i, j±1]`` (odd offset) carries the shift argument; the aligned tile carries only the
    alignment; and the odd-offset store keeps the 3-argument (scalar) call."""
    cu = _device_code(_vectorized(_stencil3_2d))
    loads = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", cu)
    assert loads, "no fp16 width-2 tile_load emitted"
    assert set(loads) == {", 4", ", 4, 1"}, \
        f"expected the aligned +-1 neighbours and the shifted centre, got {set(loads)}"
    stores = re.findall(r"tile_store<dace::float16, 2, false([^>]*)>", cu)
    assert stores and all(a == "" for a in stores), \
        f"an odd-offset store must stay per-element (widening it clobbers neighbours), got {set(stores)}"


def test_heat3d_widens_every_load():
    """heat3d's 7 stencil points split into aligned (``, 4``) and shifted (``, 4, 1``) loads with
    nothing left on the per-element path -- the case ``c13a93c97`` explicitly did not cover."""
    cu = _device_code(_vectorized(_heat3d))
    loads = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", cu)
    assert loads, "no fp16 width-2 tile_load emitted"
    assert not [a for a in loads if a == ""], "a heat3d fp16 tile load stayed on the per-element path"
    assert ", 4, 1" in loads and ", 4" in loads, f"expected both residues among the loads, got {set(loads)}"


def test_symbolic_stride_widens_only_under_its_guard():
    """An ``N``-column array leaves ``N*i + j`` with an unknown parity, and an unbacked guess there
    would emit an invalid unaligned 32-bit load. The default GPU path is ``branched_tail``, which
    guards the partial tile at runtime instead of constraining the extent, so nothing pins ``N``
    via the extent -- ``SplitMapForTileRemainder`` pins it directly instead, with a host-side
    ``N % 2`` abort, and the widening rests on THAT.

    The claim and the check are the same object, so deleting the guard tasklet must take the
    widening with it: that is the property keeping a misaligned device access out of reach."""
    sdfg = _vectorized(_stencil3_2d_symbolic)
    guards = [(st, n) for st in sdfg.states() for n in st.nodes()
              if isinstance(n, dace.nodes.Tasklet) and n.label.startswith(STRIDE_GUARD_PREFIX)]
    assert len(guards) == 1, f"expected exactly one guarded stride symbol, got {[n.label for _, n in guards]}"
    guard_state, guard_tasklet = guards[0]
    assert guard_tasklet.label == "tile_stride_div_N_2", f"guard names the wrong fact: {guard_tasklet.label}"
    code = guard_tasklet.code.as_string
    assert "(N) % 2 != 0" in code and "(N) < 2" in code, f"guard does not check N's parity: {code}"
    assert "abort()" in code, f"a violated stride promise must trap, not warn: {code}"

    loads = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", _device_code(sdfg))
    assert loads, "no fp16 width-2 tile_load emitted"
    assert set(loads) == {", 4", ", 4, 1"}, \
        f"the guarded symbolic stride must widen both residues, got {set(loads)}"

    # Same SDFG minus the check: the fact must not survive it.
    guard_state.remove_node(guard_tasklet)
    unguarded = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", _device_code(sdfg))
    assert unguarded, "no fp16 width-2 tile_load emitted"
    assert all(a == "" for a in unguarded), \
        f"a symbolic row stride was claimed aligned with no runtime check behind it, got {set(unguarded)}"


def test_the_stride_guard_compiles_wherever_its_map_ended_up():
    """The guard has to be valid C++ on the host AND valid CUDA on the device.

    It is prepended to the SDFG that owns the checked map, and that SDFG can be a nested one the
    offloading placed on the device -- where ``fprintf`` and ``abort`` are host-only and nvcc
    refuses the whole translation unit ("calling a __host__ function from a __device__ function").
    Both spellings are emitted, behind ``__CUDA_ARCH__`` / ``__HIP_DEVICE_COMPILE__``, so whichever
    compiler reads the tasklet finds the one it can call.
    """
    sdfg = _vectorized(_stencil3_2d_symbolic)
    guards = [
        n for st in sdfg.states() for n in st.nodes()
        if isinstance(n, dace.nodes.Tasklet) and n.label.startswith(STRIDE_GUARD_PREFIX)
    ]
    assert guards, "no stride guard emitted; the check below would be vacuous"
    code = guards[0].code.as_string
    assert "__CUDA_ARCH__" in code and "__HIP_DEVICE_COMPILE__" in code, \
        f"the guard is not split by compilation target, so one target cannot compile it: {code}"
    for host_only, device_only in (("fprintf", "printf"), ("abort()", "__trap()")):
        assert host_only in code and device_only in code, \
            f"the guard misses the {device_only!r} arm that device code needs: {code}"


def test_interior_transient_stride_widens():
    """A transient covering a stencil's INTERIOR is strided by ``N - 2``, and that expression is
    not provably non-negative as written -- so the coefficient test refused it even though the very
    extent fact that decides its parity, ``N = 2t + 2``, also makes the stride ``2t``. heat3d's
    fused kernel holds two such transients; before the extent rewrite was moved ahead of the test
    they were the only per-element loads left in the kernel."""
    cu = _device_code(_vectorized(_heat3d_symbolic, assume_even=True, fuse=True))
    strided = re.findall(r"tile_load<dace::float16, 2, false[^>]*>\([^,]+, \([A-Za-z_0-9]+ \+ \([^)]*ipow\(N - 2", cu)
    assert strided, "no (N-2)-strided access in the fused kernel; the test proves nothing"
    loads = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", cu)
    assert loads, "no fp16 width-2 tile_load emitted"
    assert not [a for a in loads if a == ""], \
        f"an (N-2)-strided transient stayed on the per-element path, got {collections.Counter(loads)}"


def test_symbolic_stride_widens_under_assume_even():
    """``assume_even`` turns the same symbolic stride usable: the tiled extent ``N - 2`` is then a
    guaranteed multiple of 2 (raised on when provably violated, host-guarded otherwise), so
    ``N = 2t + 2`` and every ``N*i`` term has a decidable parity. This is the precondition the
    widening needs on a symbolic shape -- without it a multi-dim stencil never widens."""
    cu = _device_code(_vectorized(_stencil3_2d_symbolic, assume_even=True))
    loads = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", cu)
    assert loads, "no fp16 width-2 tile_load emitted"
    assert set(loads) == {", 4", ", 4, 1"}, \
        f"expected the aligned +-1 neighbours and the shifted centre, got {set(loads)}"


# --------------------------------------------------------------------------------------------------
# GPU: the SASS gate and the numeric gate. Each body runs in a fresh interpreter (see
# ``_run_isolated``); the pytest wrappers only check its exit status.
# --------------------------------------------------------------------------------------------------
def _body_heat3d_sass():
    """The stencil kernels must load 32 bits at a time. ``PRMT`` is not the metric -- an aligned
    load plus a register ``PRMT`` is the intended lowering -- ``LDG.E.U16`` is, plus the total
    instruction count, which must not grow paying for the extraction."""
    sdfg = _vectorized(_heat3d, name="shifted_heat3d_sass")
    sdfg.compile()
    ops = _sass_ops(sdfg.build_folder)
    assert sum(ops.values()) > 0, "no SASS found in the built object"
    u16 = sum(v for k, v in ops.items() if k.startswith("LDG.E.U16"))
    wide = sum(v for k, v in ops.items() if k.startswith("LDG.E") and not k.startswith("LDG.E.U16"))
    # The scalar copy-in kernel keeps one 16-bit load; the two stencil kernels must have none.
    assert u16 <= 1, f"stencil kernels still load fp16 element-by-element: {u16} LDG.E.U16"
    assert wide >= 20, f"expected the widened aligned loads to appear, got {wide}"


def _body_heat3d_bitexact():
    """Bit-exact vs the NumPy fp16 oracle over the WHOLE array, so the untouched boundary planes
    and the widened window's overshoot element are both checked."""
    import numpy as np
    import cupy
    steps = 3
    rng = np.random.default_rng(0)
    A0 = rng.integers(0, 8, size=(NC, NC, NC)).astype(np.float16)
    B0 = rng.integers(0, 8, size=(NC, NC, NC)).astype(np.float16)
    expA, expB = _heat3d_oracle(A0, B0, steps)
    csr = _vectorized(_heat3d, name="shifted_heat3d_numeric").compile()
    dA, dB = cupy.asarray(A0), cupy.asarray(B0)
    csr(A=dA, B=dB, tsteps=steps)
    assert np.array_equal(cupy.asnumpy(dA).view(np.uint16), expA.view(np.uint16)), "A not bit-exact vs numpy fp16"
    assert np.array_equal(cupy.asnumpy(dB).view(np.uint16), expB.view(np.uint16)), "B not bit-exact vs numpy fp16"


def _body_symbolic_heat3d():
    """The shape heat3d is actually run in -- symbolic ``N`` under ``assume_even`` -- widens every
    load and stays bit-exact at several extents."""
    import numpy as np
    import cupy
    sdfg = _vectorized(_heat3d_symbolic, name="shifted_heat3d_sym", assume_even=True)
    loads = re.findall(r"tile_load<dace::float16, 2, false([^>]*)>", _device_code(sdfg))
    assert loads and not [a for a in loads if a == ""], "a symbolic-N heat3d load stayed per-element"
    csr = sdfg.compile()
    ops = _sass_ops(sdfg.build_folder)
    u16 = sum(v for k, v in ops.items() if k.startswith("LDG.E.U16"))
    assert u16 <= 1, f"stencil kernels still load fp16 element-by-element: {u16} LDG.E.U16"
    steps = 3
    for n in (64, 66):
        rng = np.random.default_rng(n)
        A0 = rng.integers(0, 8, size=(n, n, n)).astype(np.float16)
        B0 = rng.integers(0, 8, size=(n, n, n)).astype(np.float16)
        expA, expB = _heat3d_oracle(A0, B0, steps)
        dA, dB = cupy.asarray(A0), cupy.asarray(B0)
        csr(A=dA, B=dB, N=n, tsteps=steps)
        assert np.array_equal(cupy.asnumpy(dA).view(np.uint16), expA.view(np.uint16)), f"N={n}: A not bit-exact"
        assert np.array_equal(cupy.asnumpy(dB).view(np.uint16), expB.view(np.uint16)), f"N={n}: B not bit-exact"


@pytest.mark.gpu
def test_heat3d_sass_has_no_16bit_load():
    assert _run_isolated("_body_heat3d_sass") == 0


@pytest.mark.gpu
def test_heat3d_bitexact_including_boundaries():
    assert _run_isolated("_body_heat3d_bitexact") == 0


@pytest.mark.gpu
def test_symbolic_heat3d_bitexact_and_widened():
    assert _run_isolated("_body_symbolic_heat3d") == 0


if __name__ == "__main__":
    if len(sys.argv) > 1:  # re-entry from _run_isolated: run the named body only
        globals()[sys.argv[1]]()
    else:
        test_odd_neighbour_load_is_shifted_even_is_plain()
        test_heat3d_widens_every_load()
        test_symbolic_stride_widens_only_under_its_guard()
        test_interior_transient_stride_widens()
        test_symbolic_stride_widens_under_assume_even()
        test_heat3d_sass_has_no_16bit_load()
        test_heat3d_bitexact_including_boundaries()
        test_symbolic_heat3d_bitexact_and_widened()
