# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression guard: ``canonicalize`` must not explode polybench ``k3mm``.

``k3mm`` is ``G[:] = A @ B @ C @ D`` -- three chained MatMul LIBRARY NODES and no
explicit loops (its structural counts are ``[loops, maps, reduce, scan] == [0, 0, 0, 0]``
both at baseline and after ``LoopToMap``). Canonicalization is therefore almost a
no-op on it: the loop-centric stages find no loop to unroll, peel or fuse, the
MatMul nodes survive canon AND ``finalize_for_target('cpu')`` unexpanded, and the
generated code stays within a few KB of the untransformed baseline.

This file exists because ``k3mm`` was reported to take >900s to COMPILE after
canonicalization. That did not reproduce: measured on this tree, canon is ~0.3s,
codegen ~0.1s and the C++ compile ~2.6s (vs ~2.3s for the untransformed baseline),
with generated code growing only 5.1KB -> 5.9KB. The guard below therefore pins the
structural facts that would ALL have to break for a compile-time blowup to be
possible at all -- code growth is the only mechanism by which a canon change could
push this kernel's C++ compile into the minutes.

Assertion strategy (deliberate):

* The load-bearing assertions are the STRUCTURAL proxies -- generated-code size and
  node count. They are deterministic and independent of machine load, so they cannot
  flake.
* The wall-clock bounds are extremely generous (~100x the measured values). This box
  is shared and routinely runs concurrent pytest sweeps and nvcc builds, so a tight
  timing assertion would flake; these bounds only catch a 900s-class regression, not
  a 2x slowdown. A flaky perf test is worse than none.
"""
import time

from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus.polybench import polybench as PB

#: The CPU canonicalize knob set the numerical corpus gate uses.
_CPU = dict(target='cpu',
            peel_limit=4,
            break_anti_dependence=True,
            interchange_carry_with_map=True,
            scatter_to_guarded_maps=True)

#: Generated-code ceiling. Measured: 5105 bytes untransformed, 5861 after canon.
#: A canon change that unrolled/fused/expanded this kernel into a compile-time
#: blowup would blow past this by orders of magnitude.
_MAX_CODE_BYTES = 200_000

#: SDFG node ceiling. Measured: 11 untransformed, 17 after canon.
_MAX_NODES = 500

#: Wall-clock ceilings -- see the module docstring: ~100x measured, load-tolerant.
_MAX_CANON_SECONDS = 120.0
_MAX_COMPILE_SECONDS = 300.0


def _kernel():
    kernels = PB.collect('k3mm')
    assert kernels, "polybench corpus does not expose a 'k3mm' kernel"
    return kernels[0]


def _canonicalized():
    """A canonicalized + CPU-finalized k3mm, and the seconds canon itself took."""
    sdfg = PB.fresh_sdfg(_kernel())
    t0 = time.perf_counter()
    canonicalize(sdfg, validate=True, validate_all=False, **_CPU)
    canon_seconds = time.perf_counter() - t0
    finalize_for_target(sdfg, 'cpu')
    return sdfg, canon_seconds


def _code_bytes(sdfg):
    return sum(len(c.clean_code) for c in sdfg.generate_code())


def _node_count(sdfg):
    return sum(1 for _ in sdfg.all_nodes_recursive())


def test_k3mm_canonicalize_does_not_explode():
    """canon leaves k3mm's three MatMul library nodes alone and does not grow the code.

    Deterministic (no compile, no timing dependence) -- this is the assertion that
    actually catches a code-growth regression.
    """
    sdfg, canon_seconds = _canonicalized()

    libs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.LibraryNode)]
    assert len(libs) == 3, (f"expected k3mm's 3 chained MatMul library nodes to survive canon+finalize "
                            f"unexpanded, got {len(libs)}: {[type(n).__name__ for n in libs]}")

    nodes = _node_count(sdfg)
    assert nodes <= _MAX_NODES, f"canon exploded k3mm's SDFG: {nodes} nodes (> {_MAX_NODES}); expected ~17"

    code_bytes = _code_bytes(sdfg)
    assert code_bytes <= _MAX_CODE_BYTES, (f"canon exploded k3mm's generated code: {code_bytes} bytes "
                                           f"(> {_MAX_CODE_BYTES}); expected ~5.9KB")

    assert canon_seconds <= _MAX_CANON_SECONDS, (f"canonicalize(k3mm) took {canon_seconds:.1f}s "
                                                 f"(> {_MAX_CANON_SECONDS}s); expected ~0.3s")


def test_k3mm_canonicalized_compiles_and_is_value_preserving():
    """The canonicalized k3mm compiles in sane time and matches the polybench reference.

    Compiles for real (that is the reported failure mode), so the bound is generous;
    correctness uses the corpus's own dtype-aware criterion.
    """
    kernel = _kernel()
    arrays, psize = PB.make_inputs(kernel)
    ref = PB.reference(kernel, arrays, psize)

    sdfg, _ = _canonicalized()
    sdfg.name = f'{sdfg.name}_compile_time_guard'

    t0 = time.perf_counter()
    got = PB.run(sdfg, arrays, psize)
    compile_and_run_seconds = time.perf_counter() - t0

    assert compile_and_run_seconds <= _MAX_COMPILE_SECONDS, (
        f"canonicalized k3mm took {compile_and_run_seconds:.1f}s to compile+run "
        f"(> {_MAX_COMPILE_SECONDS}s); expected ~3s")
    assert PB.outputs_match(ref, got), "canonicalized k3mm is not value-preserving vs the polybench reference"
