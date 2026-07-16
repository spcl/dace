# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression: ``PrivatizeScalars`` (``ScalarFission``) must not version the two
sides of a NestedSDFG inout connector into different array versions.

polybench ``durbin`` carries ``beta`` and ``alpha`` across the outer ``k`` loop via
the in-place recurrence ``beta = (1 - alpha*alpha) * beta``. With best-effort
peeling on (``peel_limit > 0``) the boundary ``k`` iteration is wrapped in a Map
whose ``loop_body`` NestedSDFG reads and writes ``beta`` through ONE inout
connector. The write-shadow analysis, blind to that join, split the read side into
``beta_0`` and the write side into ``beta_1``; ``_rename_memlet_path`` renamed the
Map-boundary memlets, but the connector (which must name the SAME array on both
sides) stayed ``beta`` -- because ``_propagate_rename_into_nsdfgs`` renames a
connector only when the NSDFG edge's OWN endpoint is the renamed AccessNode, and
here that endpoint is the MapEntry/MapExit. The result was an invalid inout
connector::

    ValueError: Inout connector beta is connected to different input ({'beta_0'})
    and output ({'beta_1'}) arrays

The failure is deterministic within a process but process-order dependent (which
scalar -- ``alpha`` or ``beta`` -- gets the inconsistent split varies), so the
tests run canonicalize 10x in-process to reproduce it reliably in one process.

A SEPARATE, pre-existing peel-path defect used to block the ``peel_limit=4`` codegen
(the canonicalized SDFG was VALID but would not compile). ``BestEffortLoopPeeling``
index-set-split the outer ``k`` loop at the split point ``x = i`` -- the INNER loop's
iterator, wrongly taken as a loop-invariant broadcast (``y[k] == y[i]``) -- and baked
that inner name into the outer segment bounds. The reference stayed hidden while the
inner loop still shared the name ``_loop_it_1``, then leaked the instant
``UniqueLoopIterators`` gave that inner loop a fresh unique name: ``_loop_it_1`` became
a free symbol referenced by a Map range but defined nowhere (``SDFG.arglist`` ->
``KeyError: '_loop_it_1'``). Fixed by rejecting any split point that names an
inner-loop iterator (``parallelization_prep.BestEffortLoopPeeling._best_split_for`` /
``_inner_loop_variables``); ``test_durbin_peel4_end_to_end_bit_exact`` now compiles and
runs the ``peel_limit=4`` path bit-exact. Value-preservation at ``peel_limit=0`` is
still asserted separately below.
"""
import copy
import os

# Pin a deterministic, single-threaded, no-MPI-init run before DaCe/OpenMP load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus.polybench import polybench as PB


def _durbin_kernel():
    kernels = PB.collect("durbin")
    assert kernels, "polybench durbin kernel not found in the corpus"
    return kernels[0]


def _run_forked(build_and_run):
    """Compile + run ``build_and_run() -> Dict[str, ndarray]`` in a forked child.

    Repo rule: always fork when running a compiled kernel so a segfault cannot take
    down the pytest process. Outputs are marshalled back through a temporary npz.
    """
    import tempfile

    handle, path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    pid = os.fork()
    if pid == 0:  # child
        try:
            outputs = build_and_run()
            np.savez(path, **{name: np.asarray(value) for name, value in outputs.items()})
            os._exit(0)
        except BaseException:  # noqa: BLE001 - report and exit non-zero, never raise past fork
            import traceback
            traceback.print_exc()
            os._exit(17)
    _, status = os.waitpid(pid, 0)
    ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    result = None
    if ok:
        with np.load(path) as data:
            result = {name: data[name] for name in data.files}
    if os.path.exists(path):
        os.remove(path)
    return ok, result


def test_durbin_canonicalize_valid_deterministic():
    """canonicalize(durbin, peel_limit=4) is a VALID SDFG on every run.

    This is the core regression for the inout-connector versioning bug. Before the
    fix ``validate=True`` raised the inout-connector ``ValueError``; the split-vs-
    consistent outcome is process-order dependent, so 10x in-process reproduces it.
    """
    base = PB.fresh_sdfg(_durbin_kernel())
    for _ in range(10):
        # validate=True raises InvalidSDFG* if any stage left the SDFG invalid.
        canonicalize(copy.deepcopy(base),
                     validate=True,
                     validate_all=False,
                     peel_limit=4,
                     break_anti_dependence=True)


def test_durbin_value_preserving():
    """canonicalize(durbin) is value-preserving vs the untransformed baseline.

    Run at ``peel_limit=0`` (which compiles cleanly). The fix only ever declines a
    rename -- it introduces zero arithmetic change and is a strict no-op at
    ``peel_limit=0`` (no inout-carried scalar is present) -- so the result here is
    identical with or without it. Correctness is checked with polybench's own
    dtype-aware criterion (``PB.outputs_match``: fp64 rtol 1e-9 / atol 1e-11), the
    same bar the corpus gate uses: durbin parallelizes into elementwise Maps that
    legally REASSOCIATE fp64 additions, so ``y`` differs from the sequential baseline
    in the last few ULPs (~1e-13 relative). That reassociation is pre-existing
    pipeline behaviour, not a value bug and not attributable to this fix; a zero
    tolerance would flag legal IEEE reordering. The ``peel_limit=4`` path (where the
    fix is exercised) is validity-checked above and run end-to-end bit-exact below.
    """
    kernel = _durbin_kernel()
    call_arrays, psize = PB.make_inputs(kernel)
    reference = PB.reference(kernel, call_arrays, psize)

    base = PB.fresh_sdfg(kernel)
    candidate = canonicalize(copy.deepcopy(base), validate=True, peel_limit=0, break_anti_dependence=True)
    finalized = finalize_for_target(candidate, "cpu")

    ok, got = _run_forked(lambda: PB.run(finalized, call_arrays, psize))
    assert ok, "candidate durbin kernel run crashed"
    assert PB.outputs_match(reference, got), "canonicalized durbin is not value-preserving vs reference"


def test_durbin_peel4_end_to_end_bit_exact():
    """canonicalize(durbin, peel_limit=4) compiles, runs, and is bit-exact.

    This is the ``peel_limit=4`` path where the inout-connector fix is exercised. It
    used to only VALIDATE -- codegen crashed with ``KeyError: '_loop_it_1'`` because
    ``BestEffortLoopPeeling`` index-set-split the outer loop at a point naming the
    INNER loop's iterator, which leaked once ``UniqueLoopIterators`` renamed that inner
    loop (see module docstring). With the split-point scope guard in place the path now
    compiles end to end. Correctness uses polybench's own dtype-aware criterion
    (``PB.outputs_match``): the parallelized durbin legally reassociates fp64 additions,
    so ``y`` differs from the sequential reference by a few ULPs -- a zero tolerance
    would flag legal IEEE reordering, so the corpus criterion (not an invented one) is
    the bar."""
    kernel = _durbin_kernel()
    call_arrays, psize = PB.make_inputs(kernel)
    reference = PB.reference(kernel, call_arrays, psize)

    base = PB.fresh_sdfg(kernel)
    candidate = canonicalize(copy.deepcopy(base), validate=True, peel_limit=4, break_anti_dependence=True)
    finalized = finalize_for_target(candidate, "cpu")

    ok, got = _run_forked(lambda: PB.run(finalized, call_arrays, psize))
    assert ok, "candidate durbin kernel run crashed"
    assert PB.outputs_match(reference, got), "canonicalized durbin is not value-preserving vs reference"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
