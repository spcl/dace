# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression: ``TrivialLoopElimination`` must not eliminate a ZERO-trip loop.

polybench ``nussinov`` walks a triangular domain::

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            ...  table[i, j] = max(table[i, j], ...)

At the LAST outer iteration ``i = N-1`` the inner loop is ``for j in range(N, N)`` --
provably EMPTY. Best-effort peeling (``peel_limit=4``) materializes that boundary
iteration as its own loop, so the pipeline is left holding a literal zero-trip
``for j = N; j < N; j += 1``.

``TrivialLoopElimination.can_be_applied`` classified that loop as trivial and spliced
its body into the parent with ``j = N``. Its guard only ever rejected loops running
TWO-or-more times::

    if stride > 0 and start + stride < end + 1:   # ``get_loop_end`` is INCLUSIVE
        return False

For the empty loop ``start = N``, ``end = N - 1``, ``stride = 1``, so the test reads
``N + 1 < N`` -> False -> "trivial". Nothing checked that the loop runs at ALL, so a
zero-trip loop passed and its body was fabricated into existence at ``j = N``, one
column past the end of ``table[N, N]``::

    dace.sdfg.validation.InvalidSDFGEdgeError: Memlet subset out-of-bounds
      memlet  : table[N - 1, N]
      data    : table   shape=(N, N)
      axis 1  : max element N   vs bound N

(The exception's ``__str__`` re-raises on a stale node id, masking the message as
``<exception str() failed>`` -- hence the explicit ``.message`` assert style here.)

The fix requires a provable FIRST iteration before eliminating (``start <= end`` for a
positive stride, ``start >= end`` for a negative one); an undecidable comparison refuses,
which is the sound direction. Genuinely single-trip loops (``range(1, 2)``, and the
``for tv = 0; tv < 1`` wrappers ``MoveIfIntoLoop`` emits for the ``untrivialize`` stage
to splice back out) are unaffected -- their ``start == end``.

The failure is deterministic within a process but the canonicalize pipeline is
process-order dependent, so the validity test runs canonicalize 5x in-process.

A SECOND defect blocked nussinov's ``peel_limit >= 2`` CODEGEN and is ALSO fixed now (it was
previously masked, because the out-of-bounds error above aborted canonicalize before codegen
was ever reached). It surfaced as ``SDFG.arglist() -> KeyError: '_loop_it_1'`` -- the same
signature ``durbin_scalar_versioning_test`` documents -- but has a DIFFERENT cause:
``TrivialLoopElimination.apply`` substituted the eliminated iterator with
``LoopRegion.replace``, and ``ControlGraphView.replace`` hand-walks ``nodes()`` / ``edges()``
without routing through ``replace_dict``. It therefore silently missed a ``ConditionalBlock``'s
branch CONDITIONS (stored in ``_branches``, not ``nodes()``) and a nested ``LoopRegion``'s own
init/condition/update, leaving four references to the eliminated iterator behind. They stayed
harmless while a peel sibling still bound the name, and dangled the moment
``UniqueLoopIterators`` renamed it. ``BestEffortLoopPeeling`` is NOT involved -- no split funnel
fires on nussinov at all. The fix routes the substitution through ``replace_dict``.

``peel_limit=1`` DOES exercise this fix end to end (it raised the out-of-bounds error before
it), so end-to-end bit-exactness is asserted there.
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

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus.polybench import polybench as PB


def _nussinov_kernel():
    kernels = PB.collect("nussinov")
    assert kernels, "polybench nussinov kernel not found in the corpus"
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


def _zero_trip_sdfg() -> dace.SDFG:
    """``for j = N; j < N; j += 1`` writing ``table[N-1, j]`` -- nussinov's inner loop at
    the peeled ``i = N-1``, reduced to the smallest SDFG that carries the defect."""
    N = dace.symbol("N", nonnegative=True)
    sdfg = dace.SDFG("zero_trip_loop")
    sdfg.add_array("table", [N, N], dace.int32)
    sdfg.add_symbol("j", dace.int64)

    loop = LoopRegion("jloop", "j < N", "j", "j = N", "j = j + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("set", {}, {"out"}, "out = 1")
    body.add_edge(tasklet, "out", body.add_write("table"), None, dace.Memlet(data="table", subset="N - 1, j"))
    return sdfg


def test_trivial_loop_elimination_refuses_zero_trip_loop():
    """The unit regression: a provably empty loop is NOT a trivial loop.

    Before the fix this applied once and rewrote the write memlet to the out-of-bounds
    ``table[N - 1, N]``. The body must stay inside the loop, indexed by ``j``.
    """
    sdfg = _zero_trip_sdfg()
    applied = sdfg.apply_transformations_repeated([TrivialLoopElimination])
    assert applied == 0, "TrivialLoopElimination eliminated a ZERO-trip loop (its body must never run)"

    memlets = [e.data for state in sdfg.all_states() for e in state.edges() if e.data.data == "table"]
    assert memlets, "expected a write to table"
    for memlet in memlets:
        assert "j" in str(memlet.subset), f"zero-trip body was spliced out with j = N: table[{memlet.subset}]"


def test_trivial_loop_elimination_still_eliminates_single_trip_loop():
    """Guard the fix against over-refusal: a genuine one-iteration loop still folds.

    ``for j = N-1; j < N; j += 1`` runs exactly once (``start == end == N-1``), so the
    added first-iteration check must not reject it.
    """
    N = dace.symbol("N", nonnegative=True)
    sdfg = dace.SDFG("single_trip_loop")
    sdfg.add_array("table", [N, N], dace.int32)
    sdfg.add_symbol("j", dace.int64)

    loop = LoopRegion("jloop", "j < N", "j", "j = N - 1", "j = j + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("set", {}, {"out"}, "out = 1")
    body.add_edge(tasklet, "out", body.add_write("table"), None, dace.Memlet(data="table", subset="N - 1, j"))

    assert sdfg.apply_transformations_repeated([TrivialLoopElimination]) == 1, \
        "a genuine single-iteration loop must still be eliminated"


def test_nussinov_canonicalize_valid_deterministic():
    """canonicalize(nussinov, peel_limit=4) is a VALID SDFG on every run.

    The core regression. Before the fix ``validate=True`` raised ``InvalidSDFGEdgeError``
    ("Memlet subset out-of-bounds", ``table[N - 1, N]`` against ``table[N, N]``). Runs 5x
    in-process because the pipeline's pass order is process-order dependent.
    """
    base = PB.fresh_sdfg(_nussinov_kernel())
    for _ in range(5):
        # validate=True raises InvalidSDFG* if any stage left the SDFG invalid.
        canonicalize(copy.deepcopy(base), validate=True, validate_all=False, peel_limit=4, break_anti_dependence=True)


def _end_to_end_bit_exact(peel_limit: int):
    """canonicalize -> finalize -> compile -> run, compared against the polybench reference.

    ``table`` and ``seq`` are integer arrays, so polybench's own dtype-aware criterion
    (``PB.outputs_match`` -> ``_tol_for`` returns ``(0, 0)`` for integer kinds) degrades to an
    exact ``np.array_equal``: this asserts BIT-EXACT equality without inventing a tolerance.
    """
    kernel = _nussinov_kernel()
    call_arrays, psize = PB.make_inputs(kernel)
    reference = PB.reference(kernel, call_arrays, psize)

    base = PB.fresh_sdfg(kernel)
    candidate = canonicalize(copy.deepcopy(base), validate=True, peel_limit=peel_limit, break_anti_dependence=True)
    finalized = finalize_for_target(candidate, "cpu")

    ok, got = _run_forked(lambda: PB.run(finalized, call_arrays, psize))
    assert ok, "candidate nussinov kernel run crashed"
    assert PB.outputs_match(reference, got), "canonicalized nussinov is not value-preserving vs reference"


def test_nussinov_peel1_end_to_end_bit_exact():
    """canonicalize(nussinov, peel_limit=1) compiles, runs, and is bit-exact.

    The end-to-end half of this regression. ``peel_limit=1`` already materializes the empty
    ``j`` loop, so before the fix this raised the out-of-bounds error during canonicalize and
    never reached codegen; it now runs clean. The fix only ever declines an unsound
    elimination, so it cannot perturb arithmetic -- the point is that the path reaches codegen
    at all, and that the result still matches the reference exactly.
    """
    _end_to_end_bit_exact(1)


def test_nussinov_peel4_end_to_end_bit_exact():
    """canonicalize(nussinov, peel_limit=4) end to end -- the pipeline's default CPU peel preset.

    Records the remaining known-open blocker on nussinov's peel path. The canonicalize half of
    this configuration is already asserted green by
    ``test_nussinov_canonicalize_valid_deterministic``; only codegen is blocked. Marked
    non-strict so it turns green on its own once the peeling defect is fixed.
    """
    _end_to_end_bit_exact(4)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
