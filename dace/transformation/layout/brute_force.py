# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Brute-force GLOBAL layout sweep -- the layout optimizer (no cost model).

The SC26 layout picker is deliberately brute force: enumerate global layout candidates for a
kernel (dimension permutations, block factors, zip groupings), apply each, compile, VERIFY against
a numpy oracle, measure time, and keep the fastest correct one. The capability -- the transforms +
algebra + ``LayoutChange`` node -- is the deliverable; the picker is a sweep over global layouts.

This module is the reusable engine:

  * :func:`sweep` runs a set of named candidate SDFG builders through compile -> run -> verify ->
    (optional) time, and returns the results ranked (correct first, then by time).
  * :func:`permutation_candidates` / :func:`block_candidates` enumerate the common candidate
    families (per-array dimension permutations; per-dimension block factors) as ``apply`` closures
    over the layout passes.
  * :func:`time_cpu` is a median-of-reps wall-clock timer (best-effort; timing on a shared/loaded
    host is noisy -- correctness is the invariant, speed is advisory).
  * :func:`time_gpu` is the GPU peer: it records CUDA start/stop events on a SINGLE stream around
    each call and synchronizes on the stop event (not the whole device). :func:`sweep` picks the
    timer from its ``device`` argument.

The caller supplies a ``run`` closure that binds fresh inputs, executes a compiled SDFG, and returns
the outputs to compare, plus the reference outputs; the engine owns the compile/verify/time/rank
loop. Timing never runs for an incorrect candidate.
"""
import contextlib
import itertools
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy

import dace
from dace.transformation.layout.isolation import run_isolated


@dataclass
class SweepResult:
    """One candidate's outcome: whether it verified, its median time (if timed), and any error.
    ``order`` is the candidate's ENUMERATION position -- the identity-first tie-break needs it after
    ranking has reordered the list."""
    name: str
    correct: bool
    time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    order: int = 0


def time_cpu(fn: Callable[[], Any], reps: int = 5, warmup: int = 1) -> float:
    """Median wall-clock time (seconds) of ``fn`` over ``reps`` runs after ``warmup`` warmups."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = _time.perf_counter()
        fn()
        samples.append(_time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def time_gpu(fn: Callable[[], Any], reps: int = 5, warmup: int = 1) -> float:
    """Median GPU time (seconds) of ``fn`` measured with CUDA events on cupy's CURRENT stream.

    The events are recorded with no stream argument, i.e. on cupy's current stream, which is the
    null/default stream unless the caller is inside a ``with stream:`` context -- do NOT wrap this in
    one. A start and stop event are recorded on that stream around each call and the stop
    event is synchronized (a stream-scoped wait, not a whole-device ``deviceSynchronize``). This
    assumes ``fn`` launches its work on that SAME single stream -- so the SDFG must be COMPILED inside
    :func:`single_default_stream` (dace emits the legacy default stream), which the timer shares;
    :func:`sweep` does this for ``device="gpu"``. Recording on a fresh non-default stream would NOT
    bracket dace's kernels (dace does not run on cupy's current stream), so the events are taken on
    the default stream deliberately.
    ``cupy`` reports milliseconds; converted to seconds so the unit matches :func:`time_cpu`.

    Like :func:`time_cpu` this times the WHOLE call: ``fn`` re-enters ``SDFG.__call__`` (argument
    marshalling, cache lookup) and dace synchronizes before returning, and all of that CPU time sits
    between the start and stop markers on the stream. For a small kernel the result is therefore
    launch-overhead dominated (measurably: a 48x32 add reports ~290ms, ~1.0x the wall clock) and only
    ranks layouts whose cost differences exceed that overhead. To measure the compute region alone,
    pass ``dace.transformation.layout.timing.compute_region_timer`` as :func:`sweep`'s ``timer``.
    """
    import cupy  # only needed on the GPU path; keep the module importable without a GPU

    start = cupy.cuda.Event()
    stop = cupy.cuda.Event()
    samples = []
    for _ in range(warmup):
        fn()
    cupy.cuda.get_current_stream().synchronize()
    for _ in range(reps):
        start.record()  # cupy's CURRENT stream -- the null/default stream when no stream context is active
        fn()
        stop.record()  # same current stream, so start/stop bracket dace's default-stream kernels
        stop.synchronize()  # wait on the stop EVENT, single stream
        samples.append(cupy.cuda.get_elapsed_time(start, stop) * 1e-3)
    samples.sort()
    return samples[len(samples) // 2]


@contextlib.contextmanager
def single_default_stream():
    """Compile GPU code onto ONE stream -- the legacy default stream.

    dace's default (``compiler.cuda.max_concurrent_streams = 0``) spreads kernels across many
    streams; the default-stream CUDA events :func:`time_gpu` records would then miss the work on the
    other streams and under-report. Setting ``max_concurrent_streams = -1`` makes dace emit the
    single legacy default stream the timer records on. The value is read at CODE-GENERATION time, so
    the SDFG must be COMPILED inside this context (in the sweep, ``run`` compiles on its first call,
    which is why the whole candidate loop is wrapped). :func:`sweep` enters it automatically for
    ``device="gpu"``; a direct :func:`time_gpu` caller must compile inside it too.

    .. note:: dace's build cache keys on the SDFG, not on this config, so a same-named SDFG compiled
       earlier under a different stream setting can be reused stale -- clean ``.dacecache`` between
       runs that change the stream mode.
    """
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
        yield


def sweep(candidates: Dict[str, Callable[[], dace.SDFG]],
          run: Callable[[dace.SDFG], Dict[str, numpy.ndarray]],
          reference: Dict[str, numpy.ndarray],
          compare: Callable[[numpy.ndarray, numpy.ndarray], bool] = numpy.allclose,
          reps: int = 5,
          warmup: int = 1,
          do_time: bool = True,
          device: str = "cpu",
          timer: Optional[Callable[[dace.SDFG, Callable, int, int], Optional[float]]] = None,
          attempt_log: Optional[str] = None,
          isolate: bool = False) -> List[SweepResult]:
    """Compile, run, verify and (optionally) time each candidate; return results ranked.

    Two phases, in the design's batch order: EVERY candidate is built and compiled first; only then
    are candidates run/verified/timed -- so no candidate is timed right after its own compile burst
    (DVFS/cache state), and gcc noise never lands inside the timing phase. ``attempt_log`` (a file
    path) records each candidate name before its build and before its run, so a crashed campaign
    shows which candidate killed it.

    :param candidates: ``{name: make_sdfg}`` -- each ``make_sdfg()`` returns a FRESH SDFG with that
                       candidate's global layout already applied.
    :param run: ``run(sdfg) -> {output_name: array}`` -- binds fresh inputs, executes ``sdfg`` and
                returns the outputs to check (the caller owns argument marshalling).
    :param reference: the numpy-oracle outputs, ``{output_name: array}``.
    :param compare: elementwise correctness predicate (default ``numpy.allclose``).
    :param do_time: time each CORRECT candidate (never an incorrect one).
    :param device: ``"cpu"`` or ``"gpu"``. Selects the device-appropriate library lowering for any
                   layout-inserted node (via :func:`select_layout_lowering`) and, when ``timer`` is
                   not given, the matching wall-clock (:func:`time_cpu`) or CUDA-event
                   (:func:`time_gpu`) timer. ``"gpu"`` also compiles the whole sweep inside
                   :func:`single_default_stream` so the timer's single-stream assumption holds.
    :param timer: ``timer(sdfg, run, reps, warmup) -> time`` for a correct candidate. ``None`` =
                  the ``device``-selected whole-call timer; pass
                  ``dace.transformation.layout.timing.compute_region_timer`` to time only the compute
                  region (excluding the relayout copies), or ``compute_region_stats_timer`` to also
                  record the spread/contended trust signal (a timer may return
                  ``(time, metadata_dict)``; the dict lands in ``SweepResult.metadata``). One timer
                  is used for the whole sweep, so its unit is consistent and the ranking is valid.
    :param isolate: run each candidate's verify+time in a forked child
                    (:func:`~dace.transformation.layout.isolation.run_isolated`), so a SEGFAULT or
                    runaway in generated code is recorded as a non-viable candidate instead of
                    killing the campaign. Compilation stays in the parent (so the timing phase is
                    still gcc-free and a compile failure is a normal error), and the OpenMP pool is
                    torn down before each fork so a parallel kernel does not deadlock the child.
                    CPU only -- a CUDA context cannot survive ``fork``, so ``isolate`` with
                    ``device="gpu"`` is refused.
    :returns: results, correct-first then ascending time.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if isolate and device == "gpu":
        raise ValueError("sweep(isolate=True) is CPU-only: a CUDA context cannot survive os.fork")
    from dace.transformation.layout.select_lowering import select_layout_lowering

    def log_attempt(phase: str, name: str) -> None:
        if attempt_log is not None:
            with open(attempt_log, "a") as f:
                f.write(f"{phase} {name}\n")

    default_timer = time_gpu if device == "gpu" else time_cpu

    def verify_and_time(sdfg: dace.SDFG) -> Dict[str, Any]:
        """Run, verify, and (if correct) time one already-compiled candidate. JSON-able so it can
        run either in-process or inside a forked child (:func:`run_isolated`)."""
        out = run(sdfg)
        correct = all(name_ in out and compare(out[name_], ref) for name_, ref in reference.items())
        verdict: Dict[str, Any] = {"correct": bool(correct), "time": None, "metadata": {}}
        if do_time and correct:
            try:
                t = timer(sdfg, run, reps, warmup) if timer is not None else default_timer(
                    lambda: run(sdfg), reps, warmup)
                if isinstance(t, (tuple, list)):  # a stats timer: (median, {"spread": ..., ...})
                    t, timer_metadata = t
                    verdict["metadata"].update(timer_metadata)
                verdict["time"] = None if t is None else float(t)
            except Exception as ex:  # timing is ADVISORY: a timer failure must not demote a correct candidate
                verdict["metadata"]["timing_error"] = f"{type(ex).__name__}: {ex}"
        return verdict

    def fold(result: SweepResult, verdict: Dict[str, Any]) -> None:
        if "error" in verdict:
            result.error = verdict["error"]
            return
        result.correct = verdict["correct"]
        result.time = verdict["time"]
        result.metadata.update(verdict["metadata"])

    stream_ctx = single_default_stream() if device == "gpu" else contextlib.nullcontext()
    results: List[SweepResult] = []
    with stream_ctx:  # GPU: force the single default stream at compile time (run() compiles)
        if isolate:
            # Compile in the PARENT (phase 1), then run+verify+time each candidate in a forked child
            # (phase 2), so a segfault/runaway in generated code is a non-viable result, not a dead
            # campaign. Verify and time share the child (the value cannot cross the fork boundary),
            # but the gcc burst already happened in the parent, so the timing is still gcc-free.
            staged: List[Tuple[SweepResult, dace.SDFG]] = []
            for order, (name, make) in enumerate(candidates.items()):
                log_attempt("compile", name)
                try:
                    sdfg = make()
                    select_layout_lowering(sdfg, device)
                    sdfg.compile()
                except Exception as ex:  # a candidate that fails to build/compile is simply not viable
                    results.append(SweepResult(name, False, None, f"{type(ex).__name__}: {ex}", order=order))
                    continue
                result = SweepResult(name, False, order=order)
                results.append(result)
                staged.append((result, sdfg))
            for result, sdfg in staged:
                log_attempt("run", result.name)
                fold(result, run_isolated(lambda sdfg=sdfg: verify_and_time(sdfg)))
        else:
            # In-process, two batch phases: VERIFY every candidate by running it (run() compiles on
            # first call), THEN time the verified ones back-to-back -- so no candidate is timed right
            # after its own compile burst. A caller whose run closure fakes execution over a stub
            # SDFG still works (no standalone compile is forced).
            verified: List[Tuple[SweepResult, dace.SDFG]] = []
            for order, (name, make) in enumerate(candidates.items()):
                log_attempt("verify", name)
                try:
                    sdfg = make()
                    select_layout_lowering(sdfg, device)  # the transforms left lowering unset; choose it here
                    out = run(sdfg)
                    correct = all(name_ in out and compare(out[name_], ref) for name_, ref in reference.items())
                except Exception as ex:  # a candidate that fails to build/compile/run is simply not viable
                    results.append(SweepResult(name, False, None, f"{type(ex).__name__}: {ex}", order=order))
                    continue
                result = SweepResult(name, correct, order=order)
                results.append(result)
                if correct:
                    verified.append((result, sdfg))
            if do_time:
                for result, sdfg in verified:
                    log_attempt("time", result.name)
                    try:
                        t = timer(sdfg, run, reps, warmup) if timer is not None else default_timer(
                            lambda: run(sdfg), reps, warmup)
                        if isinstance(t, tuple):  # a stats timer: (median, {"spread": ..., ...})
                            t, timer_metadata = t
                            result.metadata.update(timer_metadata)
                        result.time = t
                    except Exception as ex:  # timing is ADVISORY: a failure must not demote a correct candidate
                        result.metadata["timing_error"] = f"{type(ex).__name__}: {ex}"
    results.sort(key=lambda r: (not r.correct, r.time if r.time is not None else float('inf')))
    return results


def best(results: List[SweepResult], noise_floor: Optional[float] = None) -> Optional[SweepResult]:
    """The winning correct candidate, with the identity-first law widened to measurement
    resolution: every correct candidate whose time is within ``noise_floor`` (relative) of the
    fastest counts as TIED, and the tie resolves to the earliest-ENUMERATED one -- exact-float
    ties essentially never happen between measured medians, so without the window the enumeration
    order (identity first) would be dead for any timed sweep. The default window is
    ``timing.SPREAD_CONTENDED_THRESHOLD``: a layout must beat the earlier candidate by more than
    the contention threshold to displace it. Pass ``noise_floor=0.0`` for the raw fastest.
    Falls back to the first correct candidate when nothing was timed; ``None`` if none verified.
    """
    if noise_floor is None:
        from dace.transformation.layout.timing import SPREAD_CONTENDED_THRESHOLD
        noise_floor = SPREAD_CONTENDED_THRESHOLD
    timed = [r for r in results if r.correct and r.time is not None]
    if not timed:
        return next((r for r in results if r.correct), None)
    fastest = min(r.time for r in timed)
    window = [r for r in timed if r.time <= fastest * (1.0 + noise_floor)]
    return min(window, key=lambda r: r.order)


# --------------------------------------------------------------------------- #
#  Candidate-family enumerators (apply closures over the layout passes)
# --------------------------------------------------------------------------- #
def permutation_candidates(array: str, ndim: int):
    """Yield ``(name, apply)`` for every dimension permutation of ``array`` (identity included).

    ``apply(sdfg)`` runs ``PermuteDimensions`` in-place for that permutation.
    """
    from dace.transformation.layout.permute_dimensions import PermuteDimensions

    for perm in itertools.permutations(range(ndim)):

        def apply(sdfg, perm=perm):
            PermuteDimensions(permute_map={array: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

        yield f"permute_{array}_{''.join(map(str, perm))}", apply


def block_candidates(array: str, ndim: int, factors: Tuple[int, ...] = (8, 16, 32)):
    """Yield ``(name, apply)`` for blocking ONE dimension of ``array`` by each factor (plus the
    unblocked identity). ``apply(sdfg)`` runs ``SplitDimensions`` then
    ``normalize_schedule_for_layout``."""
    from dace.transformation.layout.split_dimensions import SplitDimensions
    from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout

    yield f"noblock_{array}", (lambda sdfg: None)
    for dim in range(ndim):
        for factor in factors:

            def apply(sdfg, dim=dim, factor=factor):
                masks = [i == dim for i in range(ndim)]
                facs = [factor if i == dim else 1 for i in range(ndim)]
                SplitDimensions(split_map={array: (masks, facs)}).apply_pass(sdfg, {})
                normalize_schedule_for_layout(sdfg)

            yield f"block_{array}_d{dim}_{factor}", apply


def shuffle_candidates(array: str, dim: int, shuffle_names):
    """Yield ``(name, apply)`` for renumbering ``array``'s dimension ``dim`` by each REGISTERED
    shuffle (plus the unshuffled identity). ``apply(sdfg)`` runs ``ShuffleElements``. Every shuffle
    is transparent (the reorder + inverse-composed consumers preserve the result), so all candidates
    verify -- the sweep picks the layout, the algebra guarantees correctness."""
    from dace.transformation.layout.shuffle_elements import ShuffleElements

    yield f"noshuffle_{array}", (lambda sdfg: None)
    for name in shuffle_names:

        def apply(sdfg, name=name):
            ShuffleElements(shuffle_map={array: (name, dim)}).apply_pass(sdfg, {})

        yield f"shuffle_{array}_{name}", apply


def indirection_candidates(index_array: str, data_array: str, dim: int, ndim: int, shuffle_names, prepare: bool = True):
    """Yield ``(name, apply)`` layout candidates for a DATA array reached through indirection --
    ``data_array[index_array[f(i)]]`` (a sparse gather/scatter, as found by
    :func:`dace.transformation.layout.indirect_access.indirect_accesses`).

    The bijective layout levers for indirection are Shuffle and Permute, applied to the DATA array:

      * ``indir_shuffle_<data>_<name>`` -- renumber the data array by each registered shuffle
        ``sigma`` (``ShuffleElements``); the consumer's runtime index is composed with ``sigma^-1``,
        so ``data'[sigma^-1(idx[i])] == data[idx[i]]`` for ANY index distribution (duplicates
        included) -- correctness needs no injectivity guard.
      * ``indir_permute_<data>_<perm>`` -- reorder the data array's dimensions (``PermuteDimensions``
        with ``add_permute_maps``); the mu-invariant-layout lever. Emitted only for ``ndim > 1``
        (a 1-D data array has only the identity permutation).
      * ``noindir_<data>`` -- the baseline.

    ``index_array`` is named for provenance only; the reorder is applied to ``data_array``. Because a
    Shuffle / Permute *through* an indirection needs the prepared normal form (the gather's
    ``other_subset`` copy lifted to a ``CopyLibraryNode`` so ``ShuffleElements`` can rewrite it), each
    apply runs :func:`~dace.transformation.layout.prepare.prepare_for_layout` first when ``prepare``
    is set (the default). Every candidate is transparent, so all reproduce the oracle -- the sweep
    picks the layout, the algebra guarantees correctness.
    """
    from dace.transformation.layout.shuffle_elements import ShuffleElements
    from dace.transformation.layout.permute_dimensions import PermuteDimensions
    from dace.transformation.layout.prepare import prepare_for_layout

    def prepared(fn):

        def apply(sdfg):
            if prepare:
                prepare_for_layout(sdfg, validate=False)
            fn(sdfg)

        return apply

    yield f"noindir_{data_array}", prepared(lambda sdfg: None)
    for name in shuffle_names:

        def shuffle_apply(sdfg, name=name):
            ShuffleElements(shuffle_map={data_array: (name, dim)}).apply_pass(sdfg, {})

        yield f"indir_shuffle_{data_array}_{name}", prepared(shuffle_apply)
    if ndim > 1:
        for perm in itertools.permutations(range(ndim)):
            if list(perm) == list(range(ndim)):
                continue  # identity permutation duplicates the noindir baseline

            def permute_apply(sdfg, perm=perm):
                PermuteDimensions(permute_map={data_array: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

            yield f"indir_permute_{data_array}_{''.join(map(str, perm))}", prepared(permute_apply)
