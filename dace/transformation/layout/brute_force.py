# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Brute-force GLOBAL layout sweep -- the layout optimizer (no cost model): enumerate candidates, compile, verify against a numpy oracle, and time the correct ones."""
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
    """One candidate's result; ``order`` is its enumeration position, used for tie-breaks."""
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
    """Median GPU time (s) of ``fn`` via CUDA events on cupy's current stream; ``fn`` must be compiled inside :func:`single_default_stream`. Times the whole call, so small kernels are overhead-bound -- see :func:`compute_region_timer` for compute-only timing."""
    import cupy  # GPU-only; keeps the module importable without a GPU

    start = cupy.cuda.Event()
    stop = cupy.cuda.Event()
    samples = []
    for _ in range(warmup):
        fn()
    cupy.cuda.get_current_stream().synchronize()
    for _ in range(reps):
        start.record()  # current stream = null/default unless inside a stream context
        fn()
        stop.record()  # same stream; brackets dace's default-stream kernels
        stop.synchronize()  # stream-scoped wait, not device-wide
        samples.append(cupy.cuda.get_elapsed_time(start, stop) * 1e-3)
    samples.sort()
    return samples[len(samples) // 2]


@contextlib.contextmanager
def single_default_stream():
    """Force dace to emit one CUDA stream (``max_concurrent_streams=-1``) so :func:`time_gpu`'s events see all kernels. The SDFG must be COMPILED inside this context; clean ``.dacecache`` if switching stream modes -- the build cache keys on the SDFG name, not this setting."""
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
          isolate: bool = False,
          isolate_timeout: float = 900.0) -> List[SweepResult]:
    """Compile, run, verify, and (optionally) time each candidate; return results ranked (correct first, then by time). ``isolate`` forks each candidate (CPU only) so a crash doesn't kill the sweep, giving each child at most ``isolate_timeout`` seconds -- lower it for a wide sweep of small kernels, where the default would stall 15 min per hang."""
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
        """Run, verify, and (if correct) time one compiled candidate; JSON-able for :func:`run_isolated`."""
        out = run(sdfg)
        correct = all(name_ in out and compare(out[name_], ref) for name_, ref in reference.items())
        verdict: Dict[str, Any] = {"correct": bool(correct), "time": None, "metadata": {}}
        if do_time and correct:
            try:
                t = timer(sdfg, run, reps, warmup) if timer is not None else default_timer(
                    lambda: run(sdfg), reps, warmup)
                if isinstance(t, (tuple, list)):  # stats timer: (median, metadata)
                    t, timer_metadata = t
                    verdict["metadata"].update(timer_metadata)
                verdict["time"] = None if t is None else float(t)
            except Exception as ex:  # timing is advisory; a failure must not demote a correct candidate
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
    with stream_ctx:  # GPU: compile inside the single-stream context
        if isolate:
            # phase 1: compile in the parent; phase 2: run/verify/time in a forked child (segfault-safe)
            staged: List[Tuple[SweepResult, dace.SDFG]] = []
            for order, (name, make) in enumerate(candidates.items()):
                log_attempt("compile", name)
                try:
                    sdfg = make()
                    select_layout_lowering(sdfg, device)
                    sdfg.compile()
                except Exception as ex:  # build/compile failure -> not viable
                    results.append(SweepResult(name, False, None, f"{type(ex).__name__}: {ex}", order=order))
                    continue
                result = SweepResult(name, False, order=order)
                results.append(result)
                staged.append((result, sdfg))
            for result, sdfg in staged:
                log_attempt("run", result.name)
                fold(result, run_isolated(lambda sdfg=sdfg: verify_and_time(sdfg), timeout=isolate_timeout))
        else:
            # phase 1: verify all (compiles on first run); phase 2: time the verified ones back-to-back
            verified: List[Tuple[SweepResult, dace.SDFG]] = []
            for order, (name, make) in enumerate(candidates.items()):
                log_attempt("verify", name)
                try:
                    sdfg = make()
                    select_layout_lowering(sdfg, device)  # choose the device lowering the transforms left unset
                    out = run(sdfg)
                    correct = all(name_ in out and compare(out[name_], ref) for name_, ref in reference.items())
                except Exception as ex:  # build/compile/run failure -> not viable
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
                        if isinstance(t, (tuple, list)):  # stats timer: (median, metadata)
                            t, timer_metadata = t
                            result.metadata.update(timer_metadata)
                        result.time = t
                    except Exception as ex:  # timing is advisory; a failure must not demote a correct candidate
                        result.metadata["timing_error"] = f"{type(ex).__name__}: {ex}"
    results.sort(key=lambda r: (not r.correct, r.time if r.time is not None else float('inf')))
    return results


def best(results: List[SweepResult], noise_floor: Optional[float] = None) -> Optional[SweepResult]:
    """The winning candidate: correct candidates within ``noise_floor`` (relative) of the fastest tie, resolved to the earliest-enumerated. Defaults to :data:`timing.SPREAD_CONTENDED_THRESHOLD`; ``None`` if nothing verified."""
    if noise_floor is None:
        from dace.transformation.layout.timing import SPREAD_CONTENDED_THRESHOLD
        noise_floor = SPREAD_CONTENDED_THRESHOLD
    timed = [r for r in results if r.correct and r.time is not None]
    if not timed:  # nothing measured: fall back to the earliest-enumerated correct candidate
        return next((r for r in results if r.correct), None)
    fastest = min(r.time for r in timed)
    if fastest <= 0.0:
        # a non-positive median means the timer resolved nothing, and a RELATIVE window around it collapses
        # to {0.0}; keep every timed candidate in the tie and let enumeration order decide
        window = timed
    else:
        window = [r for r in timed if r.time <= fastest * (1.0 + noise_floor)]
    return min(window, key=lambda r: r.order)


# --------------------------------------------------------------------------- #
#  Candidate-family enumerators (apply closures over the layout passes)
# --------------------------------------------------------------------------- #
def permutation_candidates(array: str, ndim: int):
    """Yield ``(name, apply)`` for every dimension permutation of ``array`` (identity included); ``apply`` runs ``PermuteDimensions``."""
    from dace.transformation.layout.permute_dimensions import PermuteDimensions

    for perm in itertools.permutations(range(ndim)):

        def apply(sdfg, perm=perm):
            PermuteDimensions(permute_map={array: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

        yield f"permute_{array}_{''.join(map(str, perm))}", apply


def block_candidates(array: str, ndim: int, factors: Tuple[int, ...] = (8, 16, 32)):
    """Yield ``(name, apply)`` for blocking one dimension of ``array`` by each factor (plus unblocked identity); ``apply`` runs ``SplitDimensions`` then ``normalize_schedule_for_layout``."""
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
    """Yield ``(name, apply)`` for renumbering ``array``'s dimension ``dim`` by each registered shuffle (plus identity); ``apply`` runs ``ShuffleElements``. Every shuffle is transparent, so all candidates verify."""
    from dace.transformation.layout.shuffle_elements import ShuffleElements

    yield f"noshuffle_{array}", (lambda sdfg: None)
    for name in shuffle_names:

        def apply(sdfg, name=name):
            ShuffleElements(shuffle_map={array: (name, dim)}).apply_pass(sdfg, {})

        yield f"shuffle_{array}_{name}", apply


def indirection_candidates(index_array: str, data_array: str, dim: int, ndim: int, shuffle_names, prepare: bool = True):
    """Yield ``(name, apply)`` layout candidates for a DATA array reached through indirection (``data_array[index_array[f(i)]]``): Shuffle/Permute candidates on ``data_array`` plus the ``noindir`` baseline. Each ``apply`` runs :func:`prepare_for_layout` first (unless ``prepare=False``); every candidate is transparent, so all verify."""
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
                continue  # identity duplicates the noindir baseline

            def permute_apply(sdfg, perm=perm):
                PermuteDimensions(permute_map={data_array: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

            yield f"indir_permute_{data_array}_{''.join(map(str, perm))}", prepared(permute_apply)
