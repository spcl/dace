# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra unit tests for the brute-force GLOBAL layout sweep engine.

These complement ``brute_force_test.py`` by exercising the parts of
``dace.transformation.layout.brute_force`` that the base tests leave uncovered:

  * ``time_cpu`` actually runs ``fn`` exactly ``reps + warmup`` times;
  * ``best`` / the sweep ranking pick the FASTEST correct candidate and skip incorrect / build-failing
    ones (an incorrect candidate is never timed and never chosen as best);
  * the ``timer`` and ``compare`` hooks and the ``do_time=False`` path are honoured;
  * a candidate that returns outputs missing a reference key is flagged incorrect (not crashed);
  * ``permutation_candidates`` (3-D), ``block_candidates`` (multi-dim + default factors) and
    ``shuffle_candidates`` emit exactly the expected named candidate set; and
  * the permutation / block / shuffle families sweep end-to-end and every candidate is BIT-EXACT
    against a numpy oracle.

Only correctness invariants are asserted; timing magnitudes are never asserted (noisy on a shared
host)."""
import numpy

import dace
from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.brute_force import (SweepResult, best, block_candidates, permutation_candidates,
                                                    shuffle_candidates, sweep, time_cpu)

N = dace.symbol("N")


@dace.program
def times2plus1(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 2.0 + 1.0


@dace.program
def times3plus7(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 3.0 + 7.0


@dace.program
def scale_shift(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 1.5 - 2.0


@dace.program
def shift_down(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] - 0.5


@dace.program
def ew3d(A: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j, k] = A[i, j, k] * 2.0 + 1.0


def eval_shape(desc, n):
    """Concrete shape of ``desc`` with the sweep's symbol ``N`` bound to ``n``."""
    return tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in desc.shape)


# --------------------------------------------------------------------------- #
#  time_cpu
# --------------------------------------------------------------------------- #
def test_time_cpu_invokes_fn_reps_plus_warmup():
    """``time_cpu`` warms up ``warmup`` times then samples ``reps`` times -> exactly reps+warmup
    calls, and returns a non-negative float (the median sample)."""
    calls = []
    t = time_cpu(lambda: calls.append(1), reps=4, warmup=2)
    assert len(calls) == 6
    assert isinstance(t, float) and t >= 0.0

    single = []
    t2 = time_cpu(lambda: single.append(1), reps=1, warmup=0)
    assert len(single) == 1
    assert isinstance(t2, float) and t2 >= 0.0


# --------------------------------------------------------------------------- #
#  best() in isolation
# --------------------------------------------------------------------------- #
def test_best_selects_first_correct_or_none():
    """``best`` returns the first correct result (the list is pre-ranked) or ``None`` if none
    verified; ``SweepResult`` defaults are empty/optional."""
    default = SweepResult("x", True)
    assert default.time is None and default.error is None and default.metadata == {}

    assert best([]) is None
    assert best([SweepResult("boom", False, None, "err"), SweepResult("bad", False)]) is None

    picked = best([SweepResult("bad", False), SweepResult("good", True, 0.1), SweepResult("also", True, 0.2)])
    assert picked is not None and picked.name == "good" and picked.correct


# --------------------------------------------------------------------------- #
#  ranking: fastest correct first
# --------------------------------------------------------------------------- #
def test_sweep_ranks_fastest_correct_first():
    """Two correct candidates timed 0.9 and 0.1; the sweep ranks the faster one first and ``best``
    returns it -- proving the ranking key is (correct-first, ascending time)."""
    reference = {"C": numpy.arange(4.0)}
    candidates = {"slow": lambda: dace.SDFG("rank_slow"), "fast": lambda: dace.SDFG("rank_fast")}

    def run(sdfg):
        return {"C": numpy.arange(4.0)}

    times = iter([0.9, 0.1])  # slow enumerated first, then fast

    def stub_timer(sdfg, run_fn, reps, warmup):
        return next(times)

    results = sweep(candidates, run, reference, do_time=True, timer=stub_timer)
    assert all(r.correct for r in results)
    assert results[0].name == "fast" and results[0].time == 0.1
    assert results[1].name == "slow" and results[1].time == 0.9
    assert best(results).name == "fast"


# --------------------------------------------------------------------------- #
#  core invariant: incorrect / build-failing candidates are never timed, never best
# --------------------------------------------------------------------------- #
def test_incorrect_and_build_failure_never_timed_never_best():
    """A wrong-answer candidate and a builder that raises are both flagged not-correct, neither is
    timed (the stub timer fires only once, for the correct candidate), and ``best`` skips them."""
    _N = 8
    A = numpy.random.rand(_N)
    reference = {"C": A * 2.0 + 1.0}

    def run(sdfg):
        C = numpy.zeros(_N)
        sdfg(A=A.copy(), C=C, N=_N)
        return {"C": C}

    def boom():
        raise RuntimeError("cannot build this candidate")

    candidates = {
        "good": lambda: times2plus1.to_sdfg(simplify=True),
        "bad": lambda: times3plus7.to_sdfg(simplify=True),
        "boom": boom,
    }

    timed = []

    def stub_timer(sdfg, run_fn, reps, warmup):
        timed.append(1)
        return 0.123

    results = sweep(candidates, run, reference, do_time=True, timer=stub_timer)
    by_name = {r.name: r for r in results}

    assert by_name["good"].correct and by_name["good"].time == 0.123
    assert by_name["bad"].correct is False and by_name["bad"].time is None and by_name["bad"].error is None
    assert by_name["boom"].correct is False and by_name["boom"].time is None
    assert "cannot build" in by_name["boom"].error

    assert len(timed) == 1  # only the one correct candidate was ever timed
    assert best(results).name == "good"

    correct_flags = [r.correct for r in results]  # correct-first ordering
    assert correct_flags == sorted(correct_flags, reverse=True)


# --------------------------------------------------------------------------- #
#  compare hook is consulted
# --------------------------------------------------------------------------- #
def test_custom_compare_is_consulted():
    """The ``compare`` predicate decides correctness: a predicate that always rejects yields no
    correct candidate (best is None) even though the outputs equal the reference; the default
    ``numpy.allclose`` accepts them."""
    reference = {"C": numpy.ones(3)}
    candidates = {"id": lambda: dace.SDFG("cmp_id")}

    def run(sdfg):
        return {"C": numpy.ones(3)}

    rejected = sweep(candidates, run, reference, compare=lambda a, b: False, do_time=False)
    assert rejected[0].correct is False and best(rejected) is None

    accepted = sweep(candidates, run, reference, do_time=False)  # default allclose
    assert accepted[0].correct and best(accepted) is not None


# --------------------------------------------------------------------------- #
#  a run that omits a reference key is incorrect, not a crash
# --------------------------------------------------------------------------- #
def test_missing_output_key_is_incorrect():
    """If ``run`` returns outputs missing a reference key, the candidate is flagged incorrect (the
    ``name in out`` guard short-circuits before ``compare``), with no error and no timing."""
    reference = {"C": numpy.zeros(2), "D": numpy.ones(2)}
    candidates = {"id": lambda: dace.SDFG("miss_id")}

    def run(sdfg):
        return {"C": numpy.zeros(2)}  # "D" is absent

    results = sweep(candidates, run, reference, do_time=True, reps=1, warmup=0)
    assert results[0].correct is False
    assert results[0].error is None and results[0].time is None
    assert best(results) is None


# --------------------------------------------------------------------------- #
#  do_time=False leaves time unset; do_time=True populates it
# --------------------------------------------------------------------------- #
def test_do_time_flag_controls_timing():
    """A correct candidate has ``time is None`` under ``do_time=False`` and a non-negative float
    under ``do_time=True`` with the default whole-call timer."""
    reference = {"C": numpy.full(3, 7.0)}
    candidates = {"id": lambda: dace.SDFG("dt_id")}

    def run(sdfg):
        return {"C": numpy.full(3, 7.0)}

    off = sweep(candidates, run, reference, do_time=False)
    assert off[0].correct and off[0].time is None

    on = sweep(candidates, run, reference, do_time=True, reps=1, warmup=0)
    assert on[0].correct and on[0].time is not None and on[0].time >= 0.0


# --------------------------------------------------------------------------- #
#  enumerators: named candidate sets
# --------------------------------------------------------------------------- #
def test_permutation_candidates_3d_named_set():
    """``permutation_candidates`` yields one candidate per dimension permutation (identity included):
    for ndim=3 that is all 6 permutations, named ``permute_<arr>_<perm>``."""
    names = {name for name, _ in permutation_candidates("X", 3)}
    assert names == {
        "permute_X_012", "permute_X_021", "permute_X_102", "permute_X_120", "permute_X_201", "permute_X_210"
    }


def test_block_candidates_named_set():
    """``block_candidates`` yields the unblocked identity plus one candidate per (dimension, factor)
    pair -- across every dimension for multi-dim arrays, and with the default {8,16,32} factors."""
    multi = {name for name, _ in block_candidates("M", 2, factors=(4, 8))}
    assert multi == {"noblock_M", "block_M_d0_4", "block_M_d0_8", "block_M_d1_4", "block_M_d1_8"}

    default_1d = {name for name, _ in block_candidates("V", 1)}
    assert default_1d == {"noblock_V", "block_V_d0_8", "block_V_d0_16", "block_V_d0_32"}


def test_shuffle_candidates_named_set():
    """``shuffle_candidates`` yields the unshuffled identity plus one candidate per registered
    shuffle name (naming is lazy -- it does not require the shuffles to be registered)."""
    names = {name for name, _ in shuffle_candidates("A", 0, ["rot", "swz"])}
    assert names == {"noshuffle_A", "shuffle_A_rot", "shuffle_A_swz"}


# --------------------------------------------------------------------------- #
#  end-to-end bit-exact family sweeps
# --------------------------------------------------------------------------- #
def test_permutation_sweep_3d_bit_exact():
    """A global dimension permutation of a 3-D input is transparent (add_permute_maps wraps it), so
    all 6 permutation candidates compile and reproduce C = A*2+1 bit-exactly."""
    _N = 6
    A = numpy.random.rand(_N, _N, _N)
    reference = {"C": A * 2.0 + 1.0}

    def make_for(apply):

        def make():
            sdfg = ew3d.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    candidates = {name: make_for(apply) for name, apply in permutation_candidates("A", 3)}
    assert len(candidates) == 6

    def run(sdfg):
        C = numpy.zeros((_N, _N, _N))
        sdfg(A=A.copy(), C=C, N=_N)
        return {"C": C}

    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    picked = best(results)
    assert picked is not None and picked.correct


def test_block_family_sweep_bit_exact():
    """Blocking A by {4,8} (plus unblocked) lays it out as [N/b, b]; feeding the correspondingly
    reshaped packed-C input, every block candidate reproduces C = A*1.5-2 bit-exactly."""
    n = 16  # divisible by every candidate factor
    A_logical = numpy.random.rand(n)
    reference = {"C": A_logical * 1.5 - 2.0}

    def make_for(apply):

        def make():
            sdfg = scale_shift.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    candidates = {name: make_for(apply) for name, apply in block_candidates("A", 1, factors=(4, 8))}
    assert set(candidates) == {"noblock_A", "block_A_d0_4", "block_A_d0_8"}

    def run(sdfg):
        a_shape = eval_shape(sdfg.arrays["A"], n)
        c_shape = eval_shape(sdfg.arrays["C"], n)
        A_in = A_logical.reshape(a_shape).copy()  # fresh contiguous array (not a view)
        C = numpy.zeros(c_shape)
        sdfg(A=A_in, C=C, N=n)
        return {"C": numpy.asarray(C).reshape(n).copy()}

    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    picked = best(results)
    assert picked is not None and picked.correct


def test_shuffle_family_sweep_bit_exact():
    """A self-inverse XOR swizzle and a cyclic shift are transparent value-permutations, so every
    shuffle candidate compiles and reproduces C = A-0.5 bit-exactly."""
    register_shuffle("bf_xor5", "i ^ 5", "i ^ 5")  # self-inverse on [0,8)
    register_shuffle("bf_cyc", "(i + 1) % N", "(i + N - 1) % N")

    _N = 8
    A = numpy.random.rand(_N)
    reference = {"C": A - 0.5}

    def make_for(apply):

        def make():
            sdfg = shift_down.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    candidates = {name: make_for(apply) for name, apply in shuffle_candidates("A", 0, ["bf_xor5", "bf_cyc"])}
    assert set(candidates) == {"noshuffle_A", "shuffle_A_bf_xor5", "shuffle_A_bf_cyc"}

    def run(sdfg):
        C = numpy.zeros(_N)
        sdfg(A=A.copy(), C=C, N=_N)
        return {"C": C}

    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    picked = best(results)
    assert picked is not None and picked.correct


if __name__ == "__main__":
    test_time_cpu_invokes_fn_reps_plus_warmup()
    test_best_selects_first_correct_or_none()
    test_sweep_ranks_fastest_correct_first()
    test_incorrect_and_build_failure_never_timed_never_best()
    test_custom_compare_is_consulted()
    test_missing_output_key_is_incorrect()
    test_do_time_flag_controls_timing()
    test_permutation_candidates_3d_named_set()
    test_block_candidates_named_set()
    test_shuffle_candidates_named_set()
    test_permutation_sweep_3d_bit_exact()
    test_block_family_sweep_bit_exact()
    test_shuffle_family_sweep_bit_exact()
    print("brute_force extra tests PASS")
