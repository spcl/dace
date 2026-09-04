# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests wall-clock instrumentation of an SDFG with a steady-clock timer. """
import time
from typing import Tuple

import numpy as np

import dace
from dace.transformation.passes.instrument_with_timer import InstrumentWithTimer

BIG = 4000000


@dace.program
def add_one(a: dace.float64[16], out: dace.float64[16]):
    out[:] = a + 1.0


@dace.program
def heavy(a: dace.float64[BIG], out: dace.float64[BIG]):
    out[:] = a * 2.0 + 1.0


def instrument(prog: dace.frontend.python.parser.DaceProgram) -> dace.SDFG:
    """Build ``prog``'s SDFG, instrument it and check it still validates."""
    sdfg = prog.to_sdfg(simplify=True)
    assert InstrumentWithTimer().apply_pass(sdfg, {}) == 'time_ns'
    sdfg.validate()
    return sdfg


def run(sdfg: dace.SDFG, a: np.ndarray, out: np.ndarray) -> Tuple[int, int]:
    """Run the instrumented ``sdfg`` and return its own measurement next to one taken around the call."""
    ns = np.zeros(1, dtype=np.int64)
    csdfg = sdfg.compile()  # compiling inside the timed window would swamp the comparison
    before = time.perf_counter_ns()
    csdfg(a=a, out=out, time_ns=ns)
    return int(ns[0]), time.perf_counter_ns() - before


def test_trivial_roundtrip():
    a = np.random.rand(16)
    out = np.zeros(16)

    ns, outer = run(instrument(add_one), a, out)

    assert np.allclose(out, a + 1.0)  # instrumentation must not disturb the result
    assert 0 < ns <= outer


def test_measures_real_work():
    a = np.random.rand(BIG)
    out = np.zeros(BIG)

    ns, outer = run(instrument(heavy), a, out)

    assert np.allclose(out, a * 2.0 + 1.0)
    assert 1000000 < ns <= outer  # a 4M-element kernel cannot finish in under a millisecond
    assert ns > outer // 2  # the timer brackets the whole body, not some constant slice of it


def test_second_application_refused():
    sdfg = instrument(add_one)
    blocks = len(sdfg.nodes())
    arrays = sorted(sdfg.arrays)

    assert InstrumentWithTimer().apply_pass(sdfg, {}) is None
    assert len(sdfg.nodes()) == blocks
    assert sorted(sdfg.arrays) == arrays


def test_descriptors():
    sdfg = instrument(add_one)
    assert not sdfg.arrays['time_ns'].transient
    assert 'time_ns' in sdfg.arglist()
    assert sdfg.arrays['time_start'].transient
    assert sdfg.arrays['time_start'].shape == (1, )


if __name__ == '__main__':
    test_trivial_roundtrip()
    test_measures_real_work()
    test_second_application_refused()
    test_descriptors()
