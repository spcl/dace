# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.atomic``: atomic updates that return the old value, e.g., for the last of many updates to act."""

import numpy as np
import pytest

import dace
from dace.frontend.python import interface

N = dace.symbol("N")


@dace.program
def last_arrivers(counters: dace.int32[2], last: dace.int32[N]):
    for i in dace.map[0:N]:
        left = dace.atomic.add(counters, i % 2, -1)
        if left == 1:
            last[i] = 1


def test_last_arriver():
    sdfg = last_arrivers.to_sdfg(simplify=True)
    # Acquire-release: the last update sees what the others wrote before theirs
    assert "dace::wcr_acq_rel<false>::fixed<dace::ReductionType::Sum, int>" in "\n".join(
        c.clean_code for c in sdfg.generate_code()
    )
    counters = np.full(2, 32, np.int32)
    last = np.zeros(64, np.int32)
    sdfg(counters=counters, last=last, N=64)
    assert np.array_equal(counters, [0, 0])
    assert last.sum() == 2 and last[0::2].sum() == 1


@dace.program
def ticket_numbers(counter: dace.int64[1], tickets: dace.int64[N], step: dace.int64):
    for i in dace.map[0:N]:
        tickets[i] = dace.atomic.add(counter, 0, step)


def test_tickets_from_scalars():
    """A value from a scalar: every element gets a distinct old value."""
    counter = np.zeros(1, np.int64)
    tickets = np.zeros(50, np.int64)
    ticket_numbers(counter=counter, tickets=tickets, step=3, N=50)
    assert counter[0] == 150
    assert np.array_equal(np.sort(tickets), np.arange(0, 150, 3))


@dace.program
def per_group_last(groups: dace.int32[N], counters: dace.int32[4], last: dace.int32[N]):
    for i in dace.map[0:N]:
        g = groups[i]
        left = dace.atomic.add(counters, g, -1)  # An index read from data
        if left == 1:
            last[i] = 1


def test_index_from_data():
    groups = np.arange(64, dtype=np.int32) % 4
    counters = np.full(4, 16, np.int32)
    last = np.zeros(64, np.int32)
    per_group_last(groups=groups, counters=counters, last=last, N=64)
    assert np.array_equal(counters, [0, 0, 0, 0])
    assert np.array_equal(np.bincount(groups[last == 1], minlength=4), [1, 1, 1, 1])


@dace.program
def high_water_marks(values: dace.int32[N], mark: dace.int32[1], before: dace.int32[N]):
    for i in dace.map[0:N]:
        before[i] = dace.atomic.custom(lambda a, b: max(a, b), mark, 0, values[i])


def test_custom_maximum():
    """A function that is a known reduction updates with its atomic instruction."""
    sdfg = high_water_marks.to_sdfg(simplify=True)
    assert "dace::ReductionType::Max" in "\n".join(c.clean_code for c in sdfg.generate_code())
    values = np.random.default_rng(0).permutation(40).astype(np.int32)
    mark = np.full(1, -1, np.int32)
    before = np.zeros(40, np.int32)
    sdfg(values=values, mark=mark, before=before, N=40)
    assert mark[0] == 39
    assert np.count_nonzero(before == -1) == 1  # One update comes first, in any order
    assert np.all(np.isin(before[before != -1], values))  # The others see marks that updates set


@dace.program
def products(factors: dace.float64[N], total: dace.float64[2], before: dace.float64[N]):
    for i in dace.map[0:N]:
        before[i] = dace.atomic.custom(lambda a, b: a * b, total, (i % 2,), factors[i])


def test_custom_product():
    """A product, which no atomic instruction computes, with a tuple index."""
    factors = np.random.default_rng(1).uniform(0.5, 2, 30)
    total = np.ones(2)
    before = np.zeros(30)
    products(factors=factors, total=total, before=before, N=30)
    assert np.allclose(total, [factors[0::2].prod(), factors[1::2].prod()])
    assert np.all((before > 0) & np.isfinite(before))


def test_custom_requires_lambda():
    @dace.program
    def not_a_function(counters: dace.int32[2]):
        dace.atomic.custom(3, counters, 0, 1)

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match="lambda of two arguments"):
        not_a_function.to_sdfg()


def test_python_semantics():
    """Outside of programs, the updates are plain Python."""
    counters = np.array([5, 7])
    assert dace.atomic.add(counters, 1, -2) == 7
    assert dace.atomic.custom(lambda a, b: a * b, counters, 0, 3) == 5
    assert np.array_equal(counters, [15, 5])


def test_defined_statically():
    """``dace.atomic`` is a class of the wrappers that ``dace`` imports, which editors can see without running it."""
    assert dace.atomic is interface.atomic


if __name__ == "__main__":
    test_last_arriver()
    test_tickets_from_scalars()
    test_index_from_data()
    test_custom_maximum()
    test_custom_product()
    test_custom_requires_lambda()
    test_python_semantics()
    test_defined_statically()
