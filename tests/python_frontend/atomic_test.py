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
def atomic_add(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.add(target, 0, values[i])


@dace.program
def atomic_sub(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.sub(target, 0, values[i])


@dace.program
def atomic_min(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.min(target, 0, values[i])


@dace.program
def atomic_max(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.max(target, 0, values[i])


@dace.program
def atomic_and(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.and_(target, 0, values[i])


@dace.program
def atomic_or(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.or_(target, 0, values[i])


@dace.program
def atomic_xor(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.xor(target, 0, values[i])


@dace.program
def atomic_exch(values: dace.int64[N], target: dace.int64[1], old: dace.int64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.exch(target, 0, values[i])


@dace.program
def atomic_prod(values: dace.float64[N], target: dace.float64[1], old: dace.float64[N]):
    for i in dace.map[0:N]:
        old[i] = dace.atomic.prod(target, 0, values[i])


# Programs in which every iteration updates one element and keeps its old value
UPDATING_PROGRAMS = {
    "add": atomic_add,
    "sub": atomic_sub,
    "min": atomic_min,
    "max": atomic_max,
    "and_": atomic_and,
    "or_": atomic_or,
    "xor": atomic_xor,
    "exch": atomic_exch,
    "prod": atomic_prod,
}


@pytest.mark.parametrize(
    "name, initial, fold",
    [
        ("add", 3, lambda t, v: t + v.sum()),
        ("sub", 3, lambda t, v: t - v.sum()),
        ("min", 50, lambda t, v: min(t, v.min())),
        ("max", -50, lambda t, v: max(t, v.max())),
        ("and_", -1, lambda t, v: np.bitwise_and.reduce(v, initial=t)),
        ("or_", 0, lambda t, v: np.bitwise_or.reduce(v, initial=t)),
        ("xor", 5, lambda t, v: np.bitwise_xor.reduce(v, initial=t)),
    ],
)
def test_standard_updates(name, initial, fold):
    """Every old value is the element before some update, and the final element combines all updates."""
    values = np.random.default_rng(7).integers(-40, 40, 64)
    target = np.array([initial])
    old = np.zeros(64, np.int64)
    UPDATING_PROGRAMS[name](values=values, target=target, old=old, N=64)
    assert target[0] == fold(initial, values)
    assert old[0] == initial or initial in old  # The first update sees the initial value

    python_target, python_old = np.array([initial]), np.zeros(64, np.int64)  # The same updates in plain Python
    for i in range(64):
        python_old[i] = vars(dace.atomic)[name].__func__(python_target, 0, values[i])
    assert python_target[0] == target[0]


def test_product():
    values = np.random.default_rng(8).uniform(0.9, 1.1, 32)
    target, old = np.ones(1), np.zeros(32)
    UPDATING_PROGRAMS["prod"](values=values, target=target, old=old, N=32)
    assert np.isclose(target[0], values.prod())
    assert 1.0 in old


def test_exchange():
    """Exchanges pass the element along: the old values and the final one are the initial value and all the others."""
    values = np.arange(1, 33, dtype=np.int64)
    target, old = np.zeros(1, np.int64), np.zeros(32, np.int64)
    UPDATING_PROGRAMS["exch"](values=values, target=target, old=old, N=32)
    assert sorted([*old, target[0]]) == list(range(33))


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["min", "and_", "exch", "prod"])
def test_gpu_updates(name):
    """On GPUs: 64-bit integers, which atomic functions take as ``long long``, and products, which compare and swap."""
    sdfg = UPDATING_PROGRAMS[name].to_sdfg(simplify=True)
    sdfg.name = f"gpu_{sdfg.name}"
    sdfg.apply_gpu_transformations()
    n = 4096
    rng = np.random.default_rng(9)
    if name == "prod":
        values, target = rng.uniform(0.999, 1.001, n), np.ones(1)
    elif name == "exch":
        values, target = np.arange(1, n + 1, dtype=np.int64), np.zeros(1, np.int64)
    else:
        values, target = rng.integers(1, 1 << 40, n), np.array([(1 << 41) if name == "min" else -1], np.int64)
    initial = target[0]
    old = np.zeros(n, values.dtype)
    sdfg(values=values, target=target, old=old, N=n)
    if name == "exch":
        assert sorted([*old, target[0]]) == list(range(n + 1))
    elif name == "prod":
        assert np.isclose(target[0], values.prod())
        assert len(set(old.tolist())) == n  # No update is lost: each sees the element after the ones before it
    elif name == "min":
        assert target[0] == values.min() and initial in old
    else:
        assert target[0] == np.bitwise_and.reduce(values, initial=initial) and initial in old


@dace.program
def first_occurrences(items: dace.int32[N], first: dace.int32[N]):
    seen = np.zeros([4], dtype=np.int32)
    for i in dace.map[0:N]:
        if dace.atomic.exch(seen, items[i], 1) == 0:
            first[i] = 1


def test_old_value_reads_initialization():
    """The old value of an update reads the array: the initialization before the updates is not dead dataflow."""
    sdfg = first_occurrences.to_sdfg(simplify=True)
    assert any(  # Zeros written to the whole array, not only updates
        isinstance(node, dace.nodes.AccessNode)
        and node.data == "seen"
        and any(edge.data.wcr is None for edge in state.in_edges(node))
        for state in sdfg.states()
        for node in state.nodes()
    )
    items = np.random.default_rng(10).integers(0, 4, 64).astype(np.int32)
    first = np.zeros(64, np.int32)
    sdfg(items=items, first=first, N=64)
    assert np.array_equal(np.sort(items[first == 1]), np.unique(items))


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
    assert dace.atomic.min(counters, 0, 20) == 15
    assert dace.atomic.exch(counters, 1, 9) == 5
    assert np.array_equal(counters, [15, 9])


def test_defined_statically():
    """``dace.atomic`` is a class of the wrappers that ``dace`` imports, which editors can see without running it."""
    assert dace.atomic is interface.atomic


if __name__ == "__main__":
    test_last_arriver()
    test_tickets_from_scalars()
    test_index_from_data()
    for name, initial, fold in [("add", 3, lambda t, v: t + v.sum()), ("min", 50, lambda t, v: min(t, v.min()))]:
        test_standard_updates(name, initial, fold)
    test_product()
    for name in ["min", "and_", "exch", "prod"]:
        test_gpu_updates(name)
    test_exchange()
    test_old_value_reads_initialization()
    test_custom_maximum()
    test_custom_product()
    test_custom_requires_lambda()
    test_python_semantics()
    test_defined_statically()
