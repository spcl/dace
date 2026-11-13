# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reading the old value of a write-conflict resolution (an atomic read-modify-write), and its memory ordering.

Assigning a connector of a dynamic WCR memlet updates the data atomically and leaves the old value in the connector.
When the tasklet reads it, the update is acquire-release; otherwise, it stays relaxed.
"""

import numpy as np
import pytest

import dace
from dace.transformation.interstate import GPUTransformSDFG

N = dace.symbol("N")
P = dace.symbol("P")
R = dace.symbol("R")


def _code(sdfg: dace.SDFG) -> str:
    return "\n".join(c.clean_code for c in sdfg.generate_code())


def _check_last_arriver(sdfg: dace.SDFG, num_elements: int = 1):
    """Every thread decrements a counter from N; exactly one of them sees the old value 1."""
    n = 64
    counter = np.full(num_elements, n, np.int32)
    last = np.zeros(n, np.int32)
    sdfg(counter=counter, last=last, N=n)
    assert counter[0] == 0
    assert last.sum() == 1


def test_chained_assignment():
    @dace.program
    def chained(counter: dace.int32[1], last: dace.int32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                c >> counter(-1, lambda a, b: a + b)[0]
                l >> last[i]
                old = c = -1
                l = 1 if old == 1 else 0

    sdfg = chained.to_sdfg(simplify=True)
    assert "dace::wcr_acq_rel<false>::fixed<dace::ReductionType::Sum, int>" in _code(sdfg)
    _check_last_arriver(sdfg)


def test_read_after_assignment():
    @dace.program
    def read_after(counter: dace.int32[1], last: dace.int32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                c >> counter(-1, lambda a, b: a + b)[0]
                l >> last[i]
                c = -1
                l = 1 if c == 1 else 0

    sdfg = read_after.to_sdfg(simplify=True)
    assert "dace::wcr_acq_rel<false>::fixed<" in _code(sdfg)
    _check_last_arriver(sdfg)


def test_chained_subscript():
    @dace.program
    def chained_subscript(counter: dace.int32[2], last: dace.int32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                c >> counter(-1, lambda a, b: a + b)[:]
                l >> last[i]
                old = c[0] = -1
                l = 1 if old == 1 else 0

    sdfg = chained_subscript.to_sdfg(simplify=True)
    assert "dace::wcr_acq_rel<false>::fixed<" in _code(sdfg)
    _check_last_arriver(sdfg, num_elements=2)


def test_custom_reduction():
    """A reduction that is not built in declares its function before the update."""

    @dace.program
    def custom(counter: dace.int32[1], last: dace.int32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                c >> counter(-1, lambda a, b: a - b)[0]
                l >> last[i]
                old = c = 1
                l = 1 if old == 1 else 0

    sdfg = custom.to_sdfg(simplify=True)
    assert "dace::wcr_acq_rel<false>::fenced(" in _code(sdfg)
    _check_last_arriver(sdfg)


def test_unread_update_is_relaxed():
    @dace.program
    def unread(counter: dace.int32[1], last: dace.int32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                c >> counter(-1, lambda a, b: a + b)[0]
                l >> last[i]
                c = -1
                l = i

    sdfg = unread.to_sdfg(simplify=True)
    code = _code(sdfg)
    assert "wcr_acq_rel" not in code
    assert "dace::wcr_fixed<dace::ReductionType::Sum, int>::reduce_atomic(" in code
    counter = np.full(1, 64, np.int32)
    sdfg(counter=counter, last=np.zeros(64, np.int32), N=64)
    assert counter[0] == 0


def test_claim_slots():
    """The example in the documentation: appending to a list from many threads."""

    @dace.program
    def append_positive(values: dace.float64[N], result: dace.float64[N], count: dace.int64[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                v << values[i]
                c >> count(-1, lambda a, b: a + b)[0]
                out >> result(-1)[:]

                if v > 0:
                    slot = c = 1
                    out[slot] = v

    values = np.random.default_rng(0).standard_normal(1000)
    result = np.zeros(1000)
    count = np.zeros(1, np.int64)
    append_positive(values=values, result=result, count=count)
    assert count[0] == (values > 0).sum()
    assert np.array_equal(np.sort(result[: count[0]]), np.sort(values[values > 0]))


def test_chained_into_connectors():
    """The old values go to other connectors: an element of a two-dimensional array, through a chain."""

    @dace.program
    def claim_slots(keys: dace.int32[N, 2], count: dace.int32[4], slots: dace.int32[N, 2]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                k << keys[i, :]
                c >> count(-1, lambda a, b: a + b)[:]
                s >> slots(-1)[:, :]
                for j in range(2):
                    s[i, j] = c[k[j]] = 1

    n = 100
    keys = np.random.default_rng(0).integers(0, 4, (n, 2), dtype=np.int32)
    count = np.zeros(4, np.int32)
    slots = np.full((n, 2), -1, np.int32)
    claim_slots(keys=keys, count=count, slots=slots)
    assert np.array_equal(count, np.bincount(keys.ravel(), minlength=4))
    for key in range(4):  # Every key's slots are 0, 1, ..., its count - 1
        assert np.array_equal(np.sort(slots[keys == key]), np.arange(count[key]))


@dace.program
def last_arriver(data: dace.int32[P], counters: dace.int32[R], errors: dace.int32[1]):
    """In every round, each producer reads all data (caching it), writes its own element, and decrements the counter
    of the round. The producer that brings it to zero checks that every producer's write of that round is visible."""
    for p in dace.map[0:P]:
        for r in range(R):
            with dace.tasklet:
                seen << data(-1)[:]
                mine >> data(-1)[p]
                c >> counters(-1, lambda a, b: a + b)[r]
                err >> errors(-1, lambda a, b: a + b)[0]
                total = 0
                for i in range(P):
                    total = total + seen[i]
                mine = r + 1
                remaining = c = -1
                if remaining == 1:
                    stale = 0
                    for i in range(P):
                        if seen[i] < r + 1:
                            stale = stale + 1
                    if total < 0:  # Keeps the first reads
                        stale = stale + 1
                    err = stale


def _run_last_arriver(sdfg: dace.SDFG, producers: int, rounds: int):
    data = np.zeros(producers, np.int32)
    counters = np.full(rounds, producers, np.int32)
    errors = np.zeros(1, np.int32)
    sdfg(data=data, counters=counters, errors=errors, P=producers, R=rounds)
    assert np.all(counters == 0)
    assert np.all(data == rounds)
    assert errors[0] == 0


def test_last_arriver_sees_all_writes():
    sdfg = last_arriver.to_sdfg(simplify=True)
    _run_last_arriver(sdfg, producers=16, rounds=200)


@pytest.mark.gpu
def test_last_arriver_sees_all_writes_gpu():
    """With relaxed atomics, the same program finds stale data millions of times (cached or reused reads)."""
    sdfg = last_arriver.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    assert "dace::wcr_acq_rel<false>::fixed<" in _code(sdfg)
    _run_last_arriver(sdfg, producers=4096, rounds=100)


if __name__ == "__main__":
    test_chained_assignment()
    test_read_after_assignment()
    test_chained_subscript()
    test_custom_reduction()
    test_unread_update_is_relaxed()
    test_claim_slots()
    test_chained_into_connectors()
    test_last_arriver_sees_all_writes()
    test_last_arriver_sees_all_writes_gpu()
