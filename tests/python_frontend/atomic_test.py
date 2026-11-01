# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.atomic.add``: an atomic update that returns the old value, e.g., for the last of many updates to act."""

import numpy as np

import dace

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


def test_python_semantics():
    """Outside of programs, the update is plain Python."""
    counters = np.array([5, 7])
    assert dace.atomic.add(counters, 1, -2) == 7
    assert np.array_equal(counters, [5, 5])


if __name__ == "__main__":
    test_last_arriver()
    test_tickets_from_scalars()
    test_index_from_data()
    test_python_semantics()
