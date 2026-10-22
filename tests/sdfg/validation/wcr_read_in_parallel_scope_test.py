# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests the validation warning for code nodes that read, inside a parallel scope, a location their own iteration
contributed to through a write-conflict resolution (WCR).
"""

import warnings

import pytest

import dace

_RACE_MESSAGE = "partially-reduced"


def _wcr_then_read_sdfg(
    name: str,
    schedule: dace.ScheduleType = dace.ScheduleType.Default,
    transient_accumulator: bool = False,
) -> dace.SDFG:
    """
    Builds ``map i { A[i] -> t1 --(wcr +)--> acc[0] -> t2 -> B[i] }``.

    :param name: The SDFG name.
    :param schedule: The schedule of the map.
    :param transient_accumulator: If True, ``acc`` is a transient accessed only inside the map (private to each
                                  iteration); otherwise it is the non-transient array ``C``.
    :return: The SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [10], dace.float64)
    if transient_accumulator:
        acc_name, _ = sdfg.add_transient("acc", [1], dace.float64)
    else:
        acc_name, _ = sdfg.add_array("C", [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", dict(i="0:10"), schedule=schedule)
    t1 = state.add_tasklet("contribute", {"a"}, {"o"}, "o = a")
    t2 = state.add_tasklet("observe", {"c"}, {"b"}, "b = c")
    acc = state.add_access(acc_name)
    state.add_memlet_path(state.add_read("A"), entry, t1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_edge(t1, "o", acc, None, dace.Memlet(f"{acc_name}[0]", wcr="lambda x, y: x + y"))
    state.add_edge(acc, None, t2, "c", dace.Memlet(f"{acc_name}[0]"))
    state.add_memlet_path(t2, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    return sdfg


def _validate_without_race_warning(sdfg: dace.SDFG) -> None:
    """Validates ``sdfg`` and asserts that no WCR read race warning was emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdfg.validate()
    assert not any(_RACE_MESSAGE in str(w.message) for w in caught)


def test_wcr_read_in_parallel_map_warns():
    sdfg = _wcr_then_read_sdfg("wcr_read_parallel")
    with pytest.warns(UserWarning, match=_RACE_MESSAGE):
        sdfg.validate()


def test_wcr_read_in_sequential_map_does_not_warn():
    sdfg = _wcr_then_read_sdfg("wcr_read_sequential", schedule=dace.ScheduleType.Sequential)
    _validate_without_race_warning(sdfg)


def test_wcr_read_of_private_transient_does_not_warn():
    sdfg = _wcr_then_read_sdfg("wcr_read_private", transient_accumulator=True)
    _validate_without_race_warning(sdfg)


def test_wcr_read_after_map_does_not_warn():
    """``map i { A[i] -> t1 --(wcr +)--> } -> C[0] -> t2 -> B[0]``: the read happens after the map completes."""
    sdfg = dace.SDFG("wcr_read_after_map")
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", dict(i="0:10"))
    t1 = state.add_tasklet("contribute", {"a"}, {"o"}, "o = a")
    t2 = state.add_tasklet("observe", {"c"}, {"b"}, "b = c")
    c = state.add_access("C")
    state.add_memlet_path(state.add_read("A"), entry, t1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t1, exit_, c, src_conn="o", memlet=dace.Memlet("C[0]", wcr="lambda x, y: x + y"))
    state.add_edge(c, None, t2, "c", dace.Memlet("C[0]"))
    state.add_edge(t2, "b", state.add_write("B"), None, dace.Memlet("B[0]"))
    _validate_without_race_warning(sdfg)


if __name__ == "__main__":
    test_wcr_read_in_parallel_map_warns()
    test_wcr_read_in_sequential_map_does_not_warn()
    test_wcr_read_of_private_transient_does_not_warn()
    test_wcr_read_after_map_does_not_warn()
