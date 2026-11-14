# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""On the CPU a library node a parallel region encloses takes its single-core lowering; at top level, the library."""

import os

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.linalg import TensorTranspose
from dace.transformation.passes.canonicalize.finalize import (
    SilentPureLowering,
    canonicalize_set_fast_implementations,
    refuse_silent_pure,
)


def transpose_sdfg(in_map: bool) -> tuple[dace.SDFG, TensorTranspose]:
    """``y[t] = transpose(x[t], (2, 0, 1))`` per task ``t`` of a parallel map, or one top-level transpose."""
    sdfg = dace.SDFG(f"cpu_transpose_{'in_map' if in_map else 'top'}")
    tasks = 6 if in_map else 1
    sdfg.add_array("x", [tasks, 4, 8, 3], dace.float64)
    sdfg.add_array("y", [tasks, 3, 4, 8], dace.float64)
    sdfg.add_array("xt", [4, 8, 3], dace.float64, transient=True)
    sdfg.add_array("yt", [3, 4, 8], dace.float64, transient=True)
    state = sdfg.add_state()
    node = TensorTranspose("transpose", axes=[2, 0, 1])
    xt, yt = state.add_access("xt"), state.add_access("yt")
    state.add_edge(xt, None, node, "_inp_tensor", dace.Memlet("xt[0:4, 0:8, 0:3]"))
    state.add_edge(node, "_out_tensor", yt, None, dace.Memlet("yt[0:3, 0:4, 0:8]"))
    if in_map:
        entry, exit_node = state.add_map("tasks", dict(t="0:6"), schedule=dtypes.ScheduleType.CPU_Multicore)
        state.add_memlet_path(state.add_read("x"), entry, xt, memlet=dace.Memlet("x[t, 0:4, 0:8, 0:3]"))
        state.add_memlet_path(yt, exit_node, state.add_write("y"), memlet=dace.Memlet("y[t, 0:3, 0:4, 0:8]"))
    else:
        state.add_edge(state.add_read("x"), None, xt, None, dace.Memlet("x[0, 0:4, 0:8, 0:3]"))
        state.add_edge(yt, None, state.add_write("y"), None, dace.Memlet("y[0, 0:3, 0:4, 0:8]"))
    sdfg.validate()
    return sdfg, node


def test_a_transpose_inside_a_parallel_map_takes_the_in_thread_loop():
    """cp2k_grid_integrate's per-task transposes: 19 s through an HPTT plan per call, 4 s in-thread."""
    sdfg, node = transpose_sdfg(in_map=True)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation == "sequential"
    x = np.random.default_rng(4).random((6, 4, 8, 3))
    y = np.zeros((6, 3, 4, 8))
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y, np.transpose(x, (0, 3, 1, 2)))


@pytest.mark.skipif("HPTT_ROOT" not in os.environ, reason="HPTT is not installed")
def test_a_top_level_transpose_calls_hptt():
    sdfg, node = transpose_sdfg(in_map=False)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation == "HPTT"
    refuse_silent_pure(sdfg, dtypes.DeviceType.CPU)


def test_a_top_level_transpose_left_pure_is_refused(monkeypatch):
    """Without HPTT a top-level transpose would silently run the serial loop: finalize refuses it, by name."""
    monkeypatch.delenv("HPTT_ROOT", raising=False)
    sdfg, node = transpose_sdfg(in_map=False)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation == "pure"
    with pytest.raises(SilentPureLowering, match="TensorTranspose 'transpose' .*HPTT"):
        refuse_silent_pure(sdfg, dtypes.DeviceType.CPU)


def test_a_transpose_in_a_parallel_map_is_not_refused(monkeypatch):
    monkeypatch.delenv("HPTT_ROOT", raising=False)
    sdfg, _ = transpose_sdfg(in_map=True)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    refuse_silent_pure(sdfg, dtypes.DeviceType.CPU)


if __name__ == "__main__":
    test_a_transpose_inside_a_parallel_map_takes_the_in_thread_loop()
    if "HPTT_ROOT" in os.environ:
        test_a_top_level_transpose_calls_hptt()
