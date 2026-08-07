# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that a GPU failure inside a compiled SDFG reaches the caller, and reaches the right caller."""
import numpy as np
import pytest

import dace
from dace.codegen import common

N = dace.symbol('N')
M = dace.symbol('M')

GRID_Y_LIMIT = 65535

# An M above the grid y limit makes the launch itself invalid, which the driver reports without
# poisoning the context, so the remaining tests can keep using the GPU.
FAILING_M = GRID_Y_LIMIT + 1
WORKING_M = 8


@dace.program
def gpu_error_doubler(A: dace.float64[M, N], B: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = A[i, j] * 2.0


def build_doubler():
    """Compiles the doubler with block y == 1, so the map's M range lands directly on the grid y dimension."""
    sdfg = gpu_error_doubler.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
                node.map.gpu_block_size = [64, 1, 1]
    return sdfg.compile()


def arguments(m):
    return {
        'A': np.ones((m, 2), dtype=np.float64),
        'B': np.zeros((m, 2), dtype=np.float64),
        'N': 2,
        'M': m,
    }


def unchecked_call(csdfg, m):
    """A call that must NOT consume the SDFG's error record.

    ctypes: ``fast_call`` with ``do_gpu_check=False`` (the documented default).
    nanobind: there is no ``fast_call``; the equivalent is a plain call with
    the ``gpu_error_check`` toggle off - the check (and thus the record read)
    lives in the compiled binding, gated by that toggle.
    """
    if hasattr(csdfg, 'fast_call'):
        callargs, initargs = csdfg.construct_arguments(**arguments(m))
        csdfg.fast_call(callargs, initargs, do_gpu_check=False)
    else:
        previous = csdfg.gpu_error_check
        csdfg.gpu_error_check = False
        try:
            csdfg(**arguments(m))
        finally:
            csdfg.gpu_error_check = previous


def checked_call(csdfg, m):
    """A call that reads the SDFG's error record and raises what it holds."""
    if hasattr(csdfg, 'fast_call'):
        callargs, initargs = csdfg.construct_arguments(**arguments(m))
        csdfg.fast_call(callargs, initargs, do_gpu_check=True)
    else:
        csdfg(**arguments(m))


@pytest.mark.gpu
def test_failed_launch_reaches_caller():
    csdfg = build_doubler()
    with pytest.raises(RuntimeError, match='gpu_error_doubler'):
        csdfg(**arguments(FAILING_M))

    # Reporting the failure consumed it, so tearing down finds nothing left to complain about.
    csdfg.finalize()


@pytest.mark.gpu
def test_failure_is_not_charged_to_an_innocent_sdfg():
    failing = build_doubler()
    innocent = build_doubler()

    # An unchecked call: nothing here consumes the failing SDFG's record.
    unchecked_call(failing, FAILING_M)

    innocent(**arguments(WORKING_M))
    innocent.finalize()

    # The failure is still charged to the SDFG that actually caused it.
    with pytest.raises(RuntimeError, match='gpu_error_doubler'):
        failing.finalize()


@pytest.mark.gpu
def test_failure_survives_a_drained_runtime_slot():
    csdfg = build_doubler()

    unchecked_call(csdfg, FAILING_M)

    # Stands in for any other GPU user in the process reading the runtime's shared last-error slot.
    common.get_gpu_runtime().get_last_error()

    # This launch is valid, so only the record the generated code kept can still report the failure.
    with pytest.raises(RuntimeError, match='gpu_error_doubler'):
        checked_call(csdfg, WORKING_M)

    csdfg.finalize()


@pytest.mark.gpu
def test_reported_failure_is_not_reported_again():
    csdfg = build_doubler()

    with pytest.raises(RuntimeError, match='gpu_error_doubler'):
        checked_call(csdfg, FAILING_M)

    checked_call(csdfg, WORKING_M)

    csdfg.finalize()


if __name__ == '__main__':
    test_failed_launch_reaches_caller()
    test_failure_is_not_charged_to_an_innocent_sdfg()
    test_failure_survives_a_drained_runtime_slot()
    test_reported_failure_is_not_reported_again()
