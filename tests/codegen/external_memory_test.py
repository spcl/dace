# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests external memory allocation.
"""
import dace
import numpy as np
import pytest


@pytest.mark.parametrize('symbolic', (False, True))
def test_external_mem(symbolic):
    N = dace.symbol('N') if symbolic else 20

    @dace.program
    def tester(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)

        workspace[:] = a
        workspace += 1
        a[:] = workspace

    sdfg = tester.to_sdfg()

    # Test that there is no allocation
    code = sdfg.generate_code()[0].clean_code
    # No heap allocation in either form (plain or C++ >= 17 aligned).
    assert 'new double' not in code
    assert 'new (std::align_val_t(64)) double' not in code
    assert 'delete[]' not in code
    assert 'set_external_memory' in code

    a = np.random.rand(20)

    if symbolic:
        extra_args = dict(N=20)
    else:
        extra_args = {}

    # The nanobind interface stores no symbol values between calls (by
    # design), so the workspace queries take them per call; ctypes reuses
    # the values passed to initialize() and accepts no arguments here.
    ws_args = extra_args if dace.Config.get('compiler', 'interface') == 'nanobind' else {}

    # Test workspace size
    csdfg = sdfg.compile()
    csdfg.initialize(a, **extra_args)
    sizes = csdfg.get_workspace_sizes(**ws_args)
    assert sizes == {dace.StorageType.CPU_Heap: 20 * 8}

    # Test setting the workspace
    wsp = np.random.rand(20)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp, **ws_args)

    ref = a + 1

    csdfg(a, **extra_args)

    assert np.allclose(a, ref)
    assert np.allclose(wsp, ref)


def test_external_twobuffers():
    N = dace.symbol('N')

    @dace.program
    def tester(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)
        workspace2 = dace.ndarray([2], dace.float64, lifetime=dace.AllocationLifetime.External)

        workspace[:] = a
        workspace += 1
        workspace2[0] = np.sum(workspace)
        workspace2[1] = np.mean(workspace)
        a[0] = workspace2[0] + workspace2[1]

    sdfg = tester.to_sdfg()
    csdfg = sdfg.compile()

    # See test_external_mem: nanobind takes the symbols per call.
    ws_args = dict(N=20) if dace.Config.get('compiler', 'interface') == 'nanobind' else {}

    # Test workspace size
    a = np.random.rand(20)
    csdfg.initialize(a=a, N=20)
    sizes = csdfg.get_workspace_sizes(**ws_args)
    assert sizes == {dace.StorageType.CPU_Heap: 22 * 8}

    # Test setting the workspace
    wsp = np.random.rand(22)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp, **ws_args)

    ref = a + 1
    ref2 = np.copy(a)
    s, m = np.sum(ref), np.mean(ref)
    ref2[0] = s + m

    csdfg(a=a, N=20)

    assert np.allclose(a, ref2)
    assert np.allclose(wsp[:-2], ref)
    assert np.allclose(wsp[-2], s)
    assert np.allclose(wsp[-1], m)


if __name__ == '__main__':
    test_external_mem(False)
    test_external_mem(True)
    test_external_twobuffers()
