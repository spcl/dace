# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the distribution of the thread-blocks of a kernel over the chiplets of a GPU. """

import re

import pytest

import dace
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap

N = 512
M = 1024
K = 8

# The first grid dimension corresponds to the last map parameter and is divided by the default block
# size of 32, so the grid of ``two_dimensional`` is [M / 32, N, 1] = [32, 512, 1] thread-blocks. Over
# 6 chiplets it becomes [6, ceil(32 / 6), 512] = [6, 6, 512].
CHIPLETS = 6


@dace.program
def two_dimensional(a: dace.float64[N, M] @ dace.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:M] @ dace.ScheduleType.GPU_Device:
        a[i, j] = 1.0


@dace.program
def three_dimensional(a: dace.float64[K, N, M] @ dace.StorageType.GPU_Global):
    for k, i, j in dace.map[0:K, 0:N, 0:M] @ dace.ScheduleType.GPU_Device:
        a[k, i, j] = 1.0


def _generate(program, chiplets=None):
    """ Generates the GPU code of ``program``, targeting HIP with the given number of chiplets. """
    with dace.config.temporary_config():
        # Set explicitly, so that the test does not depend on the GPU of the machine it runs on
        dace.config.Config.set('compiler', 'cuda', 'backend', value='hip')
        if chiplets is not None:
            dace.config.Config.set('compiler', 'cuda', 'chiplet_number', value=chiplets)
        return program.to_sdfg().generate_code()[1].code


def test_chiplet_distribution():
    code = _generate(two_dimensional, CHIPLETS)

    # The chiplet ID is the first grid dimension, which the hardware round-robin scheduling maps to
    # the chiplets, and the first dimension of the map is spread over the two first grid dimensions
    assert 'dim3(6, 6, 512)' in code
    assert '(blockIdx.x * gridDim.y + blockIdx.y)' in code

    # The second dimension of the map moves to the third grid dimension
    assert re.search(r'\w+ = blockIdx\.z;', code)

    # The first grid dimension is padded to a multiple of the number of chiplets, so the blocks
    # beyond the range of the map have to be masked out
    assert re.search(r'if \(\w+ < %d\)' % M, code)


def test_chiplet_distribution_without_threadblock_map(monkeypatch):
    # Kernels without an inner thread-block map offset the block index by the thread index
    # themselves. `AddThreadBlockMap` inserts such a map into every simple kernel, so it is
    # disabled here to generate a kernel that does not have one.
    monkeypatch.setattr(AddThreadBlockMap, 'can_be_applied', lambda *args, **kwargs: False)
    code = _generate(two_dimensional, CHIPLETS)

    assert 'dim3(6, 6, 512)' in code
    assert '(blockIdx.x * gridDim.y + blockIdx.y) * 32 + threadIdx.x' in code
    assert re.search(r'if \(\w+ < %d\)' % M, code)


def test_chiplet_distribution_skipped_for_three_dimensional_grid():
    # The distribution moves the second grid dimension to the third one, so it cannot be applied to
    # a kernel that uses all three of them
    with pytest.warns(UserWarning, match='third grid dimension'):
        code = _generate(three_dimensional, CHIPLETS)

    assert 'dim3(32, 512, 8)' in code
    assert 'gridDim.y' not in code


def test_chiplet_distribution_disabled_by_default():
    default = _generate(two_dimensional)
    disabled = _generate(two_dimensional, 1)

    assert default == disabled
    assert 'dim3(32, 512, 1)' in default
    assert 'gridDim.y' not in default


def test_invalid_chiplet_number():
    with pytest.raises(ValueError, match='chiplet'):
        _generate(two_dimensional, 0)


if __name__ == '__main__':
    test_chiplet_distribution()
    test_chiplet_distribution_skipped_for_three_dimensional_grid()
    test_chiplet_distribution_disabled_by_default()
    test_invalid_chiplet_number()
