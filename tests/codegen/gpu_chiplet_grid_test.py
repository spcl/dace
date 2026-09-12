# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the distribution of the thread-blocks of a kernel over the chiplets of a GPU. """

import re
import sys
import types
import warnings

import pytest

import dace
from dace.codegen import common
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap

N = 512
M = 1024
K = 8

# Thread-block size of ``explicit_threadblock``, in (x, y) order
TBX, TBY = 64, 4

# The first grid dimension corresponds to the last map parameter and is divided by the default block
# size of 32, so the grid of ``two_dimensional`` is [M / 32, N, 1] = [32, 512, 1] thread-blocks. Over
# 6 chiplets its first dimension is padded to ceil(32 / 6) * 6 = 36, so the grid becomes [36, 512, 1]
# and every chiplet owns 6 thread-blocks of it.
CHIPLETS = 6


@dace.program
def two_dimensional(a: dace.float64[N, M] @ dace.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:M] @ dace.ScheduleType.GPU_Device:
        a[i, j] = 1.0


@dace.program
def three_dimensional(a: dace.float64[K, N, M] @ dace.StorageType.GPU_Global):
    for k, i, j in dace.map[0:K, 0:N, 0:M] @ dace.ScheduleType.GPU_Device:
        a[k, i, j] = 1.0


# A kernel with an explicit thread-block map takes its block size from that map, and its device map
# is already expressed in thread-blocks, so the grid of ``explicit_threadblock`` is
# [M / TBX, N / TBY, 1] = [16, 128, 1]. Over 6 chiplets its first dimension is padded to
# ceil(16 / 6) * 6 = 18, so the grid becomes [18, 128, 1], whose 18 thread-blocks cover the 16 of the
# first dimension, leaving 2 to be masked out.
@dace.program
def explicit_threadblock(a: dace.float64[N, M] @ dace.StorageType.GPU_Global):
    for i, j in dace.map[0:N:TBY, 0:M:TBX] @ dace.ScheduleType.GPU_Device:
        for bi, bj in dace.map[0:TBY, 0:TBX] @ dace.ScheduleType.GPU_ThreadBlock:
            a[i + bi, j + bj] = 1.0


def _fake_amdsmi(chiplets, handles=(object(), )):
    """
    Returns a stand-in for the ``amdsmi`` module that reports GPUs with ``chiplets`` chiplets.

    The module records the calls that initialize and shut it down in its ``calls`` attribute, so that
    a test can check that the query leaves it shut down again.

    :param handles: Processor handles the module reports, empty to mimic a machine without a GPU.
    """
    module = types.ModuleType('amdsmi')
    module.calls = []
    module.amdsmi_init = lambda: module.calls.append('init')
    module.amdsmi_shut_down = lambda: module.calls.append('shut_down')
    module.amdsmi_get_processor_handles = lambda: list(handles)
    module.amdsmi_get_gpu_xcd_counter = lambda handle: chiplets
    return module


def _generate(program, chiplets=None, allow_distribution=None):
    """
    Generates the GPU code of ``program``, targeting HIP with the given number of chiplets.

    :param chiplets: If not None, the value the ``compiler.cuda.chiplet_number`` configuration entry
                     is set to. Left at its default of 0, the number of chiplets is detected instead.
    :param allow_distribution: If not None, the value the ``allow_chiplet_threadblock_distribution``
                               property of every device map of the program is set to.
    """
    with dace.config.temporary_config():
        # Set explicitly, so that the test does not depend on the GPU of the machine it runs on
        dace.config.Config.set('compiler', 'cuda', 'backend', value='hip')
        if chiplets is not None:
            dace.config.Config.set('compiler', 'cuda', 'chiplet_number', value=chiplets)

        # `get_gpu_backend` caches its result for the whole process, so the backend set above only
        # reaches the code generator if that cache is cleared first. It is cleared again afterwards,
        # so that these tests do not force "hip" onto whatever runs next in the same process.
        # `get_gpu_chiplet_count` is cached for the whole process as well, warning once at most, so
        # it is cleared alongside it to keep the tests independent of the order they run in.
        common.get_gpu_backend.cache_clear()
        common.get_gpu_chiplet_count.cache_clear()
        try:
            sdfg = program.to_sdfg()
            if allow_distribution is not None:
                for node, _ in sdfg.all_nodes_recursive():
                    if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
                        node.map.allow_chiplet_threadblock_distribution = allow_distribution
            return sdfg.generate_code()[1].code
        finally:
            common.get_gpu_backend.cache_clear()
            common.get_gpu_chiplet_count.cache_clear()


def test_chiplet_distribution():
    code = _generate(two_dimensional, CHIPLETS)

    # The first grid dimension is padded to a multiple of the number of chiplets, which makes the
    # chiplet a block runs on `blockIdx.x % chiplets` under the hardware round-robin scheduling, and
    # the blocks of the first dimension are permuted so that every chiplet owns a contiguous chunk
    assert 'dim3(36, 512, 1)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6)' in code

    # The second dimension of the map keeps the second grid dimension
    assert re.search(r'\w+ = blockIdx\.y;', code)

    # The blocks that the padding adds beyond the range of the map have to be masked out
    assert re.search(r'if \(\w+ < %d\)' % M, code)


def test_chiplet_distribution_without_threadblock_map(monkeypatch):
    # Kernels without an inner thread-block map offset the block index by the thread index
    # themselves. `AddThreadBlockMap` inserts such a map into every simple kernel, so it is
    # disabled here to generate a kernel that does not have one.
    monkeypatch.setattr(AddThreadBlockMap, 'can_be_applied', lambda *args, **kwargs: False)
    code = _generate(two_dimensional, CHIPLETS)

    assert 'dim3(36, 512, 1)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6) * 32 + threadIdx.x' in code
    assert re.search(r'if \(\w+ < %d\)' % M, code)


def test_chiplet_distribution_for_three_dimensional_grid():
    # The distribution only reshapes the first grid dimension, so it applies to a kernel that uses
    # all three of them, and leaves the other two dimensions of the map on their own grid dimension
    code = _generate(three_dimensional, CHIPLETS)

    assert 'dim3(36, 512, 8)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6)' in code
    assert re.search(r'\w+ = blockIdx\.y;', code)
    assert re.search(r'\w+ = blockIdx\.z;', code)
    assert re.search(r'if \(\w+ < %d\)' % M, code)


def test_chiplet_number_detected(monkeypatch):
    # The number of chiplets is not configured, so it is detected through `amdsmi` and the grid is
    # distributed over the chiplets of the GPU without any configuration
    amdsmi = _fake_amdsmi(CHIPLETS)
    monkeypatch.setitem(sys.modules, 'amdsmi', amdsmi)

    code = _generate(two_dimensional)

    assert 'dim3(36, 512, 1)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6)' in code

    # The query initializes `amdsmi` and shuts it down again, exactly once
    assert amdsmi.calls == ['init', 'shut_down']


def test_detected_chiplet_number_is_written_back(monkeypatch):
    # The detected number replaces the 0 of the configuration entry, so that the rest of the process
    # sees the number of chiplets the code is generated for
    amdsmi = _fake_amdsmi(CHIPLETS)
    monkeypatch.setitem(sys.modules, 'amdsmi', amdsmi)

    with dace.config.temporary_config():
        dace.config.Config.set('compiler', 'cuda', 'backend', value='hip')
        common.get_gpu_backend.cache_clear()
        common.get_gpu_chiplet_count.cache_clear()
        try:
            first = two_dimensional.to_sdfg()
            assert 'dim3(36, 512, 1)' in first.generate_code()[1].code
            assert int(dace.config.Config.get('compiler', 'cuda', 'chiplet_number')) == CHIPLETS

            # The next kernel is distributed over the same number of chiplets, without querying again
            second = two_dimensional.to_sdfg()
            second.name = 'two_dimensional_second_kernel'
            assert 'dim3(36, 512, 1)' in second.generate_code()[1].code
            assert amdsmi.calls == ['init', 'shut_down']
        finally:
            common.get_gpu_backend.cache_clear()
            common.get_gpu_chiplet_count.cache_clear()


def test_chiplet_number_detection_failure_warns(monkeypatch):
    # `amdsmi` ships with ROCm, so importing it fails on a machine that generates code without it
    monkeypatch.setitem(sys.modules, 'amdsmi', None)

    with pytest.warns(UserWarning, match='amdsmi'):
        code = _generate(two_dimensional)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code


def test_chiplet_number_detection_without_gpu_warns(monkeypatch):
    # A machine with ROCm but without a GPU, a login node for instance, reports no processor handle
    amdsmi = _fake_amdsmi(CHIPLETS, handles=())
    monkeypatch.setitem(sys.modules, 'amdsmi', amdsmi)

    with pytest.warns(UserWarning, match='did not report any GPU'):
        code = _generate(two_dimensional)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code

    # `amdsmi` is shut down again even though the query failed
    assert amdsmi.calls == ['init', 'shut_down']


def test_chiplet_distribution_explicitly_disabled():
    code = _generate(two_dimensional, 1)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code

    # Disabling the distribution is deliberate, so it is not reported. Note that the label of the
    # kernel contains the name of this file, so the messages themselves have to be matched.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _generate(two_dimensional, 1)
    assert not [w for w in caught if re.search(r'chiplets?[ ,.]|chiplet_number', str(w.message))]


def test_chiplet_distribution_disabled_per_map():
    # The map opts out, so its grid is left alone even though the distribution is configured. The
    # thread-block map that codegen inserts becomes the kernel map, so this also covers the
    # propagation of the property by `AddThreadBlockMap`.
    code = _generate(two_dimensional, CHIPLETS, allow_distribution=False)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code

    # Opting out is not a misconfiguration, so it is not reported. Note that the label of the kernel
    # contains the name of this file, so the messages themselves have to be matched.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _generate(two_dimensional, CHIPLETS, allow_distribution=False)
    assert not [w for w in caught if re.search(r'chiplets?[ ,.]|chiplet_number', str(w.message))]


def test_chiplet_distribution_disabled_per_map_without_threadblock_map(monkeypatch):
    # Same, for a kernel whose map is the kernel map itself (see the test above)
    monkeypatch.setattr(AddThreadBlockMap, 'can_be_applied', lambda *args, **kwargs: False)
    code = _generate(two_dimensional, CHIPLETS, allow_distribution=False)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code


def test_chiplet_distribution_with_threadblock_map():
    code = _generate(explicit_threadblock, CHIPLETS)

    # The block size comes from the thread-block map, not from `compiler.cuda.default_block_size`,
    # while the grid is distributed over the chiplets
    assert 'dim3(%d, %d, 1)' % (TBX, TBY) in code
    assert 'dim3(18, 128, 1)' in code

    # The thread-block map maps work to the threads of the block, so the index of the distributed
    # dimension is not offset by the thread index, unlike in a kernel without such a map
    assert '(%d * ((blockIdx.x %% 6) * 3 + blockIdx.x / 6))' % TBX in code
    assert '(%d * blockIdx.y)' % TBY in code
    assert re.search(r'if \(\w+ < %d\)' % M, code)

    assert re.search(r'\w+ = threadIdx\.x;', code)
    assert re.search(r'\w+ = threadIdx\.y;', code)


def test_chiplet_distribution_disabled_per_map_with_threadblock_map():
    code = _generate(explicit_threadblock, CHIPLETS, allow_distribution=False)

    # The grid is left alone, and the block size is unaffected either way
    assert 'dim3(16, 128, 1)' in code
    assert 'dim3(%d, %d, 1)' % (TBX, TBY) in code
    assert 'blockIdx.x % ' not in code


def test_allow_chiplet_threadblock_distribution_is_serialized():
    sdfg = two_dimensional.to_sdfg()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.allow_chiplet_threadblock_distribution = False

    restored = dace.SDFG.from_json(sdfg.to_json())

    maps = [
        node.map for node, _ in restored.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device
    ]
    assert maps
    assert all(m.allow_chiplet_threadblock_distribution is False for m in maps)


def test_invalid_chiplet_number():
    # 0 is the value that asks for the number of chiplets to be detected, so only a negative number
    # of chiplets is invalid
    with pytest.raises(ValueError, match='chiplet'):
        _generate(two_dimensional, -1)


if __name__ == '__main__':
    test_chiplet_distribution()
    test_chiplet_distribution_for_three_dimensional_grid()
    test_chiplet_distribution_explicitly_disabled()
    test_chiplet_distribution_disabled_per_map()
    test_chiplet_distribution_with_threadblock_map()
    test_chiplet_distribution_disabled_per_map_with_threadblock_map()
    test_allow_chiplet_threadblock_distribution_is_serialized()
    test_invalid_chiplet_number()
