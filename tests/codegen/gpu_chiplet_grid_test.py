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

# Both CUDA code generators distribute the grid over chiplets, and the point of the tests below is
# that the two agree on every grid, permutation and mask. `compiler.cuda.implementation` selects one
# of them (`dace/codegen/codegen.py` disables the other), so every test that generates code runs
# once per implementation.
CUDA_IMPLEMENTATIONS = ('legacy', 'experimental')
implementations = pytest.mark.parametrize('implementation', CUDA_IMPLEMENTATIONS)

# The grid of a kernel without an explicit thread-block map is the map range divided by the configured
# default block size, which is ambient configuration, so `pin_codegen_configuration` pins it to the
# value the grids below state instead of reading whatever the branch this test runs on happens to
# default to.
BLOCK_X = 32

# The first grid dimension corresponds to the last map parameter and is divided by ``BLOCK_X``, so the
# grid of ``two_dimensional`` is [M / 32, N, 1] = [32, 512, 1] thread-blocks. Over 6 chiplets its first
# dimension is padded to ceil(32 / 6) * 6 = 36, so the grid becomes [36, 512, 1] and every chiplet owns
# 6 thread-blocks of it.
CHIPLETS = 6

# The thread-blocks the padding adds run past the end of the map, so the index of the distributed
# dimension is masked against the map range. Its absence is what says the distribution is off.
TRAILING_BLOCK_MASK = r'if \(\w+ < %d\)' % M


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


def fake_amdsmi(chiplets, handles=(object(), )):
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


def pin_codegen_configuration(implementation):
    """Pins every configuration entry the grids expected by these tests are a function of."""
    # Set explicitly, so that the tests do not depend on the GPU of the machine they run on
    dace.config.Config.set('compiler', 'cuda', 'backend', value='hip')
    dace.config.Config.set('compiler', 'cuda', 'implementation', value=implementation)
    dace.config.Config.set('compiler', 'cuda', 'default_block_size', value='%d,1,1' % BLOCK_X)


def generate_gpu_code(program, implementation, chiplets=None, allow_distribution=None):
    """
    Generates the GPU code of ``program``, targeting HIP with the given number of chiplets.

    :param implementation: Value of the ``compiler.cuda.implementation`` configuration entry, which
                           selects the code generator the kernel is emitted by.
    :param chiplets: If not None, the value the ``compiler.cuda.chiplet_number`` configuration entry
                     is set to. Left at its default of 0, the number of chiplets is detected instead.
    :param allow_distribution: If not None, the value the ``allow_chiplet_threadblock_distribution``
                               property of every device map of the program is set to.
    """
    with dace.config.temporary_config():
        pin_codegen_configuration(implementation)
        if chiplets is not None:
            dace.config.Config.set('compiler', 'cuda', 'chiplet_number', value=chiplets)

        # `get_gpu_backend` reads the configuration entry fresh every call (see its docstring), so the
        # backend set above reaches the code generator without any cache to clear. `get_gpu_chiplet_count`
        # is cached for the whole process, warning once at most, so it is cleared to keep the tests
        # independent of the order they run in.
        common.get_gpu_chiplet_count.cache_clear()
        try:
            sdfg = program.to_sdfg()
            if allow_distribution is not None:
                for node, _ in sdfg.all_nodes_recursive():
                    if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
                        node.map.allow_chiplet_threadblock_distribution = allow_distribution
            return sdfg.generate_code()[1].code
        finally:
            common.get_gpu_chiplet_count.cache_clear()


@implementations
def test_chiplet_distribution(implementation):
    code = generate_gpu_code(two_dimensional, implementation, CHIPLETS)

    # The first grid dimension is padded to a multiple of the number of chiplets, which makes the
    # chiplet a block runs on `blockIdx.x % chiplets` under the hardware round-robin scheduling, and
    # the blocks of the first dimension are permuted so that every chiplet owns a contiguous chunk
    assert 'dim3(36, 512, 1)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6)' in code

    # The second dimension of the map keeps the second grid dimension
    assert re.search(r'\w+ = blockIdx\.y;', code)

    # The blocks that the padding adds beyond the range of the map have to be masked out
    assert re.search(TRAILING_BLOCK_MASK, code)


@implementations
def test_chiplet_distribution_for_three_dimensional_grid(implementation):
    # The distribution only reshapes the first grid dimension, so it applies to a kernel that uses
    # all three of them, and leaves the other two dimensions of the map on their own grid dimension
    code = generate_gpu_code(three_dimensional, implementation, CHIPLETS)

    assert 'dim3(36, 512, 8)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6)' in code
    assert re.search(r'\w+ = blockIdx\.y;', code)
    assert re.search(r'\w+ = blockIdx\.z;', code)
    assert re.search(TRAILING_BLOCK_MASK, code)


@implementations
def test_chiplet_number_detected(implementation, monkeypatch):
    # The number of chiplets is not configured, so it is detected through `amdsmi` and the grid is
    # distributed over the chiplets of the GPU without any configuration
    amdsmi = fake_amdsmi(CHIPLETS)
    monkeypatch.setitem(sys.modules, 'amdsmi', amdsmi)

    code = generate_gpu_code(two_dimensional, implementation)

    assert 'dim3(36, 512, 1)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6)' in code

    # The query initializes `amdsmi` and shuts it down again, exactly once
    assert amdsmi.calls == ['init', 'shut_down']


@implementations
def test_detected_chiplet_number_is_written_back(implementation, monkeypatch):
    # The detected number replaces the 0 of the configuration entry, so that the rest of the process
    # sees the number of chiplets the code is generated for
    amdsmi = fake_amdsmi(CHIPLETS)
    monkeypatch.setitem(sys.modules, 'amdsmi', amdsmi)

    with dace.config.temporary_config():
        pin_codegen_configuration(implementation)
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
            common.get_gpu_chiplet_count.cache_clear()


@implementations
def test_chiplet_number_detection_failure_warns(implementation, monkeypatch):
    # `amdsmi` ships with ROCm, so importing it fails on a machine that generates code without it
    monkeypatch.setitem(sys.modules, 'amdsmi', None)

    with pytest.warns(UserWarning, match='amdsmi'):
        code = generate_gpu_code(two_dimensional, implementation)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code
    assert not re.search(TRAILING_BLOCK_MASK, code)


@implementations
def test_chiplet_number_detection_without_gpu_warns(implementation, monkeypatch):
    # A machine with ROCm but without a GPU, a login node for instance, reports no processor handle
    amdsmi = fake_amdsmi(CHIPLETS, handles=())
    monkeypatch.setitem(sys.modules, 'amdsmi', amdsmi)

    with pytest.warns(UserWarning, match='did not report any GPU'):
        code = generate_gpu_code(two_dimensional, implementation)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code
    assert not re.search(TRAILING_BLOCK_MASK, code)

    # `amdsmi` is shut down again even though the query failed
    assert amdsmi.calls == ['init', 'shut_down']


@implementations
def test_chiplet_distribution_explicitly_disabled(implementation):
    code = generate_gpu_code(two_dimensional, implementation, 1)

    # Turning the distribution off leaves the grid at its own size, indexed by `blockIdx.x` alone and
    # with no mask for a padding that is not there
    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code
    assert not re.search(TRAILING_BLOCK_MASK, code)

    # Disabling the distribution is deliberate, so it is not reported. Note that the label of the
    # kernel contains the name of this file, so the messages themselves have to be matched.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        generate_gpu_code(two_dimensional, implementation, 1)
    assert not [w for w in caught if re.search(r'chiplets?[ ,.]|chiplet_number', str(w.message))]


@implementations
def test_chiplet_distribution_disabled_per_map(implementation):
    # The map opts out, so its grid is left alone even though the distribution is configured. The
    # thread-block map that codegen inserts becomes the kernel map, so this also covers the
    # propagation of the property by `AddThreadBlockMap`.
    code = generate_gpu_code(two_dimensional, implementation, CHIPLETS, allow_distribution=False)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code
    assert not re.search(TRAILING_BLOCK_MASK, code)

    # Opting out is not a misconfiguration, so it is not reported. Note that the label of the kernel
    # contains the name of this file, so the messages themselves have to be matched.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        generate_gpu_code(two_dimensional, implementation, CHIPLETS, allow_distribution=False)
    assert not [w for w in caught if re.search(r'chiplets?[ ,.]|chiplet_number', str(w.message))]


@implementations
def test_chiplet_distribution_with_threadblock_map(implementation):
    code = generate_gpu_code(explicit_threadblock, implementation, CHIPLETS)

    # The block size comes from the thread-block map, not from `compiler.cuda.default_block_size`,
    # while the grid is distributed over the chiplets
    assert 'dim3(%d, %d, 1)' % (TBX, TBY) in code
    assert 'dim3(18, 128, 1)' in code

    # The thread-block map maps work to the threads of the block, so the index of the distributed
    # dimension is not offset by the thread index, unlike in a kernel without such a map
    assert '(%d * ((blockIdx.x %% 6) * 3 + blockIdx.x / 6))' % TBX in code
    assert '(%d * blockIdx.y)' % TBY in code
    assert re.search(TRAILING_BLOCK_MASK, code)

    assert re.search(r'\w+ = threadIdx\.x;', code)
    assert re.search(r'\w+ = threadIdx\.y;', code)


@implementations
def test_chiplet_distribution_disabled_per_map_with_threadblock_map(implementation):
    code = generate_gpu_code(explicit_threadblock, implementation, CHIPLETS, allow_distribution=False)

    # The grid is left alone, and the block size is unaffected either way
    assert 'dim3(16, 128, 1)' in code
    assert 'dim3(%d, %d, 1)' % (TBX, TBY) in code
    assert 'blockIdx.x % ' not in code
    assert not re.search(TRAILING_BLOCK_MASK, code)


@implementations
def test_invalid_chiplet_number(implementation):
    # 0 is the value that asks for the number of chiplets to be detected, so only a negative number
    # of chiplets is invalid
    with pytest.raises(ValueError, match='chiplet'):
        generate_gpu_code(two_dimensional, implementation, -1)


def test_legacy_chiplet_distribution_without_threadblock_map(monkeypatch):
    # Kernels without an inner thread-block map offset the block index by the thread index
    # themselves. `AddThreadBlockMap` inserts such a map into every simple kernel, so it is
    # disabled here to generate a kernel that does not have one. Only the legacy code generator
    # emits that form of kernel entry -- see
    # `test_experimental_codegen_requires_a_threadblock_map` for what the other one does instead.
    monkeypatch.setattr(AddThreadBlockMap, 'can_be_applied', lambda *args, **kwargs: False)
    code = generate_gpu_code(two_dimensional, 'legacy', CHIPLETS)

    assert 'dim3(36, 512, 1)' in code
    assert '((blockIdx.x % 6) * 6 + blockIdx.x / 6) * 32 + threadIdx.x' in code
    assert re.search(TRAILING_BLOCK_MASK, code)


def test_legacy_chiplet_distribution_disabled_per_map_without_threadblock_map(monkeypatch):
    # Same, for a kernel whose map is the kernel map itself (see the test above)
    monkeypatch.setattr(AddThreadBlockMap, 'can_be_applied', lambda *args, **kwargs: False)
    code = generate_gpu_code(two_dimensional, 'legacy', CHIPLETS, allow_distribution=False)

    assert 'dim3(32, 512, 1)' in code
    assert 'blockIdx.x % ' not in code

    # `TRAILING_BLOCK_MASK` is not asserted absent here: a kernel map that binds threads itself
    # masks its own trailing block with the very same condition, distribution or none, so in this
    # one kernel shape the mask says nothing about chiplets. The grid and the index above do.


def test_experimental_codegen_requires_a_threadblock_map(monkeypatch):
    # `ExperimentalCUDACodeGen` reads the launch configuration off an inner thread-block map, so a
    # kernel that has none is refused before any chiplet decision is reached. That is why the two
    # tests above are pinned to the legacy code generator: the kernel entry they describe has no
    # counterpart here, rather than a chiplet distribution that disagrees.
    monkeypatch.setattr(AddThreadBlockMap, 'can_be_applied', lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match='GPU_ThreadBlock map'):
        generate_gpu_code(two_dimensional, 'experimental', CHIPLETS)


def test_allow_chiplet_threadblock_distribution_is_serialized():
    # The property travels with the map, not with the code generator, so this one is not run per
    # implementation: no code is generated here.
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


if __name__ == '__main__':
    for cuda_implementation in CUDA_IMPLEMENTATIONS:
        test_chiplet_distribution(cuda_implementation)
        test_chiplet_distribution_for_three_dimensional_grid(cuda_implementation)
        test_chiplet_distribution_explicitly_disabled(cuda_implementation)
        test_chiplet_distribution_disabled_per_map(cuda_implementation)
        test_chiplet_distribution_with_threadblock_map(cuda_implementation)
        test_chiplet_distribution_disabled_per_map_with_threadblock_map(cuda_implementation)
        test_invalid_chiplet_number(cuda_implementation)
    test_allow_chiplet_threadblock_distribution_is_serialized()
