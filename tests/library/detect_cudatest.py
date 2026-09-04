# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Device lowerings of the detection primitives: ``FindFirst`` and ``ScatterConflictCheck``.

Both run the search on the GPU (``dace/runtime/include/dace/cuda/detect.cuh``) and are checked
against the host expansions on the same input, so a divergence is a device bug and not a
disagreement about what the answer is.
"""
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.libraries.standard.nodes import FindFirst
from dace.libraries.standard.nodes.find_first import INDEX_NAME, OUTPUT_CONNECTOR_NAME

N = dace.symbol('N')


def make_find_first_sdfg(implementation: str, threshold: float) -> dace.SDFG:
    """``out[0] = first i with a[i] > threshold``, with ``a`` staged to the device for CUDA."""
    on_device = implementation == 'CUDA'
    sdfg = dace.SDFG(f'find_first_{implementation}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [1], dace.int64)

    if on_device:
        sdfg.add_transient('a_dev', [N], dace.float64, storage=dace.StorageType.GPU_Global)
        copy = sdfg.add_state('stage', is_start_block=True)
        copy.add_edge(copy.add_read('a'), None, copy.add_write('a_dev'), None, dace.Memlet('a[0:N]'))
        state = sdfg.add_state_after(copy, 'search')
        source = 'a_dev'
    else:
        state = sdfg.add_state('search', is_start_block=True)
        source = 'a'

    node = FindFirst('search', predicate=f'__r_a[{INDEX_NAME}] > {threshold}', begin=0, end=N)
    node.implementation = implementation
    # Host code either way -- the CUDA expansion LAUNCHES from the host. Set explicitly because a
    # node reading device memory and writing a host scalar has no unambiguous inferred schedule.
    node.schedule = dace.ScheduleType.Sequential
    state.add_node(node)
    node.add_in_connector('__r_a')
    state.add_edge(state.add_read(source), None, node, '__r_a', dace.Memlet(f'{source}[0:N]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize('first', [0, 1, 37, 4097, -1])
def test_find_first_cuda_matches_host(first):
    """Every firing position, including none at all, answers the same on both machines."""
    n = 8192
    a = np.zeros(n, dtype=np.float64)
    if first >= 0:
        a[first:] = 1.0
    expected = n if first < 0 else first

    got_gpu = np.zeros(1, dtype=np.int64)
    make_find_first_sdfg('CUDA', 0.5)(a=a.copy(), out=got_gpu, N=n)
    got_cpu = np.zeros(1, dtype=np.int64)
    make_find_first_sdfg('OpenMP', 0.5)(a=a.copy(), out=got_cpu, N=n)

    assert got_gpu[0] == expected, f'CUDA answered {got_gpu[0]}, expected {expected}'
    assert got_cpu[0] == expected, f'OpenMP answered {got_cpu[0]}, expected {expected}'


@pytest.mark.gpu
def test_find_first_cuda_is_exact_when_most_indices_fire():
    """Dense firing: every block finds something and they all race to publish, so a device
    lowering that folded its answer into an advisory word instead of an ``atomicMin`` answers a
    firing index that is not the first."""
    n = 1 << 16
    sdfg = make_find_first_sdfg('CUDA', 0.5)
    for first in (1, 129, 1023, 5000):
        a = np.zeros(n, dtype=np.float64)
        a[first:] = 1.0
        out = np.zeros(1, dtype=np.int64)
        sdfg(a=a, out=out, N=n)
        assert out[0] == first, f'dense firing from {first}: answered {out[0]}'


@pytest.mark.gpu
def test_find_first_host_expansion_refuses_device_memory():
    """A host expansion over GPU memory would dereference a device pointer from the host. Refused
    with the knob to turn, never silently emitted."""
    sdfg = make_find_first_sdfg('CUDA', 0.5)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, FindFirst))
    node.implementation = 'OpenMP'
    with pytest.raises(NotImplementedError, match="implementation='CUDA'"):
        sdfg.expand_library_nodes()


def make_conflict_sdfg(implementation: str, capacity: int) -> dace.SDFG:
    """``count[0] = 1`` iff the device-resident index ``ip`` holds a duplicate."""
    on_device = implementation == 'CUDA'
    sdfg = dace.SDFG(f'scatter_conflict_{implementation}')
    sdfg.add_array('ip', [N], dace.int64)
    sdfg.add_array('count', [1], dace.int64)
    sdfg.add_transient('owner', [capacity], dace.int64, storage=dace.StorageType.CPU_Heap)

    if on_device:
        sdfg.add_transient('ip_dev', [N], dace.int64, storage=dace.StorageType.GPU_Global)
        copy = sdfg.add_state('stage', is_start_block=True)
        copy.add_edge(copy.add_read('ip'), None, copy.add_write('ip_dev'), None, dace.Memlet('ip[0:N]'))
        state = sdfg.add_state_after(copy, 'check')
        source = 'ip_dev'
    else:
        state = sdfg.add_state('check', is_start_block=True)
        source = 'ip'

    node = ScatterConflictCheck('check')
    node.implementation = implementation
    node.schedule = dace.ScheduleType.Sequential
    state.add_node(node)
    node.add_out_connector('_owner_out')  # optional connector: the caller-sized tag array
    state.add_edge(state.add_read(source), None, node, '_idx_in', dace.Memlet(f'{source}[0:N]'))
    state.add_edge(node, '_count_out', state.add_write('count'), None, dace.Memlet('count[0]'))
    state.add_edge(node, '_owner_out', state.add_write('owner'), None, dace.Memlet(f'owner[0:{capacity}]'))
    sdfg.validate()
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize('duplicate', [False, True])
def test_scatter_conflict_check_cuda_matches_host(duplicate):
    """A permutation flags 0 and a duplicated slot flags 1, the same on both machines."""
    n = 4096
    ip = np.arange(n, dtype=np.int64)[::-1].copy()
    if duplicate:
        ip[n // 3] = ip[2 * n // 3]
    expected = np.int64(duplicate)

    got_gpu = np.zeros(1, dtype=np.int64)
    make_conflict_sdfg('CUDA', n)(ip=ip.copy(), count=got_gpu, N=n)
    got_cpu = np.zeros(1, dtype=np.int64)
    make_conflict_sdfg('CPU', n)(ip=ip.copy(), count=got_cpu, N=n)

    assert got_gpu[0] == expected, f'CUDA answered {got_gpu[0]}, expected {expected}'
    assert got_cpu[0] == expected, f'CPU answered {got_cpu[0]}, expected {expected}'


@pytest.mark.gpu
def test_scatter_conflict_check_cuda_leaves_the_index_on_the_device():
    """The check must not copy the index back: the whole point of the device expansion is that
    only the flag crosses. A host round-trip would show up as a device-to-host copy of ``ip``."""
    sdfg = make_conflict_sdfg('CUDA', 256)
    code = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    assert 'dace::detect_collision_device' in code
    assert 'cudaMemcpyDeviceToHost' not in code.split('__dace_scatter_conflict')[-1].split('}')[0]


DETECT_ALL_POSITIVE_MAIN = """
#include <cstdio>
#include <vector>
#include "dace/cuda/detect.cuh"

int main() {
    const long long n = 1 << 20;
    std::vector<long long> host(n, 7);
    long long *dev = nullptr;
    cudaMalloc(&dev, n * sizeof(long long));

    long long out = -1;
    cudaMemcpy(dev, host.data(), n * sizeof(long long), cudaMemcpyHostToDevice);
    if (dace::detect_all_positive_device(dev, n, &out, 0) != cudaSuccess || out != 1) {
        printf("FAIL all-positive: %lld\\n", out);
        return 1;
    }
    for (long long bad : {0LL, n / 2, n - 1}) {
        host[bad] = -3;
        cudaMemcpy(dev, host.data(), n * sizeof(long long), cudaMemcpyHostToDevice);
        if (dace::detect_all_positive_device(dev, n, &out, 0) != cudaSuccess || out != 0) {
            printf("FAIL non-positive at %lld: %lld\\n", bad, out);
            return 1;
        }
        host[bad] = 7;
    }
    cudaFree(dev);
    printf("OK\\n");
    return 0;
}
"""


@pytest.mark.gpu
def test_detect_all_positive_device(tmp_path):
    """The sign check's device form, exercised directly.

    Its host twin is called from a guard the BreakAntiDependence pass emits, which runs before any
    GPU transform and so has no device path to reach this through. Compiled and run standalone
    rather than left as an untested sibling of the other two primitives."""
    nvcc = shutil.which('nvcc')
    if nvcc is None:
        pytest.skip('nvcc not on PATH')
    src = tmp_path / 'detect_all_positive.cu'
    src.write_text(DETECT_ALL_POSITIVE_MAIN)
    include = pathlib.Path(dace.__file__).parent / 'runtime' / 'include'
    binary = tmp_path / 'detect_all_positive'
    build = subprocess.run([nvcc, '-std=c++17', '-O2', f'-I{include}',
                            str(src), '-o', str(binary)],
                           capture_output=True,
                           text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout.strip() == 'OK', f'{run.stdout}\n{run.stderr}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
