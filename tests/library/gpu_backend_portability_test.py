# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the ``CUDA``-keyed GPU library nodes declare has to follow the backend, not the key's name.

The nodes that lower to device primitives -- ``Scan``, ``IntegerSort``, ``ArgReduce``, ``FindFirst``,
``ScatterConflictCheck``, ``Symmetrize`` -- register under the implementation key ``CUDA`` and emit
the backend-neutral ``gpucub`` / ``gpu*`` aliases, so one expansion serves CUDA and HIP alike. Their
ENVIRONMENT did not: it asked CMake for ``find_package(CUDA)`` and put ``cuda_runtime.h`` in the host
frame, which aborts the configure on a ROCm-only machine ("Specify CUDA_TOOLKIT_ROOT_DIR") and took
every one of them down at once, long before any code was compiled.

None of this needs a GPU: the CMake flags, the header lists and the implementation priority are all
decided in Python, and the one header question that is not (does the tile-op GPU header reach a
CUDA-toolkit-only include under hipcc?) the C preprocessor answers on its own.
"""
import contextlib
import os
import pathlib
import re
import subprocess

import pytest

import dace
import dace.libraries.sort  # noqa: F401  (registers the CUB environments)
from dace import dtypes
from dace.codegen.compiler import get_environment_flags
from dace.library import get_environments_and_dependencies
from dace.libraries.sort.nodes.integer_sort import IntegerSort
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.find_first import FindFirst, INDEX_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.libraries.standard.nodes.symmetrize import Symmetrize
from dace.transformation.passes.canonicalize.finalize import canonicalize_fast_library_priority

#: Every environment on the CUB-backed nodes' dependency chain. Each one reaches the GPU runtime
#: environment through ``dependencies``, which is what carried the CUDA package to a ROCm host.
GPU_ENVIRONMENTS = (
    'dace.libraries.standard.environments.cuda.CUDA',
    'dace.libraries.sort.environments.cub.CUB',
    'dace.libraries.sort.environments.cub.SortScratch',
    'dace.libraries.sort.environments.cub.ScanScratch',
    'dace.libraries.sort.environments.cub.ReduceScratch',
    'dace.libraries.sort.environments.cub.DetectScratch',
)

#: The one shape symbol the lowering fixtures below share.
SIZE_SYMBOL = dace.symbol('N_portability', dtype=dace.int64)

TILE_OPS_GPU_HEADER = pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'tile_ops' / 'cuda.h'


def cmake_flags(env_name: str, backend: str) -> str:
    """The CMake command line one environment contributes, under one GPU backend."""
    with dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend):
        flags, _ = get_environment_flags(get_environments_and_dependencies({env_name}))
    return ' '.join(flags)


def frame_headers(env_name: str, backend: str):
    with dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend):
        envs = get_environments_and_dependencies({env_name})
        out = []
        for env in envs:
            headers = env.headers() if callable(env.headers) else env.headers
            out += headers.get('frame', []) if isinstance(headers, dict) else list(headers)
    return out


@pytest.mark.parametrize('env_name', GPU_ENVIRONMENTS)
def test_no_cuda_package_is_requested_on_a_rocm_host(env_name: str) -> None:
    assert 'CUDA' not in cmake_flags(
        env_name, 'hip'), (f'{env_name} still asks CMake for find_package(CUDA) under the HIP backend; on a ROCm-only '
                           'machine that aborts the configure before a single libnode is compiled.')


@pytest.mark.parametrize('env_name', GPU_ENVIRONMENTS)
def test_the_cuda_package_is_still_requested_on_a_cuda_host(env_name: str) -> None:
    """Negative control: dropping the package unconditionally would pass the test above."""
    assert 'DACE_ENV_PACKAGES="CUDA"' in cmake_flags(env_name, 'cuda')


@pytest.mark.parametrize('env_name', GPU_ENVIRONMENTS)
def test_no_frame_header_names_the_cuda_runtime_on_a_rocm_host(env_name: str) -> None:
    named = [h for h in frame_headers(env_name, 'hip') if h.startswith('cuda') or h.startswith('cub/')]
    assert not named, (f'{env_name} puts {named} in the host translation unit under HIP, and ROCm '
                       'ships neither. dace/cuda/cudacommon.cuh is the one place the two runtimes '
                       'are reconciled; name that instead.')


@pytest.mark.parametrize('backend, vendors, foreign', [
    ('cuda', ('cuBLAS', 'cuSolverDn', 'cuTENSOR'), ('rocBLAS', 'rocSOLVER', 'hipTENSOR')),
    ('hip', ('rocBLAS', 'rocSOLVER', 'hipTENSOR'), ('cuBLAS', 'cuSolverDn', 'cuTENSOR')),
])
def test_canonicalize_picks_the_backend_s_own_vendor_libraries(backend, vendors, foreign) -> None:
    """The canonicalize perf tail forced a pick from a CUDA-only list, so every rocBLAS-lowered
    node fell back to the serial ``pure`` loop on a ROCm host -- the lowering existed and was
    never selected."""
    with dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend):
        priority = canonicalize_fast_library_priority(dtypes.DeviceType.GPU)
    assert set(vendors) <= set(priority), f'{backend}: {sorted(set(vendors) - set(priority))} missing from {priority}'
    assert not set(foreign) & set(priority), f'{backend}: {sorted(set(foreign) & set(priority))} leaked into {priority}'
    # The device-primitive keys serve both backends and must survive the split.
    assert {'GPUAuto', 'CUB', 'CUDA'} <= set(priority)


#: A header only the CUDA toolkit ships. ``dace/cuda/...`` is backend-neutral and must not match.
CUDA_TOOLKIT_HEADER = re.compile(r'\bcuda_\w+\.hp?p?\b|\bcub/cub\.cuh\b')


def preprocess_gpu_header(device_compiler_macro: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    """Preprocess the tile-op GPU header with only the HIP fp16/fp8 headers stubbed out.

    Every CUDA-toolkit header is deliberately left unstubbed, and ``-H`` names each header the
    preprocessor opens, so :func:`cuda_headers_reached` answers the question on a host that has the
    toolkit as well as on one that does not -- see there.
    """
    for stub in ('hip/hip_fp16.h', 'hip/hip_fp8.h'):
        path = tmp_path / stub
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#pragma once\n')
    cxx = os.environ.get('CXX', 'g++')
    cmd = [
        cxx, '-E', '-H', '-x', 'c++', '-std=c++17', f'-D{device_compiler_macro}', '-I',
        str(tmp_path),
        str(TILE_OPS_GPU_HEADER), '-o', os.devnull
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def cuda_headers_reached(result: subprocess.CompletedProcess) -> list:
    """Toolkit headers the preprocess opened, together with any whose lookup it failed.

    A returncode alone cannot decide this. Where the toolkit is installed the include RESOLVES and
    the run succeeds, so a leak reads as a pass; where it is absent the run stops at the first
    missing header and never reports the rest. ``-H`` names every header actually opened, and the
    "No such file" line names the one that was looked for, so both spellings of "the header was
    reached" count and the verdict no longer depends on what the host happens to have installed.
    """
    return sorted({m.group(0) for line in result.stderr.splitlines() for m in [CUDA_TOOLKIT_HEADER.search(line)] if m})


def test_tile_op_gpu_header_reaches_no_cuda_only_include_under_hipcc(tmp_path) -> None:
    result = preprocess_gpu_header('__HIPCC__', tmp_path)
    reached = cuda_headers_reached(result)
    assert result.returncode == 0 and not reached, (
        f'dace/tile_ops/cuda.h reaches {reached or "a CUDA-toolkit header"} when the device compiler '
        f'is hipcc:\n{result.stderr}')


def test_the_preprocessor_probe_would_notice_a_cuda_only_include(tmp_path) -> None:
    """Negative control: under ``__CUDACC__`` the same probe must see the include, or it proves nothing."""
    result = preprocess_gpu_header('__CUDACC__', tmp_path)
    assert 'cuda_fp16.h' in cuda_headers_reached(result), (
        'the probe no longer detects an unstubbed CUDA include, so its HIP twin is vacuous')


#: A CUDA runtime call (``cudaMemcpyAsync``), a toolkit type (``cudaStream_t``), a toolkit header or
#: an NVIDIA-only type. ``dace/cuda/...`` paths and the ``dace::cub::`` scratch namespace are
#: deliberately NOT matched: those are backend-neutral and naming them here would make the check
#: unsatisfiable rather than strict.
CUDA_ONLY_SYMBOL = re.compile(r'\bcuda[A-Z]\w*|\bcuda_\w+\.h|\bcub/cub\.cuh|\b__nv_\w+')

N = 4096


def scan_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('portability_scan')
    sdfg.add_array('x', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('y', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Scan('scan', op=ScanOp.SUM)
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('x'), None, node, '_scan_in', dace.Memlet(f'x[0:{N}]'))
    state.add_edge(node, '_scan_out', state.add_write('y'), None, dace.Memlet(f'y[0:{N}]'))
    return sdfg


def arg_reduce_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('portability_argreduce')
    sdfg.add_array('a', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op='max')
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet(f'a[0:{N}]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    return sdfg


def integer_sort_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('portability_integer_sort')
    sdfg.add_array('k', [N], dace.int32, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('o', [N], dace.int32, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = IntegerSort('isort')
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('k'), None, node, '_keys_in', dace.Memlet(f'k[0:{N}]'))
    state.add_edge(node, '_keys_out', state.add_write('o'), None, dace.Memlet(f'o[0:{N}]'))
    return sdfg


def find_first_sdfg() -> dace.SDFG:
    sym_n = dace.symbol('N')
    sdfg = dace.SDFG('portability_find_first')
    sdfg.add_array('a', [sym_n], dace.float64)
    sdfg.add_array('out', [1], dace.int64)
    sdfg.add_transient('a_dev', [sym_n], dace.float64, storage=dace.StorageType.GPU_Global)
    stage = sdfg.add_state('stage', is_start_block=True)
    stage.add_edge(stage.add_read('a'), None, stage.add_write('a_dev'), None, dace.Memlet('a[0:N]'))
    state = sdfg.add_state_after(stage, 'search')
    node = FindFirst('search', predicate=f'__r_a[{INDEX_NAME}] > 0.5', begin=0, end=sym_n)
    node.implementation = 'CUDA'
    # The CUDA expansion LAUNCHES from the host; a node reading device memory and writing a host
    # scalar has no unambiguous inferred schedule, so say so.
    node.schedule = dace.ScheduleType.Sequential
    state.add_node(node)
    node.add_in_connector('__r_a')
    state.add_edge(state.add_read('a_dev'), None, node, '__r_a', dace.Memlet('a_dev[0:N]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


def scatter_conflict_sdfg() -> dace.SDFG:
    sym_n = dace.symbol('N')
    sdfg = dace.SDFG('portability_scatter_conflict')
    sdfg.add_array('ip', [sym_n], dace.int64)
    sdfg.add_array('count', [1], dace.int64)
    sdfg.add_transient('owner', [N], dace.int64, storage=dace.StorageType.CPU_Heap)
    sdfg.add_transient('ip_dev', [sym_n], dace.int64, storage=dace.StorageType.GPU_Global)
    stage = sdfg.add_state('stage', is_start_block=True)
    stage.add_edge(stage.add_read('ip'), None, stage.add_write('ip_dev'), None, dace.Memlet('ip[0:N]'))
    state = sdfg.add_state_after(stage, 'check')
    node = ScatterConflictCheck('check')
    node.implementation = 'CUDA'
    node.schedule = dace.ScheduleType.Sequential
    state.add_node(node)
    node.add_out_connector('_owner_out')
    state.add_edge(state.add_read('ip_dev'), None, node, '_idx_in', dace.Memlet('ip_dev[0:N]'))
    state.add_edge(node, '_count_out', state.add_write('count'), None, dace.Memlet('count[0]'))
    state.add_edge(node, '_owner_out', state.add_write('owner'), None, dace.Memlet(f'owner[0:{N}]'))
    return sdfg


def symmetrize_sdfg() -> dace.SDFG:
    sym_m = dace.symbol('M')
    sdfg = dace.SDFG('portability_symmetrize')
    sdfg.add_array('X', [sym_m, sym_m], dace.float64)
    state = sdfg.add_state()
    node = Symmetrize('sym', row_lo='0', row_hi='M - 1', col_offset=1, col_hi='M', source_upper=True)
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('X'), None, node, '_in', dace.Memlet('X[0:M, 0:M]'))
    state.add_edge(node, '_out', state.add_write('X'), None, dace.Memlet('X[0:M, 0:M]'))
    sdfg.validate()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    return sdfg


#: The device-primitive nodes whose only GPU key is ``CUDA``. Their emitted text is what decides
#: whether that key is a name or a vendor lock-in.
DEVICE_PRIMITIVE_NODES = {
    'ArgReduce': arg_reduce_sdfg,
    'FindFirst': find_first_sdfg,
    'IntegerSort': integer_sort_sdfg,
    'Scan': scan_sdfg,
    'ScatterConflictCheck': scatter_conflict_sdfg,
    'Symmetrize': symmetrize_sdfg,
}


def generated_code(builder, backend: str) -> str:
    with dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend):
        sdfg = builder()
        sdfg.expand_library_nodes()
        return '\n'.join(obj.clean_code for obj in sdfg.generate_code())


@pytest.mark.parametrize('node_name', sorted(DEVICE_PRIMITIVE_NODES))
def test_a_cuda_keyed_expansion_emits_nothing_cuda_only_under_hip(node_name: str) -> None:
    """``CUDA`` is the key these register under, not a statement about the toolchain: their bodies
    name ``gpucub`` / ``gpu*``, which resolve per backend. Anything the toolkit alone provides --
    a ``cuda*`` call, ``cuda_runtime.h``, ``cub/cub.cuh`` -- fails to compile under hipcc."""
    leaked = sorted(set(CUDA_ONLY_SYMBOL.findall(generated_code(DEVICE_PRIMITIVE_NODES[node_name], 'hip'))))
    assert not leaked, f'{node_name} emits {leaked} under the HIP backend, and ROCm provides none of them'


@pytest.mark.parametrize('node_name', sorted(DEVICE_PRIMITIVE_NODES))
def test_the_cuda_only_check_sees_real_text(node_name: str) -> None:
    """Negative control: the same node under the CUDA backend must trip the very same pattern, or
    the check above is matching nothing and would pass on any code at all."""
    found = sorted(set(CUDA_ONLY_SYMBOL.findall(generated_code(DEVICE_PRIMITIVE_NODES[node_name], 'cuda'))))
    assert found, f'{node_name} names no CUDA symbol even under CUDA, so the HIP assertion is vacuous'


#: A bare ``cub::`` in generated code. ``gpucub::`` (the alias that follows the backend) and
#: ``dace::cub::`` (DaCe's own scratch-pool namespace) are deliberately not matched: only the
#: unqualified vendor namespace is, because under HIP nothing defines it.
BARE_CUB_NAMESPACE = re.compile(r'(?<![\w:])cub::\w+')

#: Everything ``dace.h`` can pull into a GPU translation unit.
RUNTIME_INCLUDE_DIR = pathlib.Path(dace.__file__).parent / 'runtime' / 'include'


def test_no_runtime_header_reaches_the_vendored_nvidia_cub() -> None:
    """The vendored NVIDIA CUB under ``dace/external/cub`` is unbuildable by hipcc -- it needs
    ``cuda.h`` and ``cudaError_t``. A header that reaches it, even through an ``__has_include``
    fallback that only fires when the toolkit's CUB is absent, takes down every HIP build at once,
    because ``dace.h`` includes the GPU headers for ``__HIPCC__`` too. The backend belongs to
    ``gpucub.cuh`` and nowhere else."""
    offenders = sorted(f'{path.relative_to(RUNTIME_INCLUDE_DIR)}:{n}' for path in RUNTIME_INCLUDE_DIR.rglob('*')
                       if path.is_file() and path.suffix in ('.h', '.cuh', '.hpp')
                       for n, line in enumerate(path.read_text().splitlines(), 1) if 'external/cub' in line)
    assert not offenders, f'these headers reach the vendored NVIDIA CUB: {offenders}'


def generated_gpu_code(program, backend: str, mutate=None, implementation: str = None) -> str:
    """Code for ``program`` lowered to the GPU under ``backend``, without compiling it."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend))
        if implementation is not None:
            stack.enter_context(dace.config.set_temporary('compiler', 'cuda', 'implementation', value=implementation))
        sdfg = program.to_sdfg(simplify=True)
        sdfg.apply_gpu_transformations(validate=False, simplify=False)
        if mutate is not None:
            mutate(sdfg)
        return '\n'.join(obj.clean_code for obj in sdfg.generate_code())


@dace.program
def wcr_map_exit(a: dace.float64[SIZE_SYMBOL], s: dace.float64[1]):
    for i in dace.map[0:SIZE_SYMBOL]:
        with dace.tasklet:
            x << a[i]
            o >> s(1, lambda p, q: p + q)[0]
            o = x


def test_the_map_exit_wcr_fold_is_backend_neutral() -> None:
    """A map-exit WCR accumulator folds through a block reduction before its one atomic. That fold
    is emitted by the CPU frame (it is the same code on both GPU backends), and it named the vendor
    namespace directly, so it compiled only where NVIDIA's CUB happened to be reachable."""
    code = generated_gpu_code(wcr_map_exit, 'hip')
    assert 'gpucub::BlockReduce' in code, 'the block-reduce fold did not appear; the test lowers nothing'
    assert not BARE_CUB_NAMESPACE.findall(code), BARE_CUB_NAMESPACE.findall(code)


@dace.program
def persistent_kernel(x: dace.float32[SIZE_SYMBOL], b: dace.float32[SIZE_SYMBOL]):
    for ignore in dace.map[0:1]:  # the persistent grid; 0:1 rather than 0, which lowers to a no-op
        for i in dace.map[0:SIZE_SYMBOL]:
            with dace.tasklet:
                a << x[i]
                o >> b[i]
                o = a * 2.0


def make_outermost_map_persistent(sdfg) -> None:
    for state in sdfg.states():
        for scope in state.nodes():
            if isinstance(scope, dace.sdfg.nodes.EntryNode) and state.entry_node(scope) is None:
                scope.map.schedule = dtypes.ScheduleType.GPU_Persistent


def test_the_grid_barrier_is_named_in_dace_s_own_namespace() -> None:
    """A persistent kernel takes a grid barrier as an argument. The barrier is DaCe's own class --
    it only borrows CUB's ``ThreadLoad`` -- but it was declared by reopening CUB's namespace, which
    an alias cannot do, so it could never follow the backend. It lives in ``dace::`` now."""
    code = generated_gpu_code(persistent_kernel, 'hip', make_outermost_map_persistent, implementation='legacy')
    assert 'dace::GridBarrier' in code, 'no grid barrier was emitted; the test lowers nothing'
    assert not BARE_CUB_NAMESPACE.findall(code), BARE_CUB_NAMESPACE.findall(code)
