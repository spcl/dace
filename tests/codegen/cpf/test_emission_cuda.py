# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's CUDA dialect: the device unit of the ``hip`` language, spelled for nvcc.

Both device languages share one rendering path and differ only in spelling tables and toolkit
headers, so what is asserted here is the CUDA side of that split: the unit names the CUDA runtime
and nothing of DaCe's, keeps the host entry prototype, and -- on a GPU host -- builds with nvcc and
computes what NumPy computes.
"""
import ctypes
import pathlib
import re
import shutil
import subprocess
from typing import Callable

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.codegen.cpf import CUDA_BUILD_FLAGS, cpf, render
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.nodes.scan import ScanOp

from tests.codegen.cpf.conftest import (assert_matches, assert_standalone_device, canonical_gpu_sdfg, device_scan_sdfg,
                                        entry_argtypes, render_gpu)

N = dace.symbol('N')

HostArrays = dict[str, np.ndarray]


@dace.program
def vector_add(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a + b


@dace.program
def fused_polynomial(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = (a[i] * a[i] + 1.0) * (a[i] * a[i] - 1.0)


def vector_add_in_numpy(host: HostArrays) -> HostArrays:
    return {'c': host['a'] + host['b']}


def fused_polynomial_in_numpy(host: HostArrays) -> HostArrays:
    return {'out': (host['a']**2 + 1.0) * (host['a']**2 - 1.0)}


PROGRAMS = (('vector_add', vector_add), ('fused_polynomial', fused_polynomial))
PROGRAM_IDS = [label for label, _ in PROGRAMS]
NUMPY_REFERENCES: dict[str, Callable[[HostArrays], HostArrays]] = {
    'vector_add': vector_add_in_numpy,
    'fused_polynomial': fused_polynomial_in_numpy,
}


def entry_prototypes(code: str, name: str) -> list[str]:
    return re.findall(r'^extern "C" void %s\([^)]*\)' % re.escape(name), code, re.MULTILINE)


def nvcc_library(code: str, directory: pathlib.Path, name: str) -> pathlib.Path:
    """Build ``code`` with the nvcc on PATH into a shared object and return its path."""
    nvcc = shutil.which('nvcc')
    assert nvcc is not None, 'the gpu tests build the CUDA unit with nvcc, which is not on PATH'
    source = directory / f'{name}.cu'
    source.write_text(code)
    library = directory / f'lib{name}.so'
    # Native, as DaCe's own build on a GPU host: without it a toolkit newer than the driver emits
    # PTX that the driver cannot JIT, and the kernel launch fails at run time.
    command = [nvcc, *CUDA_BUILD_FLAGS, '-arch=native', '-O2', '-shared', '-Xcompiler=-fPIC', '-o', str(library)]
    result = subprocess.run(command + [str(source)], capture_output=True, text=True)
    assert result.returncode == 0, f'{" ".join(command)} {source}\n{result.stderr}'
    return library


@pytest.mark.parametrize('label,program', PROGRAMS, ids=PROGRAM_IDS)
def test_an_offloaded_kernel_renders_as_one_cuda_unit_with_the_host_entry_prototype(label: str, program: DaceProgram):
    name = f'cpf_cuda_{label}'
    host_code = render(canonical_gpu_sdfg(program, name), language='c++').code

    _, code = render_gpu(program, name, language='cuda')

    assert_standalone_device(code, name)
    assert '#include <cuda_runtime.h>' in code and '#include <cub/cub.cuh>' in code, 'the CUDA toolkit headers'
    assert re.search(r'^__global__ void __launch_bounds__\(\d+\) %s_\w+\(' % name, code, re.MULTILINE), 'no kernel'
    assert re.search(r'= cudaLaunchKernel\(\s*\(void\*\)%s_\w+, dim3\(' % name, code), 'no kernel launch'
    assert re.search(r'cpf_gpu_check\(cudaStreamSynchronize\(gpu_streams\[', code), 'no stream synchronize'
    assert re.findall(r'\bhip[A-Z]\w*', code) == [], 'a HIP runtime name is undeclared in a CUDA unit'
    assert '__dace_' not in code, 'the unit must name nothing of the DaCe runtime'
    assert code.count('extern "C"') == 1, 'the entry is the one C-linkage function'
    assert entry_prototypes(code, name) == entry_prototypes(host_code, name) != [], 'the c++ entry prototype'


@pytest.mark.gpu
@pytest.mark.parametrize('label,program', PROGRAMS, ids=PROGRAM_IDS)
def test_a_cuda_unit_built_by_nvcc_computes_what_numpy_computes(tmp_path: pathlib.Path, label: str,
                                                                program: DaceProgram):
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.
    name = f'cpf_cuda_run_{label}'
    sdfg, code = render_gpu(program, name, language='cuda')
    entry = ctypes.CDLL(str(nvcc_library(code, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(sdfg)
    length = 1000  # not a multiple of the 128-thread block, so the last block's bounds guard is exercised
    generator = np.random.default_rng(7)
    host = {arg: generator.random(length) for arg, desc in sdfg.arglist().items() if isinstance(desc, dt.Array)}
    expected = NUMPY_REFERENCES[label](host)
    # The entry takes device pointers for its GPU_Global arrays; the caller copies to and from the device.
    device = {arg: cupy.asarray(values) for arg, values in host.items()}
    arguments = [ctypes.c_void_p(device[arg].data.ptr) if arg in device else length for arg in sdfg.arglist()]

    entry(*arguments)

    assert_matches(expected, {arg: cupy.asnumpy(device[arg]) for arg in expected}, name)


@pytest.mark.gpu
@pytest.mark.parametrize('op,functor', ((ScanOp.SUM, 'cpf_cub_plus()'), (ScanOp.PRODUCT, 'cpf_cub_multiplies()')),
                         ids=('sum', 'product'))
def test_a_device_scan_cuda_unit_builds_with_nvcc_through_cpf_functors(tmp_path: pathlib.Path, op: ScanOp,
                                                                       functor: str):
    """CCCL 3 dropped cub's ``Sum``/``Min``/``Max`` structs, so the CUDA unit passes CPF's own functors."""
    name = f'cpf_cuda_scan_{op.name.lower()}'
    code = cpf(device_scan_sdfg(name, op), language='cuda')
    assert_standalone_device(code, name)
    assert functor in code, f'the scan must pass {functor} where the DaCe operator macro was'

    nvcc_library(code, tmp_path, name)
