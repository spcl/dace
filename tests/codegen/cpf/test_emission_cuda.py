# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's CUDA dialect: the host and device units of the ``hip`` language, spelled for nvcc.

Both device languages share one rendering path and differ only in spelling tables and toolkit
headers, so what is asserted here is the CUDA side of that split: the units name the CUDA runtime
and nothing of DaCe's, the host unit keeps the host entry prototype, and -- on a GPU host -- the two
build with nvcc, link, and compute what NumPy computes.
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
from dace.codegen.cpf import CUDA_BUILD_FLAGS, Rendering, render
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.nodes.scan import ScanOp

from tests.codegen.cpf.conftest import (
    assert_matches,
    assert_standalone_units,
    canonical_gpu_sdfg,
    device_scan_sdfg,
    entry_argtypes,
    render_gpu,
)

N = dace.symbol("N")

HostArrays = dict[str, np.ndarray]


@dace.program
def vector_add(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a + b


@dace.program
def fused_polynomial(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = (a[i] * a[i] + 1.0) * (a[i] * a[i] - 1.0)


@dace.program
def conditional_stores(cond: dace.float64[N], src: dace.float64[N], a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        if cond[i] > 0.0:
            a[i] = src[i] * 2.0
        else:
            b[i] = src[i] + 1.0


def vector_add_in_numpy(host: HostArrays) -> HostArrays:
    return {"c": host["a"] + host["b"]}


def fused_polynomial_in_numpy(host: HostArrays) -> HostArrays:
    return {"out": (host["a"] ** 2 + 1.0) * (host["a"] ** 2 - 1.0)}


PROGRAMS = (("vector_add", vector_add), ("fused_polynomial", fused_polynomial))
PROGRAM_IDS = [label for label, _ in PROGRAMS]
NUMPY_REFERENCES: dict[str, Callable[[HostArrays], HostArrays]] = {
    "vector_add": vector_add_in_numpy,
    "fused_polynomial": fused_polynomial_in_numpy,
}


def entry_prototypes(code: str, name: str) -> list[str]:
    return re.findall(r'^extern "C" void %s\([^)]*\)' % re.escape(name), code, re.MULTILINE)


def nvcc_library(rendering: Rendering, directory: pathlib.Path, name: str) -> pathlib.Path:
    """Build the two units with the nvcc on PATH, each with ``-c``, into one shared object; return its path."""
    nvcc = shutil.which("nvcc")
    assert nvcc is not None, "the gpu tests build the CUDA units with nvcc, which is not on PATH"
    library = directory / f"lib{name}.so"
    # Native, as DaCe's own build on a GPU host: without it a toolkit newer than the driver emits
    # PTX that the driver cannot JIT, and the kernel launch fails at run time.
    flags = [*CUDA_BUILD_FLAGS, "-arch=native", "-O2", "-Xcompiler=-fPIC"]
    objects = []
    for suffix, text in ((".cpp", rendering.code), (".cu", rendering.device_code)):
        source = directory / f"{name}{suffix}"
        source.write_text(text)
        objects.append(str(source) + ".o")
        command = [nvcc, *flags, "-c", str(source), "-o", objects[-1]]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{' '.join(command)}\n{result.stderr}"
    command = [nvcc, *flags, "-shared", *objects, "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(command)}\n{result.stderr}"
    return library


@pytest.mark.parametrize("label,program", PROGRAMS, ids=PROGRAM_IDS)
def test_an_offloaded_kernel_renders_as_cuda_units_with_the_host_entry_prototype(label: str, program: DaceProgram):
    name = f"cpf_cuda_{label}"
    host_code = render(canonical_gpu_sdfg(program, name), language="c++").code

    rendering = render_gpu(program, name, language="cuda")

    assert_standalone_units(rendering, name)
    for unit in (rendering.code, rendering.device_code):
        assert "#include <cuda_runtime.h>" in unit, "the CUDA toolkit headers"
        assert re.findall(r"\bhip[A-Z]\w*", unit) == [], "a HIP runtime name is undeclared in a CUDA unit"
        assert "__dace_" not in unit, "the units must name nothing of the DaCe runtime"
    device = rendering.device_code
    assert re.search(r"^__global__ void __launch_bounds__\(\d+\) %s_\w+\(" % name, device, re.MULTILINE), "no kernel"
    assert re.search(r"= cudaLaunchKernel\(\s*\(void\*\)%s_\w+, dim3\(" % name, device), "no kernel launch"
    assert re.search(r"cpf_gpu_check\(cudaStreamSynchronize\(gpu_streams\[", rendering.code), "no stream synchronize"
    assert entry_prototypes(rendering.code, name) == entry_prototypes(host_code, name) != [], "the c++ entry prototype"


def test_a_nested_body_function_is_not_declared_inline_twice_in_a_cuda_unit():
    """``DACE_DFI`` already spells ``__forceinline__``, which nvcc reads as ``inline``.

    A branch inside a map becomes a nested body function, whose host-side header carries ``static inline``;
    the device spelling is prefixed to that header, so ``inline`` must be left out or nvcc rejects the
    unit with "duplicate specifier in declaration".
    """
    name = "cpf_cuda_conditional_stores"

    rendering = render_gpu(conditional_stores, name, language="cuda")

    assert_standalone_units(rendering, name)
    code = rendering.device_code
    assert re.search(r"__device__ __forceinline__ void loop_body_\w+\(", code), "no nested body function"
    assert not re.search(r"\binline\s+__device__\s+__forceinline__", code), "inline is declared twice"


@pytest.mark.gpu
@pytest.mark.parametrize("label,program", PROGRAMS, ids=PROGRAM_IDS)
def test_a_cuda_unit_built_by_nvcc_computes_what_numpy_computes(
    tmp_path: pathlib.Path, label: str, program: DaceProgram
):
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = f"cpf_cuda_run_{label}"
    rendering = render_gpu(program, name, language="cuda")
    sdfg = rendering.sdfg
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
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
@pytest.mark.parametrize(
    "op,functor", ((ScanOp.SUM, "cpf_cub_plus()"), (ScanOp.PRODUCT, "cpf_cub_multiplies()")), ids=("sum", "product")
)
def test_a_device_scan_cuda_unit_builds_with_nvcc_through_cpf_functors(
    tmp_path: pathlib.Path, op: ScanOp, functor: str
):
    """CCCL 3 dropped cub's ``Sum``/``Min``/``Max`` structs, so the CUDA unit passes CPF's own functors."""
    name = f"cpf_cuda_scan_{op.name.lower()}"
    rendering = render(device_scan_sdfg(name, op), language="cuda")
    assert functor in rendering.code + rendering.device_code, (
        f"the scan must pass {functor} where the operator macro was"
    )

    nvcc_library(rendering, tmp_path, name)
