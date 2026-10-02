# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Every kernel of the npbench polybench and misc tests must compile under the readable generator and reproduce the
legacy result, on CPU to within 1 ULP and on GPU within a dtype-aware tolerance. Each run gets a name of its own, as the
implementation is not part of the SDFG hash and a shared build folder would serve one generator's binary to the other.
"""
import importlib
import pkgutil

import numpy as np
import pytest

import dace
from dace.codegen.exceptions import CompilationError
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.validation import InvalidSDFGError
from dace.symbolic import evaluate
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, gpu_available,
                                             run_isolated, use_implementation, without_fma_contraction)

SYMBOL_SIZE = 13
# scattering_self nests eight loops around a matrix product, so at 13 it runs for hours; 5 is the smallest size for
# which the generated neigh_idx entries (1..4) index G in bounds
KERNEL_SYMBOL_SIZES = {"scattering_self": 5}
FAMILIES = ("polybench", "misc")

# Inputs drawn at random are ill-defined for these: azimint_* read uninitialized bins and spmv needs a monotonic
# row pointer
DENYLIST = {"azimint_naive", "azimint_hist", "spmv"}

# These do not lower on the GPU under a bare apply_gpu_transformations, which legacy cannot do either
GPU_DENYLIST = DENYLIST | {"contour_integral", "crc16", "go_fast", "nbody"}


def discover(family):
    """The ``(family, kernel)`` pairs of ``tests/npbench/<family>`` that are not denylisted."""
    package = importlib.import_module(f"tests.npbench.{family}")
    stems = sorted(info.name[:-len("_test")] for info in pkgutil.iter_modules(package.__path__)
                   if info.name.endswith("_test"))
    return [(family, stem) for stem in stems if stem not in DENYLIST]


KERNELS = [entry for family in FAMILIES for entry in discover(family)]
GPU_KERNELS = [(family, name) for family, name in KERNELS if name not in GPU_DENYLIST]


def load_program(family, name):
    """The kernel ``@dace.program`` in ``tests/npbench/<family>/<name>_test.py``."""
    module = importlib.import_module(f"tests.npbench.{family}.{name}_test")
    programs = [(attr, value) for attr, value in vars(module).items() if isinstance(value, DaceProgram)]
    for attr, value in programs:
        if attr == "kernel" or attr.endswith("_kernel"):
            return value
    return programs[0][1]


def make_inputs(sdfg, symbols, seed=0):
    """Deterministic inputs for the array and scalar arguments of the SDFG."""
    rng = np.random.default_rng(seed)
    inputs = {}
    for name, desc in sdfg.arglist().items():
        if name in symbols:
            continue
        npdt = np.dtype(desc.dtype.as_numpy_dtype())
        if isinstance(desc, dace.data.Scalar):
            inputs[name] = npdt.type(rng.standard_normal())
            continue
        shape = [int(evaluate(dim, symbols)) for dim in desc.shape]
        if npdt.kind == "f":
            inputs[name] = rng.standard_normal(shape).astype(npdt)
        elif npdt.kind == "c":
            inputs[name] = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(npdt)
        else:
            inputs[name] = rng.integers(1, 5, shape).astype(npdt)
    return inputs


def collect_outputs(result, call_arguments):
    """The array arguments, which may be written in place, and the returned values."""
    outputs = {name: value for name, value in call_arguments.items() if isinstance(value, np.ndarray)}
    if result is not None:
        for index, value in enumerate(result if isinstance(result, tuple) else (result, )):
            outputs[f"__return{index}"] = np.asarray(value)
    return outputs


def build_and_run(family, name, implementation, target):
    """A closure that builds and runs the kernel and returns its outputs."""

    def run():
        with use_implementation(implementation), without_fma_contraction():
            sdfg = load_program(family, name).to_sdfg(simplify=True)
            sdfg.name = f"{sdfg.name}_{implementation}_{target}"
            size = KERNEL_SYMBOL_SIZES.get(name, SYMBOL_SIZE)
            symbols = dict.fromkeys(map(str, sdfg.free_symbols), size)
            if target == "gpu":
                sdfg.apply_gpu_transformations()
            inputs = make_inputs(sdfg, symbols)
            call_arguments = {n: (v.copy() if isinstance(v, np.ndarray) else v) for n, v in inputs.items()}
            result = sdfg(**call_arguments, **symbols)
            return collect_outputs(result, call_arguments)

    return run


@pytest.mark.parametrize("family,name", KERNELS, ids=[name for _, name in KERNELS])
def test_cpu_compiles_and_matches_legacy(family, name):
    """A kernel that legacy cannot build is skipped; any failure of the readable run is a bug."""
    try:
        legacy = run_isolated(build_and_run(family, name, LEGACY, "cpu"))
    except RuntimeError as ex:
        pytest.skip(f"{family}/{name}: not buildable/runnable on legacy CPU: {ex}")
    experimental = run_isolated(build_and_run(family, name, EXPERIMENTAL, "cpu"))
    assert_outputs_equivalent(legacy, experimental, "cpu", label=f"{family}/{name}")


@pytest.mark.gpu
@pytest.mark.parametrize("family,name", GPU_KERNELS, ids=[name for _, name in GPU_KERNELS])
def test_gpu_compiles_and_matches_legacy(require_gpu, family, name):
    """CUDA does not survive a fork, so these run in-process."""
    try:
        legacy = build_and_run(family, name, LEGACY, "gpu")()
    except (InvalidSDFGError, CompilationError, IndexError) as ex:
        pytest.skip(f"{family}/{name} does not lower to GPU under apply_gpu_transformations: {ex}")
    experimental = build_and_run(family, name, EXPERIMENTAL, "gpu")()
    assert_outputs_equivalent(legacy, experimental, "gpu", label=f"{family}/{name}")


def run_or_report_skip(test, *arguments):
    """Runs one case; a case the test itself skips (legacy cannot build it) is reported and the run goes on."""
    try:
        test(*arguments)
    except pytest.skip.Exception as skipped:
        print(f"skipped {arguments[-2:]}: {skipped}")


if __name__ == "__main__":
    for family, name in KERNELS:
        run_or_report_skip(test_cpu_compiles_and_matches_legacy, family, name)
    if gpu_available():
        for family, name in GPU_KERNELS:
            run_or_report_skip(test_gpu_compiles_and_matches_legacy, None, family, name)
