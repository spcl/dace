# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end corpus check for the experimental "readable" CPU code generator.

Over the majority of the npbench + polybench corpus present in the repo
(auto-discovered from the in-repo ``tests/npbench`` ``polybench`` and ``misc``
subpackages), this builds each kernel's SDFG post-``simplify`` and asserts, on
BOTH the CPU and the GPU target, that:

1. it COMPILES under the experimental (readable) code generator (a failure raises
   a ``CompilationError``), and
2. its numeric result is IDENTICAL to the ``legacy`` code generator's.

The GPU target lowers its device tasklets through the same CPU code generator
instance, so the equivalence check exercises "both CPU code generators agree" on
CPU and inside ``__global__`` kernels alike. CPU legacy-vs-experimental must be
BIT-EXACT (repo rule: a discrepancy is a real bug, not a tolerance question); the
GPU comparison uses a tight dtype-aware tolerance (reduction/atomic ordering is
not reproducible on the device). See :mod:`tests.codegen.readable.conftest`.

Each run gives its SDFG a name unique to ``(kernel, implementation, target)`` so
the two code generators never share a ``.dacecache`` build (the implementation
flag is not part of the SDFG hash, so a shared name would serve one generator's
compiled binary to the other and mask a real divergence).
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
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)

#: Small square extent bound to every free symbol -- keeps the compile+run fast while
#: still non-trivial (a non-power-of-two catches naive stride assumptions).
SYMBOL_SIZE = 13

#: Per-kernel overrides where the uniform SYMBOL_SIZE is invalid. stockham_fft sizes its arrays as
#: ``R**K`` (radix-R, K-stage FFT), so 13**13 (~2 PiB) is unallocatable; bind small powers (N=2**6=64).
SYMBOL_OVERRIDES = {"stockham_fft": {"R": 2, "K": 6}}

#: npbench corpus subpackages under ``tests/npbench`` this sweep covers: ``polybench`` (npbench's
#: polybench set) and ``misc`` (the non-polybench npbench kernels). The heavier deep-learning /
#: weather-stencil subpackages are left out.
FAMILIES = ("polybench", "misc")

#: Excluded because the generic random inputs are ill-defined for the kernel, not because of any
#: code-generator difference. ``azimint_*`` read back provably-uninitialized reduction bins (the two
#: runs differ on uninitialized bytes). ``spmv`` needs a valid CSR ``rowptr`` (monotonic
#: non-decreasing); random values give a negative row length ``rowptr[i+1]-rowptr[i]``, so the
#: per-row scratch ``new T[<negative>]`` raises ``std::bad_array_new_length`` on legacy and
#: experimental alike -- the generic harness cannot build a structured CSR input.
DENYLIST = {"azimint_naive", "azimint_hist", "spmv"}

#: Additionally not attempted on the GPU: functional kernels whose returned/scratch container is
#: left in an unschedulable storage by a bare ``apply_gpu_transformations()`` (a lowering
#: limitation identical on legacy). A segfault from such a kernel would take down the in-process
#: GPU run (CUDA and ``os.fork`` are incompatible, so GPU cases cannot be isolated), so only the
#: kernels verified to lower cleanly are run on the GPU.
GPU_DENYLIST = DENYLIST | {"contour_integral", "crc16", "go_fast", "nbody"}


def discover(family):
    """All ``(family, kernel_stem)`` in ``tests/npbench/<family>`` (via the package's own
    ``__path__``, so no filesystem paths are hard-coded), minus the denylist."""
    package = importlib.import_module(f"tests.npbench.{family}")
    stems = sorted(info.name[:-len("_test")] for info in pkgutil.iter_modules(package.__path__)
                   if info.name.endswith("_test"))
    return [(family, stem) for stem in stems if stem not in DENYLIST]


#: The majority of the npbench + polybench corpus present in the repo. A kernel that does not
#: build/run on the LEGACY generator (needs an op or transform out of scope here) skips itself at
#: run time, so this list stays broad without hand-curation.
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
    """Deterministic inputs matched to the SDFG's argument descriptors (free symbols are
    passed separately, so they are skipped here)."""
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
    """All comparable arrays: the (possibly in-place-mutated) array arguments plus any
    returned values."""
    outputs = {name: np.asarray(value) for name, value in call_arguments.items() if hasattr(value, "shape")}
    if result is not None:
        for index, value in enumerate(result if isinstance(result, tuple) else (result, )):
            outputs[f"__return{index}"] = np.asarray(value)
    return outputs


def build_and_run(family, name, implementation, target):
    """A zero-argument closure building + running the kernel under ``implementation`` for
    ``target`` and returning ``{name: ndarray}``."""

    def run():
        with use_implementation(implementation):
            sdfg = load_program(family, name).to_sdfg(simplify=True)
            sdfg.name = f"{sdfg.name}_{implementation}_{target}"
            symbols = {symbol: SYMBOL_SIZE for symbol in map(str, sdfg.free_symbols)}
            symbols.update({s: v for s, v in SYMBOL_OVERRIDES.get(name, {}).items() if s in symbols})
            if target == "gpu":
                sdfg.apply_gpu_transformations()
            inputs = make_inputs(sdfg, symbols)
            call_arguments = {n: (v.copy() if hasattr(v, "copy") else v) for n, v in inputs.items()}
            result = sdfg(**call_arguments, **symbols)
            return collect_outputs(result, call_arguments)

    return run


@pytest.mark.parametrize("family,name", KERNELS, ids=[name for _, name in KERNELS])
def test_cpu_compiles_and_matches_legacy(require_experimental, family, name):
    """Experimental CPU codegen compiles the kernel and reproduces the legacy result bit-exactly.

    Each run is fork-isolated (repo rule), so a kernel that the LEGACY generator cannot even
    build/run here -- it needs an operator or transform out of this PR's scope -- surfaces as a
    ``RuntimeError`` from the child and is skipped, keeping the broad sweep green. A failure of
    only the EXPERIMENTAL run, or a numeric divergence, is a real readable-codegen bug and fails."""
    try:
        legacy = run_isolated(build_and_run(family, name, LEGACY, "cpu"))
    except RuntimeError as ex:
        pytest.skip(f"{family}/{name}: not buildable/runnable on legacy CPU: {ex}")
    experimental = run_isolated(build_and_run(family, name, EXPERIMENTAL, "cpu"))
    assert_outputs_equivalent(legacy, experimental, "cpu", label=f"{family}/{name}")


@pytest.mark.gpu
@pytest.mark.parametrize("family,name", GPU_KERNELS, ids=[name for _, name in GPU_KERNELS])
def test_gpu_compiles_and_matches_legacy(require_experimental, require_gpu, family, name):
    """Experimental GPU-target codegen (device tasklets via the CPU generator) compiles and
    matches legacy. CUDA and ``os.fork`` are incompatible, so these run in-process.

    As a secondary guard (a machine may lower a kernel differently), a LEGACY build/run that
    already fails structurally is skipped -- a failure of only the EXPERIMENTAL run, or a numeric
    divergence, is a real readable-codegen bug and fails."""
    try:
        legacy = build_and_run(family, name, LEGACY, "gpu")()
    except (InvalidSDFGError, CompilationError, IndexError) as ex:
        pytest.skip(f"{family}/{name} does not lower to GPU under apply_gpu_transformations: {ex}")
    experimental = build_and_run(family, name, EXPERIMENTAL, "gpu")()
    assert_outputs_equivalent(legacy, experimental, "gpu", label=f"{family}/{name}")


if __name__ == "__main__":
    for corpus, kernel in KERNELS:
        try:
            leg = run_isolated(build_and_run(corpus, kernel, LEGACY, "cpu"))
        except RuntimeError as error:
            print(f"skip {corpus}/{kernel}: legacy not buildable ({error})")
            continue
        exp = run_isolated(build_and_run(corpus, kernel, EXPERIMENTAL, "cpu"))
        assert_outputs_equivalent(leg, exp, "cpu", label=f"{corpus}/{kernel}")
        print(f"ok   {corpus}/{kernel}")
