# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Native (no-CMake) build mode: end-to-end correctness on real polybench/npbench kernels.

Each kernel is imported verbatim from the shipped ``tests/npbench/polybench`` corpus -- nothing is
re-implemented here. Every kernel is put through the standard optimization pipeline
(``simplify`` + ``LoopToMap`` + ``MapFusion``, plus ``GPUTransformSDFG`` for the GPU variant),
compiled under ``compiler.build_mode = native``, run, and its result compared against NumPy. The
NumPy ground truth is the kernel's own undecorated Python function (``DaceProgram.f``) -- the exact
oracle DaCe's own corpus tests use (e.g. ``gemm_kernel.f(...)``).
"""
import copy
import os

import numpy as np
import pytest

import dace
from dace.config import set_temporary
from dace.frontend.python.parser import DaceProgram
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import GPUTransformSDFG, LoopToMap

from tests.npbench.polybench import gemm_npbench_test, gesummv_test, jacobi_1d_test, mvt_test

#: Problem size used for every kernel: small enough to build + run fast, large enough to exercise
#: the transformed maps/loops meaningfully.
SIZE = 'mini'


def sole_program(module) -> DaceProgram:
    """The single ``@dace.program`` a corpus module defines at module scope (its kernel)."""
    programs = [value for value in vars(module).values() if isinstance(value, DaceProgram)]
    assert len(programs) == 1, f'{module.__name__} exposes {len(programs)} dace programs, expected 1'
    return programs[0]


# Each entry returns ``(program, positional_args, symbol_values)``. The positional args mirror what
# the corpus module's own ``run_*`` driver passes; ``initialize`` / ``sizes`` come from the corpus.
def make_gemm():
    ni, nj, nk = gemm_npbench_test.sizes[SIZE]
    alpha, beta, C, A, B = gemm_npbench_test.initialize(ni, nj, nk)
    return sole_program(gemm_npbench_test), [alpha, beta, C, A, B], dict(NI=ni, NJ=nj, NK=nk)


def make_gesummv():
    n = gesummv_test.sizes[SIZE]
    alpha, beta, A, B, x = gesummv_test.initialize(n)
    return sole_program(gesummv_test), [alpha, beta, A, B, x], dict(N=n)


def make_mvt():
    n = mvt_test.sizes[SIZE]
    x1, x2, y_1, y_2, A = mvt_test.initialize(n)
    return sole_program(mvt_test), [x1, x2, y_1, y_2, A], dict(N=n)


def make_jacobi_1d():
    tsteps, n = jacobi_1d_test.sizes[SIZE]
    A, B = jacobi_1d_test.initialize(n)
    return sole_program(jacobi_1d_test), [tsteps, A, B], dict(N=n)


KERNELS = {'gemm': make_gemm, 'gesummv': make_gesummv, 'mvt': make_mvt, 'jacobi_1d': make_jacobi_1d}


def numpy_reference(program: DaceProgram, args):
    """Ground truth: run the kernel's own Python function on copies, capturing mutations + return."""
    reference_args = copy.deepcopy(args)
    reference_return = program.f(*reference_args)
    return reference_args, reference_return


def apply_pipeline(sdfg: dace.SDFG, to_gpu: bool) -> None:
    """The optimization pipeline under test: simplify + LoopToMap + MapFusion (+ GPUTransform)."""
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    if to_gpu:
        sdfg.apply_transformations(GPUTransformSDFG)
    sdfg.simplify()


def run_native(program: DaceProgram, args, symbols, to_gpu: bool, cache_dir: str):
    sdfg = program.to_sdfg()
    apply_pipeline(sdfg, to_gpu)
    sdfg.build_folder = cache_dir
    with set_temporary('compiler', 'build_mode', value='native'):
        compiled = sdfg.compile()
    call_args = copy.deepcopy(args)
    result = compiled(*call_args, **symbols)
    return call_args, result


def assert_matches(reference_args, reference_return, got_args, got_return) -> None:
    for reference, got in zip(reference_args, got_args):
        if isinstance(reference, np.ndarray):
            assert np.allclose(got, reference, rtol=1e-5, atol=1e-8)
    if reference_return is not None:
        assert np.allclose(got_return, reference_return, rtol=1e-5, atol=1e-8)


def run_case(name: str, to_gpu: bool, cache_dir: str) -> None:
    # Force the pure (map-based) matmul expansion so the corpus's ``@`` operators lower to plain
    # generated code -- portable, deterministic, and exactly the codegen the native builder targets.
    import dace.libraries.blas as blas
    previous = blas.default_implementation
    blas.default_implementation = 'pure'
    try:
        program, args, symbols = KERNELS[name]()
        reference_args, reference_return = numpy_reference(program, args)
        got_args, got_return = run_native(program, args, symbols, to_gpu, cache_dir)
        assert_matches(reference_args, reference_return, got_args, got_return)
    finally:
        blas.default_implementation = previous


@pytest.mark.skipif(os.name != 'posix', reason='native build mode is Linux-only')
@pytest.mark.parametrize('name', list(KERNELS))
def test_native_corpus_cpu(name, tmp_path):
    """simplify + LoopToMap + MapFusion, native CPU build, result == NumPy."""
    run_case(name, to_gpu=False, cache_dir=str(tmp_path / 'cache'))


@pytest.mark.gpu
@pytest.mark.parametrize('name', list(KERNELS))
def test_native_corpus_gpu(name, tmp_path):
    """simplify + LoopToMap + MapFusion + GPUTransformSDFG, native CUDA build, result == NumPy."""
    run_case(name, to_gpu=True, cache_dir=str(tmp_path / 'cache'))
