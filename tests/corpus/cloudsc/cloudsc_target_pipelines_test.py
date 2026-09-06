# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration test: the four CloudSC optimization pipelines each stay NUMERICALLY
FAITHFUL to the un-transformed kernel.

Every leg starts from the same oracle -- the un-transformed ``simplify=False`` CloudSC SDFG run
sequentially under the IEEE build (:data:`~tests.corpus.cloudsc.generate_data_for_cloudsc.
IEEE_CPU_ARGS`, ``-O0 -fno-fast-math -ffp-contract=off``) on the physical input set from the dwarf
reference -- and each leg re-drives the SAME inputs and compares every output array to it. The four
legs are the four recipes this branch ships for CloudSC:

* ``canonicalize_cpu`` -- ``canonicalize(target='cpu')``, run multicore on the host.
* ``canonicalize_gpu`` -- ``canonicalize(target='gpu')`` followed by CloudSC's own offload recipe
  (:func:`~tests.corpus.cloudsc.offload_cloudsc_to_gpu.offload_cloudsc_to_gpu`), RUN ON THE DEVICE.
* ``gpu_offload_of_canonical_cpu`` -- the generic device move
  :func:`~dace.transformation.passes.canonicalize.finalize.offload_to_gpu` (whose offload step IS
  ``apply_gpu_transformations``) applied to the CPU-canonicalized graph, then
  ``finalize_for_target(..., 'gpu')``, RUN ON THE DEVICE.
* ``vectorize_canonical_cpu`` -- ``VectorizeCPUMultiDim`` on top of the CPU-canonicalized graph,
  run multicore on the host.

The two device legs deliberately use the two different offload recipes this branch ships, over the
two different canonical forms, so a red device leg separates the offload from the target preset that
fed it. They also differ in how they are CALLED, and the driver reads each descriptor rather than
assuming: CloudSC's own offload mirrors every kernel-side array to a ``gpu_<name>`` transient with
copy-in / copy-out states and is therefore called with ordinary host arrays, while ``offload_to_gpu``
runs ``apply_gpu_storage`` and leaves the non-transients themselves in ``GPU_Global``, so those
arguments must be device buffers.

The CPU-canonicalized graph is the shared baseline of three of the four legs, which is why its own
numeric leg is here rather than only in ``cloudsc_canonicalize_test``: without it a red vectorize or
offload leg cannot be attributed to the vectorizer / the offload rather than to canonicalization.

Tolerances are NOT new. Both host legs and both device legs use the ``parallel`` arm bound of
``cloudsc_canonicalize_test`` (:data:`PARALLEL_TOL`): the graphs run their maps in parallel, so the
reduction order differs from the sequential reference by construction. The device legs additionally
build with :data:`~tests.corpus.cloudsc.pipelines.STRICT_FP_CUDA_ARGS`, so nvcc contracts nothing and
approximates nothing -- the residual is device libm, not reassociation.

Cost: the ``simplify=False`` parse (cached on disk across runs) plus ONE ``canonicalize`` per target,
both module-scoped, then one compile+run per leg. Slow -- run it with a raised ``--timeout``.

Manual run::

    pytest tests/corpus/cloudsc/cloudsc_target_pipelines_test.py -v -s -m "integration and not gpu"
    pytest tests/corpus/cloudsc/cloudsc_target_pipelines_test.py -v -s -m "integration and gpu"
"""
import contextlib
import gc
import os

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.codegen import generate_code
from dace.config import set_temporary
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig
from tests.corpus.cloudsc.generate_data_for_cloudsc import IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs
from tests.corpus.cloudsc.offload_cloudsc_to_gpu import offload_cloudsc_to_gpu
from tests.corpus.cloudsc.pipelines import (STRICT_FP_CUDA_ARGS, build_reference_outputs, generate_cuda_code,
                                            gpu_is_runnable, is_device_scheduled, run_candidate)

#: CloudSC species PARAMETER constants (Fortran NCLV=5, NCLDQL=1..NCLDQV=5), baked in so the
#: species / LU loops become constant-trip. Same set the sibling canonicalize test specializes with;
#: klev / klon / kidia / kfdia stay symbolic.
SPECIES_CONSTANTS = {'nclv': 5, 'ncldql': 1, 'ncldqi': 2, 'ncldqr': 3, 'ncldqs': 4, 'ncldqv': 5}

#: Tolerance for every leg here, taken from the ``parallel`` arm of ``cloudsc_canonicalize_test``
#: (``_ARMS['parallel']``): each leg runs its maps in parallel -- OpenMP on the host, one thread per
#: element on the device -- so the reductions and WCR accumulations fold in a different order than
#: the sequential reference. It bounds reassociation, nothing else; the FP build flags are strict on
#: both sides of every comparison.
PARALLEL_TOL = 1e-10

#: Tile width for the vectorize leg. 8 doubles is the corpus harness's CPU default
#: (``tsvc_canonicalize_vectorize_corpus_test``), and the ISA is the HOST's best runnable one --
#: vectorization enforces arch-native, so a forced non-host ISA would SIGILL at run time.
VECTOR_WIDTH = 8

#: Storage classes that put an array in device memory, so the caller must pass a device buffer for it.
GPU_STORAGES = (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned)


def map_entries(sdfg: dace.SDFG):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def tile_nodes(sdfg: dace.SDFG):
    """Tile library nodes (``TileLoad`` / ``TileBinop`` / ``TileStore`` / ...) the vectorizer left."""
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.LibraryNode) and type(n).__name__.startswith('Tile')
    ]


def omp_parallel_for_count(sdfg: dace.SDFG) -> int:
    """``#pragma omp parallel for`` occurrences in the generated host code. A Map that codegen
    declines to emit a pragma for is a silent serialization, so the Map count alone proves nothing."""
    return sum(obj.code.count('#pragma omp parallel for') for obj in generate_code(sdfg) if obj.language == 'cpp')


def canonicalized(reference_file: str, target: str, out_path: str) -> str:
    """Canonicalize a fresh copy of the reference for ``target`` and persist it."""
    sdfg = dace.SDFG.from_file(reference_file)
    # The loop transforms log every refused loop; keep the test output readable.
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        canonicalize(sdfg, validate=True, validate_all=False, target=target, specialize_constants=SPECIES_CONSTANTS)
    sdfg.validate()
    sdfg.save(out_path, compress=True)
    del sdfg
    gc.collect()
    return out_path


def run_on_host(sdfg: dace.SDFG, inputs, tag: str):
    """Run ``sdfg`` multicore under the IEEE build on a private copy of ``inputs``."""
    return run_candidate(sdfg, inputs, IEEE_CPU_ARGS, sequential=False, tag=tag)


def device_resident(sdfg: dace.SDFG):
    """Non-transient arrays the offload left in ``GPU_Global`` -- the arguments the caller has to
    hand device buffers for. Empty for a mirroring offload, which keeps every argument on the host."""
    return {name for name, desc in sdfg.arrays.items() if not desc.transient and desc.storage in GPU_STORAGES}


def run_on_device(sdfg: dace.SDFG, inputs, tag: str):
    """Run an offloaded ``sdfg`` on the GPU under the same strict FP rules the host reference used,
    and return the outputs back on the host.

    nvcc contracts to FMA and approximates division / sqrt by DEFAULT, which would build the device
    leg under looser FP rules than the reference it is compared against -- :data:`STRICT_FP_CUDA_ARGS`
    turns that off. ``sequential=False`` because forcing sequential schedules would demote every
    kernel back to the host and the run would pass while proving nothing about the GPU.

    Which arguments are device buffers is READ OFF the descriptors (:func:`device_resident`), not
    assumed: the two offload recipes disagree, and guessing wrong is either a crash or -- worse -- a
    silent host run.
    """
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it

    on_device = device_resident(sdfg)
    args = {k: (cupy.asarray(v) if k in on_device and isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
    cuda_args = f'{STRICT_FP_CUDA_ARGS} {dace.Config.get("compiler", "cuda", "args")}'
    with set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        with set_temporary('compiler', 'cuda', 'args', value=cuda_args):
            out = run_candidate(sdfg, args, IEEE_CPU_ARGS, sequential=False, tag=tag)
    return {k: (cupy.asnumpy(v) if isinstance(v, cupy.ndarray) else v) for k, v in out.items()}


def assert_matches(out, reference_out, leg: str) -> None:
    report = compare_outputs(out, reference_out, rtol=PARALLEL_TOL, atol=PARALLEL_TOL)
    worst = max(((ma, mr) for ma, mr, _ in report.values()), default=(0.0, 0.0))
    print(f'{leg}: worst |abs|={worst[0]:.3e} |rel|={worst[1]:.3e} (tol={PARALLEL_TOL:.0e})')
    bad = {name: (ma, mr) for name, (ma, mr, ok) in report.items() if not ok}
    assert not bad, (f'{leg}: outputs diverge from the un-transformed reference '
                     f'(tol={PARALLEL_TOL:.0e}): {bad}')


def require_gpu() -> None:
    """Skip only for a missing toolchain / device, decided by building and running a tiny GPU
    program (:func:`gpu_is_runnable`) rather than by sniffing for nvcc."""
    if not gpu_is_runnable():
        pytest.skip('no usable GPU on this host (nvcc, driver or device missing)')


@pytest.fixture(scope='module')
def reference_file(tmp_path_factory):
    """The un-transformed CloudSC SDFG, built once (the parse is minutes) and persisted."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    del ref
    gc.collect()
    return path


@pytest.fixture(scope='module')
def reference_bundle(reference_file):
    """Oracle: ``(inputs, reference_out)`` from the un-transformed SDFG run sequentially under IEEE.
    Shared by every leg -- it is the mathematical answer, independent of how a candidate is scheduled."""
    ref = dace.SDFG.from_file(reference_file)
    bundle = build_reference_outputs(ref, regime='ieee', seed=0)
    del ref
    gc.collect()
    return bundle


@pytest.fixture(scope='module')
def canonical_cpu_file(reference_file, tmp_path_factory):
    """CloudSC canonicalized for the CPU once; the baseline of three of the four legs."""
    return canonicalized(reference_file, 'cpu', str(tmp_path_factory.mktemp('canon_cpu') / 'cloudsc_cpu.sdfgz'))


@pytest.fixture(scope='module')
def canonical_gpu_file(reference_file, tmp_path_factory):
    """CloudSC canonicalized for the GPU target preset. Still host-scheduled -- the device move is a
    separate step, which is what the ``canonicalize_gpu`` leg runs next."""
    return canonicalized(reference_file, 'gpu', str(tmp_path_factory.mktemp('canon_gpu') / 'cloudsc_gpu.sdfgz'))


@pytest.mark.integration
def test_canonicalize_cpu_is_numerically_correct(reference_bundle, canonical_cpu_file):
    """``canonicalize(target='cpu')`` expressed CloudSC as parallel Maps that reach the backend as
    OpenMP regions, and the multicore result still matches the un-transformed reference."""
    inputs, reference_out = reference_bundle
    sdfg = dace.SDFG.from_file(canonical_cpu_file)

    maps = map_entries(sdfg)
    pragmas = omp_parallel_for_count(sdfg)
    print(f'canonicalize/cpu: maps={len(maps)} omp_parallel_for={pragmas}')
    assert maps, 'canonicalization produced no Maps -- nothing was parallelized'
    assert pragmas > 0, 'no "#pragma omp parallel for" reached the generated code -- silently serial'

    out = run_on_host(sdfg, inputs, 'canon_cpu')
    assert_matches(out, reference_out, 'canonicalize/cpu')


@pytest.mark.integration
def test_vectorize_on_canonical_cpu_is_numerically_correct(reference_bundle, canonical_cpu_file):
    """``VectorizeCPUMultiDim`` on top of the CPU-canonicalized graph emitted tile ops, and the
    vectorized kernel still matches the un-transformed reference."""
    inputs, reference_out = reference_bundle
    sdfg = dace.SDFG.from_file(canonical_cpu_file)

    config = VectorizeConfig(widths=(VECTOR_WIDTH, ), target_isa=detect_host_isa(), validate=True)
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        VectorizeCPUMultiDim(config).apply_pass(sdfg, {})
    sdfg.validate()

    tiles = tile_nodes(sdfg)
    print(f'vectorize/cpu: isa={detect_host_isa()} width={VECTOR_WIDTH} tile_nodes={len(tiles)}')
    assert tiles, 'the vectorizer left no tile ops -- the leg would compare the canonical graph to itself'

    out = run_on_host(sdfg, inputs, 'vectorize_cpu')
    assert_matches(out, reference_out, 'vectorize/cpu')


@pytest.mark.gpu
@pytest.mark.integration
def test_canonicalize_gpu_is_numerically_correct(reference_bundle, canonical_gpu_file):
    """``canonicalize(target='gpu')`` plus CloudSC's own offload recipe produces device kernels whose
    result matches the un-transformed reference."""
    require_gpu()
    inputs, reference_out = reference_bundle
    sdfg = dace.SDFG.from_file(canonical_gpu_file)

    offload_cloudsc_to_gpu(sdfg)
    sdfg.validate()

    assert is_device_scheduled(sdfg), 'nothing was scheduled onto the device -- this is a host run'
    kernels = generate_cuda_code(sdfg)
    print(f'canonicalize/gpu: __global__ kernels={kernels} device_args={sorted(device_resident(sdfg))}')

    out = run_on_device(sdfg, inputs, 'canon_gpu')
    assert_matches(out, reference_out, 'canonicalize/gpu')


@pytest.mark.gpu
@pytest.mark.integration
def test_gpu_offload_of_canonical_cpu_is_numerically_correct(reference_bundle, canonical_cpu_file):
    """``offload_to_gpu`` -- whose device move is ``apply_gpu_transformations`` -- applied to the
    CPU-canonicalized (parallelized) CloudSC produces device kernels whose result matches the
    un-transformed reference."""
    require_gpu()
    inputs, reference_out = reference_bundle
    sdfg = dace.SDFG.from_file(canonical_cpu_file)

    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, 'gpu')
    sdfg.validate()

    assert is_device_scheduled(sdfg), 'nothing was scheduled onto the device -- this is a host run'
    kernels = generate_cuda_code(sdfg)
    print(f'gpu_offload/canonical_cpu: __global__ kernels={kernels} device_args={sorted(device_resident(sdfg))}')

    out = run_on_device(sdfg, inputs, 'offload_canon_cpu')
    assert_matches(out, reference_out, 'gpu_offload/canonical_cpu')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
