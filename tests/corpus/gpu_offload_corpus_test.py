# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``OffloadToAccelerator`` over the whole corpus: polybench, npbench, tsvc and tsvc_2_5.

``tests/passes/offload_to_accelerator_test.py`` and ``offload_taskloop_test.py`` pin the pass's rules
on graphs built to show one rule each. This file asks the other question -- whether the rules hold on
281 kernels nobody wrote them for -- and it is the only place that asks it: the ``corpus`` workflow
runs on CPU, so until now no job put a corpus kernel through the GPU path at all.

Two levels, because they cost three orders of magnitude apart:

The pipeline itself is the measurement driver's ``offloaded_sdfg``, imported rather than restated:
one definition of "the canon-GPU path" keeps a test and the number measured beside it about the same
thing.

* the offload itself -- canonicalize, offload, finalize, validate, emit. No compiler, no device, a
  few seconds a kernel, and it already catches the whole placement family: a scalar the pass claims
  for the device while host code still reads it fails ``validate`` with the container named.
* the run -- compile the emitted program and compare against the corpus reference. Needs a GPU and
  minutes per kernel, so it is marked ``gpu`` and lives behind that filter.

The A/B of ``optimizer.gpu_taskloop_heuristics`` is NOT here. A ratio is a measurement, not an
assertion, and it belongs in ``tests/corpus/measure_gpu_arms.py``, which reports it.
"""
import pytest

from tests.corpus import corpus_suite as suite
from tests.corpus.measure_gpu_arms import offloaded_sdfg

#: ``S``, not ``paper``: these assertions are about the graph the pass produces, and the pass does not
#: read the extents. The perf shapes belong to the measurement driver.
PRESET = 'S'
#: The arm this suite asserts about: the canonicalize-then-offload pipeline with the taskloop knob
#: at its shipped default, named rather than inherited -- a test that reads the ambient config
#: asserts about whichever configuration the box happens to carry.
ARM = 'canon'


@pytest.mark.parametrize('kind,name', suite.kernels())
def test_the_offloaded_kernel_validates_and_emits(kind: str, name: str):
    """The pass leaves a graph that validates and emits for the GPU.

    ``validate`` is the load-bearing half. It is what reports a descriptor the pass moved to the
    device while host code still reads it -- the failure the BLAS-alpha family showed as
    ``Data container "alpha_gpu" is stored as StorageType.GPU_Global but accessed on host`` -- and it
    names the container, so a regression here says which one without a debugger.
    """
    sdfg, _ = offloaded_sdfg(suite.make(kind, name, preset=PRESET), ARM, 'corpus_offload')
    sdfg.validate()
    assert sdfg.generate_code(), f'{kind}/{name} offloaded to nothing'


@pytest.mark.gpu
@pytest.mark.parametrize('kind,name', suite.kernels())
def test_the_offloaded_kernel_matches_the_reference(kind: str, name: str):
    """Compiled and run on the device, the offloaded kernel computes what the corpus oracle does."""
    ctx = suite.make(kind, name, preset=PRESET)
    sdfg, _ = offloaded_sdfg(ctx, ARM, 'corpus_offload')
    assert suite.run_matches(ctx, sdfg), f'{kind}/{name} disagrees with its reference on the GPU'
