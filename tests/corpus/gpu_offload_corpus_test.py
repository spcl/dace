# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``OffloadToAccelerator`` over the whole corpus: polybench, tsvc and tsvc_2_5.

``tests/passes/offload_to_accelerator_test.py`` and ``offload_taskloop_test.py`` pin the pass's rules
on graphs built to show one rule each. This file asks the other question -- whether the rules hold on
corpus kernels nobody wrote them for -- and it is the only place that asks it: the ``corpus`` workflow
runs on CPU, so until now no job put a corpus kernel through the GPU path at all.

Two levels, because they cost three orders of magnitude apart:

* the offload itself -- canonicalize, offload, finalize, validate, emit. No compiler, no device, a
  few seconds a kernel, and it already catches the whole placement family: a scalar the pass claims
  for the device while host code still reads it fails ``validate`` with the container named.
* the run -- compile the emitted program and compare against the corpus reference. Needs a GPU and
  minutes per kernel, so it is marked ``gpu`` and lives behind that filter.
"""

import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import (
    finalize_for_target,
    recompute_fuse_for_gpu,
    select_gpu_device_block_size,
)
from dace.transformation.passes.offloading.taskloop import encloses_device_wide_libnode, is_kernel_refused_libnode
from tests.corpus import corpus_suite as suite


def canon_gpu(sdfg: dace.SDFG) -> None:
    """The canonicalize-then-offload GPU pipeline with the taskloop knob off, named rather than inherited:
    a test that reads the ambient config asserts about whichever configuration the box carries."""
    canonicalize(sdfg, validate=False, validate_all=False, target="gpu")
    with (
        dace.config.set_temporary("optimizer", "gpu_taskloop_heuristics", value=False),
        dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1),
    ):
        recompute_fuse_for_gpu(sdfg)
        sdfg.apply_gpu_transformations()
        select_gpu_device_block_size(sdfg)
    # A map around a library call no kernel can run stays host code whole; one with nothing else stays a CPU program
    finalize_for_target(sdfg, "gpu" if offloaded(sdfg) or not host_calls_only(sdfg) else "cpu", validate=False)


def offloaded(sdfg: dace.SDFG) -> bool:
    return any(
        isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device
        for node, parent in sdfg.all_nodes_recursive()
    )


def host_calls_only(sdfg: dace.SDFG) -> bool:
    """Every top-level map holds a library call no kernel can run, which the offload keeps on the host."""
    top = [(state, node) for state in sdfg.states() for node in state.scope_children()[None]]
    maps = [(state, node) for state, node in top if isinstance(node, dace.nodes.MapEntry)]
    return bool(maps) and all(
        encloses_device_wide_libnode(state, node, state.scope_children(), is_kernel_refused_libnode)
        for state, node in maps
    )


def offloaded_sdfg(ctx: dict) -> dace.SDFG:
    return suite.build(ctx, canon_gpu, "corpus_offload")


@pytest.mark.parametrize("kind,name", suite.kernels())
def test_the_offloaded_kernel_validates_and_emits(kind: str, name: str):
    """The pass leaves a graph that validates and emits for the GPU.

    ``validate`` is the load-bearing half. It is what reports a descriptor the pass moved to the
    device while host code still reads it -- the failure the BLAS-alpha family showed as
    ``Data container "alpha_gpu" is stored as StorageType.GPU_Global but accessed on host`` -- and it
    names the container, so a regression here says which one without a debugger.
    """
    sdfg = offloaded_sdfg(suite.make(kind, name))
    sdfg.validate()
    assert sdfg.generate_code(), f"{kind}/{name} offloaded to nothing"


@pytest.mark.gpu
@pytest.mark.parametrize("kind,name", suite.kernels())
def test_the_offloaded_kernel_matches_the_reference(kind: str, name: str):
    """Compiled and run on the device, the offloaded kernel computes what the corpus oracle does."""
    ctx = suite.make(kind, name)
    sdfg = offloaded_sdfg(ctx)
    assert suite.run_matches(ctx, sdfg), f"{kind}/{name} disagrees with its reference on the GPU"
