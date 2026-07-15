# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numerical equivalence of the experimental "readable" CPU generator vs legacy.

For each curated kernel from the npbench, polybench, tsvc and tsvc_2_5 corpora:

1. build a fresh SDFG through the corpus loader,
2. apply the standard pipeline ``simplify`` + ``LoopToMap`` + ``MapFusion``
   (plus ``apply_gpu_transformations`` for the GPU target),
3. compile + run it under ``compiler.cpu.implementation = legacy`` and again
   under ``experimental``, on the SAME inputs,
4. assert the outputs match -- bit-exact on CPU (deterministic legacy result),
   tight dtype-aware ``allclose`` on GPU.

Comparison is legacy-vs-experimental (same SDFG, same host compiler, same inputs
-- only the code generator differs), so any CPU discrepancy is a real readable-
codegen bug, per the repo rule.

The whole suite is gated on :func:`experimental_available`: until the generator
lands it skips, so this file is green today. The GPU variant additionally skips
without a CUDA device.

Curated set: ~3-5 representative kernels per corpus to keep the suite fast. To
run the FULL corpora, replace each ``*_KERNELS`` list with the loader's own
enumeration, e.g. ``[k.name for k in polybench.collect()]``,
``[c["name"] for c in npbench.collect()]``, ``[k.name for k in tsvc.collect()]``,
and ``list(T25_PROGRAMS)`` (expect long build/compile times).
"""
import copy

import pytest

import dace
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc_2_5 import tsvc_2_5

# Curated, fast-to-build kernels (small elementwise / linear-algebra / stencil).
POLYBENCH_KERNELS = ["atax", "bicg", "mvt", "gesummv", "jacobi_1d"]
NPBENCH_KERNELS = ["go_fast", "azimint_hist", "arc_distance", "hdiff"]
TSVC_KERNELS = ["s000_d_single", "s1351_d_single", "s127_d_single", "vpv_d_single", "vpvtv_d_single"]
TSVC_2_5_KERNELS = ["ext_strided_load_2", "ext_floordiv_offset", "ext_war_unit"]


def tsvc_2_5_short_name(program):
    """The bare kernel name of a tsvc_2_5 ``@dace.program`` (module prefix stripped)."""
    return program.name.rsplit("tsvc_2_5_", 1)[-1]


#: bare-name -> tsvc_2_5 program object (its ``name`` carries the full module path).
TSVC_2_5_PROGRAMS = {tsvc_2_5_short_name(program): program for program in tsvc_2_5.collect()}

CASES = ([("poly", name) for name in POLYBENCH_KERNELS] + [("np", name) for name in NPBENCH_KERNELS] +
         [("tsvc", name) for name in TSVC_KERNELS] + [("tsvc25", name) for name in TSVC_2_5_KERNELS])


def apply_pipeline(sdfg, target):
    """Standard optimization pipeline: ``simplify`` + ``LoopToMap`` + ``MapFusion`` +
    length-1-array -> scalar (transient single-element buffers, incl. (1, 1) MapFusion
    scratch), applied to BOTH code generators so the comparison isolates the codegen."""
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    ConvertLengthOneArraysToScalars(single_element=True, transient_only=True).apply_pass(sdfg, {})
    sdfg.simplify()
    if target == "gpu":
        sdfg.apply_gpu_transformations()


def make_base(corpus, name):
    """Generate the input arrays + call kwargs ONCE per kernel, so both codegen
    variants run on IDENTICAL inputs. Crucial: some kernels have ``np.empty``
    scratch outputs that are not fully overwritten; a separate make_inputs per
    variant would leave them with different garbage and false-fail the compare."""
    if corpus == "poly":
        kernel = polybench.collect(name)[0]
        arrays, extra = polybench.make_inputs(kernel)
        return dict(kernel=kernel, arrays=arrays, extra=extra)
    if corpus == "np":
        descriptor = npbench.collect(name)[0]
        arrays, extra = npbench.make_inputs(descriptor)
        return dict(descriptor=descriptor, arrays=arrays, extra=extra)
    if corpus == "tsvc":
        kernel = tsvc.collect(name=name)[0]
        arrays, extra = tsvc.make_inputs(kernel)
        return dict(kernel=kernel, arrays=arrays, extra=extra)
    program = TSVC_2_5_PROGRAMS[name]
    arrays, extra = tsvc_2_5.make_inputs(program)
    return dict(program=program, arrays=arrays, extra=extra)


def build_and_run(corpus, name, implementation, target, base):
    """Build + transform + compile + run one kernel on a deep copy of the shared
    ``base`` inputs; return its output arrays. Each variant gets a unique SDFG
    name so the two ``.dacecache`` builds never collide."""
    tag = f"{implementation}_{target}"
    arrays = copy.deepcopy(base["arrays"])
    extra = base["extra"]
    if corpus == "poly":
        sdfg = polybench.fresh_sdfg(base["kernel"])
        apply_pipeline(sdfg, target)
        sdfg.name = f"{sdfg.name}_{tag}"
        sdfg.compile()(**arrays, **extra)
        return arrays
    if corpus == "np":
        sdfg = npbench.fresh_sdfg(base["descriptor"])
        apply_pipeline(sdfg, target)
        sdfg.name = f"{sdfg.name}_{tag}"
        return npbench.run_outputs(base["descriptor"], sdfg, arrays, extra)
    if corpus == "tsvc":
        sdfg = tsvc.to_sdfg(base["kernel"], tag, simplify=True)
        apply_pipeline(sdfg, target)
        sdfg.compile()(**arrays, **extra)
        return arrays
    program = base["program"]
    sdfg = program.to_sdfg(simplify=True)
    apply_pipeline(sdfg, target)
    sdfg.name = f"{sdfg.name}_{tag}"
    free = {str(symbol) for symbol in sdfg.free_symbols}
    for symbol in free:  # a hoisted guard symbol can stay free but unregistered
        if symbol not in sdfg.symbols:
            sdfg.add_symbol(symbol, dace.int64)
    symbols = {n: v for n, v in tsvc_2_5.SIZES.items() if n in free}
    sdfg.compile()(**arrays, **extra, **symbols)
    return arrays


def run_case(corpus, name, implementation, target, base):
    """Run one kernel under ``implementation``; fork on CPU, in-process on GPU."""
    work = lambda: build_and_run(corpus, name, implementation, target, base)
    with use_implementation(implementation):
        if target == "gpu":
            # CUDA and os.fork do not mix; run in-process (existing dace GPU tests do too).
            return work()
        return run_isolated(work)


@pytest.mark.parametrize(("corpus", "name"), CASES, ids=[f"{corpus}-{name}" for corpus, name in CASES])
def test_corpus_equivalence(corpus, name, target, require_experimental):
    """Readable-generator output matches legacy for one corpus kernel (CPU + GPU)."""
    base = make_base(corpus, name)  # identical inputs for both variants
    legacy = run_case(corpus, name, LEGACY, target, base)
    experimental = run_case(corpus, name, EXPERIMENTAL, target, base)
    assert_outputs_equivalent(legacy, experimental, target, label=f"{corpus}/{name}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
