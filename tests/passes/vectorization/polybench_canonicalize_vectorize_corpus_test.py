# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""polybench corpus: canonicalize / canonicalize+vectorize, value-preserving.

For every polybench kernel (see :mod:`tests.corpus.polybench`) the corpus builds a
fresh SDFG from the python frontend, runs two pipelines, and checks each end-to-end
against the untransformed baseline run (polybench ships no numpy oracle, so this is
a value-preservation check):

1. ``canon``     -- canonicalize only.
2. ``canon_vec`` -- canonicalize then multi-dim vectorize (round-robin knob).

The canonicalized SDFG + inputs + reference are computed once per kernel and shared
(phase 2 deep-copies the canon base). Phases are parametrized so pytest reports the
``x/N`` per kernel.
(``auto_optimize`` phases are intentionally omitted -- canon+vectorize is the
must-pass path here.)
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import copy

import pytest

from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.corpus.polybench import polybench

_KERNELS = [k.name for k in polybench.collect()]
_PHASES = ("canon", "canon_vec")

# Genuine per-(kernel, phase) gaps, marked xfail(strict) with the tracking reason -- NOT a blanket skip:
# a case that starts passing flips the suite red so the entry is removed. Populated from the full sweep.
# All canon_vec-phase, all in the multidim-vectorizer (canon alone passes every polybench kernel).
_XFAIL: dict = {
    ("adi", "canon_vec"): "multidim-vectorize: output diverges from baseline",
    ("deriche", "canon_vec"): "multidim-vectorize: CompilationError post-vectorize",
    ("durbin", "canon_vec"): "multidim-vectorize: KeyError __t0_split_0 (tile split)",
    ("gramschmidt", "canon_vec"): "multidim-vectorize: InvalidSDFG (state pointer) post-vectorize",
    ("lu", "canon_vec"): "multidim-vectorize: InvalidSDFG (state pointer) post-vectorize",
}


def _cases():
    out = []
    for name in _KERNELS:
        for phase in _PHASES:
            marks = (pytest.mark.xfail(reason=_XFAIL[(name, phase)], strict=True), ) if (name, phase) in _XFAIL else ()
            out.append(pytest.param(name, phase, id=f"{name}-{phase}", marks=marks))
    return out


# Round-robin multidim knob set (one config per kernel by index), mirroring the
# TSVC / npbench corpus tests.
_MULTIDIM_KNOBS = [
    dict(target_isa="AVX512", remainder_strategy="masked_tail", branch_mode="merge"),
    dict(target_isa="SCALAR", remainder_strategy="scalar_postamble", branch_mode="merge"),
    dict(target_isa="AVX512", remainder_strategy="full_mask", branch_mode="merge"),
    dict(target_isa="SCALAR", remainder_strategy="masked_tail", branch_mode="fp_factor"),
]

_BASE: dict = {}


def _multidim_pass(name):
    knobs = _MULTIDIM_KNOBS[_KERNELS.index(name) % len(_MULTIDIM_KNOBS)]
    return VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), **knobs))


def _base(name):
    """Memoized ``(canon_sdfg, call_arrays, psize, reference)`` for one kernel."""
    if name not in _BASE:
        kernel = polybench.collect(name=name)[0]
        call_arrays, psize = polybench.make_inputs(kernel)
        ref = polybench.reference(kernel, call_arrays, psize)
        sdfg = polybench.fresh_sdfg(kernel)
        canonicalize(sdfg, validate=True)
        _BASE[name] = (sdfg, call_arrays, psize, ref)
    return _BASE[name]


@pytest.mark.parametrize("name,phase", _cases())
def test_polybench_corpus(name, phase):
    canon, call_arrays, psize, ref = _base(name)
    sdfg = copy.deepcopy(canon)
    if phase == "canon_vec":
        _multidim_pass(name).apply_pass(sdfg, {})
    sdfg.validate()
    got = polybench.run(sdfg, call_arrays, psize)
    assert polybench.outputs_match(ref, got), f"{name}/{phase}: output diverges from baseline"
