# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""polybench corpus: base(simplify+loop2map+mapfusion) then multi-dim tile-op
vectorize, value-preserving.

polybench ships no numpy oracle (the original suite validated against a C dump),
so correctness is *value-preservation*: every phase is compared against the
untransformed baseline SDFG run on identical inputs (see
:mod:`tests.corpus.polybench`). A fresh SDFG is taken through the shared base
pipeline; the base SDFG + inputs + baseline reference are memoized per kernel and
each phase deep-copies the base:

* ``base``          -- base pipeline only (must be value-preserving).
* ``<isa>_<mode>``  -- base then :class:`VectorizeCPUMultiDim` at the config's
  ISA / branch mode (all ``scalar_postamble`` remainder; widths per-kernel).

Known gaps are marked ``xfail`` via ``_XFAIL[(kernel, phase)]`` with a tracking
reason; these are removed as each root cause is fixed.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import copy

import pytest

from tests.corpus.polybench import polybench
from tests.passes.vectorization.helpers.corpus_multidim import PHASES, base_pipeline, make_pass, select_widths

_KERNELS = [k.name for k in polybench.collect()]

# Known gaps keyed by (kernel, phase) -> tracking reason; removed as each is
# fixed. Populated from the corpus sweep (see project memory).
_XFAIL: dict = {}

_BASE: dict = {}


def _base(name):
    """Memoized ``(base_sdfg, call_arrays, psize, reference, widths)`` for one kernel."""
    if name not in _BASE:
        kernel = polybench.collect(name=name)[0]
        call_arrays, psize = polybench.make_inputs(kernel)
        ref = polybench.reference(kernel, call_arrays, psize)
        sdfg = polybench.fresh_sdfg(kernel, simplify=False)
        base_pipeline(sdfg)
        _BASE[name] = (sdfg, call_arrays, psize, ref, select_widths(sdfg))
    return _BASE[name]


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("phase", PHASES)
def test_polybench_corpus(name, phase):
    if (name, phase) in _XFAIL:
        pytest.xfail(_XFAIL[(name, phase)])
    base, call_arrays, psize, ref, widths = _base(name)
    sdfg = copy.deepcopy(base)
    if phase != "base":
        make_pass(widths, phase).apply_pass(sdfg, {})
    sdfg.validate()
    got = polybench.run(sdfg, call_arrays, psize)
    assert polybench.outputs_match(ref, got), f"{name}/{phase}: output diverges from baseline"
