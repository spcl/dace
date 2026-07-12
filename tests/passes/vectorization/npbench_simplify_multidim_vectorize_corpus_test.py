# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""npbench corpus: base(simplify+loop2map+mapfusion) then multi-dim tile-op vectorize; numerical correctness vs numpy.

Inputs at ``S`` dataset preset (symbols capped for speed); reference computed; fresh SDFG through
:func:`tests.passes.vectorization.helpers.corpus_multidim.base_pipeline`. base SDFG + inputs + reference memoized
per kernel; each phase deep-copies the base:

* ``base``          -- base pipeline only (value-preserving).
* ``<isa>_<mode>``  -- base then :class:`VectorizeCPUMultiDim` at config ISA / branch mode (``scalar_postamble``
  remainder; widths per-kernel).
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import copy

import pytest

from tests.corpus.npbench import npbench
from tests.passes.vectorization.helpers.corpus_multidim import PHASES, base_pipeline, make_pass, select_widths

_CORPUS = {c["name"]: c for c in npbench.collect()}
_KERNELS = sorted(_CORPUS)

_BASE: dict = {}


def _base(name):
    """Memoized ``(base_sdfg, arrays, params, reference, widths)`` for one benchmark."""
    if name not in _BASE:
        c = _CORPUS[name]
        arrays, params = npbench.make_inputs(c)
        ref = npbench.reference_outputs(c, arrays, params)
        sdfg = npbench.fresh_sdfg(c, simplify=False)
        base_pipeline(sdfg)
        _BASE[name] = (sdfg, arrays, params, ref, select_widths(sdfg))
    return _BASE[name]


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("phase", PHASES)
def test_npbench_corpus(name, phase):
    base, arrays, params, ref, widths = _base(name)
    sdfg = copy.deepcopy(base)
    # Per-(kernel, phase) name: concurrent xdist builds must not share .dacecache (race -> spurious CompilationError).
    sdfg.name = f"{sdfg.name}_{phase}"
    if phase != "base":
        make_pass(widths, phase).apply_pass(sdfg, {})
    sdfg.validate()
    got = npbench.run_outputs(_CORPUS[name], sdfg, arrays, params)
    assert npbench.outputs_match(ref, got), f"{name}/{phase}: output diverges from npbench reference"
