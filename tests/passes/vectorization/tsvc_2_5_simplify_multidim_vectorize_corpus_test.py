# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC-2.5 corpus: base(``simplify`` + ``LoopToMap`` + ``MapFusion``) then multi-dim tile-op vectorize; numerical
correctness vs numpy oracle.

TSVC-2.5 sibling of the ``*_simplify_multidim_vectorize_corpus_test`` family: the *light* P1 path (no canonicalize).
Fresh SDFG per kernel through :func:`tests.passes.vectorization.helpers.corpus_multidim.base_pipeline`
(``simplify`` -> ``LoopToMap`` -> ``MapFusion``, value-preserving); each config deep-copies the base and is checked
e2e vs oracle (:mod:`tests.corpus.tsvc_2_5.tsvc_2_5_numpy`, ``ref_<kernel>``).

A recurrence that does not parallelize under plain ``LoopToMap`` stays a loop and the tile vectorizer no-ops, so
``base`` == config there. Scope: multi-dim CPU tile path only.

Known gaps: ``xfail`` via ``_XFAIL[(kernel, phase)]``; removed as fixed.
"""
import contextlib
import inspect
import io
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import copy

import numpy as np
import pytest

import dace
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy
from tests.passes.vectorization.helpers.corpus_multidim import PHASES, base_pipeline, make_pass, select_widths

_CORPUS = tsvc_2_5.collect()
_TOL = 1e-9

# Known gaps keyed by (kernel_short_name, phase) -> tracking reason; kernel name = trailing ``<kernel>`` (see
# ``_short``). Removed as fixed.
_XFAIL: dict = {}

_BASE: dict = {}


def _short(program) -> str:
    """Trailing ``<kernel>`` name (``program.name`` doubles the module segment)."""
    return program.name.rsplit("tsvc_2_5_", 1)[-1]


def _oracle(program):
    """The numpy oracle: ``ref_`` + name with any ``ext_`` prefix dropped."""
    base = _short(program)
    return getattr(tsvc_2_5_numpy, "ref_" + (base[4:] if base.startswith("ext_") else base))


def _reference(program):
    """``(arrays, scalars, ref)``: inputs and the numpy-oracle output arrays."""
    arrays, scalars = tsvc_2_5.make_inputs(program)
    oracle = _oracle(program)
    pool = {
        **{n: a.copy() for n, a in arrays.items()},
        **scalars,
        **{s.lower(): v for s, v in tsvc_2_5.SIZES.items()},
        "n": tsvc_2_5.SIZES["LEN_1D"],
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    ref = {n: pool[n] for n in arrays}
    return arrays, scalars, ref


def _symbol_values(sdfg) -> dict:
    """Register any unbound free symbols and return their ``SIZES`` bindings."""
    free = {str(s) for s in sdfg.free_symbols}
    for s in free:
        if s not in sdfg.symbols:
            sdfg.add_symbol(s, dace.int64)
    return {s: tsvc_2_5.SIZES[s] for s in tsvc_2_5.SIZES if s in free}


def _base(program):
    """Memoized ``(base_sdfg, arrays, scalars, ref, widths)`` for one kernel."""
    key = program.name
    if key not in _BASE:
        sdfg = program.to_sdfg(simplify=False)
        with contextlib.redirect_stdout(io.StringIO()):
            base_pipeline(sdfg)
        arrays, scalars, ref = _reference(program)
        _BASE[key] = (sdfg, arrays, scalars, ref, select_widths(sdfg))
    return _BASE[key]


def _run_and_check(program, sdfg, arrays, scalars, ref, stage: str):
    """Compile + run ``sdfg`` and assert every float output matches the oracle."""
    symbols = _symbol_values(sdfg)
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        sdfg.compile()(**got, **scalars, **symbols)
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue  # index/permutation arrays are read-only inputs
        assert np.allclose(np.asarray(ref[name]), np.asarray(got[name]), rtol=_TOL, atol=_TOL, equal_nan=True), (
            f"{program.name}/{name}: {stage} diverges from numpy oracle, "
            f"max|diff|={np.nanmax(np.abs(np.asarray(ref[name]) - np.asarray(got[name]))):.3e}")


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("idx,program", list(enumerate(_CORPUS)), ids=[p.name for p in _CORPUS])
def test_tsvc_2_5_corpus(idx, program, phase):
    """base(simplify+loop2map+mapfusion) [+ multidim vectorize] -> verify vs oracle."""
    if (_short(program), phase) in _XFAIL:
        pytest.xfail(_XFAIL[(_short(program), phase)])
    base, arrays, scalars, ref, widths = _base(program)
    sdfg = copy.deepcopy(base)
    # Per-(kernel, phase) name so concurrent phase builds don't collide on shared .dacecache under the parallel sweep.
    sdfg.name = f"{sdfg.name}_{phase}"
    if phase != "base":
        make_pass(widths, phase).apply_pass(sdfg, {})
    sdfg.validate()
    _run_and_check(program, sdfg, arrays, scalars, ref, phase)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
