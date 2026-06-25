# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``canonicalize`` end-to-end correctness over the TSVC-2.5 extension corpus.

The tsvc_2 sibling of this test is :mod:`tests.canonicalize.tsvc_corpus_test`;
this one covers the harder extension kernels in :mod:`tests.corpus.tsvc_2_5`
(symbolic stride / offset / quasi-affine index patterns). For every kernel:
build the simplified SDFG, run the production ``canonicalize`` recipe, compile,
and compare against the kernel's numpy oracle in
:mod:`tests.corpus.tsvc_2_5_numpy`. ``nan``/``inf`` match as equal.

Inner constant-tile loops (``heat3d``'s tile size 8) would otherwise unroll
512x, so ``unroll_limit`` is capped to keep the sweep feasible; the rest of the
recipe matches the corpus default (``peel_limit=4``, anti-dependence breaking).
"""
import contextlib
import inspect
import io
import os

# dace lazily ``from mpi4py import MPI`` during ``to_sdfg``; steer Open MPI off
# UCX before that import so MPI_Init does not stall. ``setdefault`` defers to any
# externally-provided configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy

_PEEL_LIMIT = 4
_BREAK_ANTI_DEP = True
_UNROLL_LIMIT = 4
_TOL = 1e-9

_CORPUS = tsvc_2_5.collect()

#: Previously-xfailed carried-dependence kernels, now fixed in the canonicalize
#: passes (kept as a comment so the history is discoverable):
#:   * reduce_inner_carry    -- LICM hoisted the per-row ``s = 0.0`` init out of
#:     the outer loop (loop_invariant_code_motion: region-wide single-writer check).
#:   * scan_strided_sym /
#:     fission_dep_sym_offset -- BreakAntiDependence mis-tagged the read-behind
#:     ``a[i-K]`` offset as a renamable WAR (break_anti_dependence: ``-C`` is a
#:     negative offset under the positive-symbol assumption -> RAW).
#:   * scan_multi_5carry     -- LoopToScan's flat multi-slot rewrite mis-seeded
#:     co-located recurrences (loop_to_scan: refuse flat multi-slot -> LoopFission
#:     splits the slots -> per-slot single-carrier scans lift them correctly).


def _oracle(program):
    """The numpy oracle for a kernel: ``ref_`` + name with any ``ext_`` dropped."""
    base = program.name.rsplit("tsvc_2_5_", 1)[-1]
    return getattr(tsvc_2_5_numpy, "ref_" + (base[4:] if base.startswith("ext_") else base))


def _allclose(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=_TOL, atol=_TOL, equal_nan=True)


@pytest.mark.parametrize("program", _CORPUS, ids=[p.name for p in _CORPUS])
def test_canonicalize_tsvc_2_5_value_preserving(program):
    """Canonicalized output matches the numpy oracle (one test per kernel)."""
    arrays, scalars = tsvc_2_5.make_inputs(program)

    # Reference run: the oracle takes args by name -- arrays, scalars, and the
    # lowercased symbol values it declares (e.g. ``ssym``, ``k``).
    oracle = _oracle(program)
    pool = {
        **{
            n: a.copy()
            for n, a in arrays.items()
        },
        **scalars,
        **{
            s.lower(): v
            for s, v in tsvc_2_5.SIZES.items()
        }, "n": tsvc_2_5.SIZES["LEN_1D"]
    }  # iv_* oracles take the trip count as ``n``
    okwargs = {p: pool[p] for p in inspect.signature(oracle).parameters}
    oracle(**okwargs)
    ref = {n: pool[n] for n in arrays}

    cand = program.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(cand,
                     validate=True,
                     peel_limit=_PEEL_LIMIT,
                     break_anti_dependence=_BREAK_ANTI_DEP,
                     unroll_limit=_UNROLL_LIMIT)
    free = {str(s) for s in cand.free_symbols}
    for s in free:  # a hoisted config guard (e.g. K) can stay free but unregistered
        if s not in cand.symbols:
            cand.add_symbol(s, dace.int64)
    symbols = {s: tsvc_2_5.SIZES[s] for s in tsvc_2_5.SIZES if s in free}
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        cand.compile()(**got, **scalars, **symbols)

    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue  # index/permutation arrays are read-only inputs
        assert _allclose(
            ref[name], got[name]), (f"{program.name}/{name}: canonicalize diverges from numpy oracle, "
                                    f"max|diff|={np.nanmax(np.abs(np.asarray(ref[name]) - np.asarray(got[name]))):.3e}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
