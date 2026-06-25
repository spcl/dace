# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Full TSVC-2.5 extension corpus: canonicalize -> vectorize, value-preserving.

The tsvc_2 sibling is
:mod:`tests.passes.vectorization.tsvc_canonicalize_vectorize_corpus_test`; this
one covers the harder extension kernels in :mod:`tests.corpus.tsvc_2_5`
(symbolic stride / offset / quasi-affine index patterns), kept as a SEPARATE
corpus so the two families are tested and reported independently. For every
kernel the test:

1. **Canonicalizes** the SDFG and **verifies post-canonicalization correctness**
   end-to-end against the numpy oracle (canonicalization alone preserves
   semantics).
2. **Vectorizes** with ONE config drawn round-robin (by kernel index) from the
   legacy ``VectorizeCPU`` knob set (``test_..._legacy``) and ONE from the
   multi-dim tile-op ``VectorizeCPUMultiDim`` set (``test_..._multidim``), then
   re-checks the output against the oracle.

``nan``/``inf`` match as equal; integer index/permutation arrays are read-only
inputs and skipped. Inner constant-tile loops (e.g. ``heat3d``'s tile size 8)
would unroll 512x, so ``unroll_limit`` is capped, matching the canonicalize
sibling :mod:`tests.canonicalize.tsvc_2_5_corpus_test`.

Known vectorize gaps are marked ``xfail`` with the tracking reason -- see
``_LEGACY_XFAIL`` / ``_MULTIDIM_XFAIL``. The ``xfail`` is imperative and fires
before canonicalize/compile, so a kernel whose vectorized codegen aborts cannot
crash the run.
"""
import contextlib
import inspect
import io
import os

# dace lazily ``from mpi4py import MPI`` during ``to_sdfg``. Skip MPI_Init
# (nothing here uses MPI; left to auto-init its abort-on-error handler aborts the
# interpreter when the compile step forks) and steer Open MPI off UCX before that
# import. ``setdefault`` defers to any externally-provided configuration.
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy

_PEEL_LIMIT = 4
_BREAK_ANTI_DEP = True
_UNROLL_LIMIT = 4
_TOL = 1e-9

_CORPUS = tsvc_2_5.collect()

# Round-robin knob sets, identical to the tsvc_2 sibling (valid combinations
# only; see VectorizeCPU / VectorizeCPUMultiDim constructors).
_LEGACY_KNOBS = [
    dict(remainder_strategy="scalar", branch_normalization=True, use_fp_factor=False),
    dict(remainder_strategy="masked", branch_normalization=True, use_fp_factor=False),
    dict(remainder_strategy="scalar", branch_normalization=False, use_fp_factor=True),
]
_MULTIDIM_KNOBS = [
    dict(target_isa="AVX512", remainder_strategy="masked_tail", branch_mode="merge"),
    dict(target_isa="SCALAR", remainder_strategy="scalar_postamble", branch_mode="merge"),
    dict(target_isa="AVX512", remainder_strategy="full_mask", branch_mode="merge"),
    dict(target_isa="SCALAR", remainder_strategy="masked_tail", branch_mode="fp_factor"),
]

# Kernels with a known vectorize gap; each entry is the tracking reason. Filled
# in after the corpus matrix run.
_LEGACY_XFAIL: dict = {
    'tests_corpus_tsvc_2_5_cond_reduce_sum':
    'numerical (flaky): masked-sum diverges nondeterministically (uninit-mask remainder)',
    'tests_corpus_tsvc_2_5_cond_reduce_sym': 'codegen gap: reduction lift inside nested SDFG unsupported',
    'tests_corpus_tsvc_2_5_config_select_branch': 'codegen gap: BranchNormalization leaves a ConditionalBlock',
    'tests_corpus_tsvc_2_5_ext_gather_load': 'pass gap: could not find iedge assignment for a promoted temp',
    'tests_corpus_tsvc_2_5_ext_strided_load_ssym': 'numerical: vectorized output diverges from numpy reference',
    'tests_corpus_tsvc_2_5_masked_store_sym': 'codegen gap: generated code parse/SyntaxError',
    'tests_corpus_tsvc_2_5_neg_stride_rev': 'canon/shape gap: negative shape in data descriptor',
    'tests_corpus_tsvc_2_5_quasi_affine_floor_div_scatter':
    'Group B (loop2map): WCR survives the map body (recurrence over-parallelized)',
    'tests_corpus_tsvc_2_5_quasi_affine_reduce_even':
    'Group B (loop2map): WCR survives the map body (recurrence over-parallelized)',
    'tests_corpus_tsvc_2_5_quasi_affine_reduce_odd':
    'Group B (loop2map): WCR survives the map body (recurrence over-parallelized)',
    'tests_corpus_tsvc_2_5_reduce_inner_carry': 'codegen gap: generated C++ fails to compile',
    'tests_corpus_tsvc_2_5_scan_strided_2': 'numerical: vectorized output diverges from numpy reference',
}

_MULTIDIM_XFAIL: dict = {
    'tests_corpus_tsvc_2_5_cond_reduce_sum': 'numerical: vectorized output diverges from numpy reference',
    'tests_corpus_tsvc_2_5_ext_break_capture': 'codegen gap: generated C++ fails to compile',
    'tests_corpus_tsvc_2_5_ext_break_find_first': 'multidim tile-emit gap: a scalar store survives InsertTileLoadStore',
    'tests_corpus_tsvc_2_5_ext_break_post_body': 'multidim tile-emit gap: a scalar store survives InsertTileLoadStore',
    'tests_corpus_tsvc_2_5_ext_gather_load': 'codegen gap: generated C++ fails to compile',
    'tests_corpus_tsvc_2_5_ext_scatter_store': 'codegen gap: generated C++ fails to compile',
    'tests_corpus_tsvc_2_5_fission_gather_2body': 'multidim tile-emit gap: a scalar store survives InsertTileLoadStore',
    'tests_corpus_tsvc_2_5_fission_scatter_2body': 'codegen gap: generated C++ fails to compile',
    'tests_corpus_tsvc_2_5_fuse_move_ifs': 'multidim tile-emit gap: a scalar store survives InsertTileLoadStore',
    'tests_corpus_tsvc_2_5_masked_store_sym': 'multidim tile-emit gap: a scalar store survives InsertTileLoadStore',
    'tests_corpus_tsvc_2_5_move_if_data_dep_nest':
    'multidim tile-emit gap: a scalar store survives InsertTileLoadStore',
    'tests_corpus_tsvc_2_5_quasi_affine_floor_div_scatter':
    'Group B (loop2map): loose WCR survives the tiled map (recurrence over-parallelized)',
    'tests_corpus_tsvc_2_5_quasi_affine_reduce_even':
    'Group B (loop2map): loose WCR survives the tiled map (recurrence over-parallelized)',
    'tests_corpus_tsvc_2_5_quasi_affine_reduce_odd':
    'Group B (loop2map): loose WCR survives the tiled map (recurrence over-parallelized)',
    'tests_corpus_tsvc_2_5_reduce_inner_carry': 'numerical: vectorized output diverges from numpy reference',
    'tests_corpus_tsvc_2_5_scan_strided_2': 'numerical: vectorized output diverges from numpy reference',
}


def _oracle(program):
    """The numpy oracle for a kernel: ``ref_`` + name with any ``ext_`` dropped."""
    base = program.name.rsplit("tsvc_2_5_", 1)[-1]
    return getattr(tsvc_2_5_numpy, "ref_" + (base[4:] if base.startswith("ext_") else base))


def _allclose(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=_TOL, atol=_TOL, equal_nan=True)


def _reference(program):
    """``(arrays, scalars, ref)``: inputs and the numpy-oracle output arrays."""
    arrays, scalars = tsvc_2_5.make_inputs(program)
    oracle = _oracle(program)
    # The oracle takes args by name: arrays, scalars, and the lowercased symbol
    # values it declares (e.g. ``ssym``, ``k``); ``iv_*`` oracles take the trip
    # count as ``n``.
    pool = {
        **{
            n: a.copy()
            for n, a in arrays.items()
        },
        **scalars,
        **{
            s.lower(): v
            for s, v in tsvc_2_5.SIZES.items()
        },
        "n": tsvc_2_5.SIZES["LEN_1D"],
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    ref = {n: pool[n] for n in arrays}
    return arrays, scalars, ref


def _symbol_values(sdfg) -> dict:
    """Register any unbound free symbols and return their ``SIZES`` bindings."""
    free = {str(s) for s in sdfg.free_symbols}
    for s in free:  # a hoisted config guard (e.g. K) can stay free but unregistered
        if s not in sdfg.symbols:
            sdfg.add_symbol(s, dace.int64)
    return {s: tsvc_2_5.SIZES[s] for s in tsvc_2_5.SIZES if s in free}


def _canonicalized(program):
    """Canonicalize ``program`` (no correctness assertion -- caller checks)."""
    cand = program.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(cand,
                     validate=True,
                     peel_limit=_PEEL_LIMIT,
                     break_anti_dependence=_BREAK_ANTI_DEP,
                     unroll_limit=_UNROLL_LIMIT)
    return cand


def _run_and_check(program, sdfg, arrays, scalars, ref, stage: str):
    """Compile + run ``sdfg`` and assert every float output matches the oracle."""
    symbols = _symbol_values(sdfg)
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        sdfg.compile()(**got, **scalars, **symbols)
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue  # index/permutation arrays are read-only inputs
        assert _allclose(
            ref[name], got[name]), (f"{program.name}/{name}: {stage} diverges from numpy oracle, "
                                    f"max|diff|={np.nanmax(np.abs(np.asarray(ref[name]) - np.asarray(got[name]))):.3e}")


@pytest.mark.parametrize("idx,program", list(enumerate(_CORPUS)), ids=[p.name for p in _CORPUS])
def test_tsvc_2_5_canonicalize(idx, program):
    """Canonicalize -> verify against the numpy oracle. Canonicalization alone is
    value-preserving; this is the first of the three corpus paths (this, then
    ``+legacy`` / ``+multidim`` vectorize)."""
    arrays, scalars, ref = _reference(program)
    sdfg = _canonicalized(program)
    # Distinct sdfg.name so the canon-only build never shares a .dacecache dir with
    # the legacy arm (same base name) under the parallel sweep.
    sdfg.name = f"{sdfg.name}_canononly"
    _run_and_check(program, sdfg, arrays, scalars, ref, "canonicalization")


@pytest.mark.parametrize("idx,program", list(enumerate(_CORPUS)), ids=[p.name for p in _CORPUS])
def test_tsvc_2_5_canonicalize_then_legacy_vectorize(idx, program):
    """Canonicalize -> verify -> legacy VectorizeCPU (round-robin knob) -> verify."""
    if program.name in _LEGACY_XFAIL:
        pytest.xfail(_LEGACY_XFAIL[program.name])
    arrays, scalars, ref = _reference(program)
    sdfg = _canonicalized(program)
    _run_and_check(program, sdfg, arrays, scalars, ref, "canonicalization")
    knobs = _LEGACY_KNOBS[idx % len(_LEGACY_KNOBS)]
    VectorizeCPU(8, fail_on_unvectorizable=False, **knobs).apply_pass(sdfg, {})
    sdfg.validate()
    _run_and_check(program, sdfg, arrays, scalars, ref, "legacy vectorization")


@pytest.mark.parametrize("idx,program", list(enumerate(_CORPUS)), ids=[p.name for p in _CORPUS])
def test_tsvc_2_5_canonicalize_then_multidim_vectorize(idx, program):
    """Canonicalize -> verify -> multidim VectorizeCPUMultiDim (round-robin knob,
    K=2 when the canonicalized body is a 2-D nested map) -> verify."""
    if program.name in _MULTIDIM_XFAIL:
        pytest.xfail(_MULTIDIM_XFAIL[program.name])
    arrays, scalars, ref = _reference(program)
    sdfg = _canonicalized(program)
    # The ``name`` cache policy keys the .dacecache folder on sdfg.name; suffix the
    # multidim variant so it never shares a build directory with its legacy sibling
    # (same kernel name) under a parallel sweep.
    sdfg.name = f"{sdfg.name}_multidim"
    _run_and_check(program, sdfg, arrays, scalars, ref, "canonicalization")
    map_param_counts = [len(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]
    # K=2 only when EVERY inner map is a genuine collapsed 2-D map (mixed-K within
    # one SDFG aborts the tile pipeline), else fall back to K=1.
    if map_param_counts and min(map_param_counts) >= 2:
        vec = VectorizeCPUMultiDim(widths=(8, 8),
                                   target_isa="SCALAR",
                                   remainder_strategy="masked_tail",
                                   branch_mode="merge")
    else:
        vec = VectorizeCPUMultiDim(widths=(8, ), **_MULTIDIM_KNOBS[idx % len(_MULTIDIM_KNOBS)])
    vec.apply_pass(sdfg, {})
    sdfg.validate()
    _run_and_check(program, sdfg, arrays, scalars, ref, "multidim vectorization")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
