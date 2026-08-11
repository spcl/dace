# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Corpus-wide gate: canonicalizing an ALREADY-canonicalized SDFG must be SOUND.

Double canonicalization is not a hypothetical. ``VectorizeCPUMultiDim`` canonicalizes at its
own entry (``ENTRY_CANONICALIZE_KWARGS``), so every caller that already canonicalized -- the
whole ``canonicalize -> vectorize`` corpus -- canonicalizes twice. Two miscompiles of that
shape were found by accident rather than on purpose:

* ``SplitStatements`` replicated an in-place read-modify-write NestedSDFG body per output
  group. Mechanism (1) is a NO-OP on the first canonicalize -- at ``'prep'`` the body is
  still a raw control-flow tree. ``LoopToMap``, ~74 units later, MANUFACTURES the blocking
  NestedSDFG that the *second* run's ``'prep'`` then mis-handles.
* ``LoopToScan`` lifted into the TOP-LEVEL SDFG for a loop living inside a NestedSDFG; a
  same-named pass-through array made the wrong ``arrays`` lookup silently SUCCEED.

Both are instances of the same systematic hazard: the pipeline is ~100 units and its LATE
stages produce shapes its EARLY stages consume, so a unit that mints a name, wraps something
in a NestedSDFG, or reads a shape it also writes is only ever exercised on run 2.

Two properties are gated, over the combined TSVC + polybench + npbench corpora:

``test_second_canonicalize_is_sound``
    Run 2 must not raise, must leave a valid SDFG, must leave every memlet naming data its
    OWN SDFG defines, and -- the assertion the ``SplitStatements`` miscompile needed -- must
    still compute the reference answer. HARD: no tracked-exception list. Numerics are checked
    once, after run 2; run 1 alone is already gated by the sibling corpus suites
    (``canonicalize_numerical_corpus_test`` for polybench/npbench,
    ``tsvc_canonicalize_vectorize_corpus_test`` for TSVC), so a red here that is also red
    there is a run-1 defect, not a run-2 one.

``test_second_canonicalize_is_a_fixed_point``
    Run 2 must change NOTHING: equal content digests (``SDFG.hash_sdfg``, which already
    drops ``guid`` / ``name`` / ``cfg_list_id`` / transformation history) before and after.
    Genuine per-kernel gaps are enumerated in ``_NOT_A_FIXED_POINT`` with the observed first
    divergence, ``xfail(strict=True)`` so a kernel that starts converging flips the suite red
    and its entry is removed -- never a blanket skip.

Each kernel is canonicalized with the SAME knobs its own corpus suite uses, so run 1 lands on
the state an existing green gate already blesses and any red is attributable to run 2.
"""
import os

# Pin a deterministic, single-threaded run before DaCe/OpenMP initialize, so the
# value-preserving assertions don't flake on thread races (matches the sibling numerical
# corpus gate). dace lazily ``from mpi4py import MPI`` during ``to_sdfg``; left to auto-init,
# mpi4py installs MPI's abort-on-error handler and the compile step's fork+exec then aborts
# the interpreter partway through the sweep.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.npbench import npbench as _NB
from tests.corpus.polybench import polybench as _PB
from tests.corpus.tsvc import tsvc as _TSVC
from tests.corpus.tsvc.tsvc_numpy import REFERENCES as _TSVC_REFERENCES

#: Per-suite canonicalize knobs -- the SAME ones that suite's existing corpus test uses, so
#: run 1 reproduces a state an already-green gate blesses.
_KNOBS = {
    'tsvc': dict(validate=True, peel_limit=4, break_anti_dependence=True),
    'poly': dict(validate=True),
    'np': dict(validate=True),
}

#: npbench's fused dataset is performance-sized (e.g. N=400000); clamp to the small preset the
#: sibling corpus gates run at, so a double canonicalize + compile + run stays CI-tractable.
_NP_CAP = 32


def _kernels() -> List[Tuple[str, str]]:
    out = [('tsvc', k.name) for k in _TSVC.collect()]
    out += [('poly', k.name) for k in _PB.collect()]
    out += [('np', c["name"]) for c in _NB.collect()]
    return out


_KERNELS = _kernels()

# Kernels where the SECOND canonicalize still rewrites the graph, i.e. canonicalization has not
# converged. Each entry carries the observed FIRST divergence between the run-1 and run-2
# digests. ``xfail(strict=True)``: a kernel that starts converging fails the suite so its entry
# is deleted rather than rotting. Soundness (``test_second_canonicalize_is_sound``) is gated
# unconditionally for every one of these -- a non-fixed point is a missed-convergence defect,
# NOT a licence to compute the wrong answer.
_NOT_A_FIXED_POINT: Dict[Tuple[str, str], str] = {
    ('tsvc', 's1115_d_single'): '.nodes[1].scope_dict.11 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's1119_d_single'): '.attributes._arrays._scan_in_aa | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's112_d_single'): ".nodes[1].nodes[0].attributes.data | run1:'a' | run2:'a_split_snap'",
    ('tsvc', 's116_d_single'): '.attributes._arrays.a_index_1 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's1161_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.c_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's118_d_single'): '.attributes._arrays.bb_slice_times_a_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's121_d_single'): '.nodes | run1:len=2 | run2:len=3',
    ('tsvc', 's122_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.__sym_LEN_1D_minus_k | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's1221_d_single'):
    '.attributes._arrays._wcr_priv_for_305_scan_seed_fold_tasklet___o | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's123_d_single'): '.nodes | run1:len=3 | run2:len=2',
    ('tsvc', 's124_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.b_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's1244_d_single'): '.nodes[1].scope_dict.5 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's125_d_single'):
    ".nodes[1].nodes[0].edges[4].attributes.data.attributes.volume | run1:'symbol($LEN_2D, nonnegative=True)**3' | run2:'symbol($LEN_2D, nonnegative=True)*ipow(symbol($LEN_2D, nonnegative=True), 2)'",
    ('tsvc', 's126_d_single'): '.nodes | run1:len=3 | run2:len=2',
    ('tsvc', 's1279_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.c_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's131_d_single'): '.nodes | run1:len=2 | run2:len=3',
    ('tsvc', 's13110_d_single'):
    ".edges[0].attributes.data.attributes.assignments.yindex | run1:'(((_argmax2d_idx_for_489) % (LEN_2D)))' | run2:'(py_mod(_argmax2d_idx_for_489, LEN_2D))'",
    ('tsvc', 's141_d_single'):
    ".nodes[1].nodes[0].attributes.sdfg.nodes[1].nodes[0].nodes[1].type | run1:'Tasklet' | run2:'AccessNode'",
    ('tsvc', 's151_d_single'): '.nodes | run1:len=2 | run2:len=3',
    ('tsvc', 's152_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's171_d_single'): '.attributes._arrays._wcr_priv__Add____out_0 | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's172_d_single'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's173_d_single'): '.nodes[0].attributes.executions | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's175_d_single'): ".edges[1].src | run1:'2' | run2:'0'",
    ('tsvc', 's176_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's2101_d_single'): '.attributes._arrays.bb_slice_times_cc_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's2111_d_single'): '.attributes._arrays.aa_index | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's212_d_single'): '.attributes._arrays.a_slice_times_d_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's221_d_single'): '.attributes._arrays.b_slice_plus_a_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's222_d_single'): '.attributes._arrays.b_slice_times_c_slice_0 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's2233_d_single'):
    ".nodes[1].attributes.update_statement.string_data | run1:'_loop_it_1 = (_loop_it_1 + 1)' | run2:'_loop_it_4 = (_loop_it_4 + 1)'",
    ('tsvc', 's2244_d_single'): ".nodes[1].nodes[1].attributes.data | run1:'e' | run2:'a'",
    ('tsvc', 's2275_d_single'): '.attributes._arrays.bb_slice_times_cc_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's231_d_single'): '.attributes._arrays._scan_in_aa | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's233_d_single'):
    ".nodes[1].attributes.update_statement.string_data | run1:'_loop_it_1 = (_loop_it_1 + 1)' | run2:'_loop_it_4 = (_loop_it_4 + 1)'",
    ('tsvc', 's235_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's241_d_single'): '.nodes[1].scope_dict.6 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's242_d_single'): '.nodes[1].scope_dict.4 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's243_d_single'): ".nodes[1].nodes[0].attributes.data | run1:'a' | run2:'a_split_snap'",
    ('tsvc', 's244_d_single'): '.attributes._arrays._wcr_priv__Add____out | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's252_d_single'): ".nodes[1].nodes[0].attributes.data | run1:'t' | run2:'b'",
    ('tsvc', 's253_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.c_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's254_d_single'): ".nodes[1].nodes[1].type | run1:'Tasklet' | run2:'AccessNode'",
    ('tsvc', 's255_d_single'): '.nodes[1].scope_dict.-1 | run1:len=14 | run2:len=12',
    ('tsvc', 's261_d_single'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's271_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's2710_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.c_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's2711_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's2712_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's272_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_c_slice_d_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's273_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.c_slice_plus_a_slice_d_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's274_d_single'): '.attributes._arrays._wcr_priv__Add____out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's275_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.aa_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's276_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's277_d_single'): '.attributes._arrays.c_slice_times_d_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's278_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.a_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's279_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.c_index_0 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's281_d_single'):
    ".nodes[1].edges[12].attributes.data.attributes.data | run1:'b_index_1' | run2:'a_index_1'",
    ('tsvc', 's291_d_single'): ".nodes[1].nodes[1].type | run1:'Tasklet' | run2:'AccessNode'",
    ('tsvc', 's292_d_single'): ".nodes[1].nodes[1].type | run1:'Tasklet' | run2:'AccessNode'",
    ('tsvc', 's293_d_single'): ".nodes[1].nodes[0].attributes.data | run1:'a' | run2:'a0'",
    ('tsvc', 's3110_d_single'):
    ".edges[0].attributes.data.attributes.assignments.yindex | run1:'(((_argmax2d_idx_for_1223) % (LEN_2D)))' | run2:'(py_mod(_argmax2d_idx_for_1223, LEN_2D))'",
    ('tsvc', 's3111_d_single'): '.attributes._arrays.sum_val_plus_a_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's31111_d_single'): '.attributes._arrays.partial_plus_a_slice_1 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's3112_d_single'): '.attributes._arrays.sum | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's312_d_single'): '.attributes._arrays.prod_0 | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's313_d_single'): '.attributes._arrays.a_slice_times_b_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's315_d_single'): '.edges[0].attributes.data.attributes.assignments | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's317_d_single'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's318_d_single'): '.nodes | run1:len=5 | run2:len=4',
    ('tsvc', 's319_d_single'): '.attributes._arrays.sum_val_plus_a_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's322_d_single'):
    ".nodes[1].nodes[0].edges[8].attributes.data.attributes.subset.ranges[0].start | run1:'$_loop_it_0' | run2:'-1 + $_loop_it_0'",
    ('tsvc', 's3251_d_single'): ".nodes[1].nodes[0].edges[2].attributes.data.attributes.data | run1:'e' | run2:'c'",
    ('tsvc', 's331_d_single'):
    '.attributes._arrays._wcr_priv__assign_out__wcr_priv_pred_index___out_to__pred_index_for_1422__out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's332_d_single'):
    '.attributes._arrays._wcr_priv__assign_out__wcr_priv_for_1433_findfirst_scan___out_to__exit_i_buf_0__out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's341_d_single'): '.nodes | run1:len=2 | run2:len=3',
    ('tsvc', 's342_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.a_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's343_d_single'): '.nodes | run1:len=2 | run2:len=3',
    ('tsvc', 's351_d_single'): '.attributes._arrays.alpha_times_b_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's352_d_single'): '.attributes._arrays.a_slice_times_b_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's353_d_single'): ".nodes[1].nodes[0].attributes.data | run1:'c' | run2:'alpha'",
    ('tsvc', 's4112_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.ip_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's4113_d_single'):
    '.nodes[3].nodes[0].attributes.sdfg.attributes.symbols.ip_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's4114_d_single'): '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.k | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's4115_d_single'): '.attributes._arrays._nnr_out__priv_sum_val_1 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's4116_d_single'): '.attributes._arrays._nnr_out__priv_sum_val_1 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's4121_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's421_d_single'): '.nodes | run1:len=2 | run2:len=3',
    ('tsvc', 's422_d_single'):
    ".nodes[1].nodes[0].attributes.data | run1:'flat_2d_array' | run2:'flat_2d_array_split_snap'",
    ('tsvc', 's423_d_single'):
    ".nodes[1].nodes[0].attributes.data | run1:'flat_2d_array' | run2:'flat_2d_array_split_snap'",
    ('tsvc', 's431_d_single'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 's441_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_b_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's442_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_b_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's443_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes._arrays.a_slice_plus_b_slice_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's453_d_single'): '.nodes[1].attributes.executions | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's481_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's482_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 's491_d_single'):
    '.nodes[3].nodes[0].attributes.sdfg.attributes.symbols.a_slice_0 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vag_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.ip_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vas_d_single'):
    '.nodes[3].nodes[0].attributes.sdfg.attributes.symbols.a_slice_0 | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vbor_d_single'):
    ".nodes[1].nodes[0].attributes.sdfg.nodes[0].edges[51].attributes.data.attributes.data | run1:'b' | run2:'a'",
    ('tsvc', 'vdotr_d_single'): '.attributes._arrays.a_slice_times_b_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vif_d_single'):
    '.nodes[1].nodes[0].attributes.sdfg.attributes.symbols.b_index | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vpv_d_single'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('tsvc', 'vpvts_d_single'): '.attributes._arrays.b_slice_times_S | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vpvtv_d_single'): '.attributes._arrays.b_slice_times_c_slice | run1:PRESENT | run2:ABSENT',
    ('tsvc', 'vsumr_d_single'): '.attributes._arrays.s_1_0 | run1:ABSENT | run2:PRESENT',
    ('tsvc', 'vtv_d_single'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('poly', 'correlation'):
    '.attributes._arrays._wcr_priv__assign_out__wcr_priv_comp_mean_out_to_mean__out | run1:ABSENT | run2:PRESENT',
    ('poly', 'covariance'): ".edges[0].src | run1:'2' | run2:'0'",
    ('poly', 'deriche'): '.nodes[1].scope_dict.3 | run1:PRESENT | run2:ABSENT',
    ('poly', 'doitgen'): '.attributes._arrays.loop_body_A_0 | run1:PRESENT | run2:ABSENT',
    ('poly', 'durbin'): '.attributes._arrays.sum_0_0_0_0_0 | run1:ABSENT | run2:PRESENT',
    ('poly', 'fdtd_2d'): '.nodes[1].nodes[0].scope_dict.2 | run1:PRESENT | run2:ABSENT',
    ('poly', 'floyd_warshall'): '.nodes[1].start_block | run1:<int> | run2:<NoneType>',
    ('poly', 'gemm'): '.nodes[1].scope_dict.-1[14] | run1:19 | run2:20',
    ('poly', 'gemver'): '.attributes._arrays._wcr_priv_augassign_47_4___out | run1:PRESENT | run2:ABSENT',
    ('poly', 'gesummv'): '.nodes[1].scope_dict.5 | run1:PRESENT | run2:ABSENT',
    ('poly', 'gramschmidt'):
    '.nodes[1].nodes[1].nodes[0].attributes.sdfg.attributes._arrays._wcr_priv_augassign_59_12___out | run1:PRESENT | run2:ABSENT',
    ('poly', 'heat_3d'): '.nodes[1].nodes[0].scope_dict.10 | run1:len=31 | run2:len=32',
    ('poly', 'jacobi_1d'): '.nodes[1].nodes[0].scope_dict.1 | run1:PRESENT | run2:ABSENT',
    ('poly', 'k2mm'): '.nodes[1].scope_dict.-1[17] | run1:22 | run2:23',
    ('poly', 'lu'):
    '.attributes._arrays._wcr_priv__assign_out__wcr_priv_k_loop1_out_to_A__out | run1:ABSENT | run2:PRESENT',
    ('poly', 'ludcmp'): '.attributes._arrays.w_0_0_0_0_0 | run1:ABSENT | run2:PRESENT',
    ('poly', 'mvt'): '.attributes._arrays._wcr_priv_augassign___out | run1:ABSENT | run2:PRESENT',
    ('poly', 'nussinov'): '.attributes.symbols._skew_t_0 | run1:PRESENT | run2:ABSENT',
    ('poly', 'seidel_2d'): '.attributes._arrays.A_index | run1:PRESENT | run2:ABSENT',
    ('poly', 'symm'): '.attributes._arrays.temp2 | run1:PRESENT | run2:ABSENT',
    ('poly', 'trmm'):
    '.nodes[1].nodes[0].attributes.sdfg.nodes[0].nodes[0].nodes[0].attributes.sdfg.attributes._arrays._wcr_priv__assign_out__wcr_priv_compute_elem_ob_to_tmp__out | run1:ABSENT | run2:PRESENT',
    ('np', 'crc16'): '.nodes | run1:len=5 | run2:len=4',
    ('np', 'contour_integral'): '.nodes[1].scope_dict.0 | run1:PRESENT | run2:ABSENT',
    ('np', 'scattering_self_energies'):
    ".nodes[1].nodes[6].attributes.sdfg.nodes[0].nodes[0].nodes[0].nodes[0].nodes[0].branches[0][1].nodes[0].nodes[0].attributes.symbol_mapping.N3D | run1:'symbol($N3D, dtype=dace.int64, nonnegative=True)' | run2:'symbol($N3D, dtype=dace.int64)'",
    ('np', 'arc_distance'): '.nodes[1].scope_dict.11 | run1:PRESENT | run2:ABSENT',
    ('np', 'azimint_hist'): '.nodes[1].scope_dict.11 | run1:PRESENT | run2:ABSENT',
    ('np', 'azimint_naive'): '.nodes[1].scope_dict.-1[5] | run1:6 | run2:5',
    ('np', 'go_fast'): '.attributes._arrays.tanh_a_slice | run1:PRESENT | run2:ABSENT',
    ('np', 'mandelbrot1'): '.nodes[3].scope_dict.-1 | run1:len=20 | run2:len=22',
    ('np', 'mandelbrot2'): '.nodes[1].nodes[4].scope_dict.14 | run1:PRESENT | run2:ABSENT',
    ('np', 'nbody'): '.attributes._arrays._wcr_priv_augassign_139_4___out | run1:PRESENT | run2:ABSENT',
    ('np', 'spmv'): '.nodes[1].scope_dict.-1[2] | run1:3 | run2:2',
    ('np', 'stockham_fft'): '.attributes._arrays.j_coord | run1:PRESENT | run2:ABSENT',
    ('np', 'cavity_flow'): '.nodes[1].scope_dict.0 | run1:PRESENT | run2:ABSENT',
    ('np', 'channel_flow'):
    '.attributes._arrays.un_slice_un_slice_dt_dx_un_slice_un_slice_minus_vn_slice_dt_dy_un_slice_un_slice_0 | run1:PRESENT | run2:ABSENT',
    ('np', 'vadv'): '.attributes._arrays.bcol | run1:PRESENT | run2:ABSENT',
}


def _cases(known: Dict[Tuple[str, str], str]):
    out = []
    for suite, name in _KERNELS:
        marks = (pytest.mark.xfail(reason=known[(suite, name)], strict=True), ) if (suite, name) in known else ()
        out.append(pytest.param(suite, name, id=f"{suite}-{name}", marks=marks))
    return out


# ---------------------------------------------------------------------------
# Content digest
# ---------------------------------------------------------------------------
#: The keys ``SDFG.hash_sdfg`` removes before hashing. Stripped HERE too, because ``hash_sdfg``
#: deep-copies its argument: without this the divergence report walks a json the digest never
#: saw and can name ``guid`` -- which cannot fail this test -- as a kernel's first divergence,
#: putting a reason on its ``_NOT_A_FIXED_POINT`` entry that points at nothing.
_DIGEST_IGNORES = ('cfg_list_id', 'name', 'hash', 'orig_sdfg', 'transformation_hist', 'instrument', 'guid')

#: Dropped ON TOP of ``_DIGEST_IGNORES``. All three are display/debug metadata that cannot change
#: what the SDFG computes, and all three churn across an idempotent re-run:
#:
#: * ``label`` -- state fusion re-labels the surviving state with a fresh unique suffix
#:   (``single_state_body_0`` -> ``single_state_body_1``), which every Map node label inside it
#:   then echoes. The load-bearing identifiers -- descriptor names (dict KEYS in ``_arrays``),
#:   memlet ``data``, Map ``params`` / ``range``, connector names -- are all kept.
#: * ``source_files`` -- a list of the .py files that touched the SDFG, in touch order.
#: * ``debuginfo`` -- source line/column provenance.
_EXTRA_DROP = ('label', 'source_files', 'debuginfo')


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip(v)
            for k, v in obj.items() if k not in _EXTRA_DROP and k not in _DIGEST_IGNORES and not k.startswith('_meta_')
        }
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _digest(sdfg) -> Tuple[str, dict]:
    """``(sha256, normalized json)``. Reuses ``SDFG.hash_sdfg``'s own normalization."""
    j = _strip(json.loads(json.dumps(sdfg.to_json())))
    return sdfg.hash_sdfg(j), j


def _divergences(a: Any, b: Any, path: str = '') -> List[str]:
    """Where two normalized SDFG JSONs first differ, as ``path | run1 | run2`` strings."""
    if type(a) is not type(b):
        return [f'{path} | run1:<{type(a).__name__}> | run2:<{type(b).__name__}>']
    out = []
    if isinstance(a, dict):
        out += [f'{path}.{k} | run1:PRESENT | run2:ABSENT' for k in a if k not in b]
        out += [f'{path}.{k} | run1:ABSENT | run2:PRESENT' for k in b if k not in a]
        for k in a:
            if k in b:
                out += _divergences(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f'{path} | run1:len={len(a)} | run2:len={len(b)}']
        for i, (x, y) in enumerate(zip(a, b)):
            out += _divergences(x, y, f'{path}[{i}]')
    elif a != b:
        out.append(f'{path} | run1:{a!r:.80} | run2:{b!r:.80}')
    return out


def _misowned_memlet_data(sdfg) -> List[Tuple[str, str, str]]:
    """``[(sdfg.name, state.label, memlet.data)]`` for every memlet naming data its OWN SDFG
    does not define -- the residue a rewrite leaves when it resolves a name against the wrong
    descriptor repository (the ``LoopToScan`` bug). An empty memlet is an ORDERING edge with no
    data, so it is skipped before the lookup."""
    return [(sd.name, state.label, e.data.data) for sd in sdfg.all_sdfgs_recursive() for state in sd.states()
            for e in state.edges()
            if not e.data.is_empty() and e.data.data is not None and e.data.data not in sd.arrays]


# ---------------------------------------------------------------------------
# Per-suite fixtures: fresh SDFG, reference outputs, run + compare
# ---------------------------------------------------------------------------
def _context(suite: str, name: str, tag: str) -> Dict[str, Any]:
    """Fresh SDFG + inputs + reference for one kernel. ``tag`` uniquifies ``sdfg.name`` so the
    ``name`` cache policy does not collide two variants on one ``.dacecache`` directory."""
    if suite == 'tsvc':
        kernel = _TSVC.collect(name=name)[0]
        sdfg = _TSVC.to_sdfg(kernel, tag=tag, simplify=True)
        arrays, call = _TSVC.make_inputs(kernel, seed=1234)
        ref = {n: a.copy() for n, a in arrays.items()}
        _TSVC_REFERENCES[kernel.name](**ref, **call)
        return dict(suite=suite, sdfg=sdfg, arrays=arrays, call=call, ref=ref)
    if suite == 'poly':
        kernel = _PB.collect(name)[0]
        arrays, psize = _PB.make_inputs(kernel, size_index=0, cap=None)
        sdfg = _PB.fresh_sdfg(kernel)
        sdfg.name = f'{sdfg.name}_{tag}'
        return dict(suite=suite, sdfg=sdfg, arrays=arrays, psize=psize, ref=_PB.reference(kernel, arrays, psize))
    c = _NB.collect(name)[0]
    arrays, params = _NB.make_inputs(c, cap=_NP_CAP)
    sdfg = _NB.fresh_sdfg(c)
    sdfg.name = f'{sdfg.name}_{tag}'
    return dict(suite=suite, c=c, sdfg=sdfg, arrays=arrays, params=params, ref=_NB.reference_outputs(c, arrays, params))


def _run_matches(ctx: Dict[str, Any]) -> bool:
    """Compile + run the context's SDFG and compare against its reference."""
    sdfg = ctx['sdfg']
    if ctx['suite'] == 'poly':
        return _PB.outputs_match(ctx['ref'], _PB.run(sdfg, ctx['arrays'], ctx['psize']))
    if ctx['suite'] == 'np':
        return _NB.outputs_match(ctx['ref'], _NB.run_outputs(ctx['c'], sdfg, ctx['arrays'], ctx['params']))
    work = {n: a.copy() for n, a in ctx['arrays'].items()}
    sdfg.compile()(**work, **ctx['call'])
    # ``equal_nan`` (not a raw max|diff|) so a recurrence that legitimately overflows to inf
    # everywhere -- s232's squaring scan -- compares inf == inf. Integer arrays are read-only
    # indices. Tolerance matches the TSVC corpus gate.
    return all(
        np.allclose(np.asarray(a), np.asarray(ctx['ref'][n]), rtol=1e-9, atol=1e-9, equal_nan=True)
        for n, a in work.items() if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating) and a.size)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("suite,name", _cases({}))
def test_second_canonicalize_is_sound(suite, name):
    """A second ``canonicalize`` must not raise, invalidate, misown a name, or change the
    answer. HARD gate -- there is no tracked-exception list, because a run-2 miscompile is
    exactly the defect class this suite exists to make unlandable."""
    ctx = _context(suite, name, 'idem_sound')
    canonicalize(ctx['sdfg'], **_KNOBS[suite])
    ctx['sdfg'].validate()
    canonicalize(ctx['sdfg'], **_KNOBS[suite])
    ctx['sdfg'].validate()
    misowned = _misowned_memlet_data(ctx['sdfg'])
    assert not misowned, f'{suite}:{name}: memlets name data their own SDFG lacks: {misowned[:5]}'
    assert _run_matches(ctx), f'{suite}:{name}: a SECOND canonicalize diverges from the reference'


@pytest.mark.parametrize("suite,name", _cases(_NOT_A_FIXED_POINT))
def test_second_canonicalize_is_a_fixed_point(suite, name):
    """A second ``canonicalize`` with the same knobs must rewrite NOTHING."""
    ctx = _context(suite, name, 'idem_fix')
    canonicalize(ctx['sdfg'], **_KNOBS[suite])
    ctx['sdfg'].validate()
    first, before = _digest(ctx['sdfg'])
    canonicalize(ctx['sdfg'], **_KNOBS[suite])
    ctx['sdfg'].validate()
    second, after = _digest(ctx['sdfg'])
    assert first == second, (f'{suite}:{name}: second canonicalize rewrote the SDFG; first divergences: '
                             f'{_divergences(before, after)[:6]}')
