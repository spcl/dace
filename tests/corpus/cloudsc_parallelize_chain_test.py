# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration test: the parallelize chain on CloudSC stays numerically
faithful to the un-transformed reference, step by step.

Applies, in order, ``simplify`` -> ``TrivialTaskletElimination`` ->
``WCRToAugAssign`` -> ``ShortLoopUnroll`` -> ``PrivatizeScalars`` (scalar fission)
-> ``SymbolPropagation`` -> ``ConstantPropagation`` -> ``LoopToReduce`` ->
``LoopToMap``. After **each** step it re-runs the candidate on identical physical
inputs and compares every output array to the non-transformed reference (built with
``simplify=False``), so a divergence is pinned to the exact step that introduced it.

The frontend emits ``arr[S] += x`` accumulators as WCR writes. ``WCRToAugAssign``
turns those back into explicit ``a = a + b`` tasklets -- removing the WCR edges and
restoring the accumulation form ``LoopToReduce`` recognizes (``TrivialTaskletElimination``
first clears the copy tasklets). The steps up to ``LoopToReduce`` are value-preserving
and must stay very close to the reference (bit-exact in the ``ieee`` regime).
``LoopToReduce`` and ``LoopToMap`` reassociate accumulations and run them in
parallel, so a small relative error is allowed there.

Two build regimes (see :mod:`tests.corpus.generate_data_for_cloudsc`):

* ``ieee``    -- ``-O0``, no fast-math, no FP contraction, sequential schedules.
* ``release`` -- the configured ``-O3 -ffast-math`` flags, parallel schedules.

This is a slow integration test: it builds the full CloudSC SDFG once
(``simplify=False`` parse is minutes) and compiles it once per step.
"""
import contextlib
import copy
import gc
import os

import pytest

import dace
from dace import symbolic
from dace.sdfg.utils import specialize_symbol
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR, WCRToAugAssign
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.loop_to_scan import LoopToScan
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess
from dace.transformation.passes.scalar_fission import PrivatizeScalars
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from tests.corpus.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                    generate_cloudsc_inputs, make_sequential)

#: (ieee_build, sequential, strict_tol, relaxed_tol) per regime. ``strict_tol``
#: gates the value-preserving steps; ``relaxed_tol`` gates the reassociating
#: reduce/loop-to-map steps where a small parallel-reduction error is expected.
_REGIMES = {
    'ieee': (True, True, 1e-15, 1e-10),
    'release': (False, False, 1e-10, 1e-10),
}

#: Steps that reassociate/parallelize accumulations -> use the relaxed tolerance.
#: ``loop_to_scan`` reassociates the carry; ``loop_to_map`` may reorder fold operations.
_RELAXED_STEPS = {'loop_to_map', 'loop_to_scan', 'loop_to_reduce'}

#: Stages at which the serialize/deserialize roundtrip e2e check fires. ``simplify``
#: is the first heavy SDFG mutation (state fusion + NSDFG inlining) where most
#: roundtrip bugs surface; ``loop_to_map`` is the final transformed shape. Per-stage
#: roundtrip on every label roughly doubles wall-clock for no new bug coverage --
#: these two bracket every transform-induced descriptor / memlet / region mutation.
_ROUNDTRIP_CHECKPOINTS = {'simplify', 'loop_to_map'}

#: Index symbols whose ``<sym> + 1`` bound expression must not survive symbol
#: propagation as an interstate-edge assignment value.
_INDEX_SYMS = ('klev', 'kfdia', 'kidia')

#: Expected end-state after ``loop_to_map`` (pinned; label-independent so it
#: survives loop-numbering churn across commits). The remaining sequential
#: loops are: LU triangular back-substitution on ``zqxn`` (2 loops, mathematically
#: sequential), the vertical flux prefix-sum on ``pfcqlng`` / ``pfsqrf`` (1 loop,
#: genuine scan; a separate optional scan->elementwise+reduce transform addresses
#: it), and the level loop blocked by a ``tendency_loc_cld`` propagation
#: over-approximation (1 loop; the inner write IS ``jk``-indexed but
#: ``propagate_memlets_nested_sdfg`` widens it -- a 2nd-order propagation fix is
#: a follow-up to the ``symbols_defined_at`` enclosing-LoopRegion fix). The ICE
#: ``for_767`` ``zvqx[1]`` slot is now privatized by
#: ``PromoteConstantIndexAccess`` (per-loop atomic check-promote-verify-commit
#: lands the rewrite that the previous batched flow lost), so the species loops
#: parallelize. A change here means loop2map coverage shifted -- review before
#: bumping the numbers.
_EXPECTED_MAPS = 357
_EXPECTED_SEQUENTIAL_LOOPS = 3


def _wcr_edges(sdfg: dace.SDFG):
    """Every WCR edge across the SDFG and all nested SDFGs (recursive)."""
    return [
        e for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for e in st.edges()
        if e.data is not None and e.data.wcr is not None
    ]


def _reduce_nodes(sdfg: dace.SDFG):
    """Every ``Reduce`` library node (recursive)."""
    from dace.libraries.standard import Reduce
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]


def _map_entries(sdfg: dace.SDFG):
    """Every ``MapEntry`` node (recursive)."""
    from dace.sdfg import nodes
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _loop_regions(sdfg: dace.SDFG):
    """Every control-flow ``LoopRegion`` that still carries a loop variable."""
    from dace.sdfg.state import LoopRegion
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _free_symbol_names(sdfg: dace.SDFG):
    """Names of every free symbol across the SDFG and its nested SDFGs."""
    return {str(s) for sd in sdfg.all_sdfgs_recursive() for s in sd.free_symbols}


def _plus_one_index_assignments(sdfg: dace.SDFG):
    """Interstate-edge assignments whose value is ``klev+1`` / ``kfdia+1`` / ``kidia+1``."""
    bad = {symbolic.pystr_to_symbolic(f'{s} + 1') for s in _INDEX_SYMS}
    found = []
    for e in sdfg.all_interstate_edges():
        for lhs, rhs in e.data.assignments.items():
            try:
                if symbolic.pystr_to_symbolic(rhs) in bad:
                    found.append((lhs, rhs))
            except Exception:
                pass
    return found


#: CloudSC species PARAMETER constants (Fortran NCLV=5, NCLDQL=1..NCLDQV=5). Baked
#: in so the species/LU loops are constant-trip; klev/klon/kidia/kfdia stay symbolic.
_SPECIES_CONSTANTS = {'nclv': 5, 'ncldql': 1, 'ncldqi': 2, 'ncldqr': 3, 'ncldqs': 4, 'ncldqv': 5}
_SPECIES_NAMES = frozenset(_SPECIES_CONSTANTS)


def _specialize(sdfg):
    for name, val in _SPECIES_CONSTANTS.items():
        specialize_symbol(sdfg, name, val)


def _unroll_fixpoint(sdfg):
    # Unrolling a loop can expose more short loops (nested species loops) -> fixpoint.
    while ShortLoopUnroll().apply_pass(sdfg, {}):
        pass


def _promote_args_to_symbols(sdfg):
    """Promote read-only ``Scalar`` arguments to symbols. ``find_promotable_scalars``
    enforces full safety (rejects any Scalar that's written, in a Stream, or has
    persistent / external lifetime), so ``transients_only=False`` is safe to default
    on for the cloudsc-style ``intent(in)`` integer args.
    """
    pass_ = ScalarToSymbolPromotion()
    pass_.transients_only = False
    pass_.apply_pass(sdfg, {})


def _pmar(xform):
    return lambda sdfg: PatternMatchAndApplyRepeated([xform()]).apply_pass(sdfg, {})


def _chain():
    """The ordered ``(label, apply_fn)`` chain applied to the candidate.

    Trimmed to the passes that actually fire on cloudsc. Dropped:

    * ``constant_propagation`` -- ``SimplifyPass`` already invokes it (see
      ``dace/transformation/passes/simplify.py``), and the two ``simplify`` /
      ``simplify_after_unroll`` stages above cover the same ground. Nothing
      between them and this stage introduces fresh scalar-write-once patterns.
    * ``accumulator_to_map_and_reduce`` -- targets ``acc[c] OP= g(other_inputs, i)``
      computed-delta accumulators; cloudsc has none left after the earlier WCR
      rewrites + unrolling, so it was a no-op here.
    * The trailing ``loop_to_map_post_pcia`` -- ``PromoteConstantIndexAccess``
      checks ``LoopToMap`` eligibility internally (via its own
      ``_mappable_loops`` probe), so running L2M *before* PCIA is wasted. One
      L2M after PCIA catches everything.
    """
    return [
        ('specialize', _specialize),
        ('simplify', lambda sdfg: sdfg.simplify()),
        # Fortran ``intent(in)`` integer args (``kidia`` / ``kfdia`` / ``klev`` / ...)
        # are registered as non-transient ``Scalar`` descriptors by the frontend.
        # ``simplify``'s internal ``ScalarToSymbolPromotion`` runs with the default
        # ``transients_only=True`` and skips them; without an explicit args-promotion
        # stage they stay as Scalars, and every downstream ``kfdia_plus_1_N = (kfdia
        # + 1)`` alias pins its iedge alive (hundreds of survivors in cloudsc). Run
        # the unrestricted promotion right after the initial simplify so the rest
        # of the chain sees them as real symbols.
        ('promote_args_to_symbols', _promote_args_to_symbols),
        ('short_loop_unroll', _unroll_fixpoint),
        # Re-simplify after unrolling. A body that guards on the iteration variable
        # (e.g. ``if jm == ncldqi``) only exposes a constant condition once unrolling
        # pins ``jm``; the first ``simplify`` ran while the guard was still symbolic, so
        # it could not fold it. Unrolling (with species constants specialised) then
        # leaves dead branches like ``if 1 == 2`` whose never-taken bodies still hold
        # constant-index writes (e.g. ``zvqx[1] = ...``) that read as a loop-carried
        # conflict and block LoopToMap; this second simplify folds them away.
        ('simplify_after_unroll', lambda sdfg: sdfg.simplify()),
        # Re-propagate memlets now that unroll+simplify changed the structure and the
        # ``symbols_defined_at`` fix lets enclosing-LoopRegion loop variables count as
        # defined. Without this, NSDFG out-connector memlets that propagation had
        # previously widened (the ``tendency_loc_cld[0:5, 0:klev, 0:klon]`` shape)
        # stay stale and block ``LoopToMap``; running propagation again narrows them
        # back to the per-level access (``[0:5, 0:_loop_it_14, 0:klon]``).
        ('propagate_memlets', lambda sdfg: propagate_memlets_sdfg(sdfg)),
        ('unique_loop_iterators', lambda sdfg: UniqueLoopIterators().apply_pass(sdfg, {})),
        ('trivial_tasklet_elimination', _pmar(TrivialTaskletElimination)),
        ('wcr_to_augassign', _pmar(WCRToAugAssign)),
        ('scalar_fission', lambda sdfg: PrivatizeScalars().apply_pass(sdfg, {})),
        # ``PromoteConstantIndexAccess`` introduces a per-iteration scalar for slots
        # that block ``LoopToMap`` (cloudsc ICE ``for_767`` writing ``zvqx[1]`` inside
        # an ``if yrecldp_laericesed`` guard). Runs right after ``scalar_fission`` --
        # PCIA's input is non-transient array slots (``arr[c]``) and is unaffected by
        # the per-state scalar renames ``scalar_fission`` performs. Sliding it earlier
        # in the pipeline lets ``symbol_propagation`` / ``loop_to_scan`` see the
        # already-privatised scalars (no risk of them being seen as live-out array
        # slots that would block their own analyses).
        ('promote_constant_index_access', lambda sdfg: PromoteConstantIndexAccess().apply_pass(sdfg, {})),
        ('symbol_propagation', lambda sdfg: SymbolPropagation().apply_pass(sdfg, {})),
        # ``LoopToReduce`` lifts bare-fold reduction loops (``acc OP= arr[f(i)]``) to a
        # ``Reduce`` libnode. No-op on cloudsc today (WCRToAugAssign + the unrolling
        # earlier in the chain leave no reduction-shaped loops), but it must run before
        # LoopToScan/LoopToMap because the three matchers carve out non-overlapping
        # cases of loop-carried writes (Reduce: loop-invariant write subset; Scan:
        # loop-var-dependent write at offset != 0; Map: loop-var-dependent write at
        # offset == iteration). Keeping it in the chain locks the ordering invariant
        # against future re-introductions of reduction shapes upstream.
        ('loop_to_reduce', lambda sdfg: LoopToReduce().apply_pass(sdfg, {})),
        # ``LoopToScan`` lifts prefix-sum / running-min/max loops to a Scan libnode call.
        # The pfsqrf vertical-flux prefix-sum (``pfcqlng[i] = pfcqlng[i-1] + delta[i]``)
        # is the cloudsc target; the multi-state-wrapper matcher relaxation now accepts
        # the cloudsc shape directly.
        ('loop_to_scan', lambda sdfg: LoopToScan().apply_pass(sdfg, {})),
        # ``AugAssignToWCR`` is the inverse of the ``wcr_to_augassign`` stage near the
        # top: now that the reduction matchers (``LoopToReduce`` / ``LoopToScan``) have
        # had their pass on the ``a = a + b`` form, the explicit augmented assignments
        # that didn't get lifted are re-encoded as WCR edges so the downstream codegen
        # emits the OMP ``reduction(...)`` / atomic write directly when ``LoopToMap``
        # parallelises the enclosing loop. Without this, an unlifted scalar accumulator
        # parallelised by L2M would race on the shared write.
        ('aug_assign_to_wcr', _pmar(AugAssignToWCR)),
        ('loop_to_map', _pmar(LoopToMap)),
        # Safety-net propagation. Each ``LoopToX`` pass now runs propagation
        # internally on its rewritten subgraph, but the chain composes several
        # of them and any pass that creates new state-level memlets without
        # narrowing them (e.g. ``AugAssignToWCR`` re-encoding edges) leaves
        # state-level memlets at array extent. One final whole-SDFG sweep
        # ensures the post-chain SDFG carries tight subsets for downstream
        # consumers (codegen, RedundantArrayCopying, validation diagnostics).
        ('propagate_memlets_final', lambda sdfg: propagate_memlets_sdfg(sdfg)),
    ]


def _run(sdfg: dace.SDFG, inputs, ieee_build: bool, sequential: bool, tag: str):
    """Run ``sdfg`` once on a private copy of ``inputs`` under the given build
    regime, returning the mutated buffers. ``sdfg`` is renamed (fresh build dir)
    and run in place; the prior ``compiler.cpu.args`` is restored afterwards."""
    sdfg.name = f'cloudsc_parallelize_{tag}'
    if sequential:
        make_sequential(sdfg)
    # Specialization erases the species symbols; drop inputs the SDFG no longer takes.
    needed = set(sdfg.arglist().keys()) | {str(s) for s in sdfg.free_symbols}
    args = {k: v for k, v in copy.deepcopy(inputs).items() if k in needed}
    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        if ieee_build:
            dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        sdfg(**args)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)
    return args


def _roundtrip(sdfg: dace.SDFG, tmpdir: str, tag: str) -> dace.SDFG:
    """Save ``sdfg`` to ``.sdfgz`` and reload it. Any non-lossless serialization is
    surfaced as a divergence between the original and the reloaded SDFG when the two
    are subsequently executed on identical inputs and their outputs compared.
    """
    path = os.path.join(tmpdir, f'{tag}.sdfgz')
    sdfg.save(path, compress=True)
    return dace.SDFG.from_file(path)


@pytest.fixture(scope='module')
def reference_sdfg_file(tmp_path_factory):
    """Build the un-transformed CloudSC SDFG once and persist it; each regime
    reloads it (the build's multi-minute parse is shared across regimes)."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    return path


@pytest.mark.integration
@pytest.mark.parametrize('regime', list(_REGIMES))
def test_cloudsc_parallelize_chain(reference_sdfg_file, regime, tmp_path):
    ieee_build, sequential, strict_tol, relaxed_tol = _REGIMES[regime]

    ref = dace.SDFG.from_file(reference_sdfg_file)
    inputs = generate_cloudsc_inputs(ref, seed=0)
    reference_out = _run(ref, inputs, ieee_build, sequential, tag=f'{regime}_ref')
    del ref
    gc.collect()

    candidate = dace.SDFG.from_file(reference_sdfg_file)
    rt_dir = str(tmp_path / 'roundtrips')
    os.makedirs(rt_dir, exist_ok=True)

    for label, apply_fn in _chain():
        # The loop transforms log every refused loop; keep the test output readable.
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            apply_fn(candidate)
        candidate.validate()

        # Per-stage structural invariants of the chain.
        n_wcr, n_red = len(_wcr_edges(candidate)), len(_reduce_nodes(candidate))
        bad_assigns = _plus_one_index_assignments(candidate)
        print(f'{regime}/{label}: wcr={n_wcr} reduce={n_red} plus1_assigns={len(bad_assigns)}')
        if label == 'wcr_to_augassign':
            assert n_wcr == 0, f'{regime}/{label}: {n_wcr} WCR edges remain (all must become augmented assignments)'
        if label == 'specialize':
            leaked = _SPECIES_NAMES & _free_symbol_names(candidate)
            assert not leaked, f'{regime}/{label}: species constants not fully baked out: {leaked}'
        if label == 'symbol_propagation':
            assert not bad_assigns, (f'{regime}/{label}: bound symbol assignments survived symbol propagation: '
                                     f'{bad_assigns}')
        # Pin the map/loop counts at the final parallelization step (the single L2M
        # sweep that follows PCIA). PCIA unblocks the ICE ``for_767`` slot, dropping
        # the loop count by 1; LoopToScan handles ``pfsqrf`` once the multi-state
        # matcher relaxation catches its 3-block shape.
        if label == 'loop_to_map':
            n_maps, n_loops = len(_map_entries(candidate)), len(_loop_regions(candidate))
            print(f'{regime}/{label}: maps={n_maps} sequential_loops={n_loops}')
            assert n_maps == _EXPECTED_MAPS, f'{regime}/{label}: {n_maps} maps, expected {_EXPECTED_MAPS}'
            assert n_loops == _EXPECTED_SEQUENTIAL_LOOPS, (
                f'{regime}/{label}: {n_loops} loops stayed sequential, expected {_EXPECTED_SEQUENTIAL_LOOPS}')

        out = _run(candidate, inputs, ieee_build, sequential, tag=f'{regime}_{label}')
        tol = relaxed_tol if label in _RELAXED_STEPS else strict_tol
        report = compare_outputs(out, reference_out, rtol=tol, atol=tol)
        worst = max(((ma, mr) for ma, mr, _ in report.values()), default=(0.0, 0.0))
        print(f'{regime}/{label}: worst |abs|={worst[0]:.3e} |rel|={worst[1]:.3e} (tol={tol:.0e})')
        bad = {name: (ma, mr) for name, (ma, mr, ok) in report.items() if not ok}
        assert not bad, (f'{regime}/{label}: outputs diverge from the un-transformed reference '
                         f'(tol={tol:.0e}): {bad}')

        # Serialize/deserialize roundtrip + e2e check. Two checkpoints: after
        # ``simplify`` (the first heavy SDFG mutation -- state fusion + NSDFG
        # inlining -- where most roundtrip bugs surface) and at the end (after
        # ``loop_to_map``, the full transformed shape). Doing it on every stage
        # roughly doubles wall-clock for one extra compile + run per stage, with
        # zero new bug coverage in practice -- the simplify and final checks
        # bracket every transform-induced descriptor / memlet / region mutation
        # the pipeline can introduce.
        if label in _ROUNDTRIP_CHECKPOINTS:
            rt = _roundtrip(candidate, rt_dir, f'{regime}_{label}')
            rt_out = _run(rt, inputs, ieee_build, sequential, tag=f'{regime}_{label}_rt')
            rt_report = compare_outputs(rt_out, out, rtol=strict_tol, atol=strict_tol)
            rt_bad = {name: (ma, mr) for name, (ma, mr, ok) in rt_report.items() if not ok}
            assert not rt_bad, (
                f'{regime}/{label}: serialize/deserialize roundtrip changed the result '
                f'(strict_tol={strict_tol:.0e}): {rt_bad}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
