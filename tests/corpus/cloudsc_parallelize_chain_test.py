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
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.scalar_fission import PrivatizeScalars
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
_RELAXED_STEPS = {'loop_to_reduce', 'loop_to_map'}

#: Index symbols whose ``<sym> + 1`` bound expression must not survive symbol
#: propagation as an interstate-edge assignment value.
_INDEX_SYMS = ('klev', 'kfdia', 'kidia')

#: Expected end-state after ``loop_to_map`` (pinned; label-independent so it
#: survives loop-numbering churn across commits). Every parallelizable loop
#: becomes a Map; the only loops that stay sequential are the genuine
#: data dependences -- LU triangular solve (``zqxn``, 2 loops), the vertical
#: flux prefix-sum (``pfsqrf``) and cloud-tendency whole-array write over klev
#: (2 loops), and the loop-invariant ``zvqx`` fall-speed setup (5 unrolled
#: copies). A change here means loop2map coverage shifted -- review before
#: bumping the numbers.
_EXPECTED_MAPS = 376
_EXPECTED_SEQUENTIAL_LOOPS = 9


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


def _pmar(xform):
    return lambda sdfg: PatternMatchAndApplyRepeated([xform()]).apply_pass(sdfg, {})


def _chain():
    """The ordered ``(label, apply_fn)`` chain applied to the candidate."""
    return [
        ('specialize', _specialize),
        ('simplify', lambda sdfg: sdfg.simplify()),
        ('short_loop_unroll', _unroll_fixpoint),
        ('unique_loop_iterators', lambda sdfg: UniqueLoopIterators().apply_pass(sdfg, {})),
        ('trivial_tasklet_elimination', _pmar(TrivialTaskletElimination)),
        ('wcr_to_augassign', _pmar(WCRToAugAssign)),
        ('scalar_fission', lambda sdfg: PrivatizeScalars().apply_pass(sdfg, {})),
        ('symbol_propagation', lambda sdfg: SymbolPropagation().apply_pass(sdfg, {})),
        ('constant_propagation', lambda sdfg: ConstantPropagation().apply_pass(sdfg, {})),
        ('loop_to_reduce', lambda sdfg: LoopToReduce().apply_pass(sdfg, {})),
        ('loop_to_map', _pmar(LoopToMap)),
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
def test_cloudsc_parallelize_chain(reference_sdfg_file, regime):
    ieee_build, sequential, strict_tol, relaxed_tol = _REGIMES[regime]

    ref = dace.SDFG.from_file(reference_sdfg_file)
    inputs = generate_cloudsc_inputs(ref, seed=0)
    reference_out = _run(ref, inputs, ieee_build, sequential, tag=f'{regime}_ref')
    del ref
    gc.collect()

    candidate = dace.SDFG.from_file(reference_sdfg_file)

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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
