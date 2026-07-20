# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC e2e pipeline driver: parallelize / canon_cpu / canon_gpu, phase-checkpointed.

Every variant drives the loops to MAXIMUM (sound) parallelism -- canon runs with peeling and
anti-dependence breaking on (its defaults), parallelize peels too. ``canon_gpu`` runs every GPU
knob but never offloads; the cutoff is structural (``offload_to_gpu`` absent from ``_build_stages``),
so the graph stays CPU-runnable and the numeric check compiles it for the host.

The recipe is grouped into PHASES (consecutive stages sharing a stage label -- so canon's
``loop_to_x`` Loop2X lifts and its ``parallelize`` Loop2Map each form one phase; the parallelize
variant is classified into ``prep`` / ``loop_to_x`` / ``parallelize``). At each phase boundary
``run_pipeline`` validates, runs ``numeric_check`` against the un-transformed reference, and saves a
``.sdfgz`` checkpoint. A checkpoint on disk therefore means "this phase passed"; a re-run loads the
furthest good checkpoint and resumes past it (the multi-minute ``simplify`` / build is not repeated).

``numeric_check`` is the reference-compare closure from :func:`make_numeric_check` (self-contained --
no dace-fortran); ``None`` = structural-only (fast, no per-phase compile+run).
"""
import contextlib
import copy
import hashlib
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dace
from dace.sdfg.utils import specialize_symbol
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from dace.transformation.passes.parallelization_prep import DEFAULT_UNROLL_LIMIT
from dace.transformation.passes.parallelize import ParallelizePipeline
from dace.transformation.passes.pattern_matching import PatternMatchAndApply
Stage = Tuple[str, Callable[[dace.SDFG], None]]
Phase = Tuple[str, List[Stage]]

VARIANTS: Tuple[str, ...] = ('parallelize', 'canon_cpu', 'canon_gpu')

#: Numeric-check regimes; parameters resolved lazily by :func:`_regime_params`.
_REGIME_NAMES: Tuple[str, ...] = ('ieee', 'o3')


def _regime_params(regime: str) -> Tuple[str, bool, float, float]:
    """``(cpu_args, sequential, strict_tol, relaxed_tol)`` for a numeric-check regime. The CPU arg
    strings load lazily from ``generate_data_for_cloudsc`` so this module still imports when loaded by
    path OUTSIDE the extended tests tree -- the dace-fortran matrix test imports pipelines.py directly
    and needs only ``run_pipeline``, never the python-variant numeric helpers, whose
    ``tests.corpus.cloudsc...`` imports would otherwise fail against another repo's ``tests`` package.
    ``ieee`` (-O0, sequential) is bit-exact on value-preserving phases; ``o3`` runs parallel maps, so
    reassociating phases get the looser bound."""
    from tests.corpus.cloudsc.generate_data_for_cloudsc import IEEE_CPU_ARGS, O3_CPU_ARGS
    return {
        'ieee': (IEEE_CPU_ARGS, True, 1e-16, 1e-15),
        'o3': (O3_CPU_ARGS, False, 1e-16, 1e-12),
    }[regime]

#: Coarse phase names at (or after) which FP reassociation / parallel-reduction reorder can occur. The
#: tolerance goes relaxed on the first hit and STAYS relaxed (sticky) -- ``start`` and ``normalize``
#: (value-preserving; on cloudsc the BLAS/IV lifts there no-op) stay bit-exact, ``loop_to_x`` onward
#: (reduction/scan lifts, then loops->maps, then fusion) gets the reduction tolerance.
_REASSOC_PHASES = frozenset({'loop_to_x', 'parallelize', 'finalize'})


def specialize_stage(constants: Optional[Dict[str, int]]) -> List[Stage]:
    if not constants:
        return []

    def apply(sdfg, consts=dict(constants)):
        for name, value in consts.items():
            if name in sdfg.symbols or any(name == str(s) for s in sdfg.free_symbols):
                specialize_symbol(sdfg, name, value)

    return [('specialize', apply)]


def pretreat_stages() -> List[Stage]:
    """Shrink the raw frontend SDFG before any loop unrolling: simplify, then fuse states with
    happens-before (StateFusionExtended). Both pipelines unroll constant-trip loops -- unroll cost
    and post-unroll validation both scale with state count, so collapsing the one-state-per-statement
    frontend graph first is what keeps unrolling (and its validate) tractable. ``parallelize``
    documents that its caller must simplify beforehand; this IS that step. Value-preserving, so the
    per-phase numeric_check still holds."""

    def simplify(sdfg):
        sdfg.simplify()

    def state_fusion_extended(sdfg):
        from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
        sdfg.apply_transformations_repeated(StateFusionExtended, validate=False, validate_all=False)

    return [('simplify', simplify), ('state_fusion_extended', state_fusion_extended)]


def _stage_pass_names(unit) -> set:
    """The transformation type names a pass runs -- unwrapping ``PatternMatchAndApply`` to see the
    real transforms inside (so a ``PatternMatchAndApplyRepeated([LoopToMap()])`` reads as LoopToMap,
    not the wrapper). Avoids getattr: ``PatternMatchAndApply`` always sets ``transformations``."""
    if isinstance(unit, PatternMatchAndApply):
        return {type(t).__name__ for t in unit.transformations}
    return {type(unit).__name__}


def _parallelize_phase_of(names: set) -> str:
    """Classify a parallelize-pipeline pass into a phase label -- Loop2Map and Loop2X are their own
    phases (per the user's request), everything else is ``prep``."""
    if names & {'LoopToMap'}:
        return 'parallelize'
    if names & {'LoopToReduce'}:
        return 'loop_to_x'
    return 'prep'


def _parallelize_phases() -> List[Phase]:
    """The parallelize pipeline (single source of truth: ``ParallelizePipeline._stages()``) grouped
    into ``prep`` / ``loop_to_x`` / ``parallelize`` phases. Peeling on (``peel_limit=4``) for max
    parallelism."""
    stages = ParallelizePipeline(unroll_limit=DEFAULT_UNROLL_LIMIT, peel_limit=4)._stages()
    flat: List[Tuple[str, Stage]] = []
    for unit in stages:
        names = _stage_pass_names(unit)
        phase = _parallelize_phase_of(names)
        label = next(iter(names)) if len(names) == 1 else '+'.join(sorted(names))
        # Carry the phase label as the group key; keep the transform name for per-stage reporting.
        flat.append((phase, (label, lambda sdfg, u=unit: u.apply_pass(sdfg, {}))))
    # Group by the phase key (first element), preserving the (label, fn) stage tuples.
    phases: List[Phase] = []
    for phase, stage in flat:
        if phases and phases[-1][0] == phase:
            phases[-1][1].append(stage)
        else:
            phases.append((phase, [stage]))
    return phases


#: Coarse canon super-phases: each fine ``_build_stages`` stage label -> one of four super-phases, so
#: canon checkpoints at 4 boundaries (Loop2X and Loop2Map kept distinct) instead of ~47. GLUE labels
#: (value-preserving structural fixups that recur across regions) inherit the currently-open
#: super-phase, so a ``cascade_iedges_up`` between the loop_to_x lifts stays in ``loop_to_x`` rather
#: than opening a spurious ``normalize`` group. A fine label absent from BOTH maps trips the assert in
#: :func:`_canon_coarse_phases` -- a canon stage added upstream must be classified here, never
#: silently misfiled.
_CANON_SUPER_PHASE: Dict[str, str] = {
    # normalize: clean + semantic lifts + lower-to-loops + reduce-prep -> canonical loop form.
    'clean': 'normalize', 'loop_to_symm': 'normalize', 'lift_inv': 'normalize', 'privatize_scatter': 'normalize',
    'normalize_reduction': 'normalize', 'loop_to_syrk': 'normalize', 'loop_to_syr2k': 'normalize',
    'prep': 'normalize', 'lift_reduce': 'normalize', 'lower': 'normalize', 'reroll': 'normalize',
    'reduce': 'normalize', 'distribute': 'normalize', 'loop_to_symmetrize': 'normalize',
    # loop_to_x (Loop2X): parallelization-prep (peel/break/fission/stride) + the reduction/scan lifts.
    'peel': 'loop_to_x', 'break_antidep': 'loop_to_x', 'move_if_into_loop': 'loop_to_x', 'fission': 'loop_to_x',
    'loop_stride_permutation': 'loop_to_x', 'fuse_consecutive_loops': 'loop_to_x', 'lift_copy_loops': 'loop_to_x',
    'loop_to_x': 'loop_to_x', 'loop_to_scan': 'loop_to_x',
    # parallelize (Loop2Map): loops -> maps, scatter guards, post-l2m fusion / interchange / collapse.
    'parallelize': 'parallelize', 'parallelize_guarded': 'parallelize', 'reduction_to_wcr_map': 'parallelize',
    'scatter': 'parallelize', 'post_l2m': 'parallelize', 'loop_fuse': 'parallelize', 'lift_copy': 'parallelize',
    'interchange': 'parallelize', 'reorder': 'parallelize', 'collapse': 'parallelize',
    # ``coalesce`` runs between ``post_l2m`` and ``loop_fuse`` (pipeline.py ``_coalesce``): graph
    # prep for maximal fusion once loops are maps, so it belongs to the same super-phase as its
    # neighbours rather than opening a new checkpoint boundary.
    'coalesce': 'parallelize',
    # finalize: map fusion + einsum lift + guard hoist + WCR normalize + terminal simplify / parallelize.
    'fuse': 'finalize', 'lift': 'finalize', 'licm': 'finalize', 'hoist_guards': 'finalize',
    'normalize_wcr': 'finalize', 'revert_nonreduction_wcr': 'finalize', 'relax_powers': 'finalize', 'end': 'finalize',
}
#: Recurring value-preserving structural-fixup labels that inherit the currently-open super-phase.
_CANON_GLUE = frozenset({'cascade_iedges_up', 'ssa', 'untrivialize'})

#: The four canon super-phases, in pipeline order.
_CANON_ORDER: Tuple[str, ...] = ('normalize', 'loop_to_x', 'parallelize', 'finalize')


def _canon_coarse_phases(target: str, assume_parallel_guards: bool) -> List[Phase]:
    """The canonicalization recipe coarse-grouped into the four :data:`_CANON_ORDER` super-phases
    (``normalize`` -> ``loop_to_x`` -> ``parallelize`` -> ``finalize``), so canon checkpoints at 4
    boundaries with Loop2X (reduction/scan lifts) and Loop2Map distinct instead of ~47 fine ones.
    Sound max-parallelism defaults (peel_limit=4, break_anti_dependence=True)."""
    phases: List[Phase] = []
    current = _CANON_ORDER[0]
    for label, unit in _build_stages(target=target,
                                     peel_limit=4,
                                     break_anti_dependence=True,
                                     assume_parallel_guards=assume_parallel_guards):
        if label in _CANON_GLUE:
            super_phase = current  # inherit the open region so recurring glue doesn't split a phase
        else:
            assert label in _CANON_SUPER_PHASE, (
                f'canon stage {label!r} is unclassified -- add it to _CANON_SUPER_PHASE or _CANON_GLUE')
            super_phase = _CANON_SUPER_PHASE[label]
        current = super_phase
        stage = (label, lambda sdfg, u=unit: u.apply_pass(sdfg, {}))
        if phases and phases[-1][0] == super_phase:
            phases[-1][1].append(stage)
        else:
            phases.append((super_phase, [stage]))
    return phases


def variant_phases(variant: str,
                   constants: Optional[Dict[str, int]] = None,
                   assume_parallel_guards: bool = False) -> List[Phase]:
    """The full ordered phase list for ``variant``. ``start`` (specialize config-prop + pretreat
    simplify/state-fusion) is ONE phase; then the variant's own coarse phases (parallelize:
    prep/loop_to_x/parallelize; canon: normalize/loop_to_x/parallelize/finalize)."""
    start: List[Stage] = specialize_stage(constants) + pretreat_stages()
    phases: List[Phase] = [('start', start)]
    if variant == 'parallelize':
        phases += _parallelize_phases()
    elif variant == 'canon_cpu':
        phases += _canon_coarse_phases('cpu', assume_parallel_guards)
    elif variant == 'canon_gpu':
        phases += _canon_coarse_phases('gpu', assume_parallel_guards)
    else:
        raise ValueError(f'unknown variant {variant!r}; expected one of {VARIANTS}')
    return phases


def variant_stages(variant: str, constants: Optional[Dict[str, int]] = None) -> List[Stage]:
    """Flat view of :func:`variant_phases` (every stage, phase boundaries dropped)."""
    return [stage for _, stages in variant_phases(variant, constants) for stage in stages]


def uniquely_named(sdfg: dace.SDFG, name: str) -> dace.SDFG:
    sdfg.name = name
    return sdfg


def run_candidate(sdfg: dace.SDFG, inputs: Dict, cpu_args: str, sequential: bool, tag: str) -> Dict:
    """Run ``sdfg`` once on a private copy of ``inputs`` under ``cpu_args``, returning the mutated
    buffers. Renamed to a fresh build dir for the run then restored, so the pipeline SDFG keeps its
    identity. Specialization erases the species symbols, so args the SDFG no longer takes are dropped.
    Under ``sequential`` the maps are forced sequential first -- pass a throwaway copy there so the
    live candidate's schedules are not mutated."""
    saved_name = sdfg.name
    sdfg.name = f'cloudsc_e2e_{tag}'
    if sequential:
        from tests.corpus.cloudsc.generate_data_for_cloudsc import make_sequential
        make_sequential(sdfg)
    needed = set(sdfg.arglist().keys()) | {str(s) for s in sdfg.free_symbols}
    args = {k: v for k, v in copy.deepcopy(inputs).items() if k in needed}
    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=cpu_args)
        sdfg(**args)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)
        sdfg.name = saved_name
    return args


def build_reference_outputs(reference: dace.SDFG, regime: str = 'ieee', seed: int = 0) -> Tuple[Dict, Dict]:
    """Run the un-transformed ``reference`` ONCE under ``regime``; return ``(inputs, reference_out)``.
    ``inputs`` is the pristine seeded input dict (candidates are re-driven on private copies of it);
    ``reference_out`` is the reference's mutated output buffers. Shareable across variants so the raw
    cloudsc compiles+runs a single time for the whole matrix."""
    from tests.corpus.cloudsc.generate_data_for_cloudsc import generate_cloudsc_inputs, make_sequential
    cpu_args, sequential, _strict, _relaxed = _regime_params(regime)
    if sequential:
        make_sequential(reference)
    inputs = generate_cloudsc_inputs(reference, seed=seed)
    reference_out = run_candidate(reference, inputs, cpu_args, sequential, tag=f'{regime}_ref')
    return inputs, reference_out


def numeric_check_from(inputs: Dict, reference_out: Dict, regime: str = 'ieee') -> Callable[[dace.SDFG, str], None]:
    """A ``check(sdfg, phase_name)`` closure over a prebuilt reference (see
    :func:`build_reference_outputs`): re-drives ``sdfg`` on the identical inputs and compares every
    output array to ``reference_out``, so a divergence is pinned to the exact phase that introduced
    it. Raises ``AssertionError`` on any out-of-tolerance array. Tolerance is strict (bit-exact under
    ``ieee``) on value-preserving phases and goes relaxed -- and stays relaxed -- from the first
    reassociating phase (see :data:`_REASSOC_PHASES`)."""
    from tests.corpus.cloudsc.generate_data_for_cloudsc import compare_outputs
    cpu_args, sequential, strict_tol, relaxed_tol = _regime_params(regime)
    state = {'relaxed': False}

    def check(sdfg: dace.SDFG, phase_name: str) -> None:
        if phase_name in _REASSOC_PHASES:
            state['relaxed'] = True
        tol = relaxed_tol if state['relaxed'] else strict_tol
        # ``sequential`` mutates schedules -- run on a copy so the live candidate is untouched; ``o3``
        # (parallel) does not mutate, so run it in place (cheap).
        target = copy.deepcopy(sdfg) if sequential else sdfg
        out = run_candidate(target, inputs, cpu_args, sequential, tag=f'{regime}_{phase_name}')
        report = compare_outputs(out, reference_out, rtol=tol, atol=tol)
        worst = max(((ma, mr) for ma, mr, _ in report.values()), default=(0.0, 0.0))
        bad = {name: (ma, mr) for name, (ma, mr, ok) in report.items() if not ok}
        if bad:
            raise AssertionError(f'{phase_name}: outputs diverge from the un-transformed reference '
                                 f'(tol={tol:.0e}): {bad}')
        print(f'    numeric[{phase_name}]: worst |abs|={worst[0]:.2e} |rel|={worst[1]:.2e} tol={tol:.0e} OK')

    return check


def make_numeric_check(reference: dace.SDFG,
                       regime: str = 'ieee',
                       seed: int = 0) -> Callable[[dace.SDFG, str], None]:
    """Convenience wrapper: build the reference output from ``reference`` and return the per-phase
    check closure in one call (see :func:`build_reference_outputs` + :func:`numeric_check_from`)."""
    inputs, reference_out = build_reference_outputs(reference, regime=regime, seed=seed)
    return numeric_check_from(inputs, reference_out, regime=regime)


def _plan_signature(phases: List[Phase], constants: Optional[Dict[str, int]] = None) -> str:
    """Short digest of the phase plan (ordered names + stage counts) AND the specialize constants.
    Baked into checkpoint filenames so a changed pipeline -- or a different constant baking (e.g.
    ``nclv=5`` vs ``nclv=3``, which leaves the plan shape identical) -- mints fresh checkpoints
    instead of resuming from a stale ``.sdfgz``."""
    plan = '|'.join(f'{name}:{len(stages)}' for name, stages in phases)
    consts = ','.join(f'{k}={v}' for k, v in sorted((constants or {}).items()))
    return hashlib.sha1(f'{plan}#{consts}'.encode()).hexdigest()[:8]


def _checkpoint_path(dump_dir: Path, tag: str, sig: str, idx: int, phase_name: str) -> Path:
    return dump_dir / f'{tag}__{sig}__p{idx:02d}__{phase_name}.sdfgz'


def _checkpoint_matches(ckpt: Path, sdfg: dace.SDFG) -> bool:
    """True iff ``ckpt`` exists and deserializes to the same SDFG hash. A checkpoint that fails to
    load -- e.g. a Fortran-frontend SDFG whose ``$klon``-style symbols the reader can't sympify --
    counts as no-match, so the phase just re-saves instead of crashing the pipeline."""
    if not ckpt.exists():
        return False
    try:
        return dace.SDFG.from_file(str(ckpt)).hash_sdfg() == sdfg.hash_sdfg()
    except Exception:
        return False


def run_pipeline(sdfg: dace.SDFG,
                 variant: str,
                 dump_dir: Path,
                 constants: Optional[Dict[str, int]] = None,
                 tag: Optional[str] = None,
                 numeric_check: Optional[Callable[[dace.SDFG, str], None]] = None,
                 assume_parallel_guards: bool = False,
                 resume: bool = True) -> dace.SDFG:
    """Drive ``variant`` on ``sdfg`` to maximum parallelism, phase by phase.

    At each phase boundary: apply the phase's stages, validate once, run ``numeric_check(sdfg,
    phase_name)`` if wired (``None`` = structural-only), then save a ``.sdfgz`` checkpoint. On a
    re-run (``resume``) the furthest existing checkpoint for this ``(tag, plan)`` is loaded and the
    pipeline resumes past it. A per-phase apply/validate timing summary is printed at the end.
    """
    tag = tag or f'{variant}_{sdfg.name}'
    dump_dir.mkdir(parents=True, exist_ok=True)
    phases = variant_phases(variant, constants, assume_parallel_guards=assume_parallel_guards)
    sig = _plan_signature(phases, constants)

    start = 0
    if resume:
        for i in range(len(phases) - 1, -1, -1):
            ckpt = _checkpoint_path(dump_dir, tag, sig, i + 1, phases[i][0])
            if not ckpt.exists():
                continue
            try:
                sdfg = dace.SDFG.from_file(str(ckpt))
                start = i + 1
                print(f'{tag}: resumed from checkpoint p{i + 1:02d}_{phases[i][0]} ({ckpt.name})')
                break
            except Exception as exc:  # a truncated / stale-format checkpoint -- fall back to an earlier one
                print(f'{tag}: checkpoint {ckpt.name} unreadable ({exc}); trying an earlier one')

    timings: List[Tuple[str, float, float]] = []
    for idx in range(start, len(phases)):
        phase_name, stages = phases[idx]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            for _label, apply_fn in stages:
                apply_fn(sdfg)
        t1 = time.perf_counter()
        sdfg.validate()
        t2 = time.perf_counter()
        # numeric BEFORE save: a checkpoint on disk then means "this phase passed", so resume can
        # skip it without re-checking.
        if numeric_check is not None:
            numeric_check(sdfg, phase_name)
        ckpt = _checkpoint_path(dump_dir, tag, sig, idx + 1, phase_name)
        if not _checkpoint_matches(ckpt, sdfg):
            # Best-effort: a checkpoint is a resume/post-mortem aid, not correctness. A frontend
            # whose SDFG won't serialize (e.g. Fortran ``$klon`` symbols) must not derail the
            # numeric run.
            try:
                sdfg.save(str(ckpt), compress=True)
            except Exception as exc:
                print(f'{tag}/p{idx + 1:02d}_{phase_name}: checkpoint save skipped '
                      f'({type(exc).__name__}: {exc})')
        timings.append((f'p{idx + 1:02d}_{phase_name}', t1 - t0, t2 - t1))
        print(f'{tag}/p{idx + 1:02d}_{phase_name}: apply={t1 - t0:.2f}s validate={t2 - t1:.2f}s '
              f'stages={len(stages)} states={sdfg.number_of_nodes()} + saved')

    if numeric_check is None:
        print(f'{tag}/END: structural-only (no numeric_check wired)')
    top = sorted(timings, key=lambda r: r[1] + r[2], reverse=True)[:8]
    print(f'{tag}/TIMING top-{len(top)} slowest (apply+validate):')
    for name, ap, va in top:
        print(f'    {name}: {ap + va:.2f}s (apply={ap:.2f} validate={va:.2f})')
    return sdfg


def dump_root() -> Path:
    return Path(os.environ.get('CLOUDSC_E2E_DUMP', str(Path.home() / '.cache' / 'cloudsc_e2e')))


def _main() -> int:
    import argparse
    from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg, CLOUDSC_SYMBOLS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', choices=VARIANTS, default='parallelize', help='canon_* are heavy (many phases)')
    ap.add_argument('--numeric', action='store_true', help='wire per-phase numeric check vs the un-transformed ref')
    ap.add_argument('--regime', choices=list(_REGIME_NAMES), default='ieee', help='numeric-check build regime')
    ap.add_argument('--no-resume', dest='resume', action='store_false', help='ignore saved checkpoints')
    ap.add_argument('--assume-parallel-guards', action='store_true', help='drop runtime guards (unsound, max maps)')
    args = ap.parse_args()

    constants = {k: CLOUDSC_SYMBOLS[k] for k in ('nclv', ) if k in CLOUDSC_SYMBOLS}

    numeric = None
    if args.numeric:
        # Build once, persist, reload twice: one instance drives the reference run, one the pipeline
        # (the multi-minute simplify=False parse is not repeated).
        ref_path = dump_root() / 'cloudsc_nosimplify.sdfgz'
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        build_cloudsc_sdfg(simplify=False).save(str(ref_path), compress=True)
        numeric = make_numeric_check(dace.SDFG.from_file(str(ref_path)), regime=args.regime, seed=0)
        sdfg = uniquely_named(dace.SDFG.from_file(str(ref_path)), 'cloudsc_python')
    else:
        sdfg = uniquely_named(build_cloudsc_sdfg(simplify=False), 'cloudsc_python')

    run_pipeline(sdfg,
                 args.variant,
                 dump_root(),
                 constants=constants,
                 tag=f'{args.variant}_python',
                 numeric_check=numeric,
                 assume_parallel_guards=args.assume_parallel_guards,
                 resume=args.resume)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_main())
