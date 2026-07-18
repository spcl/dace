# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pipeline half only; .F90 sources, data init, numeric runner come from dace-fortran via
``numeric_check`` (README). canon_gpu runs every GPU knob but never offload -- the cutoff is
structural (``offload_to_gpu`` absent from ``_build_stages``), graph stays CPU-runnable.
"""
import contextlib
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dace
from dace.sdfg.utils import specialize_symbol
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from dace.transformation.passes.parallelize import ParallelizePipeline

Stage = Tuple[str, Callable[[dace.SDFG], None]]

VARIANTS: Tuple[str, ...] = ('parallelize', 'canon_cpu', 'canon_gpu')


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
    end-of-pipeline numeric_check still holds."""

    def simplify(sdfg):
        sdfg.simplify()

    def state_fusion_extended(sdfg):
        from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
        sdfg.apply_transformations_repeated(StateFusionExtended, validate=False, validate_all=False)

    return [('simplify', simplify), ('state_fusion_extended', state_fusion_extended)]


def variant_stages(variant: str, constants: Optional[Dict[str, int]] = None) -> List[Stage]:
    stages = specialize_stage(constants) + pretreat_stages()
    if variant == 'parallelize':
        for stage in ParallelizePipeline()._stages():
            stages.append((type(stage).__name__, lambda sdfg, s=stage: s.apply_pass(sdfg, {})))
    elif variant in ('canon_cpu', 'canon_gpu'):
        target = 'cpu' if variant == 'canon_cpu' else 'gpu'
        for label, unit in _build_stages(target=target):
            stages.append((label, lambda sdfg, u=unit: u.apply_pass(sdfg, {})))
    else:
        raise ValueError(f'unknown variant {variant!r}; expected one of {VARIANTS}')
    return stages


def uniquely_named(sdfg: dace.SDFG, name: str) -> dace.SDFG:
    sdfg.name = name
    return sdfg


def run_pipeline(sdfg: dace.SDFG,
                 variant: str,
                 dump_dir: Path,
                 constants: Optional[Dict[str, int]] = None,
                 tag: Optional[str] = None,
                 numeric_check: Optional[Callable[[dace.SDFG], None]] = None) -> dace.SDFG:
    """``numeric_check`` is the dace-fortran plug point; ``None`` = structural-only.

    Each subphase is timed (apply / validate separately) so the slowest stages are visible; a
    descending summary is printed at the end.
    """
    tag = tag or f'{variant}_{sdfg.name}'
    dump_dir.mkdir(parents=True, exist_ok=True)
    timings: List[Tuple[str, float, float]] = []
    for idx, (label, apply_fn) in enumerate(variant_stages(variant, constants), start=1):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            apply_fn(sdfg)
        t1 = time.perf_counter()
        sdfg.validate()
        t2 = time.perf_counter()
        dump_path = dump_dir / f'{tag}_{idx:02d}_{label}.sdfgz'
        if not (dump_path.exists() and dace.SDFG.from_file(str(dump_path)).hash_sdfg() == sdfg.hash_sdfg()):
            sdfg.save(str(dump_path), compress=True)
        timings.append((f'{idx:02d}_{label}', t1 - t0, t2 - t1))
        print(f'{tag}/{idx:02d}_{label}: apply={t1 - t0:.2f}s validate={t2 - t1:.2f}s '
              f'states={sdfg.number_of_nodes()} + saved')
    if numeric_check is not None:
        numeric_check(sdfg)
        print(f'{tag}/END: numeric check passed')
    else:
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
    ap.add_argument('--variant', choices=VARIANTS, default='parallelize', help='canon_* are heavy (164/165 stages)')
    args = ap.parse_args()

    constants = {k: CLOUDSC_SYMBOLS[k] for k in ('nclv', ) if k in CLOUDSC_SYMBOLS}
    sdfg = uniquely_named(build_cloudsc_sdfg(simplify=False), 'cloudsc_python')
    run_pipeline(sdfg, args.variant, dump_root(), constants=constants, tag=f'{args.variant}_python')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_main())
