# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pipeline half only; .F90 sources, data init, numeric runner come from dace-fortran via
``numeric_check`` (README). canon_gpu runs every GPU knob but never offload -- the cutoff is
structural (``offload_to_gpu`` absent from ``_build_stages``), graph stays CPU-runnable.
"""
import contextlib
import os
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


def variant_stages(variant: str, constants: Optional[Dict[str, int]] = None) -> List[Stage]:
    stages = specialize_stage(constants)
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
    """``numeric_check`` is the dace-fortran plug point; ``None`` = structural-only."""
    tag = tag or f'{variant}_{sdfg.name}'
    dump_dir.mkdir(parents=True, exist_ok=True)
    for idx, (label, apply_fn) in enumerate(variant_stages(variant, constants), start=1):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            apply_fn(sdfg)
        sdfg.validate()
        dump_path = dump_dir / f'{tag}_{idx:02d}_{label}.sdfgz'
        if not (dump_path.exists() and dace.SDFG.from_file(str(dump_path)).hash_sdfg() == sdfg.hash_sdfg()):
            sdfg.save(str(dump_path), compress=True)
        print(f'{tag}/{idx:02d}_{label}: validated + saved')
    if numeric_check is not None:
        numeric_check(sdfg)
        print(f'{tag}/END: numeric check passed')
    else:
        print(f'{tag}/END: structural-only (no numeric_check wired)')
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
