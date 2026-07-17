# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end: the three optimization variants stay numerically correct at every subphase, on
CloudSC built three ways.

Frontends (SDFG sources):

* ``python``    -- the pure-Python ``cloudsc_py`` program.
* ``fortran``   -- the dace-fortran lowering of the CPU ``cloudsc.F90`` (entry ``cloudscouter``).
* ``gpu_scc``   -- the dace-fortran lowering of the GPU SCC k-caching kernel
                   (``cloudsc_scc_k_caching``); OpenACC pragmas are ignored (flang ``-U_OPENACC``),
                   so the sequential body lowers and the pipelines re-discover parallelism. Only the
                   COMPUTATIONAL body is lowered -- the multistep I/O program and the driver's
                   timer/MPI wrapper are excluded. See ``GPU_SCC_INTEGRATION_REPORT.md``.

All Fortran sources, module dependencies and saturation-function includes are co-located in
``cloudsc_variants/`` so the frontend resolves everything from one location.

Three optimization variants (each validated one subphase at a time):

* ``parallelize``  -- ``ParallelizePipeline`` after a ``specialize`` step (config propagation).
* ``canon_cpu``    -- ``canonicalize`` stages, ``target='cpu'``.
* ``canon_gpu``    -- ``canonicalize`` stages, ``target='gpu'``, CUT OFF before the GPU-offloading
                      transformation (``finalize_for_target``/``offload_to_gpu`` is NOT applied), so
                      the graph stays CPU-runnable and every subphase is validated on CPU.

Config propagation: every variant begins with a ``specialize`` subphase that bakes parameter
constants (shape symbols; for the GPU kernel also physics constants / ``TECLDP`` fields) via
``specialize_symbol`` -- itself a validated, saved subphase.

Per subphase: apply, structural ``validate()`` (validate-all), and SAVE the phase SDFGz
(``<variant>_<frontend>_<idx>_<label>.sdfgz``, cached). The numerical run happens ONCE at the end on
the fully-transformed graph vs the untransformed reference. Each frontend gets a distinct
``sdfg.name`` -- the dace-fortran frontend emits single-core scalar code the same way for both
Fortran sources, so distinct names keep the three baselines' ``.dacecache`` folders from colliding.

The Fortran/GPU frontends are OPTIONAL: if ``dace_fortran`` is unavailable or the source fails to
lower, those tests SKIP -- the Python tests always run.

    python tests/corpus/cloudsc/e2e_pipelines_test.py                    # all available
    python tests/corpus/cloudsc/e2e_pipelines_test.py --only python
    pytest tests/corpus/cloudsc/e2e_pipelines_test.py -v -s -m integration
"""
import argparse
import contextlib
import copy
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg.utils import specialize_symbol
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from dace.transformation.passes.parallelize import ParallelizePipeline

from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                            make_sequential, CLOUDSC_SYMBOLS)
from tests.corpus.cloudsc.inputs_physical import get_inputs_physical_py

VARIANTS_DIR = Path(__file__).parent / 'cloudsc_variants'
FORTRAN_CPU_SRC = VARIANTS_DIR / 'cloudsc.F90'
FORTRAN_CPU_ENTRY = 'cloudscouter'
GPU_KERNEL = VARIANTS_DIR / 'cloudsc_gpu_scc_k_caching_mod.F90'
GPU_ENTRY = 'cloudsc_scc_k_caching'

STRICT_TOL = 1e-13
RELAXED_TOL = 1e-9
RELAXED_LABELS = frozenset({'reduce', 'normalize_reduction', 'reduction_to_wcr_map', 'normalize_wcr',
                            'revert_nonreduction_wcr', 'fuse', 'lift_reduce', 'end'})


# --------------------------------------------------------------------------------------------------
# Optimization-variant stage lists -- each is [(label, apply_fn)], config-prop as the first stage.
# --------------------------------------------------------------------------------------------------
def _specialize_stage(constants: Optional[Dict[str, int]]) -> List[Tuple[str, Callable]]:
    if not constants:
        return []

    def apply(sdfg, consts=dict(constants)):
        for name, value in consts.items():
            if name in sdfg.symbols or any(name == str(s) for s in sdfg.free_symbols):
                specialize_symbol(sdfg, name, value)

    return [('specialize', apply)]


def _parallelize_stages(constants: Optional[Dict[str, int]]) -> List[Tuple[str, Callable]]:
    stages = _specialize_stage(constants)
    for stage in ParallelizePipeline()._stages():
        stages.append((type(stage).__name__, (lambda sdfg, s=stage: s.apply_pass(sdfg, {}))))
    return stages


def _canonicalize_stages(constants: Optional[Dict[str, int]], target: str) -> List[Tuple[str, Callable]]:
    """``target='gpu'`` runs the GPU-knob stages but NEVER the GPU-offload finalize -- ``_build_stages``
    contains no offload, so the cutoff is structural: the graph stays CPU-runnable, validated on CPU."""
    stages = _specialize_stage(constants)
    for label, unit in _build_stages(target=target):
        stages.append((label, (lambda sdfg, u=unit: u.apply_pass(sdfg, {}))))
    return stages


# --------------------------------------------------------------------------------------------------
# Per-subphase driver.
# --------------------------------------------------------------------------------------------------
def _run_sequential(sdfg: dace.SDFG, inputs: Dict) -> Dict:
    make_sequential(sdfg)
    needed = set(sdfg.arglist().keys()) | {str(s) for s in sdfg.free_symbols}
    args = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items() if k in needed}
    saved = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        sdfg(**args)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved)
    return args


def _validate_chain(build_reference: Callable[[], dace.SDFG], make_inputs: Callable[[dace.SDFG], Dict],
                    stages: List[Tuple[str, Callable]], dump_dir: Path, tag: str, run_numeric: bool) -> None:
    """Reference once, then each subphase validated + saved; one numerical run at the end.

    :param run_numeric: when False (inputs not yet wired for this frontend), validate + save every
        phase structurally but skip compile+run -- the variant is still proven to apply cleanly.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    reference_out = inputs = None
    if run_numeric:
        ref = build_reference()
        inputs = make_inputs(ref)
        reference_out = _run_sequential(ref, inputs)
        del ref
        gc.collect()

    candidate = build_reference()
    for idx, (label, apply_fn) in enumerate(stages, start=1):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            apply_fn(candidate)
        candidate.validate()
        dump_path = dump_dir / f'{tag}_{idx:02d}_{label}.sdfgz'
        if not (dump_path.exists() and dace.SDFG.from_file(str(dump_path)).hash_sdfg() == candidate.hash_sdfg()):
            candidate.save(str(dump_path), compress=True)
        print(f'{tag}/{idx:02d}_{label}: validated + saved')

    if not run_numeric:
        print(f'{tag}/END: structural-only (no inputs wired for this frontend yet)')
        return
    final_label = stages[-1][0] if stages else 'identity'
    out = _run_sequential(copy.deepcopy(candidate), inputs)
    tol = RELAXED_TOL if final_label in RELAXED_LABELS else STRICT_TOL
    report = compare_outputs(out, reference_out, rtol=tol, atol=tol)
    bad = {name: (ma, mr) for name, (ma, mr, ok) in report.items() if not ok}
    worst = max(((ma, mr) for ma, mr, _ in report.values()), default=(0.0, 0.0))
    print(f'{tag}/END: worst |abs|={worst[0]:.3e} |rel|={worst[1]:.3e} tol={tol:.0e}'
          f'{"" if not bad else "  <-- DIVERGES: " + ", ".join(bad)}')
    assert not bad, (f'{tag}: fully-transformed output diverges from the reference: {bad}. '
                     f'Per-phase SDFGz in {dump_dir} -- bisect to localise.')


# --------------------------------------------------------------------------------------------------
# Frontends. Each gets a distinct sdfg.name (see module docstring).
# --------------------------------------------------------------------------------------------------
def _uniquely_named(sdfg: dace.SDFG, name: str) -> dace.SDFG:
    sdfg.name = name
    return sdfg


def _python_reference() -> dace.SDFG:
    return _uniquely_named(build_cloudsc_sdfg(simplify=False), 'cloudsc_python')


def _python_inputs(sdfg: dace.SDFG) -> Dict:
    return get_inputs_physical_py(sdfg, seed=0)


def _dace_fortran_reason() -> Optional[str]:
    try:
        import dace_fortran.build  # noqa: F401
    except ImportError as exc:
        return f'dace_fortran not importable: {exc}'
    return None


def _fortran_cpu_reference() -> dace.SDFG:
    from dace_fortran.build import build_sdfg
    sdfg = build_sdfg(FORTRAN_CPU_SRC.read_text(), entry=FORTRAN_CPU_ENTRY, name='cloudsc_cpu',
                      out_dir=tempfile.mkdtemp(prefix='cloudsc_cpu_'))
    return _uniquely_named(sdfg, 'cloudsc_fortran_cpu')


def _gpu_scc_reference() -> dace.SDFG:
    """Lower the GPU SCC k-caching compute kernel. Sources co-located in ``cloudsc_variants/``: the
    kernel, the I/O-free module set, the saturation-function includes. The driver and multistep sit
    alongside but are not passed (the kernel does not USE them)."""
    from dace_fortran.build import build_sdfg_from_files
    files = [GPU_KERNEL, VARIANTS_DIR / 'cloudsc_modules_clean.F90'] + sorted(VARIANTS_DIR.glob('*.func.h'))
    sdfg = build_sdfg_from_files(files, entry=GPU_ENTRY, name='cloudsc_gpu_scc',
                                 out_dir=tempfile.mkdtemp(prefix='cloudsc_gpu_'))
    return _uniquely_named(sdfg, 'cloudsc_gpu_scc')


#: Shape symbols specialized in every variant (config propagation).
_CONSTANTS = {k: CLOUDSC_SYMBOLS[k] for k in ('nclv', ) if k in CLOUDSC_SYMBOLS}

_VARIANTS = ('parallelize', 'canon_cpu', 'canon_gpu')


def _variant_stages(variant: str, constants: Optional[Dict[str, int]]) -> List[Tuple[str, Callable]]:
    if variant == 'parallelize':
        return _parallelize_stages(constants)
    if variant == 'canon_cpu':
        return _canonicalize_stages(constants, target='cpu')
    if variant == 'canon_gpu':
        return _canonicalize_stages(constants, target='gpu')
    raise ValueError(variant)


def _dump_root() -> Path:
    """Where per-subphase SDFGz are saved. Default is under ``$HOME`` (NOT ``/tmp``) so a scratch
    wipe does not erase the saved phases; override with ``CLOUDSC_E2E_DUMP``."""
    return Path(os.environ.get('CLOUDSC_E2E_DUMP', str(Path.home() / '.cache' / 'cloudsc_e2e')))


# --------------------------------------------------------------------------------------------------
# Test matrix: {parallelize, canon_cpu, canon_gpu} x {python, fortran, gpu_scc}.
# --------------------------------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize('variant', _VARIANTS)
def test_python(variant):
    _validate_chain(_python_reference, _python_inputs, _variant_stages(variant, _CONSTANTS),
                    _dump_root(), f'{variant}_python', run_numeric=True)


@pytest.mark.integration
@pytest.mark.parametrize('variant', _VARIANTS)
def test_fortran_cpu(variant):
    reason = _dace_fortran_reason()
    if reason:
        pytest.skip(reason)
    _validate_chain(_fortran_cpu_reference, _python_inputs, _variant_stages(variant, None),
                    _dump_root(), f'{variant}_fortran', run_numeric=True)


@pytest.mark.integration
@pytest.mark.parametrize('variant', _VARIANTS)
def test_gpu_scc(variant):
    reason = _dace_fortran_reason()
    if reason:
        pytest.skip(reason)
    _validate_chain(_gpu_scc_reference, _python_inputs, _variant_stages(variant, None),
                    _dump_root(), f'{variant}_gpu_scc', run_numeric=True)


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', choices=('python', 'fortran', 'gpu_scc'), help='run only one frontend')
    ap.add_argument('--variant', choices=_VARIANTS, help='run only one optimization variant')
    args = ap.parse_args()

    variants = [args.variant] if args.variant else list(_VARIANTS)
    frontends = []
    if args.only in (None, 'python'):
        frontends.append(('python', _python_reference, _python_inputs, _CONSTANTS, True))
    if args.only in (None, 'fortran', 'gpu_scc'):
        reason = _dace_fortran_reason()
        if reason:
            print(f'SKIP fortran/gpu_scc: {reason}')
        else:
            if args.only in (None, 'fortran'):
                frontends.append(('fortran', _fortran_cpu_reference, _python_inputs, None, True))
            if args.only in (None, 'gpu_scc'):
                frontends.append(('gpu_scc', _gpu_scc_reference, _python_inputs, None, True))

    failures = []
    for fname, ref, inp, consts, numeric in frontends:
        for variant in variants:
            tag = f'{variant}_{fname}'
            print(f'\n===== {tag} =====')
            try:
                _validate_chain(ref, inp, _variant_stages(variant, consts), _dump_root(), tag, run_numeric=numeric)
                print(f'{tag}: PASS')
            except Exception as exc:  # noqa: BLE001 -- top-level runner reports, does not raise
                print(f'{tag}: FAIL -- {type(exc).__name__}: {exc}')
                failures.append(tag)
    total = len(frontends) * len(variants)
    print(f'\n{total - len(failures)}/{total} passed' + (f'; FAILED: {failures}' if failures else ''))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_main())
