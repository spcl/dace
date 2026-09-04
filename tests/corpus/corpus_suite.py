# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unified adapter over the polybench, npbench, tsvc and tsvc_2_5 corpora.

The four corpus loaders expose different APIs (polybench is size-indexed with an
untransformed-SDFG baseline; npbench is descriptor-dict based with a real numpy
reference; tsvc and tsvc_2_5 are symbol-parametrized kernel registries with a
scalar numpy oracle each). This module dispatches all four behind one interface so
the canonicalize numerical and performance corpus tests can iterate the *combined*
suite uniformly.

Datasets are selected by a named preset. polybench carries five sizes
(mini..extra-large); npbench carries one fused dataset that the loader clamps via
``cap``; tsvc sweeps one extent per kernel regime; tsvc_2_5 evaluates its declared
extents at a symbol table. The presets below pick a small/fast shape (``S``, the
CI-tractable one) and each suite's own published shape (``paper``, what a reported
speedup has to be measured at).
"""
import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram
from tests.corpus.npbench import npbench as _NB
from tests.corpus.polybench import polybench as _PB
from tests.corpus.tsvc import tsvc as _TS
from tests.corpus.tsvc.tsvc_numpy import REFERENCES as _TS_REF
from tests.corpus.tsvc_2_5 import tsvc_2_5 as _T25
from tests.corpus.tsvc_2_5 import tsvc_2_5_numpy as _T25_REF


def int_or_none(value: str) -> Optional[int]:
    """``''`` -> ``None`` (npbench's "no clamp"), anything else -> its integer value."""
    return int(value) if value.strip() else None


#: ``paper`` means each suite's OWN published shape, not a CI-tractable stand-in: on a 72-thread
#: node the previous clamps put whole kernels in the tens of microseconds, where OpenMP fork/join
#: dominates and a speedup number measures the runtime rather than the code. Every one of the four
#: is env-overridable so dialling a sweep back is an export, not an edit and a redeploy.
#: preset -> polybench ``sizes`` index (0=mini .. 4=extra-large). LARGE, not EXTRA-LARGE: the
#: time-stepped kernels (heat_3d, seidel_2d, adi, fdtd_2d) carry TSTEPS as well as N, so index 4
#: costs minutes per kernel per arm and six arms times 50 reps does not fit a 4 h job.
_POLY_PRESET = {'S': 0, 'paper': int(os.environ.get('CORPUS_POLY_PAPER_INDEX', '3'))}
#: preset -> npbench integer-symbol clamp. ``None`` = npbench's own fused dataset, which is already
#: the perf-sized one (N=400000 for azimint, 4096 for spmv). Note a small native size is NOT the
#: clamp: permute_3d really is N=128 upstream.
_NP_PRESET = {'S': 32, 'paper': int_or_none(os.environ.get('CORPUS_NP_PAPER_CAP', ''))}
#: preset -> tsvc swept extent, ``(1d regime, 2d regime)``; the non-swept extent
#: follows from ``tsvc.regime_sizes``. ``S`` is the corpus's own default length.
#: ``paper`` is TSVC's own LEN_1D. The bound is s176, the one kernel quadratic in LEN_1D: the numpy
#: oracles are scalar Python loops, so its CORRECTNESS run (once per kernel, not per rep) costs
#: O(LEN_1D**2) interpreter iterations while every other kernel's is linear and free.
_TSVC_PRESET = {
    'S': (64, 16),
    'paper': (int(os.environ.get('CORPUS_TSVC_PAPER_LEN1D',
                                 '589824')), int(os.environ.get('CORPUS_TSVC_PAPER_LEN2D', '768'))),
}
#: Kernels capped BELOW the preset extent, by kernel name -> max LEN_1D.
#:
#: ``s176`` is quadratic in LEN_1D in BOTH directions and is the only one: its body is
#: ``for j in range(n//2): for i in range(n//2): a[i] += b[i+m-j-1]*c[j]``. At the raised extent the
#: KERNEL would be ~8.7e10 flops -- larger than any polybench gemm -- so it would dominate the tsvc
#: sweep on its own, and its scalar-Python oracle (tsvc_numpy.s176_d_single) would be worse still.
#: Kept at the old 32000 so the rest of the corpus can scale. Only ever LOWERS an extent, and only
#: for a ``1d`` regime kernel, so ``regime_sizes``' LEN_1D >= LEN_2D + 8 invariant cannot be broken.
TSVC_LEN1D_CAP = {'s176_d_single': int(os.environ.get('CORPUS_TSVC_S176_LEN1D', '32000'))}
#: preset -> tsvc_2_5 overrides on its ``SIZES`` symbol table. Only the ``LEN_*``
#: extents scale: the rest are tuned strides, thresholds and tile sizes that must
#: keep their values (and ``LEN_R7`` must stay a multiple of 7 -- its kernels
#: reroll by 7).
_T25_PRESET = {
    'S': {},
    'paper': {
        'LEN_1D': int(os.environ.get('CORPUS_T25_PAPER_LEN1D', '589824')),
        'LEN_2D': int(os.environ.get('CORPUS_T25_PAPER_LEN2D', '768')),
        'LEN_3D': int(os.environ.get('CORPUS_T25_PAPER_LEN3D', '96')),
        'LEN_R7': 7 * int(os.environ.get('CORPUS_T25_PAPER_R7_BLOCKS', '73728')),
    },
}

PRESETS = ('S', 'paper')

#: One seed for every corpus draw: a reference and its candidate must see identical
#: inputs, and a rerun must reproduce a failure.
_SEED = 1234


def kernels() -> List[Tuple[str, str]]:
    """All ``(suite, name)`` pairs, in corpus order: polybench, npbench, tsvc, tsvc_2_5."""
    out = [('poly', k.name) for k in _PB.collect()]
    out += [('np', c["name"]) for c in _NB.collect()]
    out += [('tsvc', k.name) for k in _TS.collect()]
    out += [('tsvc25', p.name) for p in _T25.collect()]
    return out


def t25_program(name: str) -> DaceProgram:
    """The tsvc_2_5 ``@dace.program`` with this name."""
    return [p for p in _T25.collect() if p.name == name][0]


def t25_reference(program: DaceProgram, arrays: Dict[str, np.ndarray], scalars: Dict[str, float],
                  sizes: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Run the tsvc_2_5 numpy oracle on copies of ``arrays``; return what it wrote.

    The oracle is ``ref_`` + the kernel name with any module prefix and ``ext_`` dropped,
    and it takes its arguments by name: arrays, scalars, and the lowercased symbol values
    it declares (the ``iv_*`` oracles take the trip count as ``n``).
    """
    base = program.name.rsplit('tsvc_2_5_', 1)[-1]
    oracle = vars(_T25_REF)['ref_' + (base[4:] if base.startswith('ext_') else base)]
    pool = {
        **{
            n: a.copy()
            for n, a in arrays.items()
        },
        **scalars,
        **{
            s.lower(): v
            for s, v in sizes.items()
        }, 'n': sizes['LEN_1D']
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    return {n: pool[n] for n in arrays}


def make(suite: str, name: str, preset: str = 'S') -> Dict:
    """Build inputs + reference for one kernel at ``preset``; return a context dict."""
    if suite == 'poly':
        k = _PB.collect(name)[0]
        arrays, psize = _PB.make_inputs(k, size_index=_POLY_PRESET[preset], cap=None, paper=(preset == 'paper'))
        ref = _PB.reference(k, arrays, psize)
        return dict(suite=suite, name=name, k=k, arrays=arrays, psize=psize, ref=ref)
    if suite == 'np':
        c = _NB.collect(name)[0]
        arrays, params = _NB.make_inputs(c, cap=_NP_PRESET[preset], preset=preset)
        ref = _NB.reference_outputs(c, arrays, params)
        return dict(suite=suite, name=name, c=c, arrays=arrays, params=params, ref=ref)
    if suite == 'tsvc':
        k = _TS.collect(name=name)[0]
        l1, l2 = _TS.regime_sizes(k.regime, _TSVC_PRESET[preset][0 if k.regime == '1d' else 1])
        l1 = min(l1, TSVC_LEN1D_CAP.get(name, l1))  # quadratic-in-LEN_1D kernels only; see the dict
        arrays = _TS.allocate(k, l1, l2, np.random.default_rng(_SEED))
        params = {**_TS.scalar_params(k, l1), **_TS.symbols(k, l1, l2)}
        ref = {n: a.copy() for n, a in arrays.items()}
        _TS_REF[name](**ref, **params)  # the scalar oracle writes its outputs into ref in place
        return dict(suite=suite, name=name, k=k, arrays=arrays, params=params, ref=ref)
    p = t25_program(name)
    sizes = {**_T25.SIZES, **_T25_PRESET[preset]}
    arrays, scalars = _T25.make_inputs(p, seed=_SEED, sizes=sizes)
    ref = t25_reference(p, arrays, scalars, sizes)
    return dict(suite=suite, name=name, k=p, arrays=arrays, scalars=scalars, params=sizes, ref=ref)


def build(ctx: Dict, transform: Callable, tag: str):
    """Fresh SDFG for ``ctx``'s kernel, apply ``transform`` in place, unique-name it."""
    suite = ctx['suite']
    if suite == 'poly':
        s = _PB.fresh_sdfg(ctx['k'])
    elif suite == 'np':
        s = _NB.fresh_sdfg(ctx['c'])
    elif suite == 'tsvc':
        # The loader's builder deep-copies the program's SDFG (a prior variant may have
        # mutated it); its tag is only a name uniquifier, which ``build`` applies below.
        s = _TS.to_sdfg(ctx['k'], 'corpus', simplify=True)
    else:
        s = ctx['k'].to_sdfg(simplify=True)  # re-parsed per call, so already private
    transform(s)
    s.name = f"{s.name}_{tag}"
    return s


def call_symbols(ctx: Dict, sdfg: dace.SDFG) -> Dict:
    """Non-array call kwargs for a tsvc / tsvc_2_5 SDFG. Mutates ``sdfg``, so call it
    BEFORE compiling: a transform can leave a hoisted guard symbol free but unregistered,
    and the compiled signature is what the symbol has to appear in."""
    if ctx['suite'] == 'tsvc':
        return dict(ctx['params'])
    free = {str(s) for s in sdfg.free_symbols}
    for s in free - set(sdfg.symbols):
        sdfg.add_symbol(s, dace.int64)
    return {**ctx['scalars'], **{s: v for s, v in ctx['params'].items() if s in free}}


def run_matches(ctx: Dict, sdfg: dace.SDFG) -> bool:
    """Compile + run ``sdfg`` and compare to the reference for this suite."""
    if ctx['suite'] == 'poly':
        return _PB.outputs_match(ctx['ref'], _PB.run(sdfg, ctx['arrays'], ctx['psize']))
    if ctx['suite'] == 'np':
        return _NB.outputs_match(ctx['ref'], _NB.run_outputs(ctx['c'], sdfg, ctx['arrays'], ctx['params']))
    kwargs = call_symbols(ctx, sdfg)
    got = {n: a.copy() for n, a in ctx['arrays'].items()}
    sdfg.compile()(**got, **kwargs)
    # polybench's comparison is the dtype-aware one: integers exactly (a gather index is a
    # read-only input, but an argmax index is an OUTPUT), floats at a per-precision tolerance.
    return bool(_PB.outputs_match(ctx['ref'], got))


def compiled_call(ctx: Dict, sdfg: dace.SDFG) -> Tuple[object, Dict]:
    """Return ``(compiled_sdfg, call_kwargs)`` for repeated *timed* invocation.

    The kwargs carry fresh input copies + the dataset symbols the SDFG needs.
    """
    if ctx['suite'] in ('tsvc', 'tsvc25'):
        kwargs = call_symbols(ctx, sdfg)  # may register symbols -> must precede the compile
        return sdfg.compile(), {**{n: a.copy() for n, a in ctx['arrays'].items()}, **kwargs}
    cs = sdfg.compile()
    if ctx['suite'] == 'poly':
        return cs, {**{n: v.copy() for n, v in ctx['arrays'].items()}, **ctx['psize']}
    # npbench: resolve the program's parameter names from arrays+params, then add
    # the (non-float) dataset symbols the SDFG was parametrized over.
    c = ctx['c']
    work = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in ctx['arrays'].items()}
    call = _NB._map_call(c["program"], work, ctx['params'])
    symbols = {k: v for k, v in ctx['params'].items() if not isinstance(v, float)}
    return cs, {**call, **{k: v for k, v in symbols.items() if k not in call}}


def numpy_reference(ctx: Dict) -> Dict[str, np.ndarray]:
    """The NUMPY reference outputs for ``ctx`` -- the denominator polybench and npbench SHARE.

    polybench's other reference (the untransformed SDFG, ``_PB.reference``) is kept and is what
    ``make`` still puts in ``ctx['ref']``; this is the second one, so the two can be cross-checked.
    tsvc / tsvc_2_5 have no numpy reference at all -- their oracles are scalar python loops, which
    is exactly why they divide by ``seq-cpp`` -- so asking for one raises instead of quietly
    handing back a different KIND of number.
    """
    suite = ctx['suite']
    if suite == 'poly':
        return _PB.numpy_reference(ctx['k'], ctx['arrays'], ctx['psize'])
    if suite == 'np':
        return _NB.reference_outputs(ctx['c'], ctx['arrays'], ctx['params'])
    raise ValueError(f"{suite} has no numpy reference; its oracle is a scalar python loop")


def numpy_call(ctx: Dict) -> Tuple[Callable, Dict]:
    """``(fn, kwargs)`` for repeated *timed* invocation of the numpy reference (cf.
    :func:`compiled_call`, which does the same for a compiled arm).

    The copies and the argument-name resolution happen HERE, once, outside the timed region.
    For polybench, ``_PB.restore_inputs(kwargs, ctx['arrays'])`` resets the arrays to their
    pristine values between repetitions -- the kernels write their inputs, so repetition 2 would
    otherwise measure whatever repetition 1 left behind.
    Time it WITHOUT pinning BLAS threads: the agreed np+poly baseline is PARALLEL numpy.
    """
    suite = ctx['suite']
    if suite == 'poly':
        return _PB.numpy_call(ctx['k'], ctx['arrays'], ctx['psize'])
    if suite == 'np':
        ref = ctx['c']['reference']
        work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ctx['arrays'].items()}
        return ref, _NB._map_call(ref, work, ctx['params'])
    raise ValueError(f"{suite} has no numpy reference; its oracle is a scalar python loop")
