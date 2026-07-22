# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unified adapter over the polybench and npbench corpora.

The two corpus loaders expose different APIs (polybench is size-indexed with an
untransformed-SDFG baseline; npbench is descriptor-dict based with a real numpy
reference). This module dispatches both behind one interface so the canonicalize
numerical and performance corpus tests can iterate the *combined* suite uniformly.

Datasets are selected by a named preset. polybench carries five sizes
(mini..extra-large); npbench carries one fused dataset that the loader clamps via
``cap``. The presets below pick a small/fast shape (``S``) and a larger,
performance-realistic shape (``paper``) for each suite, sized to stay
CI-tractable.
"""
from typing import Callable, Dict, List, Tuple

from tests.corpus.npbench import npbench as _NB
from tests.corpus.polybench import polybench as _PB

#: preset -> polybench ``sizes`` index (0=mini .. 4=extra-large).
_POLY_PRESET = {'S': 0, 'paper': 2}
#: preset -> npbench integer-symbol clamp (npbench's fused dataset is perf-sized,
#: e.g. N=400000; clamp to a CI-tractable shape that still exercises real loops).
_NP_PRESET = {'S': 32, 'paper': 256}

PRESETS = ('S', 'paper')


def kernels() -> List[Tuple[str, str]]:
    """All ``(suite, name)`` pairs across both corpora, polybench then npbench."""
    out = [('poly', k.name) for k in _PB.collect()]
    out += [('np', c["name"]) for c in _NB.collect()]
    return out


def make(suite: str, name: str, preset: str = 'S') -> Dict:
    """Build inputs + reference for one kernel at ``preset``; return a context dict."""
    if suite == 'poly':
        k = _PB.collect(name)[0]
        arrays, psize = _PB.make_inputs(k, size_index=_POLY_PRESET[preset], cap=None)
        ref = _PB.reference(k, arrays, psize)
        return dict(suite=suite, name=name, k=k, arrays=arrays, psize=psize, ref=ref)
    c = _NB.collect(name)[0]
    arrays, params = _NB.make_inputs(c, cap=_NP_PRESET[preset])
    ref = _NB.reference_outputs(c, arrays, params)
    return dict(suite=suite, name=name, c=c, arrays=arrays, params=params, ref=ref)


def build(ctx: Dict, transform: Callable, tag: str):
    """Fresh SDFG for ``ctx``'s kernel, apply ``transform`` in place, unique-name it."""
    s = _PB.fresh_sdfg(ctx['k']) if ctx['suite'] == 'poly' else _NB.fresh_sdfg(ctx['c'])
    transform(s)
    s.name = f"{s.name}_{tag}"
    return s


def run_matches(ctx: Dict, sdfg) -> bool:
    """Compile + run ``sdfg`` and compare to the reference for this suite."""
    if ctx['suite'] == 'poly':
        return _PB.outputs_match(ctx['ref'], _PB.run(sdfg, ctx['arrays'], ctx['psize']))
    return _NB.outputs_match(ctx['ref'], _NB.run_outputs(ctx['c'], sdfg, ctx['arrays'], ctx['params']))


def compiled_call(ctx: Dict, sdfg) -> Tuple[object, Dict]:
    """Return ``(compiled_sdfg, call_kwargs)`` for repeated *timed* invocation.

    The kwargs carry fresh input copies + the dataset symbols the SDFG needs.
    """
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
