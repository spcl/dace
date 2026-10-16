# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unified adapter over the polybench, tsvc, tsvc_2_5 and automata corpora.

The three corpus loaders expose different APIs (polybench is size-indexed with an
untransformed-SDFG baseline; tsvc and tsvc_2_5 are symbol-parametrized kernel registries
with a scalar numpy oracle each). This module dispatches all three behind one interface so
the canonicalize numerical corpus tests can iterate the *combined* suite uniformly.

Every suite runs at a small, CI-tractable shape: polybench's mini size, one tsvc extent per kernel regime,
tsvc_2_5's declared symbol table and the automata suite's fixed stream length.
"""

import inspect
from collections.abc import Callable

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram
from tests.corpus.automata import automata as AUTOMATA
from tests.corpus.polybench import polybench as _PB
from tests.corpus.tsvc import tsvc as _TS
from tests.corpus.tsvc.tsvc_numpy import REFERENCES as _TS_REF
from tests.corpus.tsvc_2_5 import tsvc_2_5 as _T25
from tests.corpus.tsvc_2_5 import tsvc_2_5_numpy as _T25_REF

#: tsvc swept extent, ``(1d regime, 2d regime)``; the non-swept extent follows from ``tsvc.regime_sizes``.
TSVC_EXTENT = (64, 16)

#: One seed for every corpus draw: a reference and its candidate must see identical
#: inputs, and a rerun must reproduce a failure.
_SEED = 1234


def kernels() -> list[tuple[str, str]]:
    """All ``(suite, name)`` pairs, in corpus order: polybench, tsvc, tsvc_2_5, automata."""
    out = [("poly", k.name) for k in _PB.collect()]
    out += [("tsvc", k.name) for k in _TS.collect()]
    out += [("tsvc25", p.name) for p in _T25.collect()]
    out += [("automata", name) for name in AUTOMATA.collect()]
    return out


def t25_program(name: str) -> DaceProgram:
    """The tsvc_2_5 ``@dace.program`` with this name."""
    return [p for p in _T25.collect() if p.name == name][0]


def t25_reference(
    program: DaceProgram, arrays: dict[str, np.ndarray], scalars: dict[str, float], sizes: dict[str, int]
) -> dict[str, np.ndarray]:
    """Run the tsvc_2_5 numpy oracle on copies of ``arrays``; return what it wrote.

    The oracle is ``ref_`` + the kernel name with any module prefix and ``ext_`` dropped,
    and it takes its arguments by name: arrays, scalars, and the lowercased symbol values
    it declares (the ``iv_*`` oracles take the trip count as ``n``).
    """
    base = program.name.rsplit("tsvc_2_5_", 1)[-1]
    oracle = vars(_T25_REF)["ref_" + (base.removeprefix("ext_"))]
    pool = {
        **{n: a.copy() for n, a in arrays.items()},
        **scalars,
        **{s.lower(): v for s, v in sizes.items()},
        "n": sizes["LEN_1D"],
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    return {n: pool[n] for n in arrays}


def make(suite: str, name: str) -> dict:
    """Build inputs + reference for one kernel; return a context dict."""
    if suite == "poly":
        k = _PB.collect(name)[0]
        arrays, psize = _PB.make_inputs(k, size_index=0, cap=None)
        ref = _PB.reference(k, arrays, psize)
        return dict(suite=suite, name=name, k=k, arrays=arrays, psize=psize, ref=ref)
    if suite == "tsvc":
        k = _TS.collect(name=name)[0]
        l1, l2 = _TS.regime_sizes(k.regime, TSVC_EXTENT[0 if k.regime == "1d" else 1])
        arrays = _TS.allocate(k, l1, l2, np.random.default_rng(_SEED))
        params = {**_TS.scalar_params(k, l1), **_TS.symbols(k, l1, l2)}
        ref = {n: a.copy() for n, a in arrays.items()}
        _TS_REF[name](**ref, **params)  # the scalar oracle writes its outputs into ref in place
        return dict(suite=suite, name=name, k=k, arrays=arrays, params=params, ref=ref)
    if suite == "automata":
        made = AUTOMATA.inputs(name, _SEED)
        ref = {n: a.copy() for n, a in made.arrays.items()}
        made.reference(**ref, **made.scalars)
        return dict(
            suite=suite,
            name=name,
            k=AUTOMATA.PROGRAMS[name],
            arrays=made.arrays,
            scalars=made.scalars,
            params=made.sizes,
            ref=ref,
        )
    p = t25_program(name)
    sizes = dict(_T25.SIZES)
    arrays, scalars = _T25.make_inputs(p, seed=_SEED, sizes=sizes)
    ref = t25_reference(p, arrays, scalars, sizes)
    return dict(suite=suite, name=name, k=p, arrays=arrays, scalars=scalars, params=sizes, ref=ref)


def build(ctx: dict, transform: Callable, tag: str):
    """Fresh SDFG for ``ctx``'s kernel, apply ``transform`` in place, unique-name it."""
    suite = ctx["suite"]
    if suite == "poly":
        s = _PB.fresh_sdfg(ctx["k"])
    elif suite == "tsvc":
        # The loader's builder deep-copies the program's SDFG (a prior variant may have
        # mutated it); its tag is only a name uniquifier, which ``build`` applies below.
        s = _TS.to_sdfg(ctx["k"], "corpus", simplify=True)
    else:
        s = ctx["k"].to_sdfg(simplify=True)  # re-parsed per call, so already private
    transform(s)
    s.name = f"{s.name}_{tag}"
    return s


def call_symbols(ctx: dict, sdfg: dace.SDFG) -> dict:
    """Non-array call kwargs for a tsvc / tsvc_2_5 SDFG. Mutates ``sdfg``, so call it
    BEFORE compiling: a transform can leave a hoisted guard symbol free but unregistered,
    and the compiled signature is what the symbol has to appear in."""
    if ctx["suite"] == "tsvc":
        return dict(ctx["params"])
    free = {str(s) for s in sdfg.free_symbols}
    for s in free - set(sdfg.symbols):
        sdfg.add_symbol(s, dace.int64)
    return {**ctx["scalars"], **{s: v for s, v in ctx["params"].items() if s in free}}


def run_matches(ctx: dict, sdfg: dace.SDFG) -> bool:
    """Compile + run ``sdfg`` and compare to the reference for this suite."""
    if ctx["suite"] == "poly":
        return _PB.outputs_match(ctx["ref"], _PB.run(sdfg, ctx["arrays"], ctx["psize"]))
    kwargs = call_symbols(ctx, sdfg)
    got = {n: a.copy() for n, a in ctx["arrays"].items()}
    sdfg.compile()(**got, **kwargs)
    # polybench's comparison is the dtype-aware one: integers exactly (a gather index is a
    # read-only input, but an argmax index is an OUTPUT), floats at a per-precision tolerance.
    return bool(_PB.outputs_match(ctx["ref"], got))
