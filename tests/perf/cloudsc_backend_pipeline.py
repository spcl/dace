# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared pipeline for the graph-backend (networkx vs rustworkx) benchmark and its
correctness test: given a fresh (un-transformed) CloudSC SDFG, specialize a handful of
scalars/symbols and fully unroll the now-constant-bounded loops this creates.

Kept out of the SDFG-shape symbols (``klev``/``klon``): those bound the kernel's large
outer loops, and specializing them would make ``LoopUnroll`` fully unroll 32-iteration
loops nested inside other 32-iteration loops -- an intractable amount of generated code
for a benchmark. ``nclv``/``ncldq*`` (the cloud-species count/indices, 5-wide) and
``kidia``/``kfdia`` (the horizontal tile bounds, exercised via ``specialize_scalar``
specifically since that is the function this benchmark is meant to cover) are small and
representative without exploding compile time.
"""
import time
from typing import Dict, Tuple, Union

import dace
from dace.sdfg.utils import specialize_scalar
from dace.transformation.interstate import LoopUnroll

from tests.corpus.cloudsc.generate_data_for_cloudsc import CLOUDSC_SYMBOLS

#: Shape/index symbols baked in via SDFG.specialize (compile-time constants).
SPECIALIZED_SYMBOLS = ('nclv', 'ncldql', 'ncldqi', 'ncldqr', 'ncldqs', 'ncldqv')
#: Scalar arguments baked in via specialize_scalar (the function this benchmark covers).
SPECIALIZED_SCALARS = ('kidia', 'kfdia')


def run_simplify(sdfg: dace.SDFG, backend: str) -> float:
    """Run ``SDFG.simplify()`` under the given graph backend. Mutates ``sdfg`` in place.

    :param sdfg: The SDFG to simplify (mutated in place).
    :param backend: ``'networkx'`` or ``'rustworkx'``.
    :returns: Elapsed seconds.
    """
    with dace.config.set_temporary('graph', 'backend', value=backend):
        t0 = time.perf_counter()
        sdfg.simplify()
        t1 = time.perf_counter()
    return t1 - t0


def specialize_and_unroll(sdfg: dace.SDFG, backend: str) -> Tuple[float, int]:
    """Specialize :data:`SPECIALIZED_SYMBOLS` / :data:`SPECIALIZED_SCALARS` ("config-prop")
    and fully unroll every loop this makes constant-bounded, under the given graph backend.
    Mutates ``sdfg`` in place.

    :param sdfg: The SDFG to transform (mutated in place).
    :param backend: ``'networkx'`` or ``'rustworkx'``.
    :returns: ``(elapsed_seconds, loop_unroll_applications)``.
    """
    with dace.config.set_temporary('graph', 'backend', value=backend):
        t0 = time.perf_counter()
        sdfg.specialize({name: CLOUDSC_SYMBOLS[name] for name in SPECIALIZED_SYMBOLS})
        for name in SPECIALIZED_SCALARS:
            specialize_scalar(sdfg, name, CLOUDSC_SYMBOLS[name])
        applied = sdfg.apply_transformations_repeated(LoopUnroll)
        sdfg.validate()
        t1 = time.perf_counter()
    return t1 - t0, applied


def run_pipeline(sdfg: dace.SDFG, backend: str) -> Dict[str, float]:
    """The full benchmarked pipeline, in the order it is reported: simplify -> config-prop
    (specialize_scalar/SDFG.specialize) + LoopUnroll -> (codegen/compile/serialize/deserialize
    are timed by the caller, which needs the intermediate SDFG for each). Mutates ``sdfg`` in
    place and returns the two front-end phase timings; see graph_backend_cloudsc_bench.py for
    the remaining phases.

    :param sdfg: The SDFG to transform (mutated in place).
    :param backend: ``'networkx'`` or ``'rustworkx'``.
    :returns: ``{'simplify': seconds, 'config_prop_loopunroll': seconds}``.
    :raises RuntimeError: If LoopUnroll found nothing to unroll (a silent no-op would make
        the downstream codegen/compile timings measure the wrong -- un-unrolled -- SDFG).
    """
    simplify_time = run_simplify(sdfg, backend)
    pass_time, applied = specialize_and_unroll(sdfg, backend)
    if applied == 0:
        raise RuntimeError(f'backend={backend!r}: LoopUnroll found nothing to unroll after simplify')
    return {'simplify': simplify_time, 'config_prop_loopunroll': pass_time}


def filtered_inputs(sdfg: dace.SDFG,
                    inputs: Dict[str, Union['object', int, float]]) -> Dict[str, Union['object', int, float]]:
    """Restrict a generated CloudSC input dict to the names ``sdfg`` actually still
    expects. ``specialize``/``specialize_scalar`` can remove a symbol/scalar from the
    compiled call signature entirely, so a stale full input dict (built once, shared
    across a transformed and an un-transformed SDFG) is not safe to pass directly.

    :param sdfg: The SDFG about to be called.
    :param inputs: A full input dict (e.g. from ``generate_cloudsc_inputs``).
    :returns: The subset of ``inputs`` whose keys are in ``sdfg.arglist()``.
    """
    expected = set(sdfg.arglist().keys())
    return {k: v for k, v in inputs.items() if k in expected}
