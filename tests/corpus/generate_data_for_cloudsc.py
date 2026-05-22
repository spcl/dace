# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Input-data generation and run-and-compare helpers for the inlined CloudSC
kernel in ``tests/corpus/cloudsc.py``, for end-to-end numerical tests.

The inlined ``cloudsc_py`` program requires no callbacks, so it compiles and
runs standalone. These helpers build a runnable SDFG, generate a
self-consistent input set, and run two SDFGs on identical inputs to check that
a transformation (simplify, SymbolPropagation, vectorization, ...) is
numerically faithful to a non-transformed reference.

Typical use::

    ref = build_cloudsc_sdfg(simplify=False)
    sut = build_cloudsc_sdfg(simplify=False)
    sut.simplify()
    SymbolPropagation().apply_pass(sut, {})
    assert run_and_compare(ref, sut)

Data generation follows the CloudSC convention: random doubles in ``[0, 1)``
for floating arrays, ones for plain integer arrays, the physically-meaningful
values from :data:`CLOUDSC_SYMBOLS` for the named integer scalars, and those
same values for the shape symbols. The small problem size (``klev = 16``,
``klon = 32``) keeps a compiled run fast.
"""
import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import sympy

import dace
from dace import dtypes
from dace.sdfg import nodes
from tests.corpus.cloudsc import cloudsc_py

#: Concrete values for the CloudSC shape symbols and the named integer scalars
#: (the cloud-species indices ``ncldq*`` and the column range ``kidia:kfdia``).
CLOUDSC_SYMBOLS: Dict[str, int] = {
    'klev': 16,
    'klon': 32,
    'nclv': 5,
    'ncldql': 1,  # liquid cloud water
    'ncldqi': 2,  # ice cloud water
    'ncldqr': 3,  # rain water
    'ncldqs': 4,  # snow
    'ncldqv': 5,  # vapour
    'kidia': 1,
    'kfdia': 32,
}


def build_cloudsc_sdfg(simplify: bool = False) -> dace.SDFG:
    """Build a fresh runnable SDFG from the inlined CloudSC program.

    :param simplify: Whether to run the simplify pipeline while building.
    :returns: A validated CloudSC SDFG.
    """
    sdfg = cloudsc_py.to_sdfg(simplify=simplify)
    sdfg.validate()
    return sdfg


def _instantiate_dim(dim) -> int:
    """Resolve one array-shape dimension to a concrete size via
    :data:`CLOUDSC_SYMBOLS`.

    :param dim: A shape dimension (int, ``dace.symbol``, or symbolic expr).
    :returns: The instantiated integer size.
    """
    if isinstance(dim, (int, sympy.Number)):
        return int(dim)
    if isinstance(dim, dace.symbol):
        return CLOUDSC_SYMBOLS[str(dim)]
    return int(sympy.sympify(dim).subs(CLOUDSC_SYMBOLS))


def generate_cloudsc_inputs(sdfg: dace.SDFG, seed: int = 0) -> Dict[str, Union[np.ndarray, int, float]]:
    """Generate a self-consistent CloudSC input set for ``sdfg``.

    Every non-transient array is filled (random doubles in ``[0, 1)``, ones for
    plain integer arrays, the :data:`CLOUDSC_SYMBOLS` value for a named integer
    scalar); length-1 arrays are passed as scalars; the shape symbols are added
    last. The result is ready to splat into ``sdfg(**inputs)``.

    :param sdfg: The CloudSC SDFG whose non-transient arrays are filled.
    :param seed: Seed for the random number generator (reproducible runs).
    :returns: A kwargs dict of arrays, scalars, and symbol values.
    """
    rng = np.random.default_rng(seed)
    arrays: Dict[str, np.ndarray] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        dims: List[int] = [_instantiate_dim(d) for d in desc.shape]
        if 'int' in str(desc.dtype):
            data = np.ones(dims, order='F').astype(np.int32)
            if name in CLOUDSC_SYMBOLS:
                data = np.zeros(dims, order='F').astype(np.int32)
                data.flat[0] = CLOUDSC_SYMBOLS[name]
            arrays[name] = data
        else:
            arrays[name] = rng.random(dims).astype(np.float64, order='F')

    inputs: Dict[str, Union[np.ndarray, int, float]] = {
        name: (data.flat[0] if sdfg.arrays[name].shape == (1, ) else data)
        for name, data in arrays.items()
    }
    inputs.update(CLOUDSC_SYMBOLS)
    return inputs


def make_sequential(sdfg: dace.SDFG) -> Tuple[int, int]:
    """Force every map and library node to a sequential schedule, in place.

    A numerical-equivalence comparison must be deterministic: CloudSC's
    parallel (OpenMP) maps reorder floating-point reductions / accumulations
    run-to-run, so two separately-compiled SDFGs differ by ~1e-5 even when they
    are the *same* computation. Running sequentially removes that noise so a
    real numerical difference between a transform and its reference stands out.

    :param sdfg: The SDFG to make sequential (mutated in place).
    :returns: ``(maps, library_nodes)`` re-scheduled.
    """
    n_maps = n_lib = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dtypes.ScheduleType.Sequential
            n_maps += 1
        elif isinstance(node, nodes.LibraryNode):
            if hasattr(node, 'schedule'):
                node.schedule = dtypes.ScheduleType.Sequential
            n_lib += 1
    return n_maps, n_lib


def run_and_compare(reference: dace.SDFG,
                    candidate: dace.SDFG,
                    seed: int = 0,
                    rtol: float = 1e-9,
                    atol: float = 1e-12,
                    sequential: bool = True,
                    verbose: bool = False) -> bool:
    """Run two CloudSC SDFGs on identical inputs and compare every shared
    non-transient output array.

    Both SDFGs are driven with the same generated inputs (a private deep copy
    each, since the call mutates the buffers in place). Only array outputs are
    compared; scalar and symbol inputs are not.

    :param reference: The reference SDFG (e.g. un-transformed).
    :param candidate: The SDFG under test.
    :param seed: Seed for input generation.
    :param rtol: Relative tolerance (``numpy.allclose``).
    :param atol: Absolute tolerance (``numpy.allclose``).
    :param sequential: Force both SDFGs sequential before running
        (:func:`make_sequential`) so the comparison is deterministic; leave on
        unless you specifically want to measure parallel behaviour.
    :param verbose: Print the max/avg difference of each mismatching array.
    :returns: ``True`` iff every shared output array matches within tolerance.
    """
    if sequential:
        make_sequential(reference)
        make_sequential(candidate)
    ref_inputs = generate_cloudsc_inputs(reference, seed)
    cand_inputs = copy.deepcopy(ref_inputs)
    reference(**ref_inputs)
    candidate(**cand_inputs)

    matches = True
    for name, ref_val in ref_inputs.items():
        if not isinstance(ref_val, np.ndarray):
            continue
        cand_val = cand_inputs[name]
        if not np.allclose(ref_val, cand_val, rtol=rtol, atol=atol, equal_nan=True):
            matches = False
            if verbose:
                diff = np.abs(ref_val - cand_val)
                print(f"{name}: max diff = {np.nanmax(diff):.6e}, avg diff = {np.nanmean(diff):.6e}")
    return matches
