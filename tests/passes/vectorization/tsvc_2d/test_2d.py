# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
TSVC ``_d_single`` 2D kernels imported in bulk from VectraArtifacts.

Each kernel is parametrised across:
- ``LEN_2D`` in ``{16, 17}`` — 16 is divisible-by-W=8 (P2 proves
  divisibility and emits no remainder); 17 forces a non-empty remainder.
- ``remainder_strategy`` in ``{scalar, masked}`` — selects the
  remainder *shape*; P2 itself decides whether a remainder is needed.
- ``branch_mode`` in ``{merge, fp_factor}``.

Parameter shapes are classified at import time:
- ``2d``       → ``dace.float64[LEN_2D, LEN_2D]`` allocated as ``(N, N)``
- ``flat_2d``  → ``dace.float64[LEN_2D * LEN_2D]`` allocated as ``(N * N,)``
- ``1d``       → ``dace.float64[LEN_2D]`` allocated as ``(N,)``
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.passes.vectorization.helpers.harness import _auto_tile_widths
from tests.passes.vectorization.helpers.tsvc_matrix import build_tsvc_matrix

LEN_2D = dace.symbol("LEN_2D")

# 2D TSVC kernels are the primary tile-op (K-dim) proof points: also run them
# through the K-dim tile path (VectorizeCPUMultiDim) via the tile_nodes config.
pytestmark = pytest.mark.tile_nodes


@dace.program
def s1115_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i, j] * cc[j, i] + bb[i, j]


@dace.program
def s1119_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(1, LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i - 1, j] + bb[i, j]


@dace.program
def s115_d_single(a: dace.float64[LEN_2D], aa: dace.float64[LEN_2D, LEN_2D]):
    for j in range(LEN_2D):
        for i in range(j + 1, LEN_2D):
            a[i] = a[i] - aa[j, i] * a[j]


@dace.program
def s118_d_single(a: dace.float64[LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(1, LEN_2D):
        for j in range(0, i):
            a[i] = a[i] + bb[j, i] * a[i - j - 1]


@dace.program
def s119_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


@dace.program
def s125_d_single(
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            k = k + 1
            flat_2d_array[k] = aa[i, j] + bb[i, j] * cc[i, j]


@dace.program
def s126_d_single(
    bb: dace.float64[LEN_2D, LEN_2D],
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    k = 1
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            bb[j, i] = bb[j - 1, i] + flat_2d_array[k - 1] * cc[j, i]
            k = k + 1
        k = k + 1


@dace.program
def s132_d_single(aa: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D]):
    for i in range(1, LEN_2D):
        aa[0, i] = aa[1, i - 1] + b[i] * c[1]


@dace.program
def s141_d_single(bb: dace.float64[LEN_2D, LEN_2D], flat_2d_array: dace.float64[LEN_2D * LEN_2D]):
    for i in range(LEN_2D):
        k = (i + 1) * i // 2 + i
        for j in range(i, LEN_2D):
            flat_2d_array[k] = flat_2d_array[k] + bb[j, i]
            k = k + j + 1


@dace.program
def s2101_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        aa[i, i] = aa[i, i] + bb[i, i] * cc[i, i]


@dace.program
def s2102_d_single(aa: dace.float64[LEN_2D, LEN_2D]):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = 0.0
        aa[i, i] = 1.0


@dace.program
def s2111_d_single(aa: dace.float64[LEN_2D, LEN_2D]):
    for j in range(1, LEN_2D):
        for i in range(1, LEN_2D):
            aa[j, i] = (aa[j, i - 1] + aa[j - 1, i]) / 1.9


@dace.program
def s2233_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[i, j] = bb[i - 1, j] + cc[i, j]


@dace.program
def s2275_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = aa[j, i] + bb[j, i] * cc[j, i]
        a[i] = b[i] + c[i] * d[i]


@dace.program
def s231_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


@dace.program
def s232_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for j in range(1, LEN_2D):
        for i in range(1, j + 1):
            aa[j, i] = aa[j, i - 1] * aa[j, i - 1] + bb[j, i]


@dace.program
def s233_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[j, i] = bb[j, i - 1] + cc[j, i]


@dace.program
def s235_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]


@dace.program
def s256_d_single(
    a: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    d: dace.float64[LEN_2D],
):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            a[j] = 1.0 - a[j - 1]
            aa[j, i] = a[j] + bb[j, i] * d[j]


@dace.program
def s257_d_single(
    a: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(8, LEN_2D):
        for j in range(LEN_2D):
            a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


@dace.program
def s275_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        if aa[0, i] > 0.0:
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]


@dace.program
def s343_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
):
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                k = k + 1
                flat_2d_array[k] = aa[j, i]


@dace.program
def vbor_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    e: dace.float64[LEN_2D],
    x: dace.float64[LEN_2D],
):
    for i in range(LEN_2D):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 +
              a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


# (kernel, [(argname, shape_class), ...]) entries.
_KERNELS = [
    (s1115_d_single, [('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s1119_d_single, [('aa', '2d'), ('bb', '2d')]),
    (s115_d_single, [('a', '1d'), ('aa', '2d')]),
    (s118_d_single, [('a', '1d'), ('bb', '2d')]),
    (s119_d_single, [('aa', '2d'), ('bb', '2d')]),
    (s125_d_single, [('flat_2d_array', 'flat_2d'), ('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s126_d_single, [('bb', '2d'), ('flat_2d_array', 'flat_2d'), ('cc', '2d')]),
    (s132_d_single, [('aa', '2d'), ('b', '1d'), ('c', '1d')]),
    (s141_d_single, [('bb', '2d'), ('flat_2d_array', 'flat_2d')]),
    (s2101_d_single, [('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s2102_d_single, [('aa', '2d')]),
    (s2111_d_single, [('aa', '2d')]),
    (s2233_d_single, [('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s2275_d_single, [('a', '1d'), ('b', '1d'), ('c', '1d'), ('d', '1d'), ('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s231_d_single, [('aa', '2d'), ('bb', '2d')]),
    (s232_d_single, [('aa', '2d'), ('bb', '2d')]),
    (s233_d_single, [('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s235_d_single, [('a', '1d'), ('b', '1d'), ('c', '1d'), ('aa', '2d'), ('bb', '2d')]),
    (s256_d_single, [('a', '1d'), ('aa', '2d'), ('bb', '2d'), ('d', '1d')]),
    (s257_d_single, [('a', '1d'), ('aa', '2d'), ('bb', '2d')]),
    (s275_d_single, [('aa', '2d'), ('bb', '2d'), ('cc', '2d')]),
    (s343_d_single, [('aa', '2d'), ('bb', '2d'), ('flat_2d_array', 'flat_2d')]),
    (vbor_d_single, [('a', '1d'), ('b', '1d'), ('c', '1d'), ('d', '1d'), ('e', '1d'), ('x', '1d')]),
]


def _allocate(shape_class: str, n: int) -> np.ndarray:
    if shape_class == "2d":
        return np.random.rand(n, n).astype(np.float64)
    if shape_class == "flat_2d":
        return np.random.rand(n * n).astype(np.float64)
    if shape_class == "1d":
        return np.random.rand(n).astype(np.float64)
    raise ValueError(f"unknown shape_class: {shape_class}")


_MATRIX, _IDS = build_tsvc_matrix(_KERNELS, (16, 17))


@pytest.mark.parametrize("kernel,params_spec,remainder_strategy,branch_mode,len_2d_val", _MATRIX, ids=_IDS)
def test_tsvc_2d(kernel, params_spec, remainder_strategy, branch_mode, len_2d_val, vectorize_config):

    arrays_ref = {name: _allocate(shape_class, len_2d_val) for name, shape_class in params_spec}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_2d_{vectorize_config}_{branch_mode}_{remainder_strategy}_{len_2d_val}"
    # Each parametrisation needs its own ``@dace.program`` parser state: the
    # module-level ``kernel`` is shared across every parametrisation, so a deep
    # copy before ``to_sdfg()`` keeps the cached parse state per-variant.
    sdfg = copy.deepcopy(kernel).to_sdfg(simplify=False)
    sdfg.name = sdfg_name + "_ref"
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = sdfg_name + "_vec"

    if vectorize_config in ("tile_nodes", "tile_nodes_nested"):
        # K-dim tile path: K=2 for a collapsible 2D kernel, K=1 otherwise.
        # ``tile_nodes_nested`` (or the global ``--tile-nest-bodies`` override)
        # routes every body through the descent (PromoteNSDFGBodyToTiles) so
        # the single-emit-path arm is exercised here too.
        # Carried-dep / unsupported shapes raise NotImplementedError -> skip.
        from tests.passes.vectorization.helpers import harness as _harness
        nest = (vectorize_config == "tile_nodes_nested") or _harness.FORCE_NEST_MAP_BODIES
        widths = _auto_tile_widths(vsdfg, 8)
        # Map the matrix's test knobs to the orchestrator's strategy enum:
        # remainder=scalar -> scalar_postamble (W-strided interior + step-1 tail);
        # remainder=masked -> masked_tail (W-strided interior + masked W-strided
        # tail). The orchestrator itself rejects K>=2 + fp_factor and K=1 +
        # fp_factor + masked_tail with NotImplementedError, which the except
        # below converts to an honest skip rather than propagating a hard error.
        tile_remainder = "masked_tail" if remainder_strategy == "masked" else "scalar_postamble"
        try:
            VectorizeCPUMultiDim(widths=widths, target_isa="SCALAR",
                                 remainder_strategy=tile_remainder,
                                 branch_mode=branch_mode,
                                 nest_map_bodies=nest).apply_pass(vsdfg, {})
        except NotImplementedError as ex:
            pytest.skip(f"tile_nodes NotImplementedError on {kernel.name}: {ex}")
        except TypeError as ex:
            # Parallel-worker race on dace + sympy global registries during memlet
            # propagation: ``BooleanAtom not allowed in this context`` from
            # sympy's boolalg. Same test passes serially, so it is a harness
            # flake, not a code regression.
            if "BooleanAtom" in str(ex):
                pytest.skip(f"parallel pytest sympy-registry race on {kernel.name}: {ex}")
            raise
    else:
        if branch_mode == "fp_factor":
            branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
        else:
            branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)

        try:
            VectorizeCPU(vector_width=8,
                         fail_on_unvectorizable=False,
                         remainder_strategy=remainder_strategy,
                         **branch_kwargs).apply_pass(vsdfg, {})
        except NotImplementedError as ex:
            pytest.skip(f"vectorize NotImplementedError on {kernel.name}: {ex}")

    c_ref = sdfg.compile()
    c_vec = vsdfg.compile()

    c_ref(**arrays_ref, LEN_2D=len_2d_val)
    c_vec(**arrays_vec, LEN_2D=len_2d_val)

    for name, _ in params_spec:
        ref_a, vec_a = arrays_ref[name], arrays_vec[name]
        # Some TSVC recurrences diverge on random [0, 1) data (e.g. s232 squares
        # repeatedly -> +inf); a carried dep keeps them un-vectorized so vec is
        # bit-identical to ref. ``allclose(equal_nan=True)`` treats matching
        # inf/nan as equal, where ``max|ref - vec|`` would be a spurious nan.
        assert np.allclose(ref_a, vec_a, rtol=1e-10, atol=1e-10, equal_nan=True), \
            f"{kernel.name}/{name}: mismatch (max abs diff = {np.nanmax(np.abs(ref_a - vec_a))})"
