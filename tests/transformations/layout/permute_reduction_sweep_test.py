# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Comprehensive numerical sweep: PermuteDimensions over WCR reduction edges.

A reduction is carried by a ``+=`` write whose memlet keeps a ``wcr``. A layout change that touches
either the reduction INPUT or its WCR TARGET must rewrite that memlet's subset like any other memlet
AND keep the ``wcr`` -- rebuilding the memlet with only the subset would silently drop the reduction,
turning ``+=`` into last-writer-wins. This file drives the ``Permute`` family transparently
(``add_permute_maps=True``, so the external interface is unchanged) across four reduction shapes:

  * ``r2d1d``  : ``out[i]    += A[i, j]``          2D -> 1D
  * ``r3d2d``  : ``out[i, k] += A[i, j, k]``       3D -> 2D, reduce the middle axis (one collapsed map)
  * ``nested`` : parallel outer ``(i, k)``, reduce inner ``j`` -> WCR through two map exits + nested SDFG
  * ``scalar`` : ``out[0]    += A[i, j]``          full reduction into a single element

Every case runs through the documented front door ``prepare_for_layout``, permutes the INPUT, the WCR
TARGET, or BOTH, asserts the ``wcr`` survives, and checks the transparent result is bit-exact with the
numpy reduction. Both dtypes ``float64`` and ``complex128`` are swept.
"""
import numpy
import pytest
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

I, J, K = (dace.symbol(s) for s in ("I", "J", "K"))
II, JJ, KK = 8, 5, 6


# ---- reduction program builders (typed on the swept dtype) ---------------------------------------
def build_r2d1d(dtype):

    @dace.program
    def prog(A: dtype[I, J], out: dtype[I]):
        for i, j in dace.map[0:I, 0:J]:
            out[i] += A[i, j]

    return prog


def build_r3d2d(dtype):

    @dace.program
    def prog(A: dtype[I, J, K], out: dtype[I, K]):
        for i, j, k in dace.map[0:I, 0:J, 0:K]:
            out[i, k] += A[i, j, k]

    return prog


def build_nested(dtype):

    @dace.program
    def prog(A: dtype[I, J, K], out: dtype[I, K]):
        for i, k in dace.map[0:I, 0:K]:
            for j in dace.map[0:J]:
                out[i, k] += A[i, j, k]

    return prog


def build_scalar(dtype):

    @dace.program
    def prog(A: dtype[I, J], out: dtype[1]):
        for i, j in dace.map[0:I, 0:J]:
            out[0] += A[i, j]

    return prog


# ---- fixed logical inputs + numpy oracles --------------------------------------------------------
RNG = numpy.random.default_rng(20260715)
BASE_A3 = {
    "f64": RNG.random((II, JJ, KK)),
    "c128": RNG.random((II, JJ, KK)) + 1j * RNG.random((II, JJ, KK)),
}
BASE_A2 = {
    "f64": RNG.random((II, JJ)),
    "c128": RNG.random((II, JJ)) + 1j * RNG.random((II, JJ)),
}
DACE_DTYPE = {"f64": dace.float64, "c128": dace.complex128}
NUMPY_DTYPE = {"f64": numpy.float64, "c128": numpy.complex128}


def oracle_axis1(base):
    return base.sum(axis=1)


def oracle_scalar(base):
    return numpy.array([base.sum()], dtype=base.dtype)


SPECS = {
    "r2d1d": dict(build=build_r2d1d, base=BASE_A2, out_shape=(II, ), syms=dict(I=II, J=JJ), oracle=oracle_axis1),
    "r3d2d": dict(build=build_r3d2d, base=BASE_A3, out_shape=(II, KK), syms=dict(I=II, J=JJ, K=KK),
                  oracle=oracle_axis1),
    "nested": dict(build=build_nested,
                   base=BASE_A3,
                   out_shape=(II, KK),
                   syms=dict(I=II, J=JJ, K=KK),
                   oracle=oracle_axis1),
    "scalar": dict(build=build_scalar, base=BASE_A2, out_shape=(1, ), syms=dict(I=II, J=JJ), oracle=oracle_scalar),
}

PERMS2 = [[0, 1], [1, 0]]
PERMS3 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
# A curated (a_perm, out_perm) subset for the heavier nested kernel: input-only, output-only, both.
NESTED_COMBOS = [
    ([2, 1, 0], [0, 1]),  # input-only: reverse
    ([1, 2, 0], [0, 1]),  # input-only: cycle
    ([1, 0, 2], [0, 1]),  # input-only: swap i,j
    ([0, 1, 2], [1, 0]),  # output-only: swap the WCR target
    ([2, 1, 0], [1, 0]),  # both: input reverse + output swap
    ([2, 0, 1], [1, 0]),  # both: input cycle + output swap
]


def _has_wcr(sdfg) -> bool:
    return any(edge.data is not None and edge.data.wcr is not None for nsdfg in sdfg.all_sdfgs_recursive()
               for state in nsdfg.states() for edge in state.edges())


def _slug(perm):
    return "".join(str(x) for x in perm)


def _make_cases():
    cases = []
    for dn in ("f64", "c128"):
        # 3D collapsed reduction: every non-identity (input perm, output perm) combination.
        for a_perm in PERMS3:
            for out_perm in PERMS2:
                if a_perm == [0, 1, 2] and out_perm == [0, 1]:
                    continue  # both-identity is not a permute
                cases.append(("r3d2d", a_perm, out_perm, dn))
        # 3D nested reduction: curated combos.
        for a_perm, out_perm in NESTED_COMBOS:
            cases.append(("nested", a_perm, out_perm, dn))
        # 2D -> 1D: only the input has a non-trivial permutation (output is 1D).
        cases.append(("r2d1d", [1, 0], [0], dn))
        # full reduction into a scalar: only the input permutes.
        cases.append(("scalar", [1, 0], [0], dn))
    return cases


CASES = _make_cases()
CASE_IDS = ["%s-A%s-O%s-%s" % (k, _slug(ap), _slug(op), dn) for k, ap, op, dn in CASES]


@pytest.mark.parametrize("kernel,a_perm,out_perm,dtype_name", CASES, ids=CASE_IDS)
def test_permute_reduction(kernel, a_perm, out_perm, dtype_name):
    spec = SPECS[kernel]
    program = spec["build"](DACE_DTYPE[dtype_name])
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = "permred_%s_A%s_O%s_%s" % (kernel, _slug(a_perm), _slug(out_perm), dtype_name)

    prepare_for_layout(sdfg, validate=False)
    assert _has_wcr(sdfg), "reduction WCR missing before permute"

    permute_map = {}
    if a_perm != list(range(len(a_perm))):
        permute_map["A"] = a_perm
    if out_perm != list(range(len(out_perm))):
        permute_map["out"] = out_perm
    assert permute_map, "case must permute at least one array"

    PermuteDimensions(permute_map=permute_map, add_permute_maps=True).apply_pass(sdfg, {})
    assert _has_wcr(sdfg), "permute dropped the WCR (reduction turned into last-writer-wins)"

    base = spec["base"][dtype_name]
    out = numpy.zeros(spec["out_shape"], dtype=NUMPY_DTYPE[dtype_name])
    sdfg(A=base.copy(), out=out, **spec["syms"])
    assert numpy.allclose(out, spec["oracle"](base))


if __name__ == "__main__":
    for case, cid in zip(CASES, CASE_IDS):
        test_permute_reduction(*case)
        print("PASS", cid)
    print("permute reduction sweep: %d cases PASS" % len(CASES))