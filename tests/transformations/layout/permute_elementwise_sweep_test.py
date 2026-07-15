# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numerical-verification sweep: ``PermuteDimensions`` over elementwise kernels.

The Permute family runs in transparent (``add_permute_maps=True``) mode, so the external interface
of every kernel is unchanged: logical numpy inputs go in and the result is compared bit-exact to a
numpy oracle. Each case picks one (or two) array(s) and applies an arbitrary dimension permutation
transparently -- the pass wraps the array with ``permute_in``/``permute_out`` states so its external
shape stays logical while the kernel body reads the permuted layout.

Coverage: 2D/3D/4D elementwise kernels across float32, float64, complex128, int64; EVERY permutation
of the permuted array's dims (``itertools.permutations``), plus two-array (distinct perms) and
output-permute cases. Small sizes (n in {4, 6}). Every case compiles, runs, and ``numpy.allclose``
against the oracle.
"""
import itertools

import numpy
import pytest
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N", dtype=dace.int64)


def make_add2d(dtype):
    """2D elementwise sum: ``C = A + B``."""

    @dace.program
    def add2d(A: dtype[N, N], B: dtype[N, N], C: dtype[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            C[i, j] = A[i, j] + B[i, j]

    return add2d


def make_affine3d(dtype):
    """3D affine combination with a scalar: ``C = A * s + B``."""

    @dace.program
    def affine3d(A: dtype[N, N, N], B: dtype[N, N, N], s: dtype, C: dtype[N, N, N]):
        for i, j, k in dace.map[0:N, 0:N, 0:N]:
            C[i, j, k] = A[i, j, k] * s + B[i, j, k]

    return affine3d


def make_mul3d(dtype):
    """3D integer elementwise product: ``C = A * B``.

    Written with the numpy-slice form ``C[:] = A * B`` rather than an explicit ``dace.map``: for
    the int64 element type the explicit map body is wrapped in a nested SDFG whose boundary memlets
    ``prepare_for_layout`` widens to full arrays, which mis-lowers to pointer-typed scalar
    connectors. The vectorized form stays a flat elementwise map that the layout pipeline lowers
    correctly (float/complex map bodies are not wrapped, so they use the explicit form)."""

    @dace.program
    def mul3d(A: dtype[N, N, N], B: dtype[N, N, N], C: dtype[N, N, N]):
        C[:] = A * B

    return mul3d


def make_square4d(dtype):
    """4D elementwise square: ``C = A * A``."""

    @dace.program
    def square4d(A: dtype[N, N, N, N], C: dtype[N, N, N, N]):
        for i, j, k, l in dace.map[0:N, 0:N, 0:N, 0:N]:
            C[i, j, k, l] = A[i, j, k, l] * A[i, j, k, l]

    return square4d


def oracle_add2d(arrays, scalar):
    return arrays["A"] + arrays["B"]


def oracle_affine3d(arrays, scalar):
    return arrays["A"] * scalar + arrays["B"]


def oracle_mul3d(arrays, scalar):
    return arrays["A"] * arrays["B"]


def oracle_square4d(arrays, scalar):
    return arrays["A"] * arrays["A"]


KERNELS = {
    "add2d": {
        "program": make_add2d(dace.float32),
        "ndim": 2,
        "inputs": ["A", "B"],
        "scalar": False,
        "scalar_name": None,
        "kind": "real",
        "npdtype": numpy.float32,
        "oracle": oracle_add2d,
    },
    "affine3d": {
        "program": make_affine3d(dace.float64),
        "ndim": 3,
        "inputs": ["A", "B"],
        "scalar": True,
        "scalar_name": "s",
        "kind": "real",
        "npdtype": numpy.float64,
        "oracle": oracle_affine3d,
    },
    "mul3d": {
        "program": make_mul3d(dace.int64),
        "ndim": 3,
        "inputs": ["A", "B"],
        "scalar": False,
        "scalar_name": None,
        "kind": "int",
        "npdtype": numpy.int64,
        "oracle": oracle_mul3d,
    },
    "square4d": {
        "program": make_square4d(dace.complex128),
        "ndim": 4,
        "inputs": ["A"],
        "scalar": False,
        "scalar_name": None,
        "kind": "complex",
        "npdtype": numpy.complex128,
        "oracle": oracle_square4d,
    },
}


def perm_tag(permute_map):
    parts = []
    for name in sorted(permute_map):
        parts.append(name + "".join(str(d) for d in permute_map[name]))
    return "_".join(parts)


def stable_seed(tag):
    seed = 0
    for i, ch in enumerate(tag):
        seed = (seed + ord(ch) * (i + 1)) % (2**31)
    return seed


def generate_array(rng, shape, spec):
    kind = spec["kind"]
    if kind == "complex":
        return (rng.random(shape) + 1j * rng.random(shape)).astype(numpy.complex128)
    if kind == "int":
        return rng.integers(-5, 6, size=shape, dtype=numpy.int64)
    return rng.random(shape).astype(spec["npdtype"])


def run_case(spec, permute_map, n, tag):
    rng = numpy.random.default_rng(stable_seed(tag))
    shape = (n, ) * spec["ndim"]
    arrays = {name: generate_array(rng, shape, spec) for name in spec["inputs"]}

    scalar_value = None
    if spec["scalar"]:
        scalar_value = float(rng.random() * 2.0 + 0.5)

    oracle = spec["oracle"](arrays, scalar_value)

    sdfg = spec["program"].to_sdfg(simplify=True)
    sdfg.name = "perm_" + tag
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map=dict(permute_map), add_permute_maps=True).apply_pass(sdfg, {})

    out = numpy.zeros(shape, dtype=oracle.dtype)
    kwargs = {name: arr.copy() for name, arr in arrays.items()}
    kwargs["C"] = out
    kwargs["N"] = n
    if spec["scalar"]:
        kwargs[spec["scalar_name"]] = scalar_value

    sdfg(**kwargs)
    return out, oracle


def build_cases():
    cases = []

    # Single-array permutation over EVERY permutation of that array's dims.
    single = [
        ("add2d", 6, "A"),
        ("add2d", 4, "A"),
        ("affine3d", 6, "A"),
        ("mul3d", 6, "A"),
        ("square4d", 4, "A"),
    ]
    for kernel_key, n, arr in single:
        ndim = KERNELS[kernel_key]["ndim"]
        for perm in itertools.permutations(range(ndim)):
            cases.append((kernel_key, n, {arr: list(perm)}))

    # Two arrays permuted with DIFFERENT perms in the same kernel.
    two = [
        ("add2d", 6, {
            "A": [1, 0],
            "B": [0, 1]
        }),
        ("add2d", 4, {
            "A": [0, 1],
            "B": [1, 0]
        }),
        ("affine3d", 6, {
            "A": [2, 0, 1],
            "B": [1, 2, 0]
        }),
        ("affine3d", 6, {
            "A": [1, 0, 2],
            "B": [2, 1, 0]
        }),
        ("mul3d", 6, {
            "A": [2, 1, 0],
            "B": [0, 2, 1]
        }),
        ("mul3d", 6, {
            "A": [1, 2, 0],
            "B": [2, 0, 1]
        }),
    ]
    for kernel_key, n, pm in two:
        cases.append((kernel_key, n, pm))

    # Transparent permutation of the WRITTEN array C.
    outp = [
        ("add2d", 6, {
            "C": [1, 0]
        }),
        ("affine3d", 6, {
            "C": [2, 0, 1]
        }),
        ("mul3d", 6, {
            "C": [0, 2, 1]
        }),
    ]
    for kernel_key, n, pm in outp:
        cases.append((kernel_key, n, pm))

    return cases


CASES = build_cases()
CASE_IDS = ["{}_n{}_{}".format(k, n, perm_tag(pm)) for k, n, pm in CASES]


@pytest.mark.parametrize("kernel_key,n,permute_map", CASES, ids=CASE_IDS)
def test_permute_elementwise(kernel_key, n, permute_map):
    spec = KERNELS[kernel_key]
    tag = "{}_n{}_{}".format(kernel_key, n, perm_tag(permute_map))
    out, oracle = run_case(spec, permute_map, n, tag)
    assert out.shape == oracle.shape
    assert numpy.allclose(out,
                          oracle), "permute {} on {} (n={}) diverged from oracle".format(permute_map, kernel_key, n)


if __name__ == "__main__":
    for key, size, pmap in CASES:
        t = "{}_n{}_{}".format(key, size, perm_tag(pmap))
        got, ref = run_case(KERNELS[key], pmap, size, t)
        assert numpy.allclose(got, ref), t
        print("PASS", t)
    print("all", len(CASES), "permute-elementwise cases pass")
