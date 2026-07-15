# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transparent PERMUTE sweep over MULTI-ARRAY kernels (numerical verification).

Every case builds a multi-array kernel, applies :class:`PermuteDimensions` in transparent mode
(``add_permute_maps=True``, so the external interface is UNCHANGED and logical inputs are passed
straight through), runs the compiled SDFG and checks it is bit-exact with a numpy oracle. A case
that only validates is worthless here -- each parametrized case COMPILES, RUNS and ``allclose``es.

Three kernel families, each in a 2D and (where meaningful) a 3D flavour so permutations are
non-trivial, with DIFFERENT subsets of the arrays permuted per case (permute one array, two arrays
with different perms, all three, and 1D-array identity permutes):

  * saxpy   y = a*x + y        (x, y read-modify-write on y)
  * blend   C = w0*A + w1*B     (A, B inputs, C output)
  * matvec  out[i] += A[i,j]*v[j]  (2D A + 1D v -> 1D out, a WCR reduction target)

The dimensions are deliberately distinct (M != N != P) so a wrong axis rewrite cannot hide behind a
square shape.
"""
import numpy
import pytest
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

M, N, P = (dace.symbol(s) for s in ("M", "N", "P"))


# ---- kernels ------------------------------------------------------------------------------------
@dace.program
def saxpy2d(a: dace.float64, x: dace.float64[M, N], y: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        y[i, j] = a * x[i, j] + y[i, j]


@dace.program
def saxpy3d(a: dace.float64, x: dace.float64[M, N, P], y: dace.float64[M, N, P]):
    for i, j, k in dace.map[0:M, 0:N, 0:P]:
        y[i, j, k] = a * x[i, j, k] + y[i, j, k]


@dace.program
def blend2d(w0: dace.float64, w1: dace.float64, A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        C[i, j] = w0 * A[i, j] + w1 * B[i, j]


@dace.program
def blend3d(w0: dace.float64, w1: dace.float64, A: dace.float64[M, N, P], B: dace.float64[M, N, P],
            C: dace.float64[M, N, P]):
    for i, j, k in dace.map[0:M, 0:N, 0:P]:
        C[i, j, k] = w0 * A[i, j, k] + w1 * B[i, j, k]


@dace.program
def matvec(A: dace.float64[M, N], v: dace.float64[N], out: dace.float64[M]):
    for i, j in dace.map[0:M, 0:N]:
        out[i] += A[i, j] * v[j]


# ---- per-kernel problem builders ----------------------------------------------------------------
# Each builder returns (program, symbols, args, output_names, oracle). ``args`` holds LOGICAL numpy
# inputs (read-only inputs plus zeroed / initial outputs); ``oracle`` is computed from the logical
# inputs BEFORE the run, so an in-place output being mutated cannot corrupt the reference.
DIMS_2D = {"M": 6, "N": 8}
DIMS_3D = {"M": 4, "N": 5, "P": 6}


def build_problem(kernel, seed):
    rng = numpy.random.default_rng(seed)
    if kernel == "saxpy2d":
        m, n = DIMS_2D["M"], DIMS_2D["N"]
        a, x, y = 2.5, rng.random((m, n)), rng.random((m, n))
        return saxpy2d, dict(DIMS_2D), {"a": a, "x": x, "y": y}, ["y"], {"y": a * x + y}
    if kernel == "saxpy3d":
        m, n, p = DIMS_3D["M"], DIMS_3D["N"], DIMS_3D["P"]
        a, x, y = -1.75, rng.random((m, n, p)), rng.random((m, n, p))
        return saxpy3d, dict(DIMS_3D), {"a": a, "x": x, "y": y}, ["y"], {"y": a * x + y}
    if kernel == "blend2d":
        m, n = DIMS_2D["M"], DIMS_2D["N"]
        w0, w1, A, B = 0.3, 0.7, rng.random((m, n)), rng.random((m, n))
        C = numpy.zeros((m, n))
        return blend2d, dict(DIMS_2D), {"w0": w0, "w1": w1, "A": A, "B": B, "C": C}, ["C"], {"C": w0 * A + w1 * B}
    if kernel == "blend3d":
        m, n, p = DIMS_3D["M"], DIMS_3D["N"], DIMS_3D["P"]
        w0, w1, A, B = 1.5, -0.5, rng.random((m, n, p)), rng.random((m, n, p))
        C = numpy.zeros((m, n, p))
        return blend3d, dict(DIMS_3D), {"w0": w0, "w1": w1, "A": A, "B": B, "C": C}, ["C"], {"C": w0 * A + w1 * B}
    if kernel == "matvec":
        m, n = DIMS_2D["M"], DIMS_2D["N"]
        A, v = rng.random((m, n)), rng.random(n)
        out = numpy.zeros(m)
        return matvec, dict(DIMS_2D), {"A": A, "v": v, "out": out}, ["out"], {"out": A @ v}
    raise ValueError(kernel)


# ---- the sweep ----------------------------------------------------------------------------------
# Each case is (kernel, permute_map). The permute_map names only the subset of arrays to relayout;
# arrays absent from it keep their logical layout. 1D arrays (matvec's v, out) carry the identity
# permute [0] to exercise that path transparently.
# yapf: disable
CASES = [
    # saxpy2d: permute x, y, both, and a mixed identity+swap
    ("saxpy2d", {"x": [1, 0]}),
    ("saxpy2d", {"y": [1, 0]}),
    ("saxpy2d", {"x": [1, 0], "y": [1, 0]}),
    ("saxpy2d", {"x": [0, 1], "y": [1, 0]}),
    # saxpy3d: every non-identity permutation of x, several of y, mixed pairs
    ("saxpy3d", {"x": [0, 2, 1]}),
    ("saxpy3d", {"x": [1, 0, 2]}),
    ("saxpy3d", {"x": [2, 1, 0]}),
    ("saxpy3d", {"x": [1, 2, 0]}),
    ("saxpy3d", {"x": [2, 0, 1]}),
    ("saxpy3d", {"y": [2, 1, 0]}),
    ("saxpy3d", {"y": [1, 2, 0]}),
    ("saxpy3d", {"y": [2, 0, 1]}),
    ("saxpy3d", {"x": [2, 1, 0], "y": [0, 2, 1]}),
    ("saxpy3d", {"x": [1, 2, 0], "y": [2, 0, 1]}),
    ("saxpy3d", {"x": [0, 2, 1], "y": [0, 2, 1]}),
    # blend2d: permute each of the three arrays and every non-empty subset
    ("blend2d", {"A": [1, 0]}),
    ("blend2d", {"B": [1, 0]}),
    ("blend2d", {"C": [1, 0]}),
    ("blend2d", {"A": [1, 0], "B": [1, 0]}),
    ("blend2d", {"A": [1, 0], "C": [1, 0]}),
    ("blend2d", {"B": [1, 0], "C": [1, 0]}),
    ("blend2d", {"A": [1, 0], "B": [1, 0], "C": [1, 0]}),
    # blend3d: single, different-per-array pairs and triples
    ("blend3d", {"A": [2, 1, 0]}),
    ("blend3d", {"B": [1, 2, 0]}),
    ("blend3d", {"C": [0, 2, 1]}),
    ("blend3d", {"A": [2, 1, 0], "B": [0, 2, 1]}),
    ("blend3d", {"A": [1, 0, 2], "B": [2, 0, 1], "C": [0, 2, 1]}),
    ("blend3d", {"A": [2, 0, 1], "C": [1, 2, 0]}),
    ("blend3d", {"A": [0, 2, 1], "B": [0, 2, 1], "C": [0, 2, 1]}),
    ("blend3d", {"B": [2, 1, 0], "C": [2, 1, 0]}),
    # matvec: permute the 2D A (the only non-trivial axis); 1D v/out are identity permutes
    ("matvec", {"A": [1, 0]}),
    ("matvec", {"A": [0, 1]}),
    ("matvec", {"A": [1, 0], "v": [0]}),
    ("matvec", {"A": [1, 0], "v": [0], "out": [0]}),
    ("matvec", {"v": [0]}),
]
# yapf: enable


def case_id(kernel, permute_map):
    parts = [name + "".join(str(d) for d in perm) for name, perm in permute_map.items()]
    return kernel + "-" + "_".join(parts)


@pytest.mark.parametrize("kernel, permute_map", CASES, ids=[case_id(k, pm) for k, pm in CASES])
def test_permute_multiarray(kernel, permute_map):
    program, symbols, args, output_names, oracle = build_problem(kernel, seed=len(kernel) + len(permute_map))

    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map=permute_map, add_permute_maps=True).apply_pass(sdfg, {})

    call = {name: (value.copy() if isinstance(value, numpy.ndarray) else value) for name, value in args.items()}
    call.update(symbols)
    sdfg(**call)

    for name in output_names:
        assert numpy.allclose(call[name], oracle[name]), f"{kernel} {permute_map}: output {name} mismatch"


if __name__ == "__main__":
    for kernel_name, pmap in CASES:
        test_permute_multiarray(kernel_name, pmap)
    print(f"permute multi-array sweep PASS ({len(CASES)} cases)")