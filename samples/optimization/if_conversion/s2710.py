# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC ``s2710`` through the CPU canonical form, with and without IfConvertParallelMapBodies.

Writes both CPF renderings to the working directory (``s2710_branchy.cpp``, ``s2710_select.cpp``), checks both
SDFGs against a NumPy reference, and times them. Set ``OMP_NUM_THREADS`` before running; see README.md.
"""

import os
import pathlib
import time
from unittest import mock

import numpy as np

import dace
from dace.codegen.cpf import cpf
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization import pipeline as cpu_pipeline

LEN_1D = dace.symbol("LEN_1D", dace.int64)
OUT = pathlib.Path.cwd()


@dace.program
def s2710(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    x: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            if LEN_1D > 10:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if x[0] > 0.0:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


def reference(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    a, b, c, d, e, x = (arrays[k].copy() for k in "abcdex")
    take = a > b
    a_new = np.where(take, a + b * d, a)
    b_new = np.where(take, b, a + e * e)
    then_c = c + d * d if a.size > 10 else d * e + 1.0
    else_c = a + d * d if x[0] > 0.0 else c + e * e
    return {"a": a_new, "b": b_new, "c": np.where(take, then_c, else_c)}


def canonical(name: str, if_convert: bool) -> dace.SDFG:
    sdfg = s2710.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    if if_convert:
        finalize_for_target(sdfg, "cpu")
    else:
        with mock.patch.object(cpu_pipeline.IfConvertParallelMapBodies, "apply_pass", return_value=None):
            finalize_for_target(sdfg, "cpu")
    (OUT / f"{name}.cpp").write_text(cpf(sdfg))
    return sdfg


def inputs(n: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)  # random a, b: the branch on a[i] > b[i] is taken half the time
    return {k: rng.random(n) for k in "abcdex"}


def median_ms(compiled, n: int, repeat: int) -> float:
    times = []
    for rep in range(repeat):
        arrays = inputs(n, seed=rep)
        start = time.perf_counter()
        compiled(**arrays, LEN_1D=n)
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times))


def main(n: int = 43_397_065, repeat: int = 7) -> None:
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, LEN_1D={n} (hpcagent-bench preset M)")
    for name, if_convert in (("s2710_branchy", False), ("s2710_select", True)):
        compiled = canonical(name, if_convert).compile()
        arrays = inputs(n)
        expected = reference(arrays)
        compiled(**arrays, LEN_1D=n)
        for key, value in expected.items():
            np.testing.assert_allclose(arrays[key], value, rtol=1e-14, atol=0.0)
        print(f"{name}: correct, median {median_ms(compiled, n, repeat):.1f} ms -> {OUT / (name + '.cpp')}")


if __name__ == "__main__":
    main()
