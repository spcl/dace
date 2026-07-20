# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""TEMPORARY repro driver -- remove before the PR is finalised.

Runs npbench's polybench/doitgen through the two pipelines the benchmark harness uses (``strict``
= parse + simplify, and ``autoopt`` = simplify + auto_optimize) and checks the result against
numpy. Kernel and initializer are verbatim from npbench main:

  npbench/benchmarks/polybench/doitgen/doitgen_dace.py
  npbench/benchmarks/polybench/doitgen/doitgen.py
  npbench/benchmarks/polybench/doitgen/doitgen_numpy.py

The ``(NQ, 1, NP)`` reshape is the point of the benchmark: it is what the in-tree test
``tests/npbench/polybench/doitgen_test.py`` stopped exercising in afd0efe48, which is why CI has
been green while npbench doitgen fails.

Usage: python doitgen_repro.py [NR NQ NP]      (defaults to 3 4 5)
"""
import copy
import sys
import traceback

import numpy as np

import dace as dc
import dace.dtypes as dtypes
import dace.transformation.auto.auto_optimize as opt

NR, NQ, NP = (dc.symbol(s, dtype=dc.int64) for s in ('NR', 'NQ', 'NP'))


@dc.program
def kernel(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))


def initialize(nr, nq, np_, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: ((i * j + k) % np_) / np_, (nr, nq, np_), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % np_) / np_, (np_, np_), dtype=datatype)
    return A, C4


def numpy_kernel(nr, nq, np_, A, C4):
    A[:] = np.reshape(np.reshape(A, (nr, nq, 1, np_)) @ C4, (nr, nq, np_))


def check(tag, got, ref):
    if got.shape != ref.shape:
        print(f"[{tag}] SHAPE MISMATCH got={got.shape} ref={ref.shape}")
        return False
    diff = np.abs(got - ref)
    ok = np.allclose(got, ref, rtol=1e-12, atol=1e-12)
    print(f"[{tag}] {'OK' if ok else 'MISMATCH'}  max_abs={diff.max():.3e}")
    if not ok:
        bad = np.argwhere(~np.isclose(got, ref, rtol=1e-12, atol=1e-12))
        print(f"[{tag}]   nbad={len(bad)}/{ref.size}  first={bad[:3].tolist()}")
    return ok


def run(tag, build, nr, nq, np_):
    A, C4 = initialize(nr, nq, np_)
    ref = A.copy()
    numpy_kernel(nr, nq, np_, ref, C4)
    try:
        sdfg = build()
    except Exception:
        print(f"[{tag}] BUILD FAILED")
        traceback.print_exc()
        return False
    got = A.copy()
    try:
        sdfg(A=got, C4=C4, NR=nr, NQ=nq, NP=np_)
    except Exception:
        print(f"[{tag}] RUN FAILED")
        traceback.print_exc()
        return False
    return check(tag, got, ref)


def main():
    nr, nq, np_ = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (3, 4, 5)
    print(f"=== doitgen NR={nr} NQ={nq} NP={np_}  dace={dc.__file__}")

    base = kernel.to_sdfg(simplify=False)

    def strict():
        sdfg = copy.deepcopy(base)
        sdfg._name = "strict"
        sdfg.simplify()
        return sdfg

    def autoopt():
        sdfg = copy.deepcopy(base)
        sdfg._name = "autoopt"
        sdfg.simplify()
        opt.auto_optimize(sdfg, dtypes.DeviceType.CPU, symbols=dict(NR=nr, NQ=nq, NP=np_))
        return sdfg

    strict_ok = run("strict ", strict, nr, nq, np_)
    autoopt_ok = run("autoopt", autoopt, nr, nq, np_)
    print(f"=== RESULT strict={'PASS' if strict_ok else 'FAIL'} autoopt={'PASS' if autoopt_ok else 'FAIL'}")
    return 0 if (strict_ok and autoopt_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
