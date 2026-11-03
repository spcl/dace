# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""After canonicalization on CPU, an array allocated per map iteration is a forced register (on the stack)."""

import numpy as np

import dace
from dace.codegen.cpf import cpf
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64)
M = dace.symbol("M", dace.int64)


@dace.program
def row_scan_fixed(A: dace.float64[N, 16], out: dace.float64[N, 16]):
    for i in dace.map[0:N]:
        tmp = np.empty((16,), dtype=np.float64)
        for j in dace.map[0:16]:
            tmp[j] = A[i, j] * 2.0
        for j in range(1, 16):
            tmp[j] = tmp[j] * tmp[j - 1] + 1.0
        for j in dace.map[0:16]:
            out[i, j] = tmp[15 - j]


@dace.program
def row_scan_dynamic(A: dace.float64[N, M], out: dace.float64[N, M]):
    for i in dace.map[0:N]:
        tmp = np.empty((M,), dtype=np.float64)
        for j in dace.map[0:M]:
            tmp[j] = A[i, j] * 2.0
        for j in range(1, M):
            tmp[j] = tmp[j] * tmp[j - 1] + 1.0
        for j in dace.map[0:M]:
            out[i, j] = tmp[M - 1 - j]


def canonical(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    finalize_for_target(sdfg, "cpu")
    return sdfg


def storage_of(sdfg: dace.SDFG, name: str) -> dace.StorageType:
    (desc,) = [desc for _, aname, desc in sdfg.arrays_recursive() if aname == name]
    return desc.storage


def reference(A: np.ndarray) -> np.ndarray:
    tmp = 2.0 * A
    for j in range(1, A.shape[1]):
        tmp[:, j] = tmp[:, j] * tmp[:, j - 1] + 1.0
    return tmp[:, ::-1]


def test_a_map_scoped_transient_is_a_forced_register():
    storage = storage_of(canonical(row_scan_fixed), "tmp")
    assert storage == dace.StorageType.Register and storage.force


def test_a_dynamically_sized_map_scoped_transient_is_a_stack_array_that_computes_the_reference():
    sdfg = canonical(row_scan_dynamic)
    storage = storage_of(sdfg, "tmp")
    assert storage == dace.StorageType.Register and storage.force
    code = cpf(sdfg)
    assert "double tmp[cpf_max(1, M)];" in code
    assert "new double" not in code
    A = np.random.default_rng(0).random((5, 7))
    out = np.zeros_like(A)
    sdfg(A=A, out=out, N=5, M=7)
    np.testing.assert_allclose(out, reference(A), rtol=1e-12)


if __name__ == "__main__":
    test_a_map_scoped_transient_is_a_forced_register()
    test_a_dynamically_sized_map_scoped_transient_is_a_stack_array_that_computes_the_reference()
