# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A cast-of-constant scalar (``dace.float16(0.125)``) is a narrowed compile-time constant.

The vectorizer must feed it to the tile intrinsic as a SINGLE-ELEMENT broadcast operand
(``_bc[1] = {(T)(const)}`` + ``tile_binop<..., true, ...>``), NOT widen it into a W-element
tile materialised by a preceding per-lane fill loop. So ``dace.float16(0.125) * b`` must
vectorize to the same broadcast shape as the un-cast literal ``0.125 * b``, including the
separate-statement form ``k = 0.125; b * dace.float16(k)`` that previously widened the
constant into a fill tile.

The see-through is restricted to SAME-DOMAIN conversions (fp -> fp among float16/32/64, or
int -> int); a cross-domain fp <-> int cast stays a real per-lane conversion. These tests
cover the emitted-code shape (broadcast flag, no fill), value-exactness vs numpy, and the
domain classifier directly.
"""
import re

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import (is_same_domain_constant,
                                                                                   numeric_constant_domain)
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy, BranchMode
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")
M = 64  # exact multiple of width 2

#: A per-lane constant fill loop ``X[__l0] = <numeric-or-cast const>;`` -- the SLOW path this
#: fix removes for a same-domain constant broadcast.
_CONST_FILL_RE = re.compile(r"\[__l\d+\]\s*=\s*(?:dace::\w+\(\(?\s*[-+0-9.]|\(?\s*[-+0-9.]+\s*\)?\s*;)")
#: ``tile_binop<T, W, 'op', bA, bB, bC>`` -- capture the three operand broadcast flags.
_BINOP_FLAGS_RE = re.compile(r"tile_binop<[^>]*'.'\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*>")


def _config(expand: bool) -> VectorizeConfig:
    return VectorizeConfig(widths=(2, ),
                           target_isa=ISA.SCALAR,
                           remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE,
                           branch_mode=BranchMode.MERGE,
                           loop_to_map_permissive=False,
                           scalar_remainder_emit="tile_k1",
                           expand_tile_nodes=expand)


def _vectorize(prog, expand: bool = True) -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(_config(expand)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def _emitted_code(sdfg: dace.SDFG) -> str:
    return "\n".join(c.clean_code for c in sdfg.generate_code())


def _has_const_fill_loop(code: str) -> bool:
    return any(_CONST_FILL_RE.search(ln) for ln in code.splitlines())


def _binop_broadcast_flags(code: str):
    return [tuple(f == "true" for f in m) for m in _BINOP_FLAGS_RE.findall(code)]


# --------------------------- kernels (module scope) ---------------------------
@dace.program
def _cast_inline16(A: dace.float16[N], C: dace.float16[N]):
    # The cast-of-constant directly inline in the binop.
    for i in dace.map[0:N]:
        C[i] = dace.float16(0.125) * A[i]


@dace.program
def _cast_separate16(A: dace.float16[N], C: dace.float16[N]):
    # A bare double constant fed through a same-domain-narrowing cast -- the form that
    # previously widened the constant into a per-lane fill tile.
    for i in dace.map[0:N]:
        k = 0.125
        C[i] = A[i] * dace.float16(k)


@dace.program
def _const_store64(C: dace.float64[N]):
    # A pure constant store (no compute consumer) still materialises a tile op -- the
    # constant is the stored value, not a broadcast operand.
    for i in dace.map[0:N]:
        C[i] = 0.5


# --------------------------- emitted-shape tests ---------------------------
@pytest.mark.parametrize("prog", [_cast_inline16, _cast_separate16])
def test_cast_constant_broadcasts_no_fill(prog):
    """The cast-of-constant vectorizes to a broadcast operand (a ``true`` flag on the
    ``tile_binop``) with NO per-lane constant fill loop."""
    code = _emitted_code(_vectorize(prog))
    assert not _has_const_fill_loop(code), "same-domain constant must not widen into a per-lane fill tile"
    flags = _binop_broadcast_flags(code)
    assert flags, "expected a tile_binop in the emitted code"
    assert any(any(f) for f in flags), "the constant operand must feed the intrinsic as a broadcast (true flag)"


def test_cast_and_separate_forms_emit_equivalent_shape():
    """``dace.float16(0.125) * b`` and ``k = 0.125; b * dace.float16(k)`` lower to the
    SAME tile-op shape: exactly one ``TileBinop`` whose constant operand is a Scalar/Symbol
    broadcast, never a widened ``TileLoad(src_kind='Symbol')`` constant tile."""
    for prog in (_cast_inline16, _cast_separate16):
        sdfg = _vectorize(prog, expand=False)
        binops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop)]
        assert len(binops) == 1, f"{prog.name}: expected exactly one TileBinop, got {len(binops)}"
        b = binops[0]
        assert "Scalar" in (b.kind_a, b.kind_b) or "Symbol" in (b.kind_a, b.kind_b), \
            f"{prog.name}: the constant operand must be a Scalar/Symbol broadcast, got kinds {(b.kind_a, b.kind_b)}"
        # No constant materialised as a per-lane tile via a Symbol-source TileLoad.
        const_fills = [
            n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad) and n.src_kind == "Symbol"
            and numeric_constant_domain(str(n.src_expr)) is not None
        ]
        assert not const_fills, f"{prog.name}: constant must not be widened into a Symbol-source fill tile"


# --------------------------- value-exactness vs numpy ---------------------------
def test_cast_inline_value_exact():
    sdfg = _vectorize(_cast_inline16)
    A = np.random.rand(M).astype(np.float16)
    C = np.zeros(M, np.float16)
    sdfg(A=A, C=C, N=M)
    ref = np.float16(0.125) * A
    assert np.allclose(C.astype(np.float32), ref.astype(np.float32), rtol=1e-2, atol=1e-2), \
        np.max(np.abs(C.astype(np.float32) - ref.astype(np.float32)))


def test_cast_separate_value_exact():
    sdfg = _vectorize(_cast_separate16)
    A = np.random.rand(M).astype(np.float16)
    C = np.zeros(M, np.float16)
    sdfg(A=A, C=C, N=M)
    ref = A * np.float16(0.125)
    assert np.allclose(C.astype(np.float32), ref.astype(np.float32), rtol=1e-2, atol=1e-2), \
        np.max(np.abs(C.astype(np.float32) - ref.astype(np.float32)))


def test_pure_const_store_still_materialises_and_is_exact():
    """A pure constant store (the constant IS the stored value, no compute consumer to
    broadcast into) keeps the materialised tile lowering and stays value-exact -- the
    see-through must not strand it as a scalar the store cannot read."""
    sdfg = _vectorize(_const_store64)
    C = np.zeros(M, np.float64)
    sdfg(C=C, N=M)
    assert np.allclose(C, np.full(M, 0.5))


# --------------------------- same-domain classifier ---------------------------
def test_same_domain_constant_classifier():
    """The see-through fires only for a same-domain narrowing (fp -> fp, int -> int) and
    is refused for a cross-domain fp <-> int cast or a non-constant symbol. Domains are
    inferred from the descriptor dtype, never hardcoded."""
    # fp literal -> fp dtype: same domain.
    assert is_same_domain_constant("0.125", dace.float16)
    assert is_same_domain_constant("0.125", dace.float64)
    # int literal -> int dtype: same domain.
    assert is_same_domain_constant("5", dace.int32)
    assert is_same_domain_constant("-3", dace.int64)
    # cross-domain: refused (kept as a real conversion).
    assert not is_same_domain_constant("0.125", dace.int32)
    assert not is_same_domain_constant("5", dace.float32)
    # non-constant / symbol: refused.
    assert not is_same_domain_constant("N", dace.float32)
    assert not is_same_domain_constant("N + 1", dace.int32)
    assert numeric_constant_domain("2.0") == "float"
    assert numeric_constant_domain("7") == "int"
    assert numeric_constant_domain("N") is None


if __name__ == "__main__":
    test_cast_constant_broadcasts_no_fill(_cast_inline16)
    test_cast_constant_broadcasts_no_fill(_cast_separate16)
    test_cast_and_separate_forms_emit_equivalent_shape()
    test_cast_inline_value_exact()
    test_cast_separate_value_exact()
    test_pure_const_store_still_materialises_and_is_exact()
    test_same_domain_constant_classifier()
