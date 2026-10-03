# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which operand dtypes the ``pure`` :class:`TileFMA` widens through ``double``, and the one
sub-fp32 dtype it must NOT.

``std::fma`` has no overload for any of DaCe's low-precision types, so the pure expansion spells a
sub-fp32 FMA as ``T(std::fma(double(a), double(b), double(c)))``. That keeps the single rounding
the node promises AND dodges the ambiguous-conversion compile error every multi-conversion device
type (``__half``, ``__nv_bfloat16``, both ``__nv_fp8_*``) produces when handed bare to an
overloaded function.

``bool_`` is one byte and so passes a naive ``bytes < 4`` width test, but widening it would emit
``bool(std::fma(...))`` -- the numeric-to-bool truncation the node's ``_cast`` guard (and its twin
in ``tile_binop.py``) exists to keep out of the generated text. The exclusion is invisible in a
value-only test: a bool FMA yields the same truthy answer either way. Hence the emitted C++ is what
is asserted here, and the table itself is pinned against a hand-written expectation.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileFMA
from dace.libraries.tileops.nodes.tile_fma import NARROW_OPERAND_CTYPES

WIDTHS = (4, )

#: Every registered dtype narrower than ``float``, C++ spelling, ``bool_`` deliberately absent.
#: Read off ``dace/dtypes.py``'s registry BY HAND -- never recomputed with the predicate under
#: test, which would make this a tautology. The two fp8 entries are what the node's previous
#: hand-written table dropped.
EXPECTED_NARROW_CTYPES = frozenset({
    "dace::float16", "dace::bfloat16", "dace::float8_e4m3fn", "dace::float8_e5m2", "int8_t", "uint8_t", "int16_t",
    "uint16_t"
})

WIDENED_DTYPES = [
    dace.float16, dace.bfloat16, dace.float8_e4m3fn, dace.float8_e5m2, dace.int8, dace.uint8, dace.int16, dace.uint16
]


def fma_sdfg(dtype: dace.dtypes.typeclass) -> dace.SDFG:
    """One state: three ``dtype`` tiles -> ``TileFMA`` -> one ``dtype`` tile, already expanded."""
    sdfg = dace.SDFG(f"fma_{dtype.to_string().replace('::', '_')}")
    for name in ("A", "B", "C", "O"):
        sdfg.add_array(name, WIDTHS, dtype, transient=False)
    state = sdfg.add_state("main")
    node = TileFMA(name="tf", widths=WIDTHS)
    state.add_node(node)
    full = ",".join(f"0:{w}" for w in WIDTHS)
    for name, conn in (("A", "_a"), ("B", "_b"), ("C", "_c")):
        state.add_edge(state.add_access(name), None, node, conn, dace.Memlet(f"{name}[{full}]"))
    state.add_edge(node, "_o", state.add_access("O"), None, dace.Memlet(f"O[{full}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


def fma_tasklet_code(dtype: dace.dtypes.typeclass) -> str:
    """The C++ body of the single tasklet the pure expansion leaves behind."""
    sdfg = fma_sdfg(dtype)
    tasklets = [n for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1, f"expected one tasklet after expansion, got {len(tasklets)}"
    return tasklets[0].code.as_string


def test_the_widened_operand_table_holds_every_subfp32_dtype_and_never_bool():
    """The table is the registry's sub-fp32 slice minus ``bool_``, fp8 pair included."""
    assert NARROW_OPERAND_CTYPES == EXPECTED_NARROW_CTYPES
    assert dace.bool_.ctype not in NARROW_OPERAND_CTYPES
    assert dace.float8_e4m3fn.ctype in NARROW_OPERAND_CTYPES
    assert dace.float8_e5m2.ctype in NARROW_OPERAND_CTYPES


def test_a_bool_tile_triple_calls_std_fma_without_casting_the_result_to_bool():
    """A bool FMA keeps the bare ``std::fma`` call: no ``bool(std::fma(...))`` truncation."""
    code = fma_tasklet_code(dace.bool_)
    assert "_o[__l0] = std::fma(_a[__l0], _b[__l0], _c[__l0]);" in code
    assert "bool(" not in code
    assert "(bool)" not in code
    assert "double(" not in code


@pytest.mark.parametrize("dtype", WIDENED_DTYPES, ids=lambda d: d.to_string())
def test_a_subfp32_tile_triple_is_widened_through_double(dtype):
    """Each sub-fp32 operand type takes the ``T(std::fma(double(...)))`` spelling."""
    code = fma_tasklet_code(dtype)
    ctype = dtype.ctype
    assert f"_o[__l0] = {ctype}(std::fma(double(_a[__l0]), double(_b[__l0]), double(_c[__l0])));" in code


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64, dace.int32, dace.int64], ids=lambda d: d.to_string())
def test_a_dtype_at_least_as_wide_as_float_calls_std_fma_at_its_own_type(dtype):
    """The negative control: the widening predicate can also say no, so a hit means something."""
    code = fma_tasklet_code(dtype)
    assert "_o[__l0] = std::fma(_a[__l0], _b[__l0], _c[__l0]);" in code
    assert "double(" not in code


@pytest.mark.parametrize("dtype", [dace.float8_e4m3fn, dace.float8_e5m2, dace.bfloat16], ids=lambda d: d.to_string())
def test_a_widened_low_precision_tile_fma_compiles_and_matches_the_reference(dtype):
    """The widened spelling is valid C++ and computes ``a * b + c`` for the fp8 / bf16 tiles.

    Every operand AND every result below is exactly representable in all three dtypes (``e5m2``
    carries two mantissa bits and is the binding one), so the expected values are the arithmetic
    itself and no rounding model enters the assertion.
    """
    npdt = dtype.as_numpy_dtype()
    sdfg = fma_sdfg(dtype)
    A = np.array([1, 2, 2, 3], dtype=npdt)
    B = np.array([2, 2, 2, 2], dtype=npdt)
    C = np.array([1, 1, 2, 2], dtype=npdt)
    O = np.zeros(len(A), dtype=npdt)
    sdfg(A=A, B=B, C=C, O=O)
    np.testing.assert_array_equal(np.asarray(O, dtype=np.float64), np.array([3.0, 5.0, 6.0, 8.0]))
