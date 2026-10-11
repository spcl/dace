# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests all spellings of optional (union-with-None) type hints on program arguments.

The ``# noqa: UP007, UP045`` comments keep ``ruff check --fix`` from rewriting the ``typing.Optional`` and
``typing.Union`` spellings into PEP 604 (``T | None``) syntax, so that every spelling remains tested.
"""

from typing import Optional, Union

import numpy as np
import pytest

import dace

N = dace.symbol("N")


def _assert_optional_array(program: dace.program) -> None:
    sdfg = program.to_sdfg(simplify=False)
    desc = sdfg.arrays["a"]
    assert isinstance(desc, dace.data.Array)
    assert desc.dtype == dace.float64
    assert desc.optional is True


def test_union_operator_on_descriptors():
    assert (dace.float64[20] | None) == Optional[dace.float64[20]]  # noqa: UP045
    assert (None | dace.float64[20]) == Union[None, dace.float64[20]]  # noqa: UP007
    assert (dace.float64 | None) == Optional[dace.float64]  # noqa: UP045
    assert (None | dace.float64) == Union[None, dace.float64]  # noqa: UP007
    assert (dace.float64[20] | dace.float32[20]) == Union[dace.float64[20], dace.float32[20]]  # noqa: UP007
    assert (dace.float64[20] | None | None) == Optional[dace.float64[20]]  # noqa: UP045


def test_typing_optional():

    @dace.program
    def tester(a: Optional[dace.float64[20]], b: dace.float64[20]):  # noqa: UP045
        b[:] = 1

    _assert_optional_array(tester)


def test_typing_union_none_last():

    @dace.program
    def tester(a: Union[dace.float64[20], None], b: dace.float64[20]):  # noqa: UP007
        b[:] = 1

    _assert_optional_array(tester)


def test_typing_union_none_first():

    @dace.program
    def tester(a: Union[None, dace.float64[20]], b: dace.float64[20]):  # noqa: UP007
        b[:] = 1

    _assert_optional_array(tester)


def test_pep604_none_last():

    @dace.program
    def tester(a: dace.float64[20] | None, b: dace.float64[20]):
        b[:] = 1

    _assert_optional_array(tester)


def test_pep604_none_first():

    @dace.program
    def tester(a: None | dace.float64[20], b: dace.float64[20]):
        b[:] = 1

    _assert_optional_array(tester)


def test_symbolic_shape():

    @dace.program
    def tester(a: dace.float64[N] | None, b: dace.float64[N]):
        b[:] = 1

    sdfg = tester.to_sdfg(simplify=False)
    assert sdfg.arrays["a"].optional is True
    assert sdfg.arrays["a"].shape == (N,)


def test_nested_optional():

    @dace.program
    def tester(a: Optional[dace.float64[20] | None], b: dace.float64[20]):  # noqa: UP045
        b[:] = 1

    _assert_optional_array(tester)


def test_forward_reference_in_union():

    @dace.program
    def tester(a: Optional["dace.float64[20]"], b: dace.float64[20]):  # noqa: UP045
        b[:] = 1

    _assert_optional_array(tester)


def test_string_annotation():
    """String annotations are also what ``from __future__ import annotations`` produces."""

    @dace.program
    def tester(a: "dace.float64[20] | None", b: "dace.float64[20]"):
        b[:] = 1

    _assert_optional_array(tester)
    sdfg = tester.to_sdfg(simplify=False)
    assert sdfg.arrays["b"].shape == (20,)
    assert sdfg.arrays["b"].optional is not True


@pytest.mark.parametrize(
    "hint",
    [
        pytest.param(dace.float64 | None, id="typeclass_pep604"),
        pytest.param(Optional[dace.float64], id="typeclass_optional"),  # noqa: UP045
        pytest.param(float | None, id="python_type_pep604"),
        pytest.param(Optional[float], id="python_type_optional"),  # noqa: UP045
        pytest.param(Union[float, None], id="python_type_union"),  # noqa: UP007
    ],
)
def test_optional_scalar(hint):

    @dace.program
    def tester(a: hint, b: dace.float64[20]):
        b[:] = a

    sdfg = tester.to_sdfg(simplify=False)
    desc = sdfg.arrays["a"]
    assert isinstance(desc, dace.data.Scalar)
    assert desc.dtype == dace.float64


@pytest.mark.parametrize(
    "hint",
    [
        pytest.param(dace.float64[20] | dace.float32[20], id="pep604"),
        pytest.param(Union[dace.float64[20], dace.float32[20]], id="union"),  # noqa: UP007
        pytest.param(Optional[dace.float64[20] | dace.float32[20]], id="optional_multiple"),  # noqa: UP045
        pytest.param(int | float, id="python_types"),
    ],
)
def test_multiple_types_unsupported(hint):

    @dace.program
    def tester(a: hint, b: dace.float64[20]):
        b[:] = 1

    with pytest.raises(SyntaxError, match="Union type hints"):
        tester.to_sdfg(simplify=False)


def test_optional_array_call():

    @dace.program
    def union_type_hints_call(a: dace.float64[20] | None, b: dace.float64[20]):
        if a is None:
            b[:] = 1
        else:
            b[:] = a + 2

    a = np.random.rand(20)
    b = np.zeros(20)
    union_type_hints_call(a, b)
    assert np.allclose(b, a + 2)

    union_type_hints_call(None, b)
    assert np.allclose(b, 1)


if __name__ == "__main__":
    test_union_operator_on_descriptors()
    test_typing_optional()
    test_typing_union_none_last()
    test_typing_union_none_first()
    test_pep604_none_last()
    test_pep604_none_first()
    test_symbolic_shape()
    test_nested_optional()
    test_forward_reference_in_union()
    test_string_annotation()
    test_optional_scalar(dace.float64 | None)
    test_optional_scalar(Optional[dace.float64])  # noqa: UP045
    test_optional_scalar(float | None)
    test_optional_scalar(Optional[float])  # noqa: UP045
    test_optional_scalar(Union[float, None])  # noqa: UP007
    test_multiple_types_unsupported(dace.float64[20] | dace.float32[20])
    test_multiple_types_unsupported(Union[dace.float64[20], dace.float32[20]])  # noqa: UP007
    test_multiple_types_unsupported(Optional[dace.float64[20] | dace.float32[20]])  # noqa: UP045
    test_multiple_types_unsupported(int | float)
    test_optional_array_call()
