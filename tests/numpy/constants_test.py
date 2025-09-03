# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import uuid
import math
import pytest


def _make_sdfg(
    code: str,
    dtype=dace.float64,
) -> dace.SDFG:
    """Generates an SDFG that writes an expression to an array.
    """
    sdfg = dace.SDFG(name=f"const_test_{str(uuid.uuid1()).replace('-', '_')}")
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_array(
        "out",
        shape=(10, ),
        dtype=dtype,
        transient=False,
    )

    state.add_mapped_tasklet(
        "comput",
        map_ranges={"__i": "0:10"},
        inputs={},
        code=f"__out = {code}",
        outputs={"__out": dace.Memlet("out[__i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def _test_sdfg(
    sdfg: dace.SDFG,
    expected,
    dtype=np.float64,
):
    out = np.zeros(10, dtype=dtype)
    sdfg.apply_gpu_transformations()
    sdfg(out=out)
    assert np.allclose(out, expected, equal_nan=True), f"Expected {expected}, but got {out[0]}"


def _perform_test(
    code,
    expected,
    dtype=np.float64,
):
    print(f"PERFORM: {code}")
    dace_dtype = dace.dtypes.dtype_to_typeclass(dtype)
    sdfg = _make_sdfg(code=code, dtype=dace_dtype)
    _test_sdfg(sdfg=sdfg, expected=expected, dtype=dtype)


@pytest.mark.gpu
def test_constant_pi_simple():
    _perform_test(code="math.pi", expected=math.pi)


@pytest.mark.gpu
def test_constant_pi_add():
    _perform_test(code="-math.pi", expected=-math.pi)
    _perform_test(code="math.pi + math.pi", expected=2 * math.pi)
    _perform_test(code="math.pi - math.pi", expected=0.)


@pytest.mark.gpu
def test_constant_pi_mult():
    _perform_test(code="(math.pi ** 2) * 2", expected=math.pi * math.pi * 2.0)
    _perform_test(code="math.pi * 2", expected=2 * math.pi)
    _perform_test(code="math.pi * 2 + math.pi", expected=2 * math.pi + math.pi)
    _perform_test(code="math.pi * math.pi * 2", expected=math.pi * math.pi * 2.0)
    _perform_test(code="math.pi / math.pi ", expected=1)
    _perform_test(code="(math.pi + math.pi) / math.pi ", expected=2)
    _perform_test(code="(math.pi * math.pi) / math.pi ", expected=math.pi)


@pytest.mark.gpu
def test_constant_pi_fun():
    _perform_test(
        code="math.sin(math.pi)",
        expected=0,
    )
    _perform_test(
        code="math.sin(math.pi * 4)",
        expected=math.sin(math.pi * 4),
    )
    _perform_test(
        code="math.sin(math.pi * 5)",
        expected=math.sin(math.pi * 5),
    )
    _perform_test(
        code="math.cos(math.pi * 4)",
        expected=math.cos(math.pi * 4),
    )
    _perform_test(
        code="math.cos(math.pi * 5)",
        expected=math.cos(math.pi * 5),
    )
    _perform_test(
        code="math.log(math.pi)",
        expected=math.log(math.pi),
    )


@pytest.mark.gpu
def test_constant_nan():
    _perform_test(code="math.nan", expected=math.nan)
    _perform_test(code="math.nan + 2", expected=math.nan)
    _perform_test(code="math.nan + 2.0", expected=math.nan)
    _perform_test(code="math.sin(math.nan + 2.0)", expected=math.nan)
    _perform_test(code="math.sin(math.nan + 2.0) ** 2", expected=math.nan)


if __name__ == "__main__":
    test_constant_pi_simple()
    test_constant_pi_add()
    test_constant_pi_mult()
    test_constant_pi_fun()
    test_constant_nan()
