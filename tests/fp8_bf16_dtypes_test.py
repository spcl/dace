# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-side registration of the low-precision dtypes bfloat16 / float8_e4m3fn /
float8_e5m2 (backed by ml_dtypes, named verbatim as ml_dtypes names them).

These tests cover the dtype registration / frontend / serialization layers only.
bfloat16 additionally has a C++ runtime type and compiles end to end -- that is
covered by tests/bfloat16_cpu_test.py and tests/bfloat16_gpu_test.py. The float8
runtime headers are not implemented yet, so those two remain Python-side only.
"""
import ml_dtypes
import numpy as np
import pytest

import dace
from dace import dtypes

LOWP = [
    ("bfloat16", ml_dtypes.bfloat16, 2, "dace::bfloat16"),
    ("float8_e4m3fn", ml_dtypes.float8_e4m3fn, 1, "dace::float8_e4m3fn"),
    ("float8_e5m2", ml_dtypes.float8_e5m2, 1, "dace::float8_e5m2"),
]


@pytest.mark.parametrize("name,scalar,nbytes,ctype", LOWP)
def test_lowp_typeclass_registered(name, scalar, nbytes, ctype):
    tc = getattr(dace, name)
    # Name follows ml_dtypes verbatim (no aliasing).
    assert tc.to_string() == name
    # Backed by the ml_dtypes scalar of the same name.
    assert tc.type is scalar
    assert tc.bytes == nbytes
    assert tc.ctype == ctype
    assert tc.as_numpy_dtype() == np.dtype(scalar)
    # Registered in the float-type set and the string registry.
    assert tc in dtypes.FLOAT_TYPES
    assert name in dtypes.TYPECLASS_STRINGS
    # Round-trips through the numpy-dtype -> typeclass map.
    assert dtypes.dtype_to_typeclass(scalar) is tc


@pytest.mark.parametrize("name,scalar,nbytes,ctype", LOWP)
def test_lowp_array_serialization_roundtrip(name, scalar, nbytes, ctype):
    desc = dace.data.Array(getattr(dace, name), [8])
    desc2 = dace.serialize.from_json(desc.to_json())
    assert desc2.dtype == getattr(dace, name)


def test_lowp_array_declaration_and_to_sdfg():
    # Declaration syntax + descriptor dtype survive to_sdfg (no compilation).
    @dace.program
    def fp8_id(a: dace.float8_e4m3fn[8], b: dace.float8_e4m3fn[8]):
        b[:] = a

    sdfg = fp8_id.to_sdfg()
    assert sdfg.arrays["a"].dtype == dace.float8_e4m3fn
    assert sdfg.arrays["a"].dtype.bytes == 1


def test_bfloat16_cast_converter_to_sdfg():
    # The dace.bfloat16(...) frontend converter resolves (it has no numpy
    # attribute, so it is sourced from ml_dtypes by name).
    @dace.program
    def cast_kernel(a: dace.float32[16], b: dace.bfloat16[16]):
        b[:] = dace.bfloat16(a)

    sdfg = cast_kernel.to_sdfg()
    assert sdfg.arrays["b"].dtype == dace.bfloat16


if __name__ == "__main__":
    for args in LOWP:
        test_lowp_typeclass_registered(*args)
        test_lowp_array_serialization_roundtrip(*args)
    test_lowp_array_declaration_and_to_sdfg()
    test_bfloat16_cast_converter_to_sdfg()
    print("fp8/bf16 dtype tests passed")
