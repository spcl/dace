# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A 16-bit float constant is written as ``constexpr float``: ``__hip_bfloat16``, ``__half`` and their CUDA twins
have no constexpr constructor, so ``constexpr __hip_bfloat16 x = 0;`` does not compile (resnet18's fused
accumulator under CPF)."""

import ml_dtypes
import numpy as np
import pytest

import dace
from dace import data


@pytest.mark.parametrize("dtype,value", [(dace.bfloat16, ml_dtypes.bfloat16(-1.0)), (dace.float16, np.float16(0.5))])
def test_a_16_bit_float_constant_is_a_constexpr_float(dtype, value):
    sdfg = dace.SDFG(f"low_precision_constant_{dtype.to_string()}")
    sdfg.add_constant("acc_init", value, data.Scalar(dtype))
    sdfg.add_array("out", [1], dtype)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("init", {}, {"o"}, "o = acc_init")
    state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "constexpr float acc_init = " in code, code
    assert f"constexpr {dtype.ctype} acc_init" not in code


def test_a_wider_constant_keeps_its_own_type():
    sdfg = dace.SDFG("wide_constant")
    sdfg.add_constant("acc_init", np.float64(0.25), data.Scalar(dace.float64))
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("init", {}, {"o"}, "o = acc_init")
    state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "constexpr double acc_init = " in code, code
