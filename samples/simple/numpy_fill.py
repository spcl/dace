# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""Simple program related to issue #1389 """

import dace
import numpy as np

from dace.codegen.exceptions import CompilationError

N = dace.symbol("N", dace.int32)


@dace.program
def fill_dace(A: dace.float64[N, N], value: dace.float64):
    A.fill(value)
#

@dace.program
def fill_dace_can_not_working_yet(A: dace.float64[N, N], value: dace.float64):
    A.fill(value + 1)
#

@dace.program
def fill_dace_does_also_not_work(A: dace.float64[N, N], value: dace.float64):
    value2 = value + 1
    A.fill(value2)
#

@dace.program
def fill_dace_will_work_but_why(A: dace.float64[N, N], value: dace.float64):
    value2 = value + 1
    A.fill(value2)
    return value2
#

@dace.program
def fill_dace_why_god_does_this_not_work(A: dace.float64[N, N], value: dace.float64):
    value2 = value + 1
    A.fill(value2)
    return value
#


N = 8
value = 0.3

input = np.ndarray((N, N)).astype(np.float64)
input_ref = np.copy(input)


try:
    input[:] = 0
    input_ref[:] = value
    fill_dace(input, value)
    assert np.allclose(input, input_ref)
except BaseException as E:
    raise ValueError(f"There was an unexpected exception:\n{str(E)}")
#

try:
    input[:] = 0
    input_ref[:] = value + 1
    fill_dace_can_not_working_yet(input, value)
    assert np.allclose(input, input_ref)
except CompilationError:
    print(f"This version should not work, because it is an expression and not a scalar, so we expect it to fail.")
except BaseException as E:
    print(f"Expected to fail with a compilation error, but instead failed with:\n{str(E)}")
#

try:
    input[:] = 0
    input_ref[:] = value + 1
    fill_dace_does_also_not_work(input, value)
    assert np.allclose(input, input_ref)
except CompilationError as E:
    print(f"This version should work, but it does not work, it fails with a compilation error telling us that `value2` is not found.\nThe error is: {str(E)}")
except BaseException as E:
    print(f"Expected to fail with a compilation error, but instead failed with:\n{str(E)}")
#


try:
    input[:] = 0
    input_ref[:] = value + 1
    z = fill_dace_will_work_but_why(input, value)
    assert np.allclose(input, input_ref)
    assert np.allclose(z, value + 1)
except BaseException as E:
    raise ValueError(f"Expected to run but failed with:\n{str(E)}")
else:
    print(f"This version should succeed and it did!")
#

try:
    input[:] = 0
    input_ref[:] = value + 1
    z = fill_dace_why_god_does_this_not_work(input, value)
    assert np.allclose(input, input_ref)
    assert np.allclose(z, value)
except BaseException as E:
    print(f"Expected this to run but it does not run, it failed with:\n{str(E)}")
else:
    raise ValueError(f"What the hell? the version succeeded?")
#
