# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import contextlib
import inspect
from copy import deepcopy as dc
from collections import OrderedDict
from typing import Callable

import dace
import numpy as np
import pytest

from numpy.random import default_rng

rng = default_rng(42)


def compare_numpy_output(device=dace.dtypes.DeviceType.CPU,
                         non_zero=False,
                         positive=False,
                         check_dtype=False,
                         validation_func=None,
                         casting=None,
                         max_value=10,
                         expect_div_by_zero=False) -> Callable[[Callable], Callable[[], None]]:
    """ Check that the `dace.program` func works identically to python
        (including errors).

        `func` will be run once as a dace program, and once using python.
        The inputs to the function will be randomly intialized arrays with
        shapes and dtypes according to the argument annotations.

        Note that this should be used *instead* of the `@dace.program`
        annotation, not along with it!

        :param device: Selects the target device for test execution.
        :param non_zero: if `True`, replace `0` inputs with `1`.
        :param positive: if `False`, floats sample from [-10.0, 10.0], and ints
                         sample from [-3, 3). Else, floats sample from
                         [0, 10.0], and ints sample from [0, 3).
        :param check_dtype: if `True`, asserts that the dtype of the result is
                            consistent between NumPy and DaCe.
        :param validation_func: If set, then it is used to validates the
                                results per element
        :param casting: If set, then the reference output is computed on the
                        cast inputs.
        :param max_value: The maximum value allowed in the inputs.
        :param expect_div_by_zero: If `True`, allows division by zero without raising an error.
    """

    def decorator(func):

        def test():
            dp = dace.program(device=device)(func)

            def get_rand_arr(ddesc):
                if type(ddesc) is dace.dtypes.typeclass:
                    # we have a scalar
                    ddesc = dace.data.Scalar(ddesc)

                if ddesc.dtype in [dace.float16, dace.float32, dace.float64]:
                    res = rng.random(ddesc.shape, dtype=getattr(np, ddesc.dtype.to_string()))
                    b = 0 if positive else -max_value
                    a = max_value
                    res = (b - a) * res + a
                    if non_zero:
                        res[res == 0] = 1
                elif ddesc.dtype in [dace.complex64, dace.complex128]:
                    res = (rng.random(ddesc.shape).astype(getattr(np, ddesc.dtype.to_string())) +
                           1j * rng.random(ddesc.shape).astype(getattr(np, ddesc.dtype.to_string())))
                    b = 0 if positive else -max_value
                    a = max_value
                    res = (b - a) * res + a
                    if non_zero:
                        res[res == 0] = 1
                elif ddesc.dtype in [dace.int8, dace.int16, dace.int32, dace.int64, dace.bool]:
                    res = rng.integers(0 if positive else -max_value, max_value, size=ddesc.shape)
                    res = res.astype(getattr(np, ddesc.dtype.to_string()))
                    if non_zero:
                        res[res == 0] = 1
                elif ddesc.dtype in [dace.uint8, dace.uint16, dace.uint32, dace.uint64]:
                    res = rng.integers(0, max_value, size=ddesc.shape)
                    res = res.astype(getattr(np, ddesc.dtype.to_string()))
                    if non_zero:
                        res[res == 0] = 1
                elif ddesc.dtype in [dace.complex64, dace.complex128]:
                    res = (rng.random(ddesc.shape).astype(getattr(np, ddesc.dtype.to_string())) +
                           1j * rng.random(ddesc.shape).astype(getattr(np, ddesc.dtype.to_string())))
                    b = 0 if positive else -10 - 10j
                    a = 10 + 10j
                    res = (b - a) * res + a
                    if non_zero:
                        res[res == 0] = 1 + 1j
                else:
                    raise ValueError("unsupported dtype {}".format(ddesc.dtype))

                if type(ddesc) is dace.data.Scalar:
                    return res[0]
                else:
                    return res

            signature = inspect.signature(func)

            inputs = OrderedDict((name, get_rand_arr(param.annotation)) for name, param in signature.parameters.items())

            dace_input = dc(inputs)
            if casting:
                reference_input = OrderedDict((name, casting(desc)) for name, desc in inputs.items())
            else:
                reference_input = dc(inputs)

            # save exceptions
            dace_thrown, numpy_thrown = None, None

            contextmgr = (pytest.warns(
                match="divide by zero encountered") if expect_div_by_zero else contextlib.nullcontext())

            try:
                if validation_func:
                    # Works only with 1D inputs of the same size!
                    reference_result = []
                    reference_input = [arr.tolist() for arr in inputs.values()]
                    for inp_args in zip(*reference_input):
                        reference_result.append(validation_func(*inp_args))
                else:
                    with contextmgr:
                        reference_result = func(**reference_input)
            except Exception as e:
                numpy_thrown = e

            try:
                if device == dace.dtypes.DeviceType.GPU:
                    sdfg = dp.to_sdfg()
                    sdfg.apply_gpu_transformations()
                    dace_result = sdfg(**dace_input)
                else:
                    dace_result = dp(**dace_input)

            except Exception as e:
                dace_thrown = e

            if dace_thrown is not None or numpy_thrown is not None:
                if dace_thrown is None or numpy_thrown is None:
                    raise_from = dace_thrown if dace_thrown is not None else numpy_thrown
                    raise AssertionError("dace threw {}: {}, but numpy threw {}: {}".format(
                        type(dace_thrown).__name__, dace_thrown,
                        type(numpy_thrown).__name__, numpy_thrown)) from raise_from
            else:
                if not isinstance(reference_result, (tuple, list)):
                    reference_result = [reference_result]
                    dace_result = [dace_result]
                    for ref, val in zip(reference_result, dace_result):
                        if ref.dtype == np.float32:
                            assert np.allclose(ref, val, equal_nan=True, rtol=1e-3, atol=1e-5)
                        else:
                            assert np.allclose(ref, val, equal_nan=True)
                        if check_dtype and not validation_func:
                            assert (ref.dtype == val.dtype)

        return test

    return decorator
