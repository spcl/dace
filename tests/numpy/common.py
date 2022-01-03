# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import inspect
from copy import deepcopy as dc
from collections import OrderedDict

import dace
import numpy as np

from numpy.random import default_rng

rng = default_rng(42)


def compare_numpy_output(non_zero=False,
                         positive=False,
                         check_dtype=False,
                         validation_func=None,
                         casting=None,
                         max_value=10):
    """ Check that the `dace.program` func works identically to python
        (including errors).

        `func` will be run once as a dace program, and once using python.
        The inputs to the function will be randomly intialized arrays with
        shapes and dtypes according to the argument annotations.

        Note that this should be used *instead* of the `@dace.program`
        annotation, not along with it!

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
    """
    def decorator(func):
        def test():
            dp = dace.program(func)

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

            try:
                if validation_func:
                    # Works only with 1D inputs of the same size!
                    reference_result = []
                    reference_input = [arr.tolist() for arr in inputs.values()]
                    for inp_args in zip(*reference_input):
                        reference_result.append(validation_func(*inp_args))
                else:
                    reference_result = func(**reference_input)
            except Exception as e:
                numpy_thrown = e

            try:
                dace_result = dp(**dace_input)
            except Exception as e:
                dace_thrown = e

            if dace_thrown is not None or numpy_thrown is not None:
                assert dace_thrown is not None and numpy_thrown is not None, "dace threw:\n{}: {}\nBut numpy threw:\n{}: {}\n".format(
                    type(dace_thrown), dace_thrown, type(numpy_thrown), numpy_thrown)
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
