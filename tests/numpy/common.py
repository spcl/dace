# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import inspect
from copy import deepcopy as dc
from collections import OrderedDict

import dace
import numpy as np


def compare_numpy_output(non_zero=False, positive=False):
    """Check that the `dace.program` func works identically to the python version (including errors).

       `func` will be run once as a dace program, and once using python. The inputs to the function
       will be randomly intialized arrays with shapes and dtypes according to the argument
       annotations.

       Note that this should be used *instead* of the `@dace.program` annotation, not along with it!

       :param non_zero: if `True`, replace `0` inputs with `1`.
       :param positive: if `False`, floats sample from [-10.0, 10.0], and ints sample from
                        [-3, 3). Else, floats sample from [0, 10.0], and ints sample from [0, 3).
    """
    def decorator(func):
        def test():
            dp = dace.program(func)

            def get_rand_arr(ddesc):
                if type(ddesc) is dace.dtypes.typeclass:
                    # we have a scalar
                    ddesc = dace.data.Scalar(ddesc)

                if ddesc.dtype in [dace.float16, dace.float32, dace.float64]:
                    res = np.random.rand(*ddesc.shape).astype(
                        getattr(np, ddesc.dtype.to_string()))
                    b = 0 if positive else -10
                    a = 10
                    res = (b - a) * res + a
                    if non_zero:
                        res[res == 0] = 1
                elif ddesc.dtype in [
                        dace.int8, dace.int16, dace.int32, dace.int64,
                        dace.bool
                ]:
                    res = np.random.randint(0 if positive else -3,
                                            3,
                                            size=ddesc.shape)
                    res = res.astype(getattr(np, ddesc.dtype.to_string()))
                    if non_zero:
                        res[res == 0] = 1
                else:
                    raise ValueError("unsupported dtype {}".format(
                        ddesc.dtype))

                if type(ddesc) is dace.data.Scalar:
                    return res[0]
                else:
                    return res

            signature = inspect.signature(func)

            inputs = OrderedDict(
                (name, get_rand_arr(param.annotation))
                for name, param in signature.parameters.items())

            dace_input = dc(inputs)
            reference_input = dc(inputs)

            # save exceptions
            dace_thrown, numpy_thrown = None, None

            try:
                reference_result = func(**reference_input)
            except Exception as e:
                numpy_thrown = e

            try:
                dace_result = dp(**dace_input)
            except Exception as e:
                dace_thrown = e

            if dace_thrown is not None or numpy_thrown is not None:
                assert dace_thrown is not None and numpy_thrown is not None, "dace threw:\n{}: {}\nBut numpy threw:\n{}: {}\n".format(
                    type(dace_thrown), dace_thrown, type(numpy_thrown),
                    numpy_thrown)
            else:
                assert np.allclose(reference_result, dace_result)

        return test

    return decorator
