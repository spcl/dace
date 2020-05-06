import dace
from copy import deepcopy as dc

import numpy as np
import torch
import torch.nn.functional as F


def compare_function(torch_name, kwargs, inputs):
    def decorator(func):
        def test():
            dace_input = dc(inputs)

            if type(inputs) is list:
                torch_input = [torch.tensor(inp) for inp in dc(inputs)]
            else:
                torch_input = torch.tensor(dc(inputs))

            torch_result = getattr(F, torch_name)(torch_input, **kwargs)
            dace_result = func(dace_input)

            assert np.allclose(torch_result, dace_result)

        return test

    return decorator


@compare_function("softmax", dict(dim=1), np.random.rand(10, 9))
@dace.program
def test_softmax_1(X: dace.float64[10, 9]):
    return F.softmax(X, 1)


@compare_function("softmax", dict(dim=0), np.random.rand(10, 9))
@dace.program
def test_softmax_0(X: dace.float64[10, 9]):
    return F.softmax(X, 0)


# currently fails
@compare_function("softmax", dict(dim=0), np.random.rand(10, 9))
@dace.program
def test_softmax_full_path(X: dace.float64[10, 9]):
    return torch.nn.functional.softmax(X, 0)


if __name__ == "__main__":
    test_softmax_0()
    test_softmax_1()
