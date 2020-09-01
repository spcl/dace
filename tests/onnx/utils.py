import os
from functools import wraps

import pytest


def parameterize_gpu(function):
    use_gpu = "ONNX_TEST_CUDA" in os.environ
    if use_gpu:
        return pytest.mark.parametrize("gpu", [True, False])(function)
    else:
        return pytest.mark.parametrize("gpu", [False])(function)

def print_when_started(function):
    @wraps(function)
    def decorated(*args, **kwargs):
        print("Running {} with args:".format(function.__name__))
        if args:
            print(args)
        if kwargs:
            print(kwargs)
        function(*args, **kwargs)
        print()
        print("-" * 100)
    return decorated
