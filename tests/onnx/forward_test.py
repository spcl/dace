from copy import copy
import tempfile
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import dace
from dace.frontend.onnx import OnnxModel

all_checks = []


class CorrectnessCheck:
    def __init__(self, params: OrderedDict):
        self.params = params

    def __call__(self, cls):
        """Apply the decorator to a class."""
        check = copy(self)
        check.model = cls

        all_checks.append(check)
        return cls

    def run(self):
        """Run the check."""
        model = self.model()

        args = tuple(torch.tensor(arg) for arg in self.params.values())

        torch_results = model(*args)

        _, fname = tempfile.mkstemp(suffix='.onnx')
        torch.onnx.export(model, args, fname)

        dace_model = OnnxModel(onnx.load(fname))
        dace_results = dace_model(*self.params.values())

        if type(torch_results) is torch.Tensor:
            assert_close(torch_results.detach(), dace_results)
        else:
            for A, B in zip(dace_results, torch_results):
                assert_close(A.detach(), B)

        os.remove(fname)


def assert_close(A, B):
    diff = np.linalg.norm(A - B) / (A.shape[0] * A.shape[1])
    assert diff < 1e-5


@CorrectnessCheck(
    OrderedDict([('x', np.random.rand(32, 784).astype(np.float32))]))
class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.dense1 = nn.Linear(784, 256)
        self.dense2 = nn.Linear(256, 64)
        self.dense3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        return x

@CorrectnessCheck(
    OrderedDict([('x', np.random.rand(32, 784).astype(np.float32)),
                 ('y', np.random.rand(784, 32).astype(np.float32))]))
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.dense1 = nn.Linear(784, 256)
        self.dense2 = nn.Linear(256, 64)
        self.dense3 = nn.Linear(64, 10)

    def forward(self, x, y):
        x = x + y.t()
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        return x


def test_all():
    for check in all_checks:
        check.run()


if __name__ == "__main__":
    test_all()
