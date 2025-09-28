import pytest
import numpy as np

import dace.libraries.onnx as donnx
from dace.frontend.python.module import DaceModule

import torch
import torch.nn as nn
import torch.nn.functional as F

from dace.testing.utils import torch_tensors_close


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3, 3))
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


@pytest.mark.torch
def test_lenet(sdfg_name, use_cpp_dispatcher):

    input = torch.rand(8, 1, 32, 32, dtype=torch.float32)

    net = LeNet()
    dace_net = LeNet()
    dace_net.load_state_dict(net.state_dict())
    dace_net = DaceModule(dace_net, sdfg_name=sdfg_name, compile_torch_extension=use_cpp_dispatcher)

    torch_output = net(torch.clone(input))
    dace_output = dace_net(torch.clone(input))
    dace_net.sdfg.expand_library_nodes()

    torch_tensors_close("output", torch_output, dace_output)
