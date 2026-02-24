# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import os

import numpy as np
import torch
import torch.utils.cpp_extension
from dace.codegen import targets, compiler
from dace.codegen.codeobject import CodeObject
from torch import nn

import dace
from dace.libraries.torch import PyTorch
from tests.utils import torch_tensors_close

op_source = """
#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/NamedTensorUtils.h>

using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;

Tensor myadd(const Tensor& self, const Tensor& other) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("myops::myadd", "")
    .typed<decltype(myadd)>();
  return op.call(self, other);
}

TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}

Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}

class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
 public:
  static Tensor forward(
      AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
    at::AutoDispatchBelowADInplaceOrView g;
    return myadd(self, other);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto grad_output = grad_outputs[0];
    return {grad_output, grad_output};
  }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
  return MyAddFunction::apply(self, other)[0];
}

TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}

TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", myadd_autograd);
}
"""


@pytest.mark.torch
def test_extension():
    program = CodeObject("myadd",
                         op_source,
                         "cpp",
                         targets.cpu.CPUCodeGen,
                         "MyAddFunction",
                         environments={PyTorch.full_class_path()})

    BUILD_PATH = os.path.join('.dacecache', "pt_extension")
    compiler.generate_program_folder(None, [program], BUILD_PATH)
    torch.utils.cpp_extension.load(
        name='pt_extension',
        sources=[os.path.join(BUILD_PATH, 'src', 'cpu', 'myadd.cpp')],
        is_python_module=False,
    )
    torch.ops.myops.myadd(torch.randn(32, 32), torch.rand(32, 32))


@pytest.mark.torch
def test_module_with_constant():

    @dace.ml.module(sdfg_name="test_module_with_constant")
    class Module(nn.Module):

        def forward(self, x):
            return x + 1

    inp = torch.ones((5, 5))
    output = Module()(inp)

    torch_tensors_close("output", inp + 1, output.cpu())


if __name__ == "__main__":
    test_extension()
    test_module_with_constant()
