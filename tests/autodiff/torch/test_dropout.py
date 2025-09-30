from typing import Literal, Union
import pytest
import torch
from torch import nn
from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close


@pytest.mark.torch
def test_dropout_fwd_training():
    p = 0.5
    module = nn.Dropout(p=p).train()
    dace_module = DaceModule(module, dummy_inputs=(torch.ones(10, 10), ), training=True)

    # dropout will set some of these to zero
    test_data = torch.randint(1, 10, (10, 10)).float()

    out = dace_module(torch.clone(test_data))
    zeroed = out == 0

    scale = 1 / (1 - p)
    torch_tensors_close("output", test_data[~zeroed] * scale, out[~zeroed])


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("p", [0, 0.99, 0.6, 0.5])
def test_dropout_bwd(p: Union[float, Literal[0]]):
    module = nn.Dropout(p=p).train()
    sdfg_name = f"test_dropout_{str(p).replace('.', '_')}_bwd"
    dace_module = DaceModule(module,
                             sdfg_name=sdfg_name,
                             dummy_inputs=(torch.ones(10, 10), ),
                             backward=True,
                             training=True)

    test_data = torch.randint(1, 10, (10, 10)).float()
    test_data.requires_grad = True
    dy = torch.rand_like(test_data)

    out = dace_module(torch.clone(test_data))

    zeroed = out == 0
    scale = 1 / (1 - p)
    # check that fwd was correct
    torch_tensors_close("output", test_data[~zeroed] * scale, out[~zeroed])

    out.backward(dy)

    # check that the gradient is correct:
    zeros = torch.zeros_like(test_data.grad)
    # check that zeroed values are zero in the grad
    torch_tensors_close("grad_zeroed", zeros[zeroed], test_data.grad[zeroed])

    # check that non-zeroed values are correct
    torch_tensors_close("grad_zeroed", dy[~zeroed] * scale, test_data.grad[~zeroed])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
