import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close


@pytest.mark.long
@pytest.mark.torch
@pytest.mark.autodiff
def test_bert_encoder_backward(sdfg_name):
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])
    ptmodel = BertLayer(BertConfig(
        hidden_act="relu",
        hidden_size=hidden_size,
        attn_implementation="eager",
    )).eval()

    dace_model = DaceModule(ptmodel, sdfg_name=sdfg_name, training=False, backward=True, auto_optimize=False)

    ptinput = torch.clone(input)
    ptinput.requires_grad = True
    ptmodel(ptinput)[0].sum().backward()

    dace_input = torch.clone(input)
    dace_input.requires_grad = True
    dace_model(dace_input)[0].sum().backward()

    torch_tensors_close("input_grad", ptinput.grad, dace_input.grad)

    for (name, dace_param), (pt_name, pt_param) in zip(ptmodel.named_parameters(), dace_model.named_parameters()):
        assert 'model.' + name == pt_name, f"Parameter name mismatch: expected 'model.{name}', got '{pt_name}'"
        torch_tensors_close(name, pt_param.grad, dace_param.grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
