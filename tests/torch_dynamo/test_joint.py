# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Joint compilation of training graphs: one SDFG with a forward and a backward phase."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch._dynamo.backends.common import aot_autograd  # noqa: E402
from torch._functorch.aot_autograd import make_boxed_func  # noqa: E402
from torch._functorch.partitioners import default_partition, min_cut_rematerialization_partition  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402
from dace.frontend.ml.torch.dynamo import joint as jt  # noqa: E402
from dace.frontend.ml.torch.dynamo.importer import ImportResult, JointImportResult  # noqa: E402
from dace.sdfg.analysis.schedule_tree import treenodes as tn  # noqa: E402


def _models(make_model):
    torch.manual_seed(0)
    model, reference = make_model(), make_model()
    reference.load_state_dict(model.state_dict())
    return model, reference


def _assert_gradients(model, reference, inputs=()):
    for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)
    for x, x_ref in inputs:
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)


def _mlp():
    return nn.Sequential(nn.Linear(6, 10), nn.GELU(), nn.Linear(10, 3), nn.Softmax(-1))


@pytest.mark.torch
def test_one_sdfg_with_two_phases():
    backend = DaceBackend()
    model, reference = _models(_mlp)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    x = torch.randn(5, 6, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = compiled(x)
    torch.testing.assert_close(out, reference(x_ref), rtol=1e-4, atol=1e-5)
    out.square().sum().backward()
    reference(x_ref).square().sum().backward()
    _assert_gradients(model, reference, [(x, x_ref)])

    assert backend.compile_count == 1
    result = backend.last_result
    assert isinstance(result, JointImportResult)
    phases = [c for c in result.stree.children if not isinstance(c, tn.StateBoundaryNode)]
    assert [type(c) for c in phases] == [tn.IfScope, tn.ElseScope]
    assert jt.PHASE_SYMBOL in phases[0].condition.as_string
    # Containers only one phase uses (primals that are not saved, tangents, gradients) are optional arguments
    arguments = result.sdfg.arglist()
    tangents = [spec.name for spec in result.backward.inputs if spec.name.startswith("tangents")]
    assert tangents and all(arguments[t].optional for t in tangents)


@pytest.mark.torch
def test_interleaved_calls():
    """Two forward calls before their backward calls: the saved values live in AOTAutograd, not in the SDFG."""
    backend = DaceBackend()
    model, reference = _models(_mlp)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    a, b = torch.randn(4, 6), torch.randn(7, 6)
    out_a, out_b = compiled(a), compiled(b)
    out_b.square().sum().backward()
    (out_a * 3).sum().backward()
    reference(b).square().sum().backward()
    (reference(a) * 3).sum().backward()
    _assert_gradients(model, reference)
    assert backend.compile_count == 1


@pytest.mark.torch
def test_recomputing_partitioner():
    """With the min-cut partitioner, the backward phase recomputes forward operators instead of saving them."""
    backend = DaceBackend(partitioner=min_cut_rematerialization_partition)
    model, reference = _models(_mlp)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for n in (3, 8):
        x = torch.randn(n, 6)
        compiled(x).square().sum().backward()
        reference(x).square().sum().backward()
        _assert_gradients(model, reference)
        model.zero_grad()
        reference.zero_grad()
    assert backend.compile_count == 1


class _Normalized(nn.Module):
    """Batch normalization in training mode updates its running statistics (input mutations of the graph)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x):
        return torch.tanh(self.norm(self.fc(x)))


@pytest.mark.torch
def test_buffer_mutation():
    backend = DaceBackend()
    model, reference = _models(_Normalized)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for n in (6, 9):
        x = torch.randn(n, 5)
        compiled(x).square().sum().backward()
        reference(x).square().sum().backward()
        _assert_gradients(model, reference)
        torch.testing.assert_close(model.norm.running_mean, reference.norm.running_mean)
        torch.testing.assert_close(model.norm.running_var, reference.norm.running_var)
        model.zero_grad()
        reference.zero_grad()
    assert backend.compile_count == 1


@pytest.mark.torch
def test_inference_is_not_joint():
    backend = DaceBackend()
    model, reference = _models(_mlp)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    x = torch.randn(4, 6)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), reference(x), rtol=1e-4, atol=1e-5)
    assert backend.compile_count == 1
    assert isinstance(backend.last_result, ImportResult)


def _trace_plan(model, x, partition):
    """The joint plan of ``model`` (no DaCe compilation): AOTAutograd with a recording partition function."""
    plans = []

    def partition_fn(joint, joint_inputs, **kwargs):
        forward, backward = partition(joint, joint_inputs, **kwargs)
        plans.append(jt.plan_from_partition(joint, forward, backward, kwargs["num_fwd_outputs"]))
        return forward, backward

    backend = aot_autograd(
        fw_compiler=lambda gm, _: make_boxed_func(gm.forward),
        bw_compiler=lambda gm, _: make_boxed_func(gm.forward),
        partition_fn=partition_fn,
    )
    torch.compile(model, backend=backend, dynamic=True)(x).sum().backward()
    return plans[0]


@pytest.mark.torch
def test_plan_phases():
    torch.manual_seed(0)
    model, x = _mlp(), torch.randn(5, 6)

    plan = _trace_plan(model, x, default_partition)
    forward, backward = plan.forward_nodes(), plan.backward_nodes()
    tangents = {n for n in plan.joint.graph.nodes if n.name.startswith("tangents")}
    assert not forward & tangents
    # Without recomputation, the backward phase computes no operator the forward phase computes
    assert not {n for n in forward & backward if n.op == "call_function"}
    assert not backward & {plan.nodes()[name] for name in plan.saved}

    recomputing = _trace_plan(model, x, min_cut_rematerialization_partition)
    shared = recomputing.forward_nodes() & recomputing.backward_nodes()
    assert {n for n in shared if n.op == "call_function"}, "expected recomputed operators"


@pytest.mark.torch
def test_plan_rejects_foreign_operators():
    """A partitioned graph with operators the joint graph does not have cannot be compiled jointly."""
    torch.manual_seed(0)
    plan = _trace_plan(_mlp(), torch.randn(5, 6), default_partition)
    backward = torch.fx.GraphModule(plan.backward, plan.backward.graph)
    operator = next(n for n in backward.graph.nodes if n.op == "call_function")
    operator.name = "foreign_operator"
    assert jt.plan_from_partition(plan.joint, plan.forward, backward, plan.num_fwd_outputs) is None


if __name__ == "__main__":
    for test in (
        test_one_sdfg_with_two_phases,
        test_interleaved_calls,
        test_recomputing_partitioner,
        test_buffer_mutation,
        test_inference_is_not_joint,
        test_plan_phases,
        test_plan_rejects_foreign_operators,
    ):
        torch._dynamo.reset()
        test()
