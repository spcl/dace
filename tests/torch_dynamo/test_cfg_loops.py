# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Control-flow capture of the loop shapes found in chunked and recycling inference code: strided
chunk loops with a ragged last chunk writing a preallocated buffer in place, ``chunk_layer``-style loops with closures
and dicts, and recycling loops that carry lists and dicts of tensors with a ``None`` first iteration. Every case is
compiled once for all sizes and compared with eager PyTorch.
"""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402
from dace.frontend.ml.torch.dynamo.cfg.blocks import ControlFlowBackend  # noqa: E402
from dace.sdfg.nodes import AccessNode  # noqa: E402
from dace.sdfg.state import LoopRegion  # noqa: E402


def _check(fn, inputs, expected_compiles=1, loops=1, **options):
    """Compares ``fn`` compiled with control-flow capture against eager PyTorch on every input."""
    backend = ControlFlowBackend(**options)
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        for args in inputs:
            args = args if isinstance(args, tuple) else (args,)
            expected, got = fn(*args), compiled(*args)
            if isinstance(expected, dict):
                assert got.keys() == expected.keys()
                for key in expected:
                    torch.testing.assert_close(got[key], expected[key], rtol=1e-5, atol=1e-5)
            else:
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    assert "cfg" in backend.kinds(), backend.events
    assert backend.compile_count == expected_compiles, backend.events
    regions = [r for r in backend.last_sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert len(regions) == loops, [r.label for r in regions]
    return backend


@pytest.mark.torch
def test_strided_chunk_loop_writes_in_place():
    """``out[i:i+c] = f(x[i:i+c])`` over a symbolic size with a ragged last chunk: one loop, one buffer."""

    def f(x):
        out = x.new_zeros(x.shape)
        for i in range(0, x.shape[0], 4):
            out[i : i + 4] = torch.tanh(x[i : i + 4]) * 2
        return out

    backend = _check(f, [torch.randn(10, 3), torch.randn(8, 3), torch.randn(13, 3)])
    # The chunk write is a strided write into the loop-carried buffer: no copy of the whole buffer inside the loop
    sdfg = backend.last_sdfg
    loop = next(r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))
    buffers = [name for name in sdfg.arrays if name.startswith("cfg") and name.endswith("_out")]
    assert buffers, list(sdfg.arrays)
    extent = str(sdfg.arrays[buffers[0]].shape[0])
    for state in loop.states():
        for e in state.edges():
            if isinstance(e.src, AccessNode) and isinstance(e.dst, AccessNode) and e.dst.data in buffers:
                assert str(e.data.subset.size()[0]) != extent, f"whole-buffer copy {e.data} in the loop"


@pytest.mark.torch
def test_functional_slice_scatter_loop():
    def f(x):
        out = x.new_zeros(x.shape)
        for i in range(0, x.shape[0], 4):
            out = torch.slice_scatter(out, torch.tanh(x[i : i + 4]) * 2, 0, i, i + 4)
        return out

    _check(f, [torch.randn(10, 3), torch.randn(8, 3), torch.randn(13, 3)])


def _layer(a, b):
    return {"y": torch.sigmoid(a) * b + 1}


def _chunk_layer(layer, inputs, chunk_size):
    """A chunked layer: ceil trip count, a counter incremented in the body, closures, dicts in and out."""
    n = next(iter(inputs.values())).shape[0]
    no_chunks = n // chunk_size + (n % chunk_size != 0)
    i = 0
    out = None
    for _ in range(no_chunks):
        select_chunk = lambda t: t[i : i + chunk_size]  # noqa: E731 - a closure over i and chunk_size
        chunks = {k: select_chunk(v) for k, v in inputs.items()}
        output_chunk = layer(**chunks)
        if out is None:
            out = {k: v.new_zeros((n,) + v.shape[1:]) for k, v in output_chunk.items()}
        for k, v in output_chunk.items():
            out[k][i : i + chunk_size] = v
        i += chunk_size
    return out


@pytest.mark.torch
def test_chunk_layer_with_closures_and_dicts():
    def f(a, b):
        return _chunk_layer(_layer, {"a": a, "b": b}, 4)

    _check(f, [(torch.randn(10, 3), torch.randn(10, 3)), (torch.randn(7, 3), torch.randn(7, 3))])


class _Block(nn.Module):
    """One recycling iteration: previous embeddings arrive as a list (``None`` in the first iteration)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def iteration(self, feats, prevs):
        m_prev, z_prev = reversed([prevs.pop() for _ in range(2)])
        if None in [m_prev, z_prev]:
            m_prev = feats.new_zeros(feats.shape)
            z_prev = feats.new_zeros(feats.shape)
        m = torch.tanh(self.fc(feats + m_prev))
        z = torch.sigmoid(self.fc(feats * 0.5 + z_prev))
        return {"m": m, "z": z, "sum": m + z}, m, z

    def forward(self, x):
        prevs = [None, None]
        for cycle_no in range(x.shape[-1]):
            feats = x[..., cycle_no]
            outputs, m, z = self.iteration(feats, prevs)
            prevs = [m, z]
        return outputs


@pytest.mark.torch
def test_recycling_loop_with_symbolic_iterations():
    """A symbolic number of recycles, a list of carried tensors, a dict result, and ``x[..., i]`` indexing."""
    torch.manual_seed(0)
    # (Dynamo gives equal sizes within one input the same symbol, so the recycle dimension differs from the others)
    _check(_Block().eval(), [torch.randn(5, 3, 4), torch.randn(6, 3, 2), torch.randn(7, 3, 5)])


@pytest.mark.torch
def test_constant_loop_is_captured_with_capture_loops_all():
    """``capture_loops='all'`` keeps a fixed number of recycles as a loop (the graph holds one iteration, not three)."""
    torch.manual_seed(0)
    model = _Block().eval()

    def f(x):
        prevs = [None, None]
        for cycle_no in range(3):
            outputs, m, z = model.iteration(x[..., cycle_no], prevs)
            prevs = [m, z]
        return outputs

    _check(f, [torch.randn(5, 3, 3), torch.randn(4, 3, 3)], capture_loops="all")
    # Without it the loop is unrolled by Dynamo (no capture)
    backend = ControlFlowBackend()
    compiled = torch.compile(f, backend=backend, dynamic=True)
    x = torch.randn(5, 3, 3)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x)["sum"], f(x)["sum"], rtol=1e-5, atol=1e-5)
    assert "cfg" not in backend.kinds()


@pytest.mark.torch
def test_tuple_index_with_loop_counter():
    """``x[:, i]`` and ``x[..., i] = v`` with a loop counter: torch does not accept symbolic ints inside index tuples."""

    def f(x):
        out = x.new_zeros(x.shape)
        for i in range(x.shape[1]):
            out[:, i] = x[:, i] * (x[..., i].sum() > 0)
        return out

    _check(f, [torch.randn(3, 5), torch.randn(4, 7)])


@pytest.mark.torch
def test_list_slicer_with_ragged_chunks():
    """A list of slices as the index (``empty_slicer`` idiom) with a ragged last chunk, accumulated in place."""

    def f(z):
        out = z.new_zeros(z.shape)
        for i in range(0, z.shape[1], 4):
            slicer = [slice(None) for _ in range(z.dim())]
            slicer[1] = slice(i, i + 4)
            out[slicer] += torch.tanh(z[slicer])
        return out

    _check(f, [torch.randn(3, 10, 2), torch.randn(2, 8, 2), torch.randn(4, 13, 2)])


@pytest.mark.torch
def test_control_flow_option_on_the_default_backend():
    """``DaceBackend(control_flow=True)`` is the production entry point; the capture state is on the backend."""

    def f(x):
        y = x[0] * 0
        for i in range(x.shape[0]):
            y = y + x[i] * i
        return y

    backend = DaceBackend(control_flow=True)
    compiled = torch.compile(f, backend=backend, dynamic=True)
    with torch.no_grad():
        for x in (torch.randn(4, 3), torch.randn(7, 3)):
            torch.testing.assert_close(compiled(x), f(x), rtol=1e-5, atol=1e-5)
    assert backend.compile_count == 1 and "cfg" in backend.control_flow.kinds()
    with pytest.raises(ValueError, match="capture_loops"):
        DaceBackend(control_flow=True, capture_loops="sometimes")


@pytest.mark.torch
def test_list_append_in_loop_falls_back():
    """A list that grows in every iteration cannot be a loop variable: Dynamo unrolls the loop (and specializes)."""

    def f(z):
        chunks = []
        for i in range(0, z.shape[0], 4):
            chunks.append(torch.tanh(z[i : i + 4]))  # noqa: PERF401 - the growing list is the point
        return torch.cat(chunks, dim=0)

    backend = ControlFlowBackend()
    compiled = torch.compile(f, backend=backend, dynamic=True)
    with torch.no_grad():
        for z in (torch.randn(10, 3), torch.randn(8, 3)):
            torch.testing.assert_close(compiled(z), f(z), rtol=1e-5, atol=1e-5)
    assert "cfg" not in backend.kinds()


if __name__ == "__main__":
    test_strided_chunk_loop_writes_in_place()
    test_functional_slice_scatter_loop()
    test_chunk_layer_with_closures_and_dicts()
    test_recycling_loop_with_symbolic_iterations()
    test_constant_loop_is_captured_with_capture_loops_all()
    test_tuple_index_with_loop_counter()
    test_list_slicer_with_ragged_chunks()
    test_control_flow_option_on_the_default_backend()
    test_list_append_in_loop_falls_back()
