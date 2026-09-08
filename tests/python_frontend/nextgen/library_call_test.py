# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for calls into a LIBRARY's registered replacements, in the shape the ONNX
op registry uses: the replacement is keyed on the package's public path
(``dace.libraries.onnx.ONNXGather``) while the object it names lives in a
private submodule under a versioned class name
(``dace.libraries.onnx.nodes.onnx_op_registry.ONNXGather_13``), and every
operand -- inputs, outputs and attributes alike -- is passed by KEYWORD.

The library here is a stand-in so the mechanism is testable without the ONNX
extra installed; the two properties it exercises are the ones that dropped
``donnx.ONNXGather(...)`` to a Python callback:

1. the callee's registry-facing qualname is the one the registry KNOWS, not
   necessarily the object's own ``__module__.__name__``, and
2. a zero-output call's stand-in container may come from a keyword operand.
"""
import contextlib
import types

import numpy as np

import dace
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import nextgen
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: A library package whose public path is not where its objects are defined.
fake_library = types.ModuleType('nextgen_fake_library')


class ScaleOp_13:  # noqa: N801 -- mirrors the ONNX registry's versioned class names
    """Defined here (a "private submodule"), exported as ``ScaleOp``."""


fake_library.ScaleOp = ScaleOp_13


def _scale_op(pv, sdfg, state, data: str, output: str, factor: float):
    """Writes ``data * factor`` into ``output``; a stand-in for a library node
    expansion, which likewise writes its outputs and returns nothing."""
    descriptor = sdfg.arrays[data]
    state.add_mapped_tasklet('scale', {'__i': f'0:{descriptor.shape[0]}'}, {'__inp': Memlet(data=data, subset='__i')},
                             f'__out = __inp * {factor}', {'__out': Memlet(data=output, subset='__i')},
                             external_edges=True)
    return None


def _scale_op_descriptor(input_descs, **_kwargs):
    return ()  # Zero-output: the call writes into its own ``output=`` operand


@contextlib.contextmanager
def _registered():
    """Register the stand-in library for the duration of one test. The
    registry is global, so a leftover entry would show up in unrelated runs
    (``schedule_tree/registry_parity_test.py`` enumerates it)."""
    oprepo.replaces('nextgen_fake_library.ScaleOp')(_scale_op)
    oprepo.infers_descriptor('nextgen_fake_library.ScaleOp')(_scale_op_descriptor)
    try:
        yield
    finally:
        del oprepo.Replacements._rep['nextgen_fake_library.ScaleOp']
        del oprepo.Replacements._dtype_rep['nextgen_fake_library.ScaleOp']


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_library_call_through_the_public_module_path():
    """``fake_library.ScaleOp(...)`` resolves to the name the registry is keyed
    on, even though the class's own qualified name is a different one."""

    @dace.program
    def scaled(inp: dace.float64[4], out: dace.float64[4]):
        fake_library.ScaleOp(data=inp, output=out, factor=2.0)

    inp, out = np.random.rand(4), np.zeros(4)
    with _registered():
        # The SDFG build stays inside: the call is DEFERRED, so the expansion
        # runs the replacement again when the tree is converted.
        tree = nextgen.parse_program(scaled, inp, out)
        assert not _nodes_of_type(tree, tn.PythonCallbackNode)
        calls = _nodes_of_type(tree, tn.ReplacementCallNode)
        assert len(calls) == 1 and calls[0].qualname == 'nextgen_fake_library.ScaleOp'
        compiled = tree.as_sdfg().compile()
    compiled(inp=inp, out=out)
    assert np.allclose(out, inp * 2.0)


def test_callee_qualname_prefers_the_registered_spelling():
    """The resolution itself: the object's canonical name is not registered, so
    the alias-resolved module path is what ``resolve_callee`` reports."""
    from dace.frontend.python.nextgen.semantics import inference

    canonical = f'{ScaleOp_13.__module__}.{ScaleOp_13.__name__}'
    with _registered():
        assert not inference._registered_qualname(canonical)
        assert inference._registered_qualname('nextgen_fake_library.ScaleOp')


def test_unregistered_library_call_still_degrades():
    """An unregistered callable is unaffected by the candidate search: no
    spelling of it is registered, so it stays interpreter work (here through
    preprocessing's own callable detection, which runs first)."""

    @dace.program
    def unknown(inp: dace.float64[4], out: dace.float64[4]):
        fake_library.MissingOp(data=inp, output=out)

    fake_library.MissingOp = ScaleOp_13
    try:
        with _registered():
            tree = nextgen.parse_program(unknown, np.zeros(4), np.zeros(4))
    finally:
        del fake_library.MissingOp
    assert not _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(_nodes_of_type(tree, tn.PythonCallbackNode)) == 1


if __name__ == '__main__':
    test_library_call_through_the_public_module_path()
    test_callee_qualname_prefers_the_registered_spelling()
    test_unregistered_library_call_still_degrades()
