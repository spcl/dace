# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``LayoutChange`` library node: a first-class, chainable layout-algebra relayout.

A ``LayoutChange`` node carries a layout-algebra op sequence (``Permute``/``Block``/``Unblock``/
``Pad``/...) from its input array's (packed-C) layout to its output array's laid-out layout. It has
one input ``_inp`` and one output ``_out`` and lowers the WHOLE (simplified) op sequence at once:

  * ``pure``     -- one materialized mapped-tasklet copy (``build_relayout_sdfg``). Handles ANY op
                    sequence; runs on the GPU when the arrays are ``GPU_Global``.
  * ``cuTENSOR`` -- if the sequence is a pure dimension permutation, delegate to a ``TensorTranspose``
                    (``cutensorPermute``); otherwise fall back to ``pure``.
  * ``HPTT``     -- CPU analog: a pure permutation delegates to ``TensorTranspose`` (HPTT); otherwise
                    ``pure``.

This is the naive "one impl / libnode per operation" dispatch: permutations map to the tensor-
transpose libraries, everything else (Block/Unblock/Pad/combinations) to the pure relayout map.
Two adjacent nodes fold into one via ``fold_layout_changes`` (concatenate op sequences +
``simplify_ops``).
"""
import json
from typing import List, Optional, Tuple

import dace
from dace import library, nodes, properties
from dace.transformation.transformation import ExpandTransformation

from dace.libraries.layout.algebra import ops_from_list, ops_to_list, simplify_ops
from dace.libraries.layout.lowering import build_relayout_sdfg, relayout_map


def _as_permutation(out_map, logical_shape) -> Optional[Tuple[int, ...]]:
    """If ``out_map`` is a pure dimension permutation of the packed-C input, return its axis list
    (``out.shape[i] == in.shape[axes[i]]``); otherwise ``None`` (blocked / padded / zipped)."""
    ndim = len(logical_shape)
    if len(out_map.digits) != ndim or out_map.element is not None or out_map.shuffles:
        return None
    perm = []
    for dg in out_map.digits:
        if str(dg.stride) != '1':
            return None
        if str(dace.symbolic.simplify(dg.extent - logical_shape[dg.dim])) != '0':
            return None
        perm.append(dg.dim)
    if sorted(perm) != list(range(ndim)):
        return None
    return tuple(perm)


def _build_transpose_sdfg(label, in_desc, out_desc, axes, impl) -> dace.SDFG:
    """A nested SDFG that relayouts ``_inp`` to ``_out`` via a ``TensorTranspose`` with the given
    implementation (``cuTENSOR`` / ``HPTT``)."""
    from dace.libraries.linalg import TensorTranspose  # avoid import loop

    sdfg = dace.SDFG(f"{label}_sdfg")
    sdfg.add_array("_inp", in_desc.shape, in_desc.dtype, in_desc.storage, strides=in_desc.strides)
    sdfg.add_array("_out", out_desc.shape, out_desc.dtype, out_desc.storage, strides=out_desc.strides)
    state = sdfg.add_state(f"{label}_state", is_start_block=True)
    rin = state.add_read("_inp")
    rout = state.add_write("_out")
    tt = TensorTranspose(f"{label}_tt", axes=list(axes))
    tt.implementation = impl
    state.add_node(tt)
    state.add_edge(rin, None, tt, "_inp_tensor", dace.Memlet.from_array("_inp", sdfg.arrays["_inp"]))
    state.add_edge(tt, "_out_tensor", rout, None, dace.Memlet.from_array("_out", sdfg.arrays["_out"]))
    return sdfg


@library.expansion
class ExpandPure(ExpandTransformation):
    """Materialize the whole op sequence as ONE mapped-tasklet relayout copy."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        in_desc, out_desc = node.validate(parent_sdfg, parent_state)
        return build_relayout_sdfg(node.label, in_desc, out_desc, node.op_sequence())


@library.expansion
class ExpandCuTensor(ExpandTransformation):
    """Pure permutation -> ``cutensorPermute`` (GPU); anything else -> ``pure``."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        in_desc, out_desc = node.validate(parent_sdfg, parent_state)
        ops = node.op_sequence()
        _, out_map, _ = relayout_map(list(in_desc.shape), ops)
        axes = _as_permutation(out_map, list(in_desc.shape))
        if axes is None:
            return ExpandPure.expansion(node, parent_state, parent_sdfg)
        return _build_transpose_sdfg(node.label, in_desc, out_desc, axes, "cuTENSOR")


@library.expansion
class ExpandHPTT(ExpandTransformation):
    """Pure permutation -> HPTT tensor transpose (CPU); anything else -> ``pure``."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        in_desc, out_desc = node.validate(parent_sdfg, parent_state)
        ops = node.op_sequence()
        _, out_map, _ = relayout_map(list(in_desc.shape), ops)
        axes = _as_permutation(out_map, list(in_desc.shape))
        if axes is None:
            return ExpandPure.expansion(node, parent_state, parent_sdfg)
        return _build_transpose_sdfg(node.label, in_desc, out_desc, axes, "HPTT")


@library.node
class LayoutChange(nodes.LibraryNode):
    """Relayout ``_inp`` (packed-C) to ``_out`` (laid-out) via a layout-algebra op sequence.

    The op sequence is stored (JSON-encoded) in the ``ops`` property; :meth:`op_sequence` decodes it
    back to op dataclasses. The sequence is simplified before lowering, so cancelling chains fold to
    a plain copy.
    """

    implementations = {
        "pure": ExpandPure,
        "cuTENSOR": ExpandCuTensor,
        "HPTT": ExpandHPTT,
    }
    default_implementation = "pure"

    ops = properties.Property(dtype=str, default="[]", desc="JSON-encoded layout-algebra op sequence")

    def __init__(self, name, ops=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inp"}, outputs={"_out"}, **kwargs)
        if ops is None:
            self.ops = "[]"
        elif isinstance(ops, str):
            self.ops = ops
        else:
            self.ops = json.dumps(ops_to_list(list(ops)))

    def op_sequence(self) -> List:
        """Decode the stored op sequence back to op dataclasses."""
        return ops_from_list(json.loads(self.ops))

    def validate(self, sdfg, state):
        """Return ``(in_desc, out_desc)`` for the ``_inp`` / ``_out`` arrays in the parent SDFG."""
        in_desc = out_desc = None
        for e in state.in_edges(self):
            if e.dst_conn == "_inp":
                in_desc = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_out":
                out_desc = sdfg.arrays[e.data.data]
        if in_desc is None:
            raise ValueError("LayoutChange: missing the input array on connector '_inp'.")
        if out_desc is None:
            raise ValueError("LayoutChange: missing the output array on connector '_out'.")
        if in_desc.dtype != out_desc.dtype:
            raise ValueError("LayoutChange: input and output dtypes must match.")
        if in_desc.storage != out_desc.storage:
            raise ValueError("LayoutChange: input and output storage must match.")
        return in_desc, out_desc


def add_layout_change(sdfg: dace.SDFG, state: dace.SDFGState, in_name: str, out_name: str, ops: List) -> LayoutChange:
    """Create the laid-out ``out_name`` array, add a ``LayoutChange`` node relaying ``in_name`` to it,
    and wire the access nodes. Returns the node."""
    in_desc = sdfg.arrays[in_name]
    _, _, out_shape = relayout_map(list(in_desc.shape), ops)
    if out_name in sdfg.arrays:
        sdfg.remove_data(out_name, validate=False)
    sdfg.add_array(name=out_name,
                   shape=out_shape,
                   dtype=in_desc.dtype,
                   storage=in_desc.storage,
                   transient=in_desc.transient,
                   lifetime=in_desc.lifetime,
                   find_new_name=False)

    node = LayoutChange(f"relayout_{in_name}_to_{out_name}", ops=ops)
    rin = state.add_read(in_name)
    rout = state.add_write(out_name)
    state.add_node(node)
    state.add_edge(rin, None, node, "_inp", dace.Memlet.from_array(in_name, in_desc))
    state.add_edge(node, "_out", rout, None, dace.Memlet.from_array(out_name, sdfg.arrays[out_name]))
    return node


def fold_layout_changes(first: LayoutChange, second: LayoutChange) -> LayoutChange:
    """Fold two chained layout changes ``A->B``, ``B->C`` into one ``A->C``: concatenate their op
    sequences and simplify. Returns a fresh node (the caller re-wires ``A`` and ``C``)."""
    merged = simplify_ops(first.op_sequence() + second.op_sequence())
    return LayoutChange(f"{first.label}_then_{second.label}", ops=merged)
