# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a layout-algebra op sequence to a materialized relayout in an SDFG.

``build_relayout`` takes an identity-layout (packed-C) input array and an op sequence, computes the
resulting ``LayoutMap`` (after ``simplify_ops``), creates the output array with the laid-out shape,
and emits ONE mapped-tasklet copy that reads the input at its logical index and writes the output at
the physical position dictated by the digit tuple:

    for each logical index i:  out[ (i//stride0)%extent0, (i//stride1)%extent1, ... ] = in[i0, i1, ...]

A Permute becomes a transpose, a Block becomes a reshape-copy, an identity sequence (e.g.
``Block∘Unblock``) becomes a plain copy. This is the core of the ``LayoutChange`` node's expansion:
``relayout_map`` computes the (simplified ops, output LayoutMap, output shape) and
``build_relayout_sdfg`` emits the copy into a standalone nested SDFG with ``_inp``/``_out`` arrays.
"""
from typing import List, Tuple

import dace

from dace.libraries.layout.algebra import LayoutMap, compose_ops, identity_map, simplify_ops


def _digit_expr(digit) -> str:
    """The physical index of one output digit: ``(__i<dim> // stride) % extent``, simplified."""
    idx = dace.symbolic.pystr_to_symbolic(f"__i{digit.dim}")
    expr = (idx // digit.stride) % digit.extent
    return str(dace.symbolic.simplify(expr))


def relayout_map(logical_shape: List, ops: List) -> Tuple[List, LayoutMap, List]:
    """Simplify ``ops`` and compose them on the packed-C identity of ``logical_shape``.

    :return: ``(simplified_ops, out_map, out_shape)`` where ``out_shape`` is the laid-out
             (simplified) array shape.
    """
    dims = list(range(len(logical_shape)))
    simplified = simplify_ops(ops)
    out_map = compose_ops(simplified, base=identity_map(logical_shape, dims))
    if out_map.shuffles:
        # A net Shuffle renumbers elements by a bijection sigma -- that is a data-dependent
        # reorder, not a digit reshape this copy can express. It is lowered by the
        # ShuffleElements pass (which materializes sigma / sigma^{-1} as C functions and rewrites
        # consumers), so refuse here rather than silently emitting an unshuffled copy.
        raise NotImplementedError(
            "LayoutChange cannot lower a net Shuffle; apply dace.transformation.layout.ShuffleElements "
            "for the value-permutation, then LayoutChange for the remaining digit reshape.")
    out_shape = [dace.symbolic.simplify(e) for e in out_map.shape()]
    return simplified, out_map, out_shape


def _emit_relayout_copy(state: dace.SDFGState, in_name: str, out_name: str, logical_shape: List,
                        out_map: LayoutMap) -> None:
    """Emit the single mapped-tasklet copy ``out[digits] = in[logical]`` into ``state``."""
    dims = list(range(len(logical_shape)))
    map_ranges = {f"__i{d}": f"0:{logical_shape[d]}" for d in dims}
    read_index = ", ".join(f"__i{d}" for d in dims)
    write_index = ", ".join(_digit_expr(dg) for dg in out_map.digits)
    state.add_mapped_tasklet(
        name=f"relayout_{in_name}_to_{out_name}",
        map_ranges=map_ranges,
        inputs={"__inp": dace.Memlet.simple(in_name, read_index)},
        code="__out = __inp",
        outputs={"__out": dace.Memlet.simple(out_name, write_index)},
        external_edges=True,
    )


def build_relayout(sdfg: dace.SDFG, state: dace.SDFGState, in_name: str, out_name: str, ops: List) -> dace.SDFGState:
    """Emit a relayout ``out = layout(ops)(in)`` into ``state``.

    ``in_name`` must be an existing packed-C (identity-layout) array. ``out_name`` is created (or
    replaced) with the laid-out shape. The op sequence is normalized with ``simplify_ops`` first, so
    cancelling sequences (Block∘Unblock, Permute∘Permute⁻¹, ...) lower to a plain copy.
    """
    in_desc = sdfg.arrays[in_name]
    logical_shape = list(in_desc.shape)
    _, out_map, out_shape = relayout_map(logical_shape, ops)

    if out_name in sdfg.arrays:
        sdfg.remove_data(out_name, validate=False)
    sdfg.add_array(name=out_name,
                   shape=out_shape,
                   dtype=in_desc.dtype,
                   storage=in_desc.storage,
                   transient=in_desc.transient,
                   lifetime=in_desc.lifetime,
                   find_new_name=False)

    _emit_relayout_copy(state, in_name, out_name, logical_shape, out_map)
    return state


def build_relayout_sdfg(label: str, in_desc, out_desc, ops: List) -> dace.SDFG:
    """Build a standalone SDFG that relayouts ``_inp`` (shape/strides of ``in_desc``) to ``_out``
    (shape/strides of ``out_desc``) via ``ops`` -- the nested SDFG a ``LayoutChange.expand()``
    returns. Both descriptors come from the parent SDFG (the node's in/out arrays)."""
    logical_shape = list(in_desc.shape)
    _, out_map, _ = relayout_map(logical_shape, ops)

    sdfg = dace.SDFG(f"{label}_sdfg")
    sdfg.add_array("_inp", in_desc.shape, in_desc.dtype, in_desc.storage, strides=in_desc.strides)
    sdfg.add_array("_out", out_desc.shape, out_desc.dtype, out_desc.storage, strides=out_desc.strides)
    state = sdfg.add_state(f"{label}_state", is_start_block=True)
    _emit_relayout_copy(state, "_inp", "_out", logical_shape, out_map)
    return sdfg
