# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a layout-algebra op sequence to a materialized relayout in an SDFG: one mapped-tasklet copy
from the logical index to the digit-tuple physical position."""
from typing import List, Tuple

import dace

from dace.libraries.layout.algebra import LayoutMap, compose_ops, identity_map, simplify_ops


def _digit_expr(digit) -> str:
    """The physical index of one output digit: ``(__i<dim> // stride) % extent``, simplified."""
    idx = dace.symbolic.pystr_to_symbolic(f"__i{digit.dim}")
    expr = (idx // digit.stride) % digit.extent
    return str(dace.symbolic.simplify(expr))


def relayout_map(logical_shape: List, ops: List) -> Tuple[List, LayoutMap, List]:
    """Simplify ``ops`` and compose them on the packed-C identity of ``logical_shape``; returns
    ``(simplified_ops, out_map, out_shape)``."""
    dims = list(range(len(logical_shape)))
    simplified = simplify_ops(ops)
    out_map = compose_ops(simplified, base=identity_map(logical_shape, dims))
    if out_map.shuffles:
        # net Shuffle is a data-dependent reorder, not a digit reshape; ShuffleElements lowers that
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
    """Emit a relayout ``out = layout(ops)(in)`` into ``state``; ``in_name`` must be packed-C,
    ``out_name`` is created/replaced with the laid-out shape."""
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
    """Build a standalone SDFG relayouting ``_inp`` (``in_desc``) to ``_out`` (``out_desc``) via ``ops``
    -- the nested SDFG ``LayoutChange.expand()`` returns."""
    logical_shape = list(in_desc.shape)
    _, out_map, _ = relayout_map(logical_shape, ops)

    sdfg = dace.SDFG(f"{label}_sdfg")
    sdfg.add_array("_inp", in_desc.shape, in_desc.dtype, in_desc.storage, strides=in_desc.strides)
    sdfg.add_array("_out", out_desc.shape, out_desc.dtype, out_desc.storage, strides=out_desc.strides)
    state = sdfg.add_state(f"{label}_state", is_start_block=True)
    _emit_relayout_copy(state, "_inp", "_out", logical_shape, out_map)
    return sdfg
