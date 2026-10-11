# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ArgMin`` / ``ArgMax``: Fortran ``MINLOC`` / ``MAXLOC``, numpy ``argmin`` / ``argmax``.

Two reductions keep every write-conflict resolution a scalar ``min`` / ``max``: the first finds the extreme value,
the second the smallest (largest, for ``back``) index holding it.
"""

import dace
from dace import SDFG, SDFGState, dtypes, library, properties
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.standard.helper import connector_array, fortran_axis, map_ranges, reduced_coordinates
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.utils import prod


def extreme_index_sdfg(node: "ExtremeIndex", state: SDFGState, sdfg: SDFG) -> SDFG:
    """Without ``dim``, ``_idx`` is the subscript of the extreme element, one entry per dimension, as ``MINLOC``
    returns it; with ``dim``, the index along ``dim`` per remaining coordinate."""
    nsdfg = SDFG(f"{node.label}_sdfg")
    shape = connector_array(nsdfg, "_x", sdutil.in_edge_with_name(node, state, "_x").data, sdfg)
    connector_array(nsdfg, "_idx", sdutil.out_edge_with_name(node, state, "_idx").data, sdfg)
    masked = "_mask" in node.in_connectors
    if masked:
        connector_array(nsdfg, "_mask", sdutil.in_edge_with_name(node, state, "_mask").data, sdfg)
    rank = len(shape)
    axis = fortran_axis(node, node.dim, rank)
    dtype = nsdfg.arrays["_x"].dtype
    index_dtype = nsdfg.arrays["_idx"].dtype

    out_ranges, out = reduced_coordinates(shape, axis)
    out_shape = [1] if axis is None else [s for d, s in enumerate(shape) if d != axis] or [1]
    nsdfg.add_transient("__best_val", out_shape, dtype)
    nsdfg.add_transient("__best_idx", out_shape, index_dtype)

    # Without ``dim``, an element is identified by its row-major position among all of them
    if axis is None:
        position = " + ".join(f"__i{d} * {symstr(prod(shape[d + 1 :]))}" for d in range(rank))
        positions = prod(shape)
    else:
        position, positions = f"__i{axis}", shape[axis]
    no_match = -1 if node.back else symstr(positions)
    identity = dtypes.max_value(dtype) if node.FUNC == "min" else dtypes.min_value(dtype)
    pick = "max" if node.back else "min"
    offset = 1 if node.one_based else 0

    init = nsdfg.add_state(f"{node.label}_init")
    init.add_mapped_tasklet(
        "init",
        out_ranges,
        inputs={},
        code=f"__val = {identity}\n__idx = {no_match}",
        outputs={"__val": dace.Memlet(f"__best_val[{out}]"), "__idx": dace.Memlet(f"__best_idx[{out}]")},
        external_edges=True,
    )

    everything = map_ranges(shape)
    element = ", ".join(f"__i{d}" for d in range(rank))
    selected = "__m != 0" if masked else "True"

    def inputs() -> dict[str, dace.Memlet]:
        reads = {"__in": dace.Memlet(f"_x[{element}]")}
        if masked:
            reads["__m"] = dace.Memlet(f"_mask[{element}]")
        return reads

    find_value = nsdfg.add_state_after(init, f"{node.label}_value")
    find_value.add_mapped_tasklet(
        "value",
        everything,
        inputs=inputs(),
        code=f"__out = __in if {selected} else {identity}",
        outputs={"__out": dace.Memlet(f"__best_val[{out}]", wcr=f"lambda a, b: {node.FUNC}(a, b)")},
        external_edges=True,
    )

    find_index = nsdfg.add_state_after(find_value, f"{node.label}_index")
    find_index.add_mapped_tasklet(
        "index",
        everything,
        inputs={**inputs(), "__best": dace.Memlet(f"__best_val[{out}]")},
        code=f"__out = ({position}) if ({selected} and __in == __best) else {no_match}",
        outputs={"__out": dace.Memlet(f"__best_idx[{out}]", wcr=f"lambda a, b: {pick}(a, b)")},
        external_edges=True,
    )

    result = nsdfg.add_state_after(find_index, f"{node.label}_result")
    if axis is None:
        # One subscript per dimension, decoded from the row-major position
        # A one-element memlet makes the connector a scalar
        targets = ["__out"] if rank == 1 else [f"__out[{d}]" for d in range(rank)]
        code = "\n".join(
            f"{target} = (__in // {symstr(prod(shape[d + 1 :]))}) % {symstr(shape[d])} + {offset}"
            for d, target in enumerate(targets)
        )
        decode = result.add_tasklet("decode", {"__in"}, {"__out"}, code)
        result.add_edge(result.add_read("__best_idx"), None, decode, "__in", dace.Memlet("__best_idx[0]"))
        result.add_edge(decode, "__out", result.add_write("_idx"), None, dace.Memlet(f"_idx[0:{rank}]"))
    else:
        result.add_mapped_tasklet(
            "result",
            out_ranges,
            inputs={"__in": dace.Memlet(f"__best_idx[{out}]")},
            code=f"__out = __in + {offset}",
            outputs={"__out": dace.Memlet(f"_idx[{out}]")},
            external_edges=True,
        )
    return nsdfg


class ExpandExtremeIndexPure(ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return extreme_index_sdfg(node, parent_state, parent_sdfg)


@library.expansion
class ExpandArgMinPure(ExpandExtremeIndexPure):
    pass


@library.expansion
class ExpandArgMaxPure(ExpandExtremeIndexPure):
    pass


@properties.make_properties
class ExtremeIndex(nodes.LibraryNode):
    """The index of the extreme element of ``_x``, whole-array or along ``dim``, among the elements an optional
    ``_mask`` selects."""

    FUNC: str

    one_based = properties.Property(dtype=bool, default=True, desc="Return a 1-based index, as Fortran does.")
    back = properties.Property(dtype=bool, default=False, desc="Of equal elements, take the last instead of the first.")
    dim = properties.Property(
        dtype=int, default=None, allow_none=True, desc="Fortran 1-based axis; None reduces the whole array."
    )

    def __init__(self, name, *, one_based=True, back=False, dim=None, mask=False, **kwargs):
        super().__init__(name, inputs={"_x", "_mask"} if mask else {"_x"}, outputs={"_idx"}, **kwargs)
        self.one_based = one_based
        self.back = back
        self.dim = dim

    def validate(self, sdfg, state):
        rank = sdutil.in_edge_with_name(self, state, "_x").data.subset.dims()
        result = sdutil.out_edge_with_name(self, state, "_idx").data.subset
        if fortran_axis(self, self.dim, rank) is None:
            if result.num_elements() != rank:
                raise ValueError(f"{self.label}: _idx must hold one subscript per dimension of _x")
        elif result.dims() != max(rank - 1, 1):
            raise ValueError(f"{self.label}: _idx must have rank {max(rank - 1, 1)}")


@library.node
class ArgMin(ExtremeIndex):
    """Index of the smallest element: Fortran ``MINLOC``, numpy ``argmin``."""

    implementations = {"pure": ExpandArgMinPure}
    default_implementation = "pure"
    FUNC = "min"


@library.node
class ArgMax(ExtremeIndex):
    """Index of the largest element: Fortran ``MAXLOC``, numpy ``argmax``."""

    implementations = {"pure": ExpandArgMaxPure}
    default_implementation = "pure"
    FUNC = "max"


def add_extreme_index(node_class, state: SDFGState, x, idx, one_based, back, dim, mask) -> list:
    node = node_class(node_class.__name__.lower(), one_based=one_based, back=back, dim=dim, mask=mask is not None)
    state.add_edge(state.add_read(x), None, node, "_x", mm.Memlet(x))
    if mask is not None:
        state.add_edge(state.add_read(mask), None, node, "_mask", mm.Memlet(mask))
    state.add_edge(node, "_idx", state.add_write(idx), None, mm.Memlet(idx))
    return []


@oprepo.replaces("dace.libraries.standard.argmin")
@oprepo.replaces("dace.libraries.standard.ArgMin")
def argmin_libnode(pv, sdfg: SDFG, state: SDFGState, x, idx, *, one_based=True, back=False, dim=None, mask=None):
    return add_extreme_index(ArgMin, state, x, idx, one_based, back, dim, mask)


@oprepo.replaces("dace.libraries.standard.argmax")
@oprepo.replaces("dace.libraries.standard.ArgMax")
def argmax_libnode(pv, sdfg: SDFG, state: SDFGState, x, idx, *, one_based=True, back=False, dim=None, mask=None):
    return add_extreme_index(ArgMax, state, x, idx, one_based, back, dim, mask)
