# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fortran's reductions of a logical mask: ``ALL``, ``ANY`` and ``COUNT``, whole-array or along one ``dim``.

Every non-zero mask value is true: a ``LOGICAL`` mask may hold any non-zero value for ``.TRUE.``.
"""

import dace
from dace import dtypes, library, nodes, properties
from dace.libraries.standard.helper import connector_array, fortran_axis, map_ranges
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg import utils as sdutil
from dace.transformation.transformation import ExpandTransformation


def reduction_sdfg(node: "MaskReduction", state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
    """Casts the mask to the node's ``element`` type (``mask != 0``) and reduces it with a ``Reduce`` node, which picks
    its own implementation."""
    in_edge = sdutil.in_edge_with_name(node, state, node.INPUT_CONNECTOR_NAME)
    out_edge = sdutil.out_edge_with_name(node, state, node.OUTPUT_CONNECTOR_NAME)
    nsdfg = dace.SDFG(f"{node.label}_sdfg")
    shape = connector_array(nsdfg, node.INPUT_CONNECTOR_NAME, in_edge.data, sdfg)
    connector_array(nsdfg, node.OUTPUT_CONNECTOR_NAME, out_edge.data, sdfg)
    nsdfg.add_transient("_truth", shape, node.ELEMENT)

    cast = nsdfg.add_state(f"{node.label}_cast")
    index = ", ".join(f"__i{d}" for d in range(len(shape)))
    cast.add_mapped_tasklet(
        f"{node.label}_cast",
        map_ranges(shape),
        inputs={"__in": dace.Memlet(f"{node.INPUT_CONNECTOR_NAME}[{index}]")},
        code="__out = __in != 0",
        outputs={"__out": dace.Memlet(f"_truth[{index}]")},
        external_edges=True,
    )

    reduce_state = nsdfg.add_state_after(cast, f"{node.label}_reduce")
    axis = fortran_axis(node, node.dim, len(shape))
    reduce = Reduce(f"{node.label}_reduce", node.WCR, axes=None if axis is None else [axis], identity=node.IDENTITY)
    reduce_state.add_node(reduce)
    reduce_state.add_edge(
        reduce_state.add_read("_truth"), None, reduce, "_in", dace.Memlet.from_array("_truth", nsdfg.arrays["_truth"])
    )
    out_name = node.OUTPUT_CONNECTOR_NAME
    reduce_state.add_edge(
        reduce, "_out", reduce_state.add_write(out_name), None, dace.Memlet.from_array(out_name, nsdfg.arrays[out_name])
    )
    return nsdfg


class ExpandMaskReduction(ExpandTransformation):
    """A ``Reduce`` node of the mask's truth values."""

    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return reduction_sdfg(node, parent_state, parent_sdfg)


class ExpandMaskShortCircuit(ExpandTransformation):
    """A sequential loop over a rank-1 mask that stops at the first element deciding ``ALL`` or ``ANY``."""

    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        in_edge = sdutil.in_edge_with_name(node, parent_state, node.INPUT_CONNECTOR_NAME)
        shape = in_edge.data.subset.size()
        if len(shape) != 1 or fortran_axis(node, node.dim, 1) is not None:
            raise NotImplementedError(f"{node.label}: the short circuit reduces a whole rank-1 mask")
        n = shape[0]
        # ALL stops at the first false element, ANY at the first true one
        decided_by = node.IDENTITY == 0

        @dace.program
        def short_circuit(_mask, _out):
            _out[0] = not decided_by
            for i in range(n):
                if (_mask[i] != 0) == decided_by:
                    _out[0] = decided_by
                    break

        nsdfg = dace.SDFG(f"{node.label}_sdfg")
        connector_array(nsdfg, "_mask", in_edge.data, parent_sdfg)
        connector_array(nsdfg, "_out", sdutil.out_edge_with_name(node, parent_state, "_out").data, parent_sdfg)
        return short_circuit.to_sdfg(_mask=nsdfg.arrays["_mask"], _out=nsdfg.arrays["_out"], simplify=True)


# A library expansion belongs to one node class
@library.expansion
class ExpandAllReduction(ExpandMaskReduction):
    pass


@library.expansion
class ExpandAllShortCircuit(ExpandMaskShortCircuit):
    pass


@library.expansion
class ExpandAnyReduction(ExpandMaskReduction):
    pass


@library.expansion
class ExpandAnyShortCircuit(ExpandMaskShortCircuit):
    pass


@library.expansion
class ExpandCountReduction(ExpandMaskReduction):
    pass


@properties.make_properties
class MaskReduction(nodes.LibraryNode):
    """A reduction of a mask, whole-array or along ``dim``, by ``WCR`` from ``IDENTITY`` over ``ELEMENT`` values."""

    INPUT_CONNECTOR_NAME = "_mask"
    OUTPUT_CONNECTOR_NAME = "_out"
    ELEMENT: type[dtypes.typeclass] = dtypes.bool_
    WCR: str
    IDENTITY: int

    dim = properties.Property(dtype=int, default=-1, desc="Fortran 1-based reduction axis; -1 reduces the whole array.")

    def __init__(self, name, dim=-1, *args, **kwargs):
        super().__init__(
            name, *args, inputs={self.INPUT_CONNECTOR_NAME}, outputs={self.OUTPUT_CONNECTOR_NAME}, **kwargs
        )
        self.dim = dim

    def validate(self, sdfg, state):
        in_edge = sdutil.in_edge_with_name(self, state, self.INPUT_CONNECTOR_NAME)
        sdutil.out_edge_with_name(self, state, self.OUTPUT_CONNECTOR_NAME)
        fortran_axis(self, self.dim, in_edge.data.subset.dims())


@library.node
class AllNode(MaskReduction):
    """Fortran ``ALL(mask [, dim])``: whether every element is true."""

    implementations = {"reduction": ExpandAllReduction, "sequential": ExpandAllShortCircuit}
    default_implementation = "reduction"
    WCR = "lambda a, b: a and b"
    IDENTITY = 1


@library.node
class AnyNode(MaskReduction):
    """Fortran ``ANY(mask [, dim])``: whether some element is true."""

    implementations = {"reduction": ExpandAnyReduction, "sequential": ExpandAnyShortCircuit}
    default_implementation = "reduction"
    WCR = "lambda a, b: a or b"
    IDENTITY = 0


@library.node
class CountLibraryNode(MaskReduction):
    """Fortran ``COUNT(mask [, dim])``: the number of true elements."""

    implementations = {"pure": ExpandCountReduction}
    default_implementation = "pure"
    INPUT_CONNECTOR_NAME = "_cnt_in"
    OUTPUT_CONNECTOR_NAME = "_cnt_out"
    ELEMENT = dtypes.int32
    WCR = "lambda a, b: a + b"
    IDENTITY = 0
