# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MergeLibraryNode``: the per-element select, Fortran ``MERGE(tsource, fsource, mask)`` and NumPy
``where(mask, t, f)``. Every operand broadcasts against the result by the NumPy rule, with its shape
taken from its own memlet, which covers Fortran's scalar ``MERGE`` variants too."""
import dace
from dace import library, nodes
from dace.libraries.standard.helper import broadcast_indices, broadcast_map_expansion
from dace.transformation.transformation import ExpandTransformation
from typing import List


@library.expansion
class ExpandPure(ExpandTransformation):
    """One map doing the per-element select."""
    environments: List[type] = []

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        t, f, mask, out = node.validate(parent_sdfg, parent_state)
        cls = MergeLibraryNode
        inputs = {cls.TRUE_CONNECTOR_NAME: t, cls.FALSE_CONNECTOR_NAME: f, cls.MASK_CONNECTOR_NAME: mask}
        return broadcast_map_expansion(node.label, parent_sdfg, {
            c: (e.data, None)
            for c, e in inputs.items()
        }, (cls.OUTPUT_CONNECTOR_NAME, out.data), '_mrg_out_v = _mrg_t_v if _mrg_mask_v else _mrg_f_v')


@library.node
class MergeLibraryNode(nodes.LibraryNode):
    """Per-element select: ``_mrg_out = _mrg_t if _mrg_mask else _mrg_f``, each input broadcast against
    the result."""

    implementations = {"pure": ExpandPure}
    default_implementation = "pure"

    TRUE_CONNECTOR_NAME = "_mrg_t"
    FALSE_CONNECTOR_NAME = "_mrg_f"
    MASK_CONNECTOR_NAME = "_mrg_mask"
    OUTPUT_CONNECTOR_NAME = "_mrg_out"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs=[self.TRUE_CONNECTOR_NAME, self.FALSE_CONNECTOR_NAME, self.MASK_CONNECTOR_NAME],
                         outputs=[self.OUTPUT_CONNECTOR_NAME],
                         **kwargs)

    def validate(self, sdfg, state):
        """:returns: The edges on the true, false, mask and output connectors, in that order.

        :raises ValueError: unless each connector has exactly one edge and every input broadcasts to the output.
        """
        edges = [[e for e in state.in_edges(self) if e.dst_conn == c]
                 for c in (self.TRUE_CONNECTOR_NAME, self.FALSE_CONNECTOR_NAME, self.MASK_CONNECTOR_NAME)]
        edges.append([e for e in state.out_edges(self) if e.src_conn == self.OUTPUT_CONNECTOR_NAME])
        if any(len(es) != 1 for es in edges):
            raise ValueError(f"{type(self).__name__} expects exactly one edge per connector")
        result = edges[3][0].data.subset.size()
        for edge in (es[0] for es in edges[:3]):
            try:
                broadcast_indices(edge.data.subset.size(), result)
            except ValueError as ex:
                raise ValueError(f"{type(self).__name__}: {edge.dst_conn}: {ex}") from ex
        return tuple(es[0] for es in edges)
