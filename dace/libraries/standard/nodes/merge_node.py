# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MergeLibraryNode``: the per-element select, Fortran ``MERGE(tsource, fsource, mask)`` and NumPy
``where(mask, t, f)``. Every operand broadcasts against the result by the NumPy rule, with its shape
taken from its own memlet, which covers Fortran's scalar ``MERGE`` variants too."""
import dace
from dace import library, nodes
from dace.libraries.standard.helper import broadcast_indices, broadcast_map_expansion
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandPure(ExpandTransformation):
    """One map doing the per-element select, with both sources converted to the type of the result."""
    environments = []

    @classmethod
    def expansion(cls, node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        from dace.frontend.python.replacements.utils import cast_str  # Avoid import loop

        t, f, mask, out = node.validate(parent_sdfg, parent_state)
        result_type = parent_sdfg.arrays[out.data.data].dtype

        def source(connector: str, edge) -> str:
            value = f'{connector}_v'
            if parent_sdfg.arrays[edge.data.data].dtype == result_type:
                return value
            return f'{cast_str(result_type)}({value})'

        inputs = {node.TRUE_CONNECTOR_NAME: t, node.FALSE_CONNECTOR_NAME: f, node.MASK_CONNECTOR_NAME: mask}
        code = (f'{node.OUTPUT_CONNECTOR_NAME}_v = {source(node.TRUE_CONNECTOR_NAME, t)} '
                f'if {node.MASK_CONNECTOR_NAME}_v else {source(node.FALSE_CONNECTOR_NAME, f)}')
        return broadcast_map_expansion(node.label, parent_sdfg, {
            c: (e.data, None)
            for c, e in inputs.items()
        }, (node.OUTPUT_CONNECTOR_NAME, out.data), code)


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
