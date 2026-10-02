# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Broadcast`` library node: Fortran ``SPREAD`` and NumPy ``broadcast_to``."""
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.standard.helper import broadcast_indices, broadcast_map_expansion
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandBroadcastPure(ExpandTransformation):
    """One map writing each output element from the source element it broadcasts from."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        src, dst, axis = node.validate(parent_sdfg, parent_state)
        return broadcast_map_expansion(node.label, parent_sdfg, {'_src': (src.data, axis)}, ('_dst', dst.data),
                                       '_dst_v = _src_v')


@dace.library.node
class Broadcast(dace.sdfg.nodes.LibraryNode):
    """Replicate a source array across the destination's shape.

    * ``dim`` an integer: Fortran ``SPREAD``, the 1-based axis of the destination the source lacks.
      ``NCOPIES`` is that axis' extent in the destination.
    * ``dim`` ``None``: NumPy ``broadcast_to``, right-align the axes and stretch those of extent 1.
    """

    implementations = {"pure": ExpandBroadcastPure}
    default_implementation = "pure"

    dim = dace.properties.Property(dtype=int,
                                   default=1,
                                   allow_none=True,
                                   desc="Fortran 1-based axis position of the new replicated dimension, or "
                                   "None for a right-aligned NumPy broadcast.")

    def __init__(self, name, *, dim=1, **kwargs):
        super().__init__(name, inputs={"_src"}, outputs={"_dst"}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        """:returns: ``(src_edge, dst_edge, axis)``, ``axis`` being the 0-based axis the source lacks, or
            ``None`` for the NumPy rule.

        :raises ValueError: if the shapes cannot broadcast under the selected rule.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or in_edges[0].dst_conn != "_src":
            raise ValueError("Broadcast requires a `_src` input")
        if len(out_edges) != 1 or out_edges[0].src_conn != "_dst":
            raise ValueError("Broadcast requires a `_dst` output")
        dst_shape = out_edges[0].data.subset.size()
        axis = None if self.dim is None else self.dim - 1
        if axis is not None and not 0 <= axis < len(dst_shape):
            raise ValueError(f"Broadcast: dim={self.dim} out of range for dst rank-{len(dst_shape)}")
        try:
            broadcast_indices(in_edges[0].data.subset.size(), dst_shape, axis)
        except ValueError as ex:
            raise ValueError(f"Broadcast: {ex}") from ex
        return in_edges[0], out_edges[0], axis


@oprepo.replaces('dace.libraries.standard.broadcast')
@oprepo.replaces('dace.libraries.standard.Broadcast')
def broadcast_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, src, dst, *, dim=1):
    node = Broadcast("broadcast", dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read(src), None, node, '_src', mm.Memlet(src))
    state.add_edge(node, '_dst', state.add_write(dst), None, mm.Memlet(dst))
    return []
