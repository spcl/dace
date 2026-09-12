# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Broadcast`` library node -- Fortran ``SPREAD`` and NumPy ``broadcast_to``.

Two spellings of one replication, picked by ``dim``:

* ``dim`` an integer -- Fortran ``SPREAD(SOURCE, DIM, NCOPIES)``: insert a new axis at
  position ``DIM`` (1-based) of size ``NCOPIES`` and replicate ``SOURCE`` along it.  The
  result is rank ``rank(SOURCE) + 1``, and the source uses the other output axes
  positionally.
* ``dim`` ``None`` -- NumPy ``broadcast_to``: right-align the source's axes against the
  result's, stretch every source axis of extent 1 across the result axis it lines up
  with, and leave the leading result axes to be filled by the source entire.  This is
  the general case; SPREAD is only the shape it cannot say, because inserting an axis in
  the MIDDLE is not a right-aligned stretch.

Pure expansion: a single Map writes the broadcasted output in parallel, one tasklet per
output element reading the source position the rule above gives it.  DaCe's scheduler
picks the OpenMP / GPU lowering for the Map.
"""
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandBroadcastPure(ExpandTransformation):
    """Pure expansion -- a Map writing each output position from the matching source element."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        desc_src, desc_dst, dim_zero = node.validate(parent_sdfg, parent_state)
        dtype = desc_src.dtype.base_type
        src_shape = list(desc_src.shape)
        dst_shape = list(desc_dst.shape)
        out_rank = len(dst_shape)

        sdfg = dace.SDFG(node.label + "_sdfg")
        # Carry the operand strides, not just the shape: a compact declaration reads a strided view (``a[:, 0:2*h:2]``) at the wrong offsets and returns a plausible array of the right shape and dtype.
        sdfg.add_array("_src", src_shape, dtype, strides=desc_src.strides, storage=desc_src.storage)
        sdfg.add_array("_dst", dst_shape, dtype, strides=desc_dst.strides, storage=desc_dst.storage)

        state = sdfg.add_state()
        map_rng = {f"__o{d}": f"0:{dst_shape[d]}" for d in range(out_rank)}
        if dim_zero is None:
            # NumPy: right-align, and read a source axis of extent 1 at 0 for every
            # iteration of the result axis it lines up with.
            offset = out_rank - len(src_shape)
            src_subs = ["0" if extent == 1 else f"__o{offset + k}" for k, extent in enumerate(src_shape)]
        else:
            # SPREAD: skip the inserted axis -- the source is rank ``out_rank - 1`` and uses
            # the OTHER output iterators positionally.
            src_subs = [f"__o{d}" for d in range(out_rank) if d != dim_zero]
        if not src_subs:  # rank-0 source -> length-1 broadcast over the new axis
            src_subs = ["0"]
        src_sub = ", ".join(src_subs)
        dst_sub = ", ".join([f"__o{d}" for d in range(out_rank)])
        state.add_mapped_tasklet(
            name="_broadcast",
            map_ranges=map_rng,
            inputs={"__in": dace.Memlet(f"_src[{src_sub}]")},
            code="__out = __in",
            outputs={"__out": dace.Memlet(f"_dst[{dst_sub}]")},
            external_edges=True,
        )
        return sdfg


@dace.library.node
class Broadcast(dace.sdfg.nodes.LibraryNode):
    """Replicate a source array across the destination's shape.

    * ``dim`` an integer  --  Fortran ``SPREAD``: the 1-based axis into which the new
      replicated dimension is inserted.  ``rank(dst)`` must be ``rank(src) + 1``, and
      ``NCOPIES`` is read off the destination descriptor, so there is no separate
      property for it.
    * ``dim`` ``None``  --  NumPy ``broadcast_to``: right-align and stretch, for any
      ``rank(dst) >= rank(src)``.

    ``dim`` defaults to ``1`` so an unqualified node keeps meaning SPREAD.
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
        """:returns: ``(desc_src, desc_dst, dim_zero)``, ``dim_zero`` being the 0-based
            insert position, or ``None`` for the right-aligned NumPy broadcast.

        :raises ValueError: if the shapes cannot broadcast under the selected rule.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or in_edges[0].dst_conn != "_src":
            raise ValueError("Broadcast requires a `_src` input")
        if len(out_edges) != 1 or out_edges[0].src_conn != "_dst":
            raise ValueError("Broadcast requires a `_dst` output")
        desc_src = sdfg.arrays[in_edges[0].data.data]
        desc_dst = sdfg.arrays[out_edges[0].data.data]
        src_rank = len(desc_src.shape)
        dst_rank = len(desc_dst.shape)
        if self.dim is None:
            if dst_rank < src_rank:
                raise ValueError(f"Broadcast: dst rank must be at least the src rank; "
                                 f"got src={src_rank}, dst={dst_rank}")
            offset = dst_rank - src_rank
            for k, extent in enumerate(desc_src.shape):
                if extent == 1:
                    continue
                # Tri-state: only an answer of False is a proven mismatch. Symbolic extents that
                # cannot be decided here are left to the caller, as everywhere else in DaCe.
                if symbolic.equal(extent, desc_dst.shape[offset + k]) is False:
                    raise ValueError(f"Broadcast: src axis {k} has extent {extent}, which neither is 1 "
                                     f"nor matches dst axis {offset + k} ({desc_dst.shape[offset + k]})")
            return desc_src, desc_dst, None
        if dst_rank != src_rank + 1:
            raise ValueError(f"Broadcast: dst rank must be src rank + 1; got src={src_rank}, dst={dst_rank}")
        if not (1 <= self.dim <= dst_rank):
            raise ValueError(f"Broadcast: dim={self.dim} out of range for dst rank-{dst_rank}")
        return desc_src, desc_dst, self.dim - 1


@oprepo.replaces('dace.libraries.standard.broadcast')
@oprepo.replaces('dace.libraries.standard.Broadcast')
def broadcast_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, src, dst, *, dim=1):
    src_in = state.add_read(src)
    dst_w = state.add_write(dst)
    node = Broadcast("broadcast", dim=dim)
    state.add_node(node)
    state.add_edge(src_in, None, node, '_src', mm.Memlet(src))
    state.add_edge(node, '_dst', dst_w, None, mm.Memlet(dst))
    return []
