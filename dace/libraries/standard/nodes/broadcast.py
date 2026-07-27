# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Broadcast`` library node -- Fortran ``SPREAD``.

``SPREAD(SOURCE, DIM, NCOPIES)`` inserts a new rank-1 axis at position ``DIM``
(1-based) of size ``NCOPIES``, replicating ``SOURCE`` along it; result rank is ``rank(SOURCE) + 1``.
"""
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
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
        sdfg.add_array("_src", src_shape, dtype)
        sdfg.add_array("_dst", dst_shape, dtype)

        state = sdfg.add_state()
        map_rng = {f"__o{d}": f"0:{dst_shape[d]}" for d in range(out_rank)}
        # Source indices: skip the inserted ``dim_zero`` axis -- the source
        # is ``rank(dst) - 1`` and uses the OTHER output iterators positionally.
        src_subs = []
        for d in range(out_rank):
            if d == dim_zero:
                continue
            src_subs.append(f"__o{d}")
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
    """Fortran ``SPREAD(SOURCE, DIM, NCOPIES)`` -- broadcast along a new axis; ``NCOPIES`` is inferred
    from the destination descriptor at expansion time."""

    implementations = {"pure": ExpandBroadcastPure}
    default_implementation = "pure"

    dim = dace.properties.Property(dtype=int,
                                   default=1,
                                   desc="Fortran 1-based axis position of the new replicated dimension.")

    def __init__(self, name, *, dim=1, **kwargs):
        super().__init__(name, inputs={"_src"}, outputs={"_dst"}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        """-> ``(desc_src, desc_dst, dim_zero)``. Raises if the destination's rank isn't
        ``rank(src)+1`` or ``dim`` is out of range."""
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
        if dst_rank != src_rank + 1:
            raise ValueError(f"Broadcast: dst rank must be src rank + 1; got src={src_rank}, dst={dst_rank}")
        if not (1 <= self.dim <= dst_rank):
            raise ValueError(f"Broadcast: dim={self.dim} out of range for dst rank-{dst_rank}")
        return desc_src, desc_dst, self.dim - 1
