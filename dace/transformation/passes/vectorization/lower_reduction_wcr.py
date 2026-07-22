# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a map-body reduction WCR into the tile-foldable aug-assign form.

``NestInnermostMapBodyIntoNSDFG`` wraps an innermost map body in a NestedSDFG; a scalar
reduction WCR that wrote the map exit (``acc (op)= ...``) is duplicated by
``nest_state_subgraph`` onto the inner body edge ``src -[wcr]-> acc`` (``acc`` a
non-transient output connector). That inner WCR is a loose reduction the tile emitter
cannot fold. Replace it with the explicit ``acc = acc <op> src`` form (read the
accumulator back): a tile input + scalar output that ``ConvertTaskletsToTileOps`` folds to
a ``TileReduce``. The cross-tile reduction stays on the boundary ``NSDFG -> AccessNode
-[wcr]-> MapExit`` chain (OpenMP ``reduction(op:acc)`` clause).

``WCRToAugAssign`` refuses this rewrite -- its nested-SDFG guard correctly rejects a
generic cross-iteration-reduction revert (dropping the WCR would clobber). It is sound
here only because the tile emitter folds the result to a ``TileReduce``.
"""
import copy

from dace import SDFG, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.dataflow.wcr_conversion import _wcr_augassign_body


def lower_reduction_wcr_in_body(inner_sdfg: SDFG, tiled: bool = True) -> int:
    """Resolve every ``src -[wcr]-> acc`` reduction edge (``acc`` a non-transient pure-sink
    :class:`~dace.sdfg.nodes.AccessNode`) inside ``inner_sdfg``, leaving no loose in-body WCR.

    Scoped to one body NestedSDFG (the caller passes each freshly-nested body). ``src`` is an
    AccessNode -- ``NormalizeWCRSource`` has already made every WCR AccessNode-sourced.

    :param inner_sdfg: The body NestedSDFG to rewrite in place.
    :param tiled: ``True`` (a tiled body) rewrites the edge to ``acc = acc <op> src`` -- a
        tile-in + scalar-out reduction the walker folds to a ``TileReduce``. ``False`` (a step-1
        postamble tail, never tiled) just drops the WCR: the body writes one element per
        iteration and the boundary ``NSDFG -> AccessNode -[wcr]-> MapExit`` chain already sums
        across iterations, so no in-body fold is needed.
    :returns: Number of reduction WCR edges resolved.
    """
    rewritten = 0
    for state in inner_sdfg.all_states():
        for edge in list(state.edges()):
            memlet = edge.data
            if memlet is None or memlet.wcr is None:
                continue
            dst = edge.dst
            if not (isinstance(dst, nodes.AccessNode) and state.out_degree(dst) == 0):
                continue
            desc = inner_sdfg.arrays.get(dst.data)
            if desc is None or desc.transient or not isinstance(edge.src, nodes.AccessNode):
                continue
            if not tiled:
                edge.data.wcr = None
                rewritten += 1
                continue
            acc, acc_subset = dst.data, memlet.subset
            src_subset = memlet.get_src_subset(edge, state)
            tasklet = state.add_tasklet('reduce_accum', {'__in1', '__in2'}, {'__out'},
                                        f"__out = {_wcr_augassign_body(memlet.wcr)}")
            state.add_edge(state.add_access(acc), None, tasklet, '__in1',
                           Memlet(data=acc, subset=copy.deepcopy(acc_subset)))
            # ``__in2`` reads the reduction addend from ``edge.src`` -- the ``_wcr_priv_*`` buffer
            # ``NormalizeWCRSource`` interposed, a SCALAR (rank 1) regardless of the accumulator's
            # rank. Its subset must match ``edge.src``'s descriptor, NOT ``acc_subset``: a 2-D
            # single-element accumulator connector (``C[i, j]`` -> ``(1, 1)``) gives a rank-2
            # ``acc_subset`` that on the rank-1 scalar source trips "subset does not match node
            # dimension".
            if src_subset is not None:
                in2_subset = copy.deepcopy(src_subset)
            else:
                src_desc = inner_sdfg.arrays.get(edge.src.data)
                in2_subset = (subsets.Range.from_array(src_desc)
                              if src_desc is not None else copy.deepcopy(acc_subset))
            state.add_edge(edge.src, None, tasklet, '__in2', Memlet(data=edge.src.data, subset=in2_subset))
            state.add_edge(tasklet, '__out', dst, None, Memlet(data=acc, subset=copy.deepcopy(acc_subset)))
            state.remove_edge(edge)
            rewritten += 1
    return rewritten
