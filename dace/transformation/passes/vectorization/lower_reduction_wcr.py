# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a map-body reduction WCR into the tile-foldable aug-assign form.

Nesting an innermost map body duplicates its reduction WCR onto an inner
``src -[wcr]-> acc`` edge, a loose reduction the tile emitter cannot fold.
Rewrite it to ``acc = acc <op> src`` (tile-in + scalar-out), which
``ConvertTaskletsToTileOps`` folds to a ``TileReduce``. The cross-tile reduction
stays on the boundary ``NSDFG -> AccessNode -[wcr]-> MapExit`` chain.

``WCRToAugAssign`` refuses this rewrite generically; it is sound here only
because the tile emitter folds the result to a ``TileReduce``.
"""
import copy

from dace import SDFG, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.dataflow.wcr_conversion import _wcr_augassign_body
from dace.ordered import OrderedSet


def lower_reduction_wcr_in_body(inner_sdfg: SDFG, tiled: bool = True) -> int:
    """Resolve every ``src -[wcr]-> acc`` reduction edge in ``inner_sdfg``, leaving no loose WCR.

    :param inner_sdfg: The body NestedSDFG to rewrite in place.
    :param tiled: ``True`` rewrites to ``acc = acc <op> src`` (folds to ``TileReduce``).
        ``False`` (step-1 postamble, never tiled) just drops the WCR: the boundary chain
        already sums across iterations.
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
            tasklet = state.add_tasklet('reduce_accum', OrderedSet(('__in1', '__in2')), {'__out'},
                                        f"__out = {_wcr_augassign_body(memlet.wcr)}")
            state.add_edge(state.add_access(acc), None, tasklet, '__in1',
                           Memlet(data=acc, subset=copy.deepcopy(acc_subset)))
            # __in2 subset must match edge.src's rank-1 scalar descriptor, not acc_subset:
            # a rank-2 acc_subset (e.g. C[i,j] -> (1,1)) trips a dimension-mismatch check.
            if src_subset is not None:
                in2_subset = copy.deepcopy(src_subset)
            else:
                src_desc = inner_sdfg.arrays.get(edge.src.data)
                in2_subset = (subsets.Range.from_array(src_desc) if src_desc is not None else copy.deepcopy(acc_subset))
            state.add_edge(edge.src, None, tasklet, '__in2', Memlet(data=edge.src.data, subset=in2_subset))
            state.add_edge(tasklet, '__out', dst, None, Memlet(data=acc, subset=copy.deepcopy(acc_subset)))
            state.remove_edge(edge)
            rewritten += 1
    return rewritten
