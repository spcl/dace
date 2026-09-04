# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Read a value the same iteration just stored from a register instead of from memory.

TSVC ``s323`` is the shape::

    for i in range(1, N):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]

``a[i]`` is written and read at the SAME ``i``, so it carries nothing; the only loop-carried
edge is ``b[i-1] -> b[i]``, and substituting ``a[i]`` leaves ``b[i] = b[i-1] + c[i]*d[i] +
c[i]*e[i]`` -- a unit-coefficient first-order recurrence, i.e. a prefix sum. Nothing here has
to run sequentially.

The round trip through ``a`` hides that. ``a`` is read AND written at loop-variant subsets, so
every carrier-based matcher counts it as a second carrier alongside ``b``, and the ``b`` update
reads ``a`` rather than the value it was handed. ``SplitStatements`` cannot separate the two
statements either: ``a[i]`` pulls the ``a`` group first while ``b[i-1]`` pulls the ``b`` group
first, and two arrays wanting opposite orders is a refusal.

Forwarding the stored value settles both. The value goes into a transient on its way to the
array, the in-iteration reader is fed from the transient, and the store to ``a`` STAYS -- ``a``
is a declared output. Only one array, ``b``, then crosses between the two statements, so
``SplitStatements`` orders them and ``LoopToScan`` sees the prefix sum it always could lift.

The same-iteration proof is structural, not name-based. The write and the reads sit on ONE
access node in ONE state, where dataflow order makes the out-edges observe what the in-edge
wrote, and every read subset must be the SAME single-element subset as the write. A body that
reads ``a[i-1]`` instead has a different subset -- the value of an earlier iteration, which
this state never wrote -- and is refused. Anything the pass cannot see through is refused too:
a second access node for the array in the state, a view, a ``may_alias`` descriptor, a WCR or
dynamic memlet, or a producer/consumer in another scope.
"""
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, Memlet, data as dt, properties, subsets
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


def point() -> subsets.Range:
    """A fresh one-element subset for the forwarding transient -- subsets are never shared."""
    return subsets.Range([(0, 0, 1)])


def in_same_scope(state: SDFGState, node: nodes.Node, other: nodes.Node) -> bool:
    """Whether ``node`` and ``other`` sit under the same map entry (or both at the top).

    A producer or consumer one scope out reaches the access node through a map boundary, where
    the memlet is the PROPAGATED subset of a whole iteration space rather than the element this
    pass forwards.
    """
    return state.entry_node(node) is state.entry_node(other)


def loop_body_states(sdfg: SDFG) -> 'OrderedDict[SDFGState, List[str]]':
    """Every state inside a loop, mapped to the iteration variables of the loops enclosing it.

    Built by descent from the ``LoopRegion`` s rather than by ascent from the states, so it needs
    no control-flow parent bookkeeping and keeps the pipeline's insertion order.
    """
    found: 'OrderedDict[SDFGState, List[str]]' = OrderedDict()
    for region in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(region, LoopRegion) or not region.loop_variable:
            continue
        for state in region.all_states():
            found.setdefault(state, []).append(region.loop_variable)
    return found


def forwardable_store(state: SDFGState, node: nodes.AccessNode,
                      loop_variables: List[str]) -> Optional[Tuple[Any, List[Any], subsets.Range]]:
    """Match a store on ``node`` that the same state reads back at the same element.

    :param state: The state holding the candidate access node.
    :param node: The access node to test.
    :param loop_variables: Iteration variables of the enclosing loops; the store must depend on
                           one, or there is no per-iteration value to forward and no carrier to
                           remove.
    :returns: ``(write_edge, read_edges, write_subset)``, or ``None`` if the shape is not proven.
    """
    sdfg = state.sdfg
    desc = sdfg.arrays.get(node.data)
    if desc is None or isinstance(desc, dt.View) or not isinstance(desc, dt.Array):
        return None
    # Read off the descriptor, not its __dict__: the property is stored under `_may_alias`, so a
    # dict lookup by the public name silently never refuses anything.
    if desc.may_alias:
        return None
    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)
    if len(in_edges) != 1 or not out_edges:
        return None
    # A second access node for the same array is a second store or a second read whose order
    # against this one is spelled elsewhere; the forwarding story then needs more than this state.
    if any(other is not node and other.data == node.data for other in state.data_nodes()):
        return None
    write = in_edges[0]
    if write.data is None or write.data.wcr is not None or write.data.dynamic:
        return None
    if not isinstance(write.src, (nodes.Tasklet, nodes.AccessNode)) or not in_same_scope(state, node, write.src):
        return None
    write_subset = write.data.get_dst_subset(write, state)
    if not isinstance(write_subset, subsets.Range) or write_subset.num_elements() != 1:
        return None
    if not any(v in {str(s) for s in write_subset.free_symbols} for v in loop_variables):
        return None
    for read in out_edges:
        if read.data is None or read.data.wcr is not None or read.data.dynamic:
            return None
        if not isinstance(read.dst, (nodes.Tasklet, nodes.AccessNode)) or not in_same_scope(state, node, read.dst):
            return None
        if read.data.get_src_subset(read, state) != write_subset:
            return None
    return write, out_edges, write_subset


def forward_one(state: SDFGState, node: nodes.AccessNode, write: Any, reads: List[Any],
                write_subset: subsets.Range) -> None:
    """Route the stored value through a fresh transient and feed the same-iteration reads from it."""
    sdfg = state.sdfg
    name, _ = sdfg.add_scalar(f'{node.data}_fwd', sdfg.arrays[node.data].dtype, transient=True, find_new_name=True)
    forwarded = state.add_access(name)
    src, src_conn = write.src, write.src_conn
    # Where the stored value CAME from is preserved verbatim (a tasklet connector carries no
    # subset; an access node's element does); only where it lands changes.
    origin = copy.deepcopy(write.data.get_src_subset(write, state)) if isinstance(src, nodes.AccessNode) else None
    state.remove_edge(write)
    state.add_edge(src, src_conn, forwarded, None, Memlet(data=name, subset=point(), other_subset=origin))
    state.add_edge(forwarded, None, node, None,
                   Memlet(data=node.data, subset=copy.deepcopy(write_subset), other_subset=point()))
    for read in reads:
        dst, dst_conn = read.dst, read.dst_conn
        # Where the read LANDED is the reader's business and is preserved verbatim; only where it
        # came from changes.
        landing = copy.deepcopy(read.data.get_dst_subset(read, state)) if isinstance(dst, nodes.AccessNode) else None
        state.remove_edge(read)
        state.add_edge(forwarded, None, dst, dst_conn, Memlet(data=name, subset=point(), other_subset=landing))


@properties.make_properties
@xf.explicit_cf_compatible
class ForwardStoreToLoad(ppl.Pass):
    """Feed an in-iteration read from the stored value instead of from the array it was stored to.

    See the module docstring for the shape, the same-iteration proof and the refusals.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Forward every proven same-iteration store-to-load in every loop body.

        :param sdfg: The SDFG to rewrite.
        :returns: The number of stores forwarded, or ``None`` if the SDFG was left untouched.
        """
        count = 0
        for nested in sdfg.all_sdfgs_recursive():
            for state, loop_variables in loop_body_states(nested).items():
                for node in list(state.data_nodes()):
                    match = forwardable_store(state, node, loop_variables)
                    if match is None:
                        continue
                    forward_one(state, node, *match)
                    count += 1
        return count or None
