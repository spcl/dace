# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Read a value the same iteration just stored from a register instead of from memory."""

import copy
from collections import OrderedDict
from typing import Any

from dace import SDFG, Memlet, properties, subsets
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


def point() -> subsets.Range:
    """A fresh one-element subset for the forwarding transient -- subsets are never shared."""
    return subsets.Range([(0, 0, 1)])


def in_same_scope(state: SDFGState, node: nodes.Node, other: nodes.Node) -> bool:
    """Whether ``node`` and ``other`` sit under the same map entry (or both at the top)."""
    return state.entry_node(node) is state.entry_node(other)


def loop_body_states(sdfg: SDFG) -> "OrderedDict[SDFGState, list[str]]":
    """Every state inside a loop, mapped to the iteration variables of the loops enclosing it."""
    found: OrderedDict[SDFGState, list[str]] = OrderedDict()
    for region in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(region, LoopRegion) or not region.loop_variable:
            continue
        for state in region.states():
            found.setdefault(state, []).append(region.loop_variable)
    return found


def forwardable_store(
    state: SDFGState, node: nodes.AccessNode, loop_variables: list[str]
) -> tuple[Any, list[Any], subsets.Range] | None:
    """Match a store on ``node`` that the same state reads back at the same element."""
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


def forward_one(
    state: SDFGState, node: nodes.AccessNode, write: Any, reads: list[Any], write_subset: subsets.Range
) -> None:
    """Route the stored value through a fresh transient and feed the same-iteration reads from it."""
    sdfg = state.sdfg
    name, _ = sdfg.add_scalar(f"{node.data}_fwd", sdfg.arrays[node.data].dtype, transient=True, find_new_name=True)
    forwarded = state.add_access(name)
    src, src_conn = write.src, write.src_conn
    # Where the stored value CAME from is preserved verbatim (a tasklet connector carries no
    # subset; an access node's element does); only where it lands changes.
    origin = copy.deepcopy(write.data.get_src_subset(write, state)) if isinstance(src, nodes.AccessNode) else None
    state.remove_edge(write)
    state.add_edge(src, src_conn, forwarded, None, Memlet(data=name, subset=point(), other_subset=origin))
    state.add_edge(
        forwarded, None, node, None, Memlet(data=node.data, subset=copy.deepcopy(write_subset), other_subset=point())
    )
    for read in reads:
        dst, dst_conn = read.dst, read.dst_conn
        # Where the read LANDED is the reader's business and is preserved verbatim; only where it came from changes.
        landing = copy.deepcopy(read.data.get_dst_subset(read, state)) if isinstance(dst, nodes.AccessNode) else None
        state.remove_edge(read)
        state.add_edge(forwarded, None, dst, dst_conn, Memlet(data=name, subset=point(), other_subset=landing))


@properties.make_properties
@xf.explicit_cf_compatible
class ForwardStoreToLoad(ppl.Pass):
    """Feed an in-iteration read from the stored value instead of from the array it was stored to."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Forward every proven same-iteration store-to-load in every loop body."""
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
