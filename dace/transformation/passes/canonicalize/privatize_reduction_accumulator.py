# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize WCR-on-array-element reductions to a scalar accumulator.

After ``AugAssignToWCR + LoopToMap`` (the canonicalize reduction-to-WCR-map
stage), a scalar-reduction loop becomes a parallel map with a WCR write
pointing at an array element, e.g. ``dot[0]``. CPU codegen lowers that WCR
to ``reduce_atomic`` on the array slot -- correct but contended, and
OpenMP's ``reduction(...)`` clause is most portable with a true scalar
variable.

This pass rewrites the shape::

    ... -> Map (parallel) -> body { acc WCR-+= val } -> MapExit
                                                        |
                                                        v (WCR memlet, arr[c])
                                                      arr[c]

into::

    init state:     arr[c] -> scalar
    map state:      ... -> Map -> body { scalar WCR-+= val } -> MapExit -> scalar
    writeback state: scalar -> arr[c]

After this rewrite the per-thread reduction target is a transient scalar.
The downstream WCR codegen can then emit ``#pragma omp parallel for
reduction(op:scalar)`` and the runtime handles per-thread privatization +
the final tree-reduce. Equivalent for sum/product/min/max/and/or/xor; the
scalar seed is whatever ``arr[c]`` held before the map (preserves the
original WCR semantics).

The init's seed-read AND the writeback are unconditional, so this stays
value-preserving even if zero iterations of the map run.
"""
from typing import Optional

from dace import SDFG, data, dtypes, memlet as mm, properties, subsets
from dace.sdfg import SDFGState, nodes
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


@properties.make_properties
@xf.explicit_cf_compatible
class PrivatizeReductionAccumulator(ppl.Pass):
    """Convert WCR-on-array-element reductions to WCR-on-scalar + init + writeback."""

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.States
                | ppl.Modifies.Descriptors)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for state in list(sdfg.all_states()):
            for map_entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                map_exit = state.exit_node(map_entry)
                # WCR writes are on the MapExit's IN-edges (tasklet -> MapExit
                # IN_<name> connector); the matching OUT-edge then forwards the
                # value out of the scope to the eventual array AccessNode.
                for iedge in list(state.in_edges(map_exit)):
                    if _privatize_if_eligible(sdfg, state, map_entry, map_exit, iedge):
                        count += 1
        return count or None


def _privatize_if_eligible(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit,
                           iedge) -> bool:
    """Return True if the WCR was rewritten to target a scalar accumulator."""
    if iedge.data is None or iedge.data.wcr is None:
        return False
    in_conn = iedge.dst_conn
    if not in_conn or not in_conn.startswith("IN_"):
        return False
    out_conn = "OUT_" + in_conn[3:]
    out_edges = [e for e in state.out_edges(map_exit) if e.src_conn == out_conn]
    if len(out_edges) != 1:
        return False
    oedge = out_edges[0]
    arr_node = oedge.dst
    if not isinstance(arr_node, nodes.AccessNode):
        return False
    desc = sdfg.arrays.get(arr_node.data)
    if desc is None or isinstance(desc, data.Scalar):
        # Already a scalar -- nothing to do.
        return False
    # The IN-edge memlet (tasklet -> MapExit) carries the per-iteration write
    # subset -- the slot the reduction actually touches each iter. The OUT-edge
    # carries the union-over-iterations subset (often the whole array range),
    # which is not what we want for the writeback shape.
    write_subset = iedge.data.subset
    if write_subset is None or write_subset.num_elements() != 1:
        return False
    # The slot must not depend on the map's parameters: it has to be the same
    # slot for every iteration. (We're aggregating into a single accumulator;
    # if the slot were a function of the map parameter this wouldn't be a
    # reduction.)
    map_param_set = set(map_entry.map.params)
    if any(s in map_param_set for s in (str(x) for x in write_subset.free_symbols)):
        return False

    parent_graph: ControlFlowRegion = state.parent_graph

    # Allocate a transient scalar to hold the accumulator.
    scalar_name, _ = sdfg.add_scalar(f"_priv_{arr_node.data}", dtype=desc.dtype, transient=True, find_new_name=True)

    # Is the seed value WRITTEN INSIDE THIS STATE (kernel's own ``acc[c] = init``
    # tasklet, fused with the map)? If so the seed source is that write's
    # AccessNode -- NOT a fresh read of the caller-passed buffer (which could
    # hold arbitrary input data). When the in-state init exists, do an
    # *in-state* privatization (no priv_init/priv_wb states); otherwise fall
    # back to the cross-state init + writeback.
    in_state_init_an = None
    for n in state.nodes():
        if not isinstance(n, nodes.AccessNode):
            continue
        if n is arr_node or n.data != arr_node.data:
            continue
        if state.in_degree(n) == 0:
            continue
        # Reject the AN if any inbound edge is itself a WCR write (then it's a
        # different reduction target, not an init).
        if any(e.data is not None and e.data.wcr is not None for e in state.in_edges(n)):
            continue
        in_state_init_an = n
        break

    # --- Redirect the WCR target: rewrite both the in-edge (tasklet -> MapExit)
    # and the out-edge (MapExit -> AccessNode) to refer to the scalar.
    wcr = iedge.data.wcr
    iedge.data.data = scalar_name
    iedge.data.subset = subsets.Range([(0, 0, 1)])
    iedge.data.wcr = wcr  # keep

    if in_state_init_an is not None:
        # In-state pattern: seed _priv_dot from the in-state init AN; emit the
        # writeback in the same state, going BACK into the original ``arr_node``.
        # No priv_init/priv_wb states needed.
        seed_an = state.add_access(scalar_name)
        state.add_edge(in_state_init_an, None, seed_an, None,
                       mm.Memlet(data=arr_node.data, subset=subsets.Range.from_string(str(write_subset))))
        # The map's WCR output now goes to a fresh _priv_dot AN ...
        new_scalar_an = state.add_write(scalar_name)
        state.add_edge(map_exit, oedge.src_conn, new_scalar_an, None,
                       mm.Memlet(data=scalar_name, subset=subsets.Range([(0, 0, 1)])))
        state.remove_edge(oedge)
        # ... then copy back to the post-map ``arr_node`` (writeback).
        state.add_edge(new_scalar_an, None, arr_node, None,
                       mm.Memlet(data=arr_node.data, subset=subsets.Range.from_string(str(write_subset))))
        return True

    # --- Cross-state pattern (no in-state init): init state BEFORE the
    # current state seeds ``_priv_dot`` from the surviving ``arr_node``;
    # writeback state AFTER copies back.
    init_state = parent_graph.add_state_before(state, label=f"priv_init_{scalar_name}")
    init_r = init_state.add_read(arr_node.data)
    init_w = init_state.add_write(scalar_name)
    init_state.add_edge(init_r, None, init_w, None,
                        mm.Memlet(data=arr_node.data, subset=subsets.Range.from_string(str(write_subset))))

    new_scalar_an = state.add_write(scalar_name)
    state.add_edge(map_exit, oedge.src_conn, new_scalar_an, None,
                   mm.Memlet(data=scalar_name, subset=subsets.Range([(0, 0, 1)])))
    state.remove_edge(oedge)
    if state.degree(arr_node) == 0:
        state.remove_node(arr_node)

    wb_state = parent_graph.add_state_after(state, label=f"priv_wb_{scalar_name}")
    wb_r = wb_state.add_read(scalar_name)
    wb_w = wb_state.add_write(arr_node.data)
    wb_state.add_edge(wb_r, None, wb_w, None,
                      mm.Memlet(data=arr_node.data, subset=subsets.Range.from_string(str(write_subset))))

    return True
