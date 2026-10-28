# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn the in-place updates ``X[s] = X[s] + r`` a refused loop carries into ``X[s] += r`` (WCR), so it maps.

lavamd's box loop accumulates each box's forces into the box's own slice of ``fv``::

    for l in range(n_boxes):
        fv[box_offsets[l] : box_offsets[l] + ppb, k] += forces(l)

The slice start is data, so ``LoopToMap`` cannot prove two boxes write apart and keeps the loop sequential. The
update is an accumulation, though: as ``fv[...] += forces`` it is the same arithmetic whichever boxes overlap, and a
loop whose only conflicting writes are WCR ones is a map (the WCR stays atomic across iterations). ``AugAssignToWCR``
leaves such an update alone inside its own map -- every map index addresses one element, so the map has no conflict
of its own -- which is right until an enclosing loop is to be lifted.

Only the containers a refused loop names are rewritten, only when their updates as WCR let it map
(:meth:`~dace.transformation.passes.parallelization.parallelize_loops.ParallelizeLoops.rescuing_wcr`), and only their
in-place updates by a free tasklet or one in a top-level map.
"""

from typing import Any

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.interstate.loop_to_map import Refusal
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.eliminate_copy_back import EliminateCopyBack
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops


def updated_in_state(state: SDFGState) -> set[str]:
    """The containers a tasklet in ``state`` both reads and writes: the candidates for an in-place update."""
    names: set[str] = set()
    for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
        reads = {e.data.data for e in state.in_edges(tasklet) if not e.data.is_empty()}
        names |= reads & {e.data.data for e in state.out_edges(tasklet) if not e.data.is_empty()}
    return names


def updated_containers(loop: LoopRegion, per_state: dict[SDFGState, set[str]]) -> set[str]:
    """:func:`updated_in_state` over ``loop``'s states, each computed once into ``per_state`` (nested loops share
    their states' entries)."""
    names: set[str] = set()
    for state in [b for b in loop.all_control_flow_blocks() if isinstance(b, SDFGState)]:
        if state not in per_state:
            per_state[state] = updated_in_state(state)
        names |= per_state[state]
    return names


def updates_of(state: SDFGState, name: str) -> list[tuple[nodes.Tasklet, AugAssignToWCR | None]]:
    """Each tasklet in ``state`` that reads and writes ``name``, with its matched ``X -> [map ->] tasklet [-> map] ->
    X`` update that may become WCR (``None`` when it is no such update)."""
    found = []
    for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
        ins = [e for e in state.in_edges(tasklet) if not e.data.is_empty() and e.data.data == name]
        outs = [e for e in state.out_edges(tasklet) if not e.data.is_empty() and e.data.data == name]
        if not ins or not outs:
            continue
        found.append((tasklet, update_match(state, tasklet, ins, outs)))
    return found


def update_match(state: SDFGState, tasklet: nodes.Tasklet, ins: list, outs: list) -> AugAssignToWCR | None:
    """The update of a free tasklet (``f[j] = f[j] - d`` inside a pair loop) or one in a top-level map."""
    entry = state.entry_node(tasklet)
    if entry is not None and state.entry_node(entry) is not None:
        return None
    if len(ins) != 1 or len(outs) != 1:
        return None
    source = state.memlet_path(ins[0])[0].src
    target = state.memlet_path(outs[0])[-1].dst
    if not (isinstance(source, nodes.AccessNode) and isinstance(target, nodes.AccessNode)):
        return None
    match = {
        AugAssignToWCR.input: state.node_id(source),
        AugAssignToWCR.tasklet: state.node_id(tasklet),
        AugAssignToWCR.output: state.node_id(target),
    }
    expr = 0
    if entry is not None:
        match[AugAssignToWCR.map_entry] = state.node_id(entry)
        match[AugAssignToWCR.map_exit] = state.node_id(state.exit_node(entry))
        expr = 1
    xform = AugAssignToWCR()
    xform.setup_match(state.sdfg, state.parent_graph.cfg_id, state.block_id, match, expr)
    return xform if xform.can_be_applied(state, expr, state.sdfg, permissive=True) else None


def loop_states(loop: LoopRegion) -> list[SDFGState]:
    return [b for b in loop.all_control_flow_blocks() if isinstance(b, SDFGState)]


def all_convertible(loop: LoopRegion, names: set[str]) -> bool:
    """Whether every update of every container in ``names`` becomes WCR: the loop maps only with all of them."""
    return all(xform is not None for name in names for st in loop_states(loop) for _, xform in updates_of(st, name))


def convert_updates(loop: LoopRegion, name: str) -> int:
    """Turn every update of ``name`` in ``loop`` into WCR; :returns: how many. A rewrite may fission its state, so the
    states are listed afresh after each."""
    converted = 0
    while True:
        for state in loop_states(loop):
            xform = next((x for _, x in updates_of(state, name) if x is not None), None)
            if xform is not None:
                xform.apply(state, state.sdfg)
                converted += 1
                break
        else:
            return converted


def read_only_by_updates(sdfg: SDFG, loop: LoopRegion, name: str) -> bool:
    """Whether every read of ``name`` in ``loop`` is a tasklet that writes ``name`` back: an update's own read. Any
    other reader (householder's ``V @ Q[k:, :]`` beside ``Q[k:, :] -= ...``) sees the accumulated value, which a WCR
    does not order."""
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        for node in block.data_nodes():
            if node.data != name:
                continue
            for edge in block.out_edges(node):
                if edge.data.is_empty():
                    continue
                reader = block.memlet_path(edge)[-1].dst
                if not isinstance(reader, nodes.Tasklet):
                    return False
                if not any(not e.data.is_empty() and e.data.data == name for e in block.out_edges(reader)):
                    return False
    return name not in loop_analysis.control_reads(loop)


def writes_move_with_the_iteration(loop: LoopRegion, name: str) -> bool:
    """Whether every write of ``name`` in ``loop`` addresses through the loop variable or a symbol the body defines:
    a scatter whose iterations mostly write apart. A write of the same elements every iteration (``out += w[r] *
    x``) is a reduction over the loop, which the loop's own parallel axes serve better than atomics across it."""
    moving = loop_analysis.body_local_symbols(loop) | {loop.loop_variable}
    found = False
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        for node in block.data_nodes():
            if node.data != name:
                continue
            for edge in block.in_edges(node):
                if edge.data.is_empty():
                    continue
                subset = edge.data.get_dst_subset(edge, block)
                if subset is None or not {str(sym) for sym in subset.free_symbols} & moving:
                    return False
                found = True
    return found


class CarriedAccumulationToWCR(ppl.Pass):
    """WCR for the in-place accumulations that alone keep a loop from mapping (``EliminateCopyBack`` first)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many updates became WCR, or ``None`` if none."""
        EliminateCopyBack().apply_pass(sdfg, {})
        converted = 0
        for owner in sdfg.all_sdfgs_recursive():
            per_state: dict[SDFGState, set[str]] = {}
            for loop in [r for r in owner.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                # A loop with no in-place update has nothing to rewrite: no LoopToMap probe (cloudsc has hundreds).
                updated = updated_containers(loop, per_state)
                if not updated:
                    continue
                # The refused containers whose updates, as WCR, let the loop map; none when the loop maps without
                names = ParallelizeLoops().rescuing_wcr(
                    loop,
                    lambda r: (
                        r.kind is Refusal.NON_UNIQUE_WRITE
                        and r.container in updated
                        and writes_move_with_the_iteration(loop, r.container)
                        and read_only_by_updates(owner, loop, r.container)
                    ),
                )
                if not names or not all_convertible(loop, names):
                    continue
                for name in sorted(names):
                    converted += convert_updates(loop, name)
                per_state.clear()
        return converted or None
