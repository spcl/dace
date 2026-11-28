# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fission induction-variable updates out of compound loop bodies."""

import ast
import copy
from typing import Any

import dace
from dace import SDFG, dtypes, nodes, properties, symbolic
from dace.sdfg import SDFGState
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.split_statements import value_edges
from dace.transformation.passes.loop_to_reduce import _chase_forward_to_accum, _one_elem, _uses


def _is_iv_eligible_tasklet(tasklet: nodes.Tasklet, state: SDFGState, loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``tasklet`` alone (ignoring siblings) satisfies the IV pattern."""
    if tasklet.code.language != dtypes.Language.Python:
        return False
    try:
        tree = ast.parse((tasklet.code.as_string or "").strip())
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    rhs = tree.body[0].value
    if not isinstance(rhs, ast.BinOp) or type(rhs.op) not in (ast.Add, ast.Mult):
        return False
    if isinstance(rhs.left, ast.Name) and isinstance(rhs.right, ast.Constant):
        var_conn, const_val = rhs.left.id, rhs.right.value
    elif isinstance(rhs.right, ast.Name) and isinstance(rhs.left, ast.Constant):
        var_conn, const_val = rhs.right.id, rhs.left.value
    else:
        return False
    if not isinstance(const_val, (int, float)):
        return False

    in_edges = value_edges(state.in_edges(tasklet))
    out_edges = value_edges(state.out_edges(tasklet))
    if len(in_edges) != 1 or len(out_edges) != 1:
        return False
    (in_edge,) = in_edges
    (write_edge,) = out_edges
    # A WCR store is ``acc = wcr(acc, rhs)``: not an IV recurrence, so not fissionable as one.
    if in_edge.data.wcr is not None or write_edge.data.wcr is not None:
        return False
    if in_edge.dst_conn != var_conn:
        return False
    if not isinstance(write_edge.dst, nodes.AccessNode):
        return False

    write_subset = write_edge.data.subset
    if _one_elem(write_subset, sdfg.facts()) != 1:
        return False
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    if _uses(write_subset, loop_var_sym):
        return False

    final_accum, final_subset, _ = _chase_forward_to_accum(state, sdfg, write_edge.dst, write_subset, sdfg.facts())
    if final_accum not in sdfg.arrays or _uses(final_subset, loop_var_sym):
        return False

    src = in_edge.src
    if not isinstance(src, nodes.AccessNode):
        return False
    return True


def _tasklet_component(state: SDFGState, tasklet: nodes.Tasklet) -> dict[nodes.Node, None]:
    """BFS from ``tasklet`` over the VALUE edges of ``state``."""
    seen: dict[nodes.Node, None] = {tasklet: None}
    frontier = [tasklet]
    while frontier:
        n = frontier.pop()
        for e in value_edges(state.in_edges(n)) + value_edges(state.out_edges(n)):
            for nb in (e.src, e.dst):
                if nb not in seen:
                    seen[nb] = None
                    frontier.append(nb)
    return seen


def _is_copy_tasklet(t: nodes.Tasklet) -> bool:
    """A pure passthrough tasklet (``__out = __inp``) -- the frontend emits one of these for every ``arr[k] = <expr>``
    to copy from a compute transient back to the named accumulator."""
    if t.code.language != dtypes.Language.Python:
        return False
    code = (t.code.as_string or "").strip()
    return code in ("__out = __inp", "__out = (__inp)")


def _is_isolated_iv_component(state: SDFGState, tasklet: nodes.Tasklet) -> dict[nodes.Node, None] | None:
    """Return the IV tasklet's component if every tasklet in it is either the IV tasklet itself or a pure copy (``__out
    = __inp``) -- otherwise ``None``."""
    component = _tasklet_component(state, tasklet)
    for n in component:
        if isinstance(n, nodes.Tasklet) and n is not tasklet and not _is_copy_tasklet(n):
            return None
        # The fission runs the component FIRST and would drop a boundary-crossing ordering edge.
        for e in state.in_edges(n) + state.out_edges(n):
            if e.data is not None and e.data.is_empty() and (e.src not in component or e.dst not in component):
                return None
    return component


def _split_iv_component_to_sibling_loop(
    loop: LoopRegion, state: SDFGState, component: dict[nodes.Node, None], sdfg: SDFG, parent: ControlFlowRegion
) -> None:
    """Move the IV component into a fresh sibling loop *before* ``loop``."""
    iv_loop = copy.deepcopy(loop)
    iv_loop.label = f"_iv_split_{loop.label}"
    parent.add_node(iv_loop, ensure_unique_name=True)  # derived label; wired below by object ref
    # Wire the new IV loop where ``loop`` is wired today: every edge that points at ``loop`` now points at ``iv_loop``
    in_edges = list(parent.in_edges(loop))
    for ie in in_edges:
        parent.remove_edge(ie)
        parent.add_edge(ie.src, iv_loop, ie.data)
    parent.add_edge(iv_loop, loop, dace.InterstateEdge())

    # ``iv_loop`` is a deep copy of ``loop``: keep the IV component in the copy and everything else in the original.
    iv_state = list(iv_loop.nodes())[0]
    for original, copied in zip(list(state.nodes()), list(iv_state.nodes())):
        if original in component:
            state.remove_node(original)
        else:
            iv_state.remove_node(copied)


@properties.make_properties
@xf.explicit_cf_compatible
class HoistInductionVariableUpdates(ppl.Pass):
    """Fission IV-eligible updates out of compound loop bodies into sibling loops."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Fission every compound-body loop in ``sdfg`` (and nested SDFGs) that carries an independent IV update."""
        hoisted = 0
        # Iterate over a snapshot: fissioning adds new sibling loops that we
        # don't want to walk into in the same pass invocation.
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                if self._try_hoist(cfg, sd):
                    hoisted += 1
        return hoisted or None

    def _try_hoist(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Attempt to fission one IV component out of ``loop``; return success."""
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None or stride != 1:
            return False
        blocks = list(loop.nodes())
        if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
            return False
        state = blocks[0]
        tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
        if len(tasklets) < 2:
            return False
        parent = loop.parent_graph
        if parent is None:
            return False
        for tasklet in tasklets:
            if not _is_iv_eligible_tasklet(tasklet, state, loop, sdfg):
                continue
            component = _is_isolated_iv_component(state, tasklet)
            if component is None:
                continue
            # The split matches nodes by container: another statement on one of these would go with the component
            containers = {n.data for n in component if isinstance(n, nodes.AccessNode)}
            if any(n.data in containers for n in state.data_nodes() if n not in component):
                continue
            _split_iv_component_to_sibling_loop(loop, state, component, sdfg, parent)
            return True
        return False


__all__ = ["HoistInductionVariableUpdates"]
