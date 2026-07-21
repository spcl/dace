# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fission induction-variable updates out of compound loop bodies.

The existing :class:`~dace.transformation.passes.canonicalize.induction_variable_substitution.InductionVariableSubstitution`
pass only matches *single-tasklet* loop bodies of the shape ``accum = accum OP const``.
Many real kernels (TSVC ``s317`` variants, ICON config-scalar updates, cloudsc
species loops) carry such an IV update *next to* unrelated per-iteration work --
e.g. ::

    for i in range(N):
        scale[0] = scale[0] * 0.99       # loop-invariant slot, IV-eligible
        b[i]     = b[i]     + 1.0        # per-iteration work, NOT IV-eligible

The two statements are *independent* (they share no AccessNodes, no symbols), so
the IV update can be lifted into its own sibling loop without changing semantics.
That sibling loop is then single-tasklet -- exactly what
``InductionVariableSubstitution`` recognises -- and collapses to its ``O(1)``
closed form ``scale[0] = scale[0] * 0.99 ** N``.

This pass runs **before** ``InductionVariableSubstitution`` in the canonicalize
recipe; together they recover the O(N) → O(1) speedup on real-world kernels
where the IV update was muddled with unrelated loop body work.

Scope today:

* Loop body is a single ``SDFGState`` with ≥ 2 tasklets.
* One of those tasklets matches the IV predicate from
  ``induction_variable_substitution._extract_iv`` (Python ``__out = __in OP const``,
  loop-invariant read/write subset, single in/out edge).
* That tasklet's *data-flow component* (the tasklet plus AccessNodes reached
  through its in/out edges, walking only through transients) is *isolated* --
  no edge connects it to any other tasklet in the state.

Out of scope (potential follow-ups):

* multi-state loop bodies (no fission across state boundaries);
* IV components that share an external AccessNode with another statement
  (would require value-flow analysis of the sharing);
* multiple independent IV updates in the same body (this pass fissions one per
  invocation; re-running picks up the next).
"""
import ast
import copy
from typing import Optional, Set, Tuple

import dace
from dace import SDFG, dtypes, nodes, properties, symbolic
from dace.sdfg import SDFGState
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_to_reduce import _chase_forward_to_accum, _one_elem, _uses


def _is_iv_eligible_tasklet(tasklet: nodes.Tasklet, state: SDFGState, loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``tasklet`` alone (ignoring siblings) satisfies the IV pattern.

    Mirrors the per-tasklet checks of
    :func:`induction_variable_substitution._extract_iv` -- Python body of the
    shape ``__out = __in OP const`` over a loop-invariant accumulator slot --
    but without the "loop body has exactly one tasklet" requirement.
    """
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

    in_edges = [e for e in state.in_edges(tasklet) if e.data is not None and not e.data.is_empty()]
    out_edges = [e for e in state.out_edges(tasklet) if e.data is not None and not e.data.is_empty()]
    if len(in_edges) != 1 or len(out_edges) != 1:
        return False
    (in_edge, ) = in_edges
    (write_edge, ) = out_edges
    if in_edge.dst_conn != var_conn:
        return False
    if not isinstance(write_edge.dst, nodes.AccessNode):
        return False

    write_subset = write_edge.data.subset
    if _one_elem(write_subset) != 1:
        return False
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    if _uses(write_subset, loop_var_sym):
        return False

    final_accum, final_subset = _chase_forward_to_accum(state, sdfg, write_edge.dst, write_subset)
    if final_accum not in sdfg.arrays or _uses(final_subset, loop_var_sym):
        return False

    src = in_edge.src
    if not isinstance(src, nodes.AccessNode):
        return False
    return True


def _tasklet_component(state: SDFGState, tasklet: nodes.Tasklet) -> Set:
    """BFS from ``tasklet`` through every connected node in ``state``.

    Walks both in- and out-edges, into AccessNodes and through them; the returned
    set is the connected component of ``tasklet`` in the state's data-flow graph.
    """
    seen: Set = {tasklet}
    frontier = [tasklet]
    while frontier:
        n = frontier.pop()
        for e in list(state.in_edges(n)) + list(state.out_edges(n)):
            for nb in (e.src, e.dst):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
    return seen


def _is_copy_tasklet(t: nodes.Tasklet) -> bool:
    """A pure passthrough tasklet (``__out = __inp``) -- the frontend emits one
    of these for every ``arr[k] = <expr>`` to copy from a compute transient
    back to the named accumulator. It carries no computation, only data flow,
    so it's safe to fold into the IV component."""
    if t.code.language != dtypes.Language.Python:
        return False
    code = (t.code.as_string or "").strip()
    return code in ('__out = __inp', '__out = (__inp)')


def _is_isolated_iv_component(state: SDFGState, tasklet: nodes.Tasklet) -> Optional[Set]:
    """Return the IV tasklet's component if every tasklet in it is either the
    IV tasklet itself or a pure copy (``__out = __inp``) -- otherwise ``None``.

    Copy tasklets are the frontend's ``compute -> tmp -> assign-copy -> accum``
    staging shape; they belong to the IV's data-flow chain, not to any other
    statement, so admitting them keeps real-world SDFGs (with that staging)
    matchable while still refusing genuine cross-statement entanglement.
    """
    component = _tasklet_component(state, tasklet)
    for n in component:
        if isinstance(n, nodes.Tasklet) and n is not tasklet and not _is_copy_tasklet(n):
            return None
    return component


def _split_iv_component_to_sibling_loop(loop: LoopRegion, state: SDFGState, component: Set, sdfg: SDFG,
                                        parent: ControlFlowRegion) -> None:
    """Move the IV component into a fresh sibling loop *before* ``loop``.

    The sibling loop is a deep copy of ``loop`` with the body state's contents
    replaced by the IV component (the rest of ``loop``'s body is left alone),
    spliced into ``parent``'s CFG right before ``loop`` so that subsequent
    statements see the IV update applied. ``loop``'s body state then has the
    IV-component nodes removed.

    :param loop: The original compound-body loop being fissioned.
    :param state: ``loop``'s single body state.
    :param component: The IV component nodes to move (one Tasklet + its AccessNodes).
    :param sdfg: The SDFG owning ``loop``.
    :param parent: The CFG that owns ``loop`` directly.
    """
    iv_loop = copy.deepcopy(loop)
    iv_loop.label = f"_iv_split_{loop.label}"
    parent.add_node(iv_loop, ensure_unique_name=True)  # derived label; wired below by object ref
    # Wire the new IV loop where ``loop`` is wired today: every edge that points
    # at ``loop`` now points at ``iv_loop``; an unconditional edge connects
    # ``iv_loop`` -> ``loop`` so the original loop runs after the IV pre-pass.
    in_edges = list(parent.in_edges(loop))
    for ie in in_edges:
        parent.remove_edge(ie)
        parent.add_edge(ie.src, iv_loop, ie.data)
    parent.add_edge(iv_loop, loop, dace.InterstateEdge())

    # ``iv_loop`` is currently a deep copy of ``loop`` and therefore contains a
    # copy of the entire compound body. Strip everything that isn't part of the
    # IV component (i.e., keep only the IV's deepcopy-equivalent nodes), and on
    # the original side strip the IV component out so the residual body still
    # contains the other statements.
    iv_state = list(iv_loop.nodes())[0]
    component_labels = {id_for(n) for n in component}
    for n in list(iv_state.nodes()):
        if id_for(n) not in component_labels:
            iv_state.remove_node(n)
    for n in list(state.nodes()):
        if id_for(n) in component_labels:
            state.remove_node(n)


def id_for(n) -> Tuple:
    """A stable "structural id" for a node, so we can match deepcopy-pair nodes
    by their visible shape (label/data + type) without relying on Python ``id``.

    ``label`` is missing only on ``Node``, ``EntryNode`` and ``ExitNode``, which are abstract
    bases; every node that can sit in a state resolves it.
    """
    if isinstance(n, nodes.AccessNode):
        return ('access', n.data)
    if isinstance(n, nodes.Tasklet):
        return ('tasklet', n.label, n.code.as_string)
    return (type(n).__name__, n.label)


@properties.make_properties
@xf.explicit_cf_compatible
class HoistInductionVariableUpdates(ppl.Pass):
    """Fission IV-eligible updates out of compound loop bodies into sibling loops.

    Composes with :class:`~dace.transformation.passes.canonicalize.induction_variable_substitution.InductionVariableSubstitution`:
    this pass produces the single-statement loops the IVSub pass requires, and
    IVSub then collapses them to ``O(1)`` closed-form straight-line code.
    """

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Fission every compound-body loop in ``sdfg`` (and nested SDFGs) that
        carries an independent IV update.

        :returns: The number of loops fissioned, or ``None`` if none.
        """
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
            _split_iv_component_to_sibling_loop(loop, state, component, sdfg, parent)
            return True
        return False


__all__ = ['HoistInductionVariableUpdates']
