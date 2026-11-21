# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Closed-form substitution for simple induction-variable loops."""

import ast
import copy
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import numpy as np
import sympy

from dace import SDFG, data, dtypes, nodes, properties, subsets, symbolic
from dace import memlet as mm
from dace.frontend.python import astutils
from dace.optionals import required
from dace.sdfg import SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.sdfg.narrowing import as_basic, as_expr, as_state
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.dead_carried_store import reaches
from dace.transformation.passes.canonicalize.split_statements import value_edges
from dace.transformation.passes.loop_to_reduce import _chase_forward_to_accum, _one_elem, _uses, data_in_edges

#: Builtin names the closed-form expression may mention; it is spliced verbatim into a tasklet body.
SPLICEABLE_BUILTINS = dict.fromkeys(["True", "False", "None", "abs", "min", "max", "int", "float"])

#: AST binop type -> closed-form template ``(init, c, n) -> str``.
CLOSED_FORM: dict[type[ast.operator], Callable[[str, str, str], str]] = {
    ast.Add: lambda init, c, n: f"(({init}) + ({c}) * ({n}))",
    ast.Mult: lambda init, c, n: f"(({init}) * (({c}) ** ({n})))",
}


class _UnwrapTypecasts(ast.NodeTransformer):
    """Strip ``dace.<typeclass>(x)`` calls -- the frontend's defensive type casts around symbolic operands (e.g. ``__in1
    + dace.float64(step)``) -- by replacing each such call with its single argument."""

    TYPECAST_NAMES: dict[str, None] = dict.fromkeys(dtypes.TYPECLASS_STRINGS)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # Match ``dace.<typeclass>(x)``: ``func`` is Attribute(value=Name('dace'), attr=typeclass)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dace"
            and node.func.attr in self.TYPECAST_NAMES
            and len(node.args) == 1
            and not node.keywords
        ):
            return node.args[0]
        return node


@properties.make_properties
@xf.explicit_cf_compatible
class InductionVariableSubstitution(ppl.Pass):
    """Eliminate a single-tasklet ``acc = acc OP const`` loop via closed form."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        # Fixed point: substituting a primary IV frees the symbols derived from it.
        count = 0
        for _round in range(1000):
            progressed = False
            # ``SDFG.free_symbols`` re-derives itself from the whole graph on EVERY access
            sdfg_free_symbols = sdfg.free_symbols
            for node, _parent in list(sdfg.all_nodes_recursive()):
                if not isinstance(node, LoopRegion):
                    continue
                # An early exit makes the trip count data-dependent.
                if loop_analysis.loop_jumps(node):
                    continue
                # Whole-loop collapse, use-site expansion, edge IV, derived symbol, branch hoist, unrolled counter.
                region = node.parent_graph
                # Spelled out, not a tuple of functions: each name is looked up when the round runs.
                if (
                    _try_substitute(region, node, sdfg, sdfg_free_symbols)
                    or try_substitute_use_site_iv(region, node, sdfg, sdfg_free_symbols)
                    or _try_substitute_iedge_iv(region, node, sdfg, sdfg_free_symbols)
                    or _try_substitute_derived_symbol(region, node, sdfg, sdfg_free_symbols)
                    or _hoist_branch_uniform_iv(region, node, sdfg, sdfg_free_symbols)
                    or sink_chained_rewrite(region, node, sdfg, sdfg_free_symbols)
                ):
                    count += 1
                    progressed = True
                    break  # SDFG mutated -> restart the scan on fresh node list
            if not progressed:
                break
        return count or None


def sink_chained_iv_update(loop: LoopRegion, sdfg: SDFG) -> bool:
    """Move a counter update that reaches its step through SSA temporaries to the bottom of the body."""
    from dace.transformation.passes.loop_fission import _linear_blocks

    order = _linear_blocks(loop, allow_assignments=True) if loop.loop_variable else None
    if order is None:
        return False
    chain = [loop.out_edges(block)[0] for block in order[:-1]]
    writers: dict[str, int] = {}
    for edge in chain:
        for name in edge.data.assignments:
            writers[name] = writers.get(name, 0) + 1
    for position, edge in enumerate(chain):
        step = straddled_counter_step(loop, sdfg, chain, position, writers)
        if step is not None:
            move_update_to_bottom(loop, order, chain, position, step)
            return True
    return False


def straddled_counter_step(
    loop: LoopRegion, sdfg: SDFG, chain: list[Edge[InterstateEdge]], position: int, writers: dict[str, int]
) -> symbolic.SymbolicType | None:
    """The constant step of the counter edge ``chain[position]`` updates, if its reads straddle that edge."""
    edge = chain[position]
    if len(edge.data.assignments) != 1:
        return None
    ((name, rhs),) = edge.data.assignments.items()
    if name not in sdfg.symbols or writers[name] != 1 or _symbol_updated_in_other_loop(sdfg, loop, name):
        return None
    resolved = symbolic.pystr_to_symbolic(rhs)
    if not isinstance(resolved, sympy.Expr):
        return None  # a comparison or a boolean: a guard, not a counter
    temps = {
        symbolic.pystr_to_symbolic(lhs): symbolic.pystr_to_symbolic(value)
        for earlier in chain[:position]
        for lhs, value in earlier.data.assignments.items()
        if writers[lhs] == 1 and lhs in sdfg.symbols
    }
    for _depth in range(len(temps)):
        resolved = resolved.subs(temps)
    step = symbolic.simplify(resolved - symbolic.pystr_to_symbolic(name))
    if not as_basic(step).is_number or step == 0 or _consistent_use_side(loop, edge, name) is not None:
        return None
    return step


def move_update_to_bottom(
    loop: LoopRegion,
    order: list[ControlFlowBlock],
    chain: list[Edge[InterstateEdge]],
    position: int,
    step: symbolic.SymbolicType,
) -> None:
    """Drop the counter update on ``chain[position]``: later blocks read ``k + step``, a new bottom edge bumps ``k``."""
    edge = chain[position]
    (name,) = edge.data.assignments
    bumped = f"({name} + {symbolic.symstr(step)})"
    edge.data.assignments = {}
    for later in chain[position + 1 :]:
        later.data.replace_dict({name: bumped}, replace_keys=False)
    for block in order[position + 1 :]:
        block.replace_dict({name: bumped})
    bottom = loop.add_state(f"{loop.label}_iv_bottom")
    loop.add_edge(order[-1], bottom, InterstateEdge(assignments={name: bumped}))


def sink_chained_rewrite(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]) -> bool:
    """:func:`sink_chained_iv_update` in the signature the other rewrites share."""
    return sink_chained_iv_update(loop, sdfg)


def _try_substitute(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]) -> bool:
    """Return True if the loop matched and was replaced; False otherwise."""
    info = _extract_iv(loop, sdfg, sdfg_free_symbols)
    if info is None:
        return False
    accum, accum_subset, op_type, const_val, trip_count = info

    # The closed-form RHS reads the seed via the tasklet's ``__in`` connector, NOT via a bare ``accum[subset]``
    # expression -- the SDFG dataflow is what actually wires the read.
    closed = CLOSED_FORM[op_type]("__in", const_val, symbolic.symstr(trip_count))

    _replace_loop_with_closed_form(parent, loop, accum, accum_subset, closed, sdfg)
    return True


def _is_loop_invariant_symbol(name: str, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]) -> bool:
    """Whether ``name`` refers to an SDFG symbol/constant that the loop does not redefine in its body (so its value is
    stable across iterations)."""
    if name == loop.loop_variable:
        return False
    if name not in sdfg.symbols and name not in sdfg.constants and name not in sdfg_free_symbols:
        return False
    # An assignment anywhere under the loop, a nested branch included, makes it vary.
    return not any(name in e.data.assignments for e in loop.all_interstate_edges())


def step_is_loop_invariant(
    step_names: Iterable[object], loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> bool:
    """Whether a candidate IV STEP, built from ``step_names``, is loop-invariant."""
    return all(_is_loop_invariant_symbol(str(s), loop, sdfg, sdfg_free_symbols) for s in step_names)


class TaskletIV(NamedTuple):
    """One tasklet's ``__out = __in OP const`` induction-variable shape."""

    accum: str  #: container the update eventually writes (after any staging transients)
    subset: str  #: the single-element, loop-invariant accumulator slot
    op_type: type[ast.operator]  #: ``ast.Add`` / ``ast.Mult``
    const_val: Any  #: numeric literal, or a source string over loop-invariant symbols
    in_edge: MultiConnectorEdge[
        mm.Memlet
    ]  #: the carried read edge into the tasklet (``in_edge.dst`` is the tasklet itself)
    write_edge: MultiConnectorEdge[mm.Memlet]  #: the tasklet's update write edge
    reads_accum: bool  #: the carried read traces back to ``accum[subset]`` (a true recurrence)


def _extract_iv(
    loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> tuple[str, str, type, object, object] | None:
    """Pattern-match a SINGLE-tasklet loop body."""
    if not loop.loop_variable:
        return None

    # Body shape first -- the three bound parses below are sympy round trips, and almost no
    # loop is a single-tasklet body.
    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return None
    state = blocks[0]

    tasklet = None
    for n in state.nodes():
        if isinstance(n, nodes.Tasklet):
            if tasklet is not None:
                return None
            tasklet = n
        elif not isinstance(n, nodes.AccessNode):
            return None
    if tasklet is None:
        return None

    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    iv = extract_tasklet_iv(tasklet, state, loop, sdfg, sdfg_free_symbols)
    if iv is None or not iv.reads_accum:
        return None
    # The whole loop goes: a sink other than the accumulator is a copy of its running value that would go with it
    if any(n.data != iv.accum for n in state.sink_nodes()):
        return None

    return iv.accum, iv.subset, iv.op_type, iv.const_val, trip_count(loop, start, end, stride, sdfg)


def trip_count(
    loop: LoopRegion,
    start: symbolic.SymbolicType,
    end: symbolic.SymbolicType,
    stride: symbolic.SymbolicType,
    sdfg: SDFG,
) -> symbolic.SymbolicType:
    """How often ``loop`` over the inclusive ``[start, end]`` runs; an empty range runs it zero times."""
    trip = symbolic.simplify(symbolic.int_floor(end - start, stride) + 1)
    return trip if loop_analysis.provably_runs(loop, 0, sdfg.facts()) else sympy.Max(0, trip)


def tasklet_increment(
    tasklet: nodes.Tasklet, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> tuple[type, str, Any] | None:
    """The operator, the carrier connector and the loop-invariant increment (a number or source text) of a Python
    tasklet ``__out = __in OP step``."""
    if tasklet.code.language != dtypes.Language.Python:
        return None

    try:
        tree = ast.parse((tasklet.code.as_string or "").strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    # Strip the frontend's defensive ``dace.<typeclass>(...)`` casts before pattern
    # matching, so ``__in1 + dace.float64(step)`` matches identically to ``__in1 + step``.
    rhs = _UnwrapTypecasts().visit(tree.body[0].value)
    if not isinstance(rhs, ast.BinOp) or type(rhs.op) not in CLOSED_FORM:
        return None

    # The carrier is a bare input connector, the increment anything loop-invariant (``2.5``, a symbol ``step``).
    in_connector_names = dict.fromkeys(tasklet.in_connectors.keys())

    def _is_carrier(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in in_connector_names

    # EXACTLY one side may be a connector.
    if _is_carrier(rhs.left) and not _is_carrier(rhs.right):
        var_conn, other = rhs.left.id, rhs.right
    elif _is_carrier(rhs.right) and not _is_carrier(rhs.left):
        var_conn, other = rhs.right.id, rhs.left
    else:
        return None

    if isinstance(other, ast.Constant):
        if not isinstance(other.value, (int, float)):
            return None
        const_val = other.value
    else:
        for sub in ast.walk(other):
            if not isinstance(sub, ast.Name) or sub.id == "dace" or sub.id in SPLICEABLE_BUILTINS:
                continue  # a dtype-cast root, not an operand
            # Every remaining name must be a loop-invariant symbol.
            if not step_is_loop_invariant([sub.id], loop, sdfg, sdfg_free_symbols):
                return None
        # Spliced into the closed form as source, ``dace.float64(step)`` included.
        const_val = ast.unparse(other)
    return type(rhs.op), var_conn, const_val


def extract_tasklet_iv(
    tasklet: nodes.Tasklet, state: SDFGState, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> TaskletIV | None:
    """Match ONE tasklet against the ``__out = __in OP const`` IV shape, ignoring its siblings."""
    increment = tasklet_increment(tasklet, loop, sdfg, sdfg_free_symbols)
    if increment is None:
        return None
    op_type, var_conn, const_val = increment

    in_edges = value_edges(state.in_edges(tasklet))
    out_edges = value_edges(state.out_edges(tasklet))
    if len(in_edges) != 1 or len(out_edges) != 1:
        return None
    (in_edge,) = in_edges
    (write_edge,) = out_edges
    # A WCR store is ``acc = wcr(acc, rhs)``, not ``acc = rhs``, so it is not the recurrence the
    # closed form below solves.
    if in_edge.data.wcr is not None or write_edge.data.wcr is not None:
        return None
    if in_edge.dst_conn != var_conn:
        return None
    if not isinstance(write_edge.dst, nodes.AccessNode):
        return None

    write_subset = write_edge.data.subset
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    if _uses(write_subset, loop_var_sym):
        return None
    if _one_elem(write_subset, sdfg.facts()) != 1:
        return None

    final_accum, final_subset, _ = _chase_forward_to_accum(state, sdfg, write_edge.dst, write_subset, sdfg.facts())
    if final_accum not in sdfg.arrays:
        return None
    # An integer accumulator truncates every iteration, which no closed form of a fractional step reproduces
    if isinstance(const_val, float) and np.issubdtype(sdfg.arrays[final_accum].dtype.type, np.integer):
        return None
    # The eventual write subset must also be loop-invariant.
    if _uses(final_subset, loop_var_sym):
        return None

    # Trace the input back to the same accumulator slot: only then is the tasklet a
    # RECURRENCE (``acc = acc OP c``) rather than an unrelated read that happens to add.
    src = in_edge.src
    if not isinstance(src, nodes.AccessNode):
        return None
    desc = sdfg.arrays.get(src.data)
    if desc is None:
        return None
    src_name, src_subset = src.data, in_edge.data.subset
    # ``is_empty()`` FIRST: an empty in-edge is an ORDERING edge (the frontend hangs one off the
    # accumulator to sequence a WAR), never the staging copy that would redirect the trace.
    staged = value_edges(state.in_edges(src))
    if desc.transient and len(staged) == 1:
        pred = staged[0]
        if not isinstance(pred.src, nodes.AccessNode) or pred.data.subset is None:
            return None
        src_name, src_subset = pred.src.data, pred.data.subset
    reads_accum = src_name == final_accum and str(src_subset) == str(final_subset)

    return TaskletIV(final_accum, str(final_subset), op_type, const_val, in_edge, write_edge, reads_accum)


def closed_form_state(
    parent: ControlFlowRegion,
    label: str,
    accum_name: str,
    accum_subset: str,
    closed_form: str,
    is_start_block: bool = False,
) -> SDFGState:
    """A fresh state whose single tasklet writes ``closed_form`` back to ``accum_name[accum_subset]``."""

    new_state = parent.add_state(label, is_start_block=is_start_block)
    accum_r = new_state.add_read(accum_name)
    accum_w = new_state.add_write(accum_name)
    # add_tasklet turns a connector set into the in/out-connector dict verbatim
    tasklet = new_state.add_tasklet(
        label + "_tlt", dict.fromkeys(["__in"]), dict.fromkeys(["__out"]), f"__out = {closed_form}"
    )
    new_state.add_edge(accum_r, None, tasklet, "__in", mm.Memlet(data=accum_name, subset=accum_subset))
    new_state.add_edge(tasklet, "__out", accum_w, None, mm.Memlet(data=accum_name, subset=accum_subset))
    return new_state


def _replace_loop_with_closed_form(
    parent: ControlFlowRegion, loop: LoopRegion, accum_name: str, accum_subset: str, closed_form: str, sdfg: SDFG
) -> None:
    """Swap ``loop`` for a state whose tasklet writes the closed form back to ``accum_name[accum_subset]``."""
    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))

    new_state = closed_form_state(parent, loop.label + "_iv_closed", accum_name, accum_subset, closed_form, was_start)

    for e in in_edges:
        parent.add_edge(e.src, new_state, e.data)
    for e in out_edges:
        parent.add_edge(new_state, e.dst, e.data)
    parent.remove_node(loop)


# Use-site closed-form substitution (SCEV expansion) for a DATA accumulator


def trip_index(loop: LoopRegion, start: symbolic.SymbolicType, stride: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Trip index ``t`` of the iteration running with ``loop_variable``: ``loop_var == start + t*stride``."""
    return symbolic.simplify(symbolic.int_floor(symbolic.pystr_to_symbolic(loop.loop_variable) - start, stride))


class ReplaceConnectorWithClosedForm(ast.NodeTransformer):
    """Splice ``closed_form`` in for every READ of input connector ``conn`` in a tasklet body."""

    def __init__(self, conn: str, closed_form: str) -> None:
        self.conn = conn
        self.closed_form = closed_form

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.conn or not isinstance(node.ctx, ast.Load):
            return node
        return ast.parse(self.closed_form, mode="eval").body


def splice_closed_form(tasklet: nodes.Tasklet, conn: str, closed_form: str) -> None:
    """Rewrite ``tasklet`` so every read of ``conn`` evaluates the IV's closed form instead."""
    tree = ReplaceConnectorWithClosedForm(conn, closed_form).visit(ast.parse((tasklet.code.as_string or "").strip()))
    ast.fix_missing_locations(tree)
    tasklet.code = properties.CodeBlock(astutils.unparse(tree), dtypes.Language.Python)


class UseSitePlan(NamedTuple):
    """The validated rewrite for one data-accumulator IV inside a multi-statement body."""

    pre_node: nodes.AccessNode  #: accumulator version holding the value on entry to the iteration
    post_node: nodes.AccessNode  #: accumulator version the IV update writes
    chain: list[nodes.AccessNode]  #: staging transients between the IV tasklet and ``post_node``
    pre_reads: list[
        MultiConnectorEdge[mm.Memlet]
    ]  #: edges whose consumer reads the PRE-update value (``t`` updates applied)
    post_reads: list[
        MultiConnectorEdge[mm.Memlet]
    ]  #: edges whose consumer reads the POST-update value (``t + 1`` applied)


def landing_version(
    state: SDFGState, sdfg: SDFG, iv: TaskletIV, pre_node: nodes.AccessNode
) -> tuple[list[nodes.AccessNode], nodes.AccessNode] | None:
    """The staging transients the update's write passes on its way to the accumulator, and the version it lands on."""
    chain: list[nodes.AccessNode] = []
    cur = iv.write_edge.dst
    while isinstance(cur, nodes.AccessNode) and cur.data != iv.accum:
        desc = sdfg.arrays.get(cur.data)
        if desc is None or not desc.transient or state.in_degree(cur) != 1 or state.out_degree(cur) != 1:
            return None
        chain.append(cur)
        cur = state.out_edges(cur)[0].dst
    if not isinstance(cur, nodes.AccessNode) or cur.data != iv.accum or cur is pre_node:
        return None
    return chain, cur


def slot_reads(
    state: SDFGState, iv: TaskletIV, entry_versions: list[nodes.AccessNode], post_node: nodes.AccessNode
) -> tuple[list[MultiConnectorEdge[mm.Memlet]], list[MultiConnectorEdge[mm.Memlet]]] | None:
    """The reads of the entry versions and of the updated version, each a whole-slot read by a parseable Python
    tasklet the closed form can be spliced into."""
    pre_reads: list[MultiConnectorEdge[mm.Memlet]] = []
    post_reads: list[MultiConnectorEdge[mm.Memlet]] = []
    for node in entry_versions + [post_node]:
        bucket = post_reads if node is post_node else pre_reads
        for e in state.out_edges(node):
            if e is iv.in_edge:
                continue  # the IV update's own carried read, which the rewrite deletes
            # An ordering edge out of a version we are about to delete would lose the order it enforces.
            if e.data is None or e.data.is_empty():
                return None
            if not isinstance(e.dst, nodes.Tasklet) or e.dst.code.language != dtypes.Language.Python:
                return None
            if e.dst_conn is None or e.data.data != iv.accum or str(e.data.subset) != iv.subset:
                return None
            try:
                ast.parse((e.dst.code.as_string or "").strip())
            except SyntaxError:
                return None  # unparseable consumer -> cannot splice the closed form into it
            bucket.append(e)
    return pre_reads, post_reads


def plan_use_site_substitution(
    state: SDFGState, sdfg: SDFG, tasklet: nodes.Tasklet, iv: TaskletIV
) -> UseSitePlan | None:
    """Validate that ``iv``'s recurrence can be expanded at its use sites, and enumerate them."""
    pre_node = iv.in_edge.src
    if not isinstance(pre_node, nodes.AccessNode) or pre_node.data != iv.accum:
        return None  # the carried read is staged through a transient -> entry value not directly readable
    if iv.in_edge.data.data != iv.accum or str(iv.in_edge.data.subset) != iv.subset:
        return None

    landed = landing_version(state, sdfg, iv, pre_node)
    if landed is None:
        return None
    chain, post_node = landed

    # Ordering (empty) edges are not writes; the frontend hangs them around the update to sequence the WAR.
    versions = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == iv.accum]
    if [n for n in versions if data_in_edges(state, n)] != [post_node]:
        return None  # a second write to the slot -> the value a read sees is no longer this recurrence
    (write_in,) = data_in_edges(state, post_node)
    if write_in.data.wcr is not None:  # an accumulation into the slot, not this recurrence's store
        return None
    written = write_in.data.subset if write_in.data.data == iv.accum else write_in.data.dst_subset
    if written is None or str(written) != iv.subset:
        return None
    # Every unwritten version holds the ENTRY value
    entry_versions = [n for n in versions if n is not post_node]
    reachable_from_update = dict.fromkeys(sdutil.dfs_conditional(state, sources=[post_node]))
    if any(n in reachable_from_update for n in entry_versions):
        return None

    reads = slot_reads(state, iv, entry_versions, post_node)
    if reads is None:
        return None
    pre_reads, post_reads = reads
    # Ordering edges INTO the update are vacuous once the write is gone
    if post_reads and len(state.in_edges(post_node)) != 1:
        return None
    return UseSitePlan(pre_node, post_node, chain, pre_reads, post_reads)


def try_substitute_use_site_iv(
    parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> bool:
    """Expand a DATA accumulator's closed form at every use site, then delete the recurrence."""
    if not loop.loop_variable:
        return False
    # Body shape first -- the three bound parses below are sympy round trips.
    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return False
    state = blocks[0]
    tasklets = []
    for n in state.nodes():
        if isinstance(n, nodes.Tasklet):
            tasklets.append(n)
        elif not isinstance(n, nodes.AccessNode):
            return False  # a Map / NestedSDFG scope hides the per-iteration execution order
    if len(tasklets) < 2:
        return False  # a single-statement body is _try_substitute's whole-loop collapse instead

    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride == 0:
        return False

    for tasklet in tasklets:
        iv = extract_tasklet_iv(tasklet, state, loop, sdfg, sdfg_free_symbols)
        if iv is None or not iv.reads_accum:
            continue
        plan = plan_use_site_substitution(state, sdfg, tasklet, iv)
        if plan is None:
            continue
        apply_use_site_substitution(parent, loop, state, iv, plan, start, end, stride)
        return True
    return False


def apply_use_site_substitution(
    parent: ControlFlowRegion,
    loop: LoopRegion,
    state: SDFGState,
    iv: TaskletIV,
    plan: UseSitePlan,
    start: symbolic.SymbolicType,
    end: symbolic.SymbolicType,
    stride: symbolic.SymbolicType,
) -> None:
    """Commit the rewrite :func:`plan_use_site_substitution` validated."""

    t = trip_index(loop, start, stride)
    # Splice first (the edges still name their consumer + connector), then re-point.
    for reads, applied in ((plan.pre_reads, t), (plan.post_reads, symbolic.simplify(t + 1))):
        count = symbolic.symstr(applied)
        for e in reads:
            splice_closed_form(e.dst, e.dst_conn, CLOSED_FORM[iv.op_type](e.dst_conn, iv.const_val, count))
    # Post-update readers now compute from the entry value, so they read the entry version.
    for e in plan.post_reads:
        state.remove_edge(e)
        state.add_edge(plan.pre_node, e.src_conn, e.dst, e.dst_conn, mm.Memlet(data=iv.accum, subset=iv.subset))

    state.remove_node(iv.in_edge.dst)  # the IV tasklet
    for n in plan.chain:
        state.remove_node(n)
    state.remove_node(plan.post_node)
    # Drop every version of the slot nothing reads any more, along with the ordering edges that
    # only sequenced it against the write we just deleted.
    for n in [v for v in state.nodes() if isinstance(v, nodes.AccessNode) and v.data == iv.accum]:
        if not value_edges(state.out_edges(n)):
            state.remove_node(n)

    # The loop no longer updates the accumulator; materialise the value it used to leave behind.
    trips = trip_count(loop, start, end, stride, parent.sdfg)
    closed = CLOSED_FORM[iv.op_type]("__in", iv.const_val, symbolic.symstr(trips))
    iv_post = closed_form_state(parent, loop.label + "_iv_use_post", iv.accum, iv.subset, closed)
    for oe in list(parent.out_edges(loop)):
        parent.add_edge(iv_post, oe.dst, oe.data)
        parent.remove_edge(oe)
    parent.add_edge(loop, iv_post, InterstateEdge())


# Iedge-based IV substitution (multi-statement bodies)


def _symbol_updated_in_other_loop(sdfg: SDFG, loop: LoopRegion, sym_name: str) -> bool:
    """Whether ``sym_name`` is also stepped by a loop nested inside ``loop``."""
    inner = [r for r in loop.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r is not loop]
    return any(sym_name in (e.data.assignments or {}) for region in inner for e in region.edges())


def branch_increments(br: ControlFlowRegion) -> dict[str, list[tuple[Edge[InterstateEdge], symbolic.SymbolicType]]]:
    """Per symbol, the interstate edges of ``br`` that add a number to it, with that number."""
    incs: dict[str, list[tuple[Edge[InterstateEdge], symbolic.SymbolicType]]] = {}
    for e in br.edges():
        for lhs, rhs in (e.data.assignments or {}).items():
            try:
                delta = symbolic.simplify(
                    as_expr(symbolic.pystr_to_symbolic(rhs)) - as_expr(symbolic.pystr_to_symbolic(lhs))
                )
            except Exception:
                continue
            if as_basic(delta).is_number:
                incs.setdefault(lhs, []).append((e, delta))
    return incs


def uniform_increment(
    loop: LoopRegion,
    cb: ConditionalBlock,
    branches: list[ControlFlowRegion],
    per: list[dict],
    sym: str,
    sdfg: SDFG,
    sdfg_free_symbols: set[str],
) -> tuple[Any, str] | None:
    """The step every branch adds to ``sym`` once, and the side of the conditional the hoisted increment goes on:
    ``'after'`` (increment then read, s124) hoists before it, ``'before'`` after it."""
    if sym == loop.loop_variable or (sym not in sdfg.symbols and sym not in sdfg_free_symbols):
        return None
    if any(len(p[sym]) != 1 for p in per):
        return None  # a branch increments sym more than once
    steps = dict.fromkeys(p[sym][0][1] for p in per)
    if len(steps) != 1:
        return None  # branches disagree on the step
    branch_edges = dict.fromkeys(id(p[sym][0][0]) for p in per)
    if any(sym in (e2.data.assignments or {}) and id(e2) not in branch_edges for e2 in loop.all_interstate_edges()):
        return None  # sym also written outside the per-branch increments (incl. nested) -> not clean
    if sym in {str(s) for code in cb.get_meta_codeblocks() for s in code.get_free_symbols()}:
        return None  # a guard reading the counter: hoisted before it, the increment would change the branch taken
    # Branches not reading ``sym`` do not vote (none reading -> 'after')
    sides = dict.fromkeys(_consistent_use_side(br, p[sym][0][0], sym) for br, p in zip(branches, per))
    sides.pop("unused", None)
    if not sides:
        return next(iter(steps)), "after"
    if len(sides) == 1 and next(iter(sides)) in ("after", "before"):
        return next(iter(steps)), next(iter(sides))
    return None


def hoist_increment(loop: LoopRegion, cb: ConditionalBlock, per: list[dict], sym: str, step: Any, side: str) -> None:
    """Strip ``sym``'s increment from every branch and plant one interstate edge on ``side`` of ``cb``."""
    for p in per:
        e, _ = p[sym][0]
        assigns = dict(e.data.assignments)
        assigns.pop(sym, None)
        e.data.assignments = assigns
    # Asked before adding the hoist state, an isolated second source until it is wired in.
    was_start = loop.start_block is cb
    hoist = loop.add_state(cb.label + "_iv_hoist")
    new_rhs = symbolic.symstr(symbolic.pystr_to_symbolic(sym) + step)
    if side == "after":
        for ie in list(loop.in_edges(cb)):
            loop.add_edge(ie.src, hoist, ie.data)
            loop.remove_edge(ie)
        loop.add_edge(hoist, cb, InterstateEdge(assignments={sym: new_rhs}))
        if was_start:
            loop.start_block = loop.node_id(hoist)
    else:
        for oe in list(loop.out_edges(cb)):
            loop.add_edge(hoist, oe.dst, oe.data)
            loop.remove_edge(oe)
        loop.add_edge(cb, hoist, InterstateEdge(assignments={sym: new_rhs}))


def _hoist_branch_uniform_iv(
    parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> bool:
    """Hoist an increment every branch of a body ``ConditionalBlock`` performs identically onto one iedge, before the
    conditional for post-increment reads and after it for pre-increment reads."""
    for cb in [b for b in loop.nodes() if isinstance(b, ConditionalBlock)]:
        conds = [c for c, _ in cb.branches]
        branches = [br for _, br in cb.branches]
        if len(branches) < 2 or all(c is not None for c in conds):
            continue  # no else branch -> a path skips the increment -> unsound to hoist
        per = [branch_increments(br) for br in branches]
        common = set.intersection(*[set(p) for p in per]) if per else set()
        # sorted: the first symbol hoisted decides whether a second co-incrementing IV (s124/s126/s128) still
        # passes its guards, so the order must not depend on string hashing.
        for sym in sorted(common):
            found = uniform_increment(loop, cb, branches, per, sym, sdfg, sdfg_free_symbols)
            if found is None:
                continue
            hoist_increment(loop, cb, per, sym, *found)
            return True
    return False


def _consistent_use_side(loop: LoopRegion, iv_edge: Edge[InterstateEdge], sym_name: str) -> str | None:
    """Which side of the IV increment ``iv_edge`` every use of ``sym_name`` runs on."""
    before = dict.fromkeys(sdutil.dfs_conditional(loop, sources=[iv_edge.src], reverse=True))
    before[iv_edge.src] = None
    after = dict.fromkeys(sdutil.dfs_conditional(loop, sources=[iv_edge.dst]))
    after[iv_edge.dst] = None

    saw_before = saw_after = False
    # Block uses: any body block that reads ``sym``
    for b in loop.nodes():
        if sym_name in (str(s) for s in b.free_symbols):
            if b in before:
                saw_before = True
            elif b in after:
                saw_after = True
            else:
                return None  # neither strictly before nor after the increment
    # Interstate-edge uses (RHS assignments / condition), excluding the IV edge.
    for e in loop.edges():
        if e is iv_edge or sym_name not in (str(s) for s in e.data.free_symbols):
            continue
        if e.dst in before:  # the edge completes at-or-before the increment's source
            saw_before = True
        elif e.src in after:  # the edge starts at-or-after the increment's destination
            saw_after = True
        else:
            return None
    if saw_before and saw_after:
        return None  # straddles the increment -> per-block offsets needed (unsupported)
    if saw_before:
        return "before"
    if saw_after:
        return "after"
    return "unused"  # no body use -> either offset reproduces the (absent) reads


def _preloop_symbol_value(parent: ControlFlowRegion, loop: LoopRegion, sym_name: str) -> symbolic.SymbolicType | None:
    """The value ``sym_name`` holds on ENTRY to ``loop``, or ``None`` if it is not locally provable."""
    in_edges = list(parent.in_edges(loop))
    if not in_edges:
        return None
    vals: dict[symbolic.SymbolicType, None] = {}
    for e in in_edges:
        rhs = (e.data.assignments or {}).get(sym_name)
        if rhs is None:
            return None
        try:
            vals[symbolic.pystr_to_symbolic(rhs)] = None
        except Exception:
            return None
    if len(vals) != 1:
        return None
    (val,) = vals
    if loop.loop_variable in (str(s) for s in as_basic(val).free_symbols):
        return None  # references the loop variable, which is undefined before the loop
    return val


def derived_definition(
    loop: LoopRegion, e: Any, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> tuple[str, symbolic.SymbolicType] | None:
    """``(sym, f)`` when the unconditional edge ``e`` alone defines the SDFG symbol ``sym := f(i)`` from the loop
    variable and loop invariants."""
    loop_var = loop.loop_variable
    if e.data.condition.as_string not in ("1", "True", "(1)") or len(e.data.assignments) != 1:
        return None
    ((sym, rhs),) = e.data.assignments.items()
    if sym == loop_var or (sym not in sdfg.symbols and sym not in sdfg_free_symbols):
        return None
    try:
        rhs_expr = symbolic.pystr_to_symbolic(rhs)
    except Exception:
        return None
    # An array-dependent value is a data gather: inlined, it would put an array read in memlet subsets
    # (cloudsc's ``iorder_at = IORDER(JL,JM)``).
    if symbolic.arrays(rhs):
        return None
    free = dict.fromkeys(str(s) for s in rhs_expr.free_symbols)
    if sym in free:
        return None  # self-reference -> a recurrence, not a derived symbol
    if not all(s == loop_var or _is_loop_invariant_symbol(s, loop, sdfg, sdfg_free_symbols) for s in free):
        return None  # depends on another loop-carried symbol -> substitute that first (fixed point)
    if any(oe is not e and sym in (oe.data.assignments or {}) for oe in loop.all_interstate_edges()):
        return None  # sym written elsewhere (incl. a NESTED loop, s141) -> not a clean single definition
    return sym, rhs_expr


def derived_body_value(
    parent: ControlFlowRegion, loop: LoopRegion, e: Any, sym: str, rhs_expr: symbolic.SymbolicType
) -> symbolic.SymbolicType | None:
    """The value of ``sym`` the body's uses read: ``f(i)`` after the definition, ``f(i - stride)`` before it when the
    definition runs every iteration and the entry value is ``f(start - stride)``."""
    side = _consistent_use_side(loop, e, sym)
    if side in ("after", "unused"):
        return rhs_expr
    if side != "before":
        return None  # uses straddle the definition -> per-block offsets needed (unsupported)
    start = loop_analysis.get_init_assignment(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or stride is None:
        return None
    if any(oe.data.condition.as_string not in ("1", "True", "(1)") for oe in loop.edges()):
        return None
    entry = _preloop_symbol_value(parent, loop, sym)
    if entry is None:
        return None
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    if symbolic.simplify(entry - rhs_expr.subs(loop_var_sym, start - stride)) != 0:
        return None
    return symbolic.simplify(rhs_expr.subs(loop_var_sym, loop_var_sym - stride))


def materialize_derived_value(
    parent: ControlFlowRegion, loop: LoopRegion, sym: str, rhs_expr: symbolic.SymbolicType
) -> None:
    """Assign the last iteration's ``sym`` after the loop, kept from before it when the loop may not run and a reader
    follows."""
    end = loop_analysis.get_loop_end(loop)
    if end is None:
        return
    start, stride = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_stride(loop)
    last = end if start is None or stride is None else start + symbolic.int_floor(end - start, stride) * stride
    post_val = symbolic.symstr(rhs_expr.subs(symbolic.pystr_to_symbolic(loop.loop_variable), last))
    dsym_post = parent.add_state(loop.label + "_dsym_post")
    for oe in list(parent.out_edges(loop)):
        parent.add_edge(dsym_post, oe.dst, oe.data)
        parent.remove_edge(oe)
    spare = None if start is None or stride is None else trip_count(loop, start, end, stride, loop.sdfg) - 1
    if (
        spare is None
        or symbolic.provably_nonnegative(symbolic.simplify(spare), loop.sdfg.facts())
        or not loop_analysis.read_outside(loop.sdfg, loop, {sym}, set())
    ):
        parent.add_edge(loop, dsym_post, InterstateEdge(assignments={sym: post_val}))
        return
    ran = ConditionalBlock(loop.label + "_dsym_ran")
    branch = ControlFlowRegion(loop.label + "_dsym_set")
    first = branch.add_state(is_start_block=True)
    branch.add_edge(first, branch.add_state(), InterstateEdge(assignments={sym: post_val}))
    ran.add_branch(properties.CodeBlock(f"({symbolic.symstr(spare)}) >= 0"), branch)
    parent.add_node(ran)
    parent.add_edge(loop, ran, InterstateEdge())
    parent.add_edge(ran, dsym_post, InterstateEdge())


def _try_substitute_derived_symbol(
    parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> bool:
    """Inline a symbol defined purely by a loop-variable expression (``sym := f(i)``, no self-reference)."""
    if not loop.loop_variable:
        return False
    for e in loop.edges():
        found = derived_definition(loop, e, sdfg, sdfg_free_symbols)
        if found is None:
            continue
        sym, rhs_expr = found
        body_expr = derived_body_value(parent, loop, e, sym, rhs_expr)
        if body_expr is None:
            continue
        e.data.assignments = {}
        loop.replace_dict({sym: symbolic.symstr(body_expr)})
        materialize_derived_value(parent, loop, sym, rhs_expr)
        return True
    return False


def staged_iedge_rhs(rhs: str, src_state: SDFGState, sdfg: SDFG) -> symbolic.SymbolicType | None:
    """The arithmetic an interstate assignment's RHS stands for when it is staged through a transient, or ``None`` when
    ``rhs`` is not such a staging and should be read as written."""
    desc = sdfg.arrays.get(rhs)
    if desc is None or not desc.transient or desc.total_size != 1 or not isinstance(src_state, SDFGState):
        return None
    writes = [n for n in src_state.data_nodes() if n.data == rhs and src_state.in_degree(n) > 0]
    if len(writes) != 1 or src_state.in_degree(writes[0]) != 1:
        return None
    in_edge = src_state.in_edges(writes[0])[0]
    tasklet = in_edge.src
    if (
        not isinstance(tasklet, nodes.Tasklet)
        or src_state.in_degree(tasklet) > 0
        or src_state.out_degree(tasklet) != 1
        or tasklet.has_side_effects(sdfg)
    ):
        return None
    code = tasklet.code
    if code.language is not dtypes.Language.Python or len(code.code) != 1:
        return None
    stmt = code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if astutils.rname(stmt.targets[0]) != in_edge.src_conn:
        return None
    value = _UnwrapTypecasts().visit(copy.deepcopy(stmt.value))
    try:
        return symbolic.pystr_to_symbolic(astutils.unparse(value))
    except Exception:
        # A body that is not an arithmetic expression (a call, a comparison) carries no value this can stand in for.
        return None


def iedge_iv(loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]) -> tuple[Any, str, Any] | None:
    """The body's one unconditional interstate edge ``sym := sym + step``, ``sym`` and ``step``: ``sym`` an SDFG
    symbol no other edge writes, ``step`` a literal or loop-invariant."""
    iv_candidate = None
    for e in loop.edges():
        if e.data.condition.as_string not in ("1", "True", "(1)"):
            return None
        if not e.data.assignments:
            continue
        if len(e.data.assignments) != 1:
            continue
        ((lhs, rhs),) = e.data.assignments.items()
        try:
            staged = staged_iedge_rhs(rhs, e.src, sdfg)
            rhs_expr = symbolic.pystr_to_symbolic(rhs) if staged is None else staged
            lhs_sym = symbolic.pystr_to_symbolic(lhs)
            diff = symbolic.simplify(rhs_expr - as_expr(lhs_sym))
        except Exception:
            continue  # a comparison or another non-arithmetic right-hand side
        # A numeric literal is always admissible
        if not as_basic(diff).is_number:
            if not as_basic(diff).free_symbols or not step_is_loop_invariant(
                as_basic(diff).free_symbols, loop, sdfg, sdfg_free_symbols
            ):
                continue
        if lhs == loop.loop_variable:
            continue
        if lhs not in sdfg.symbols and lhs not in sdfg_free_symbols:
            continue
        other_writers = [oe for oe in loop.all_interstate_edges() if oe is not e and lhs in oe.data.assignments]
        if other_writers:
            continue
        if iv_candidate is not None:
            return None  # >1 IV pattern; defer to a future multi-IV extension
        iv_candidate = (e, lhs, diff)
    return iv_candidate


def iedge_body_offset(loop: LoopRegion, iv_edge: Any, sym_name: str, norm_iter: Any) -> Any | None:
    """The trips of the increment the body sees: one more when the increment leaves the empty start block or every
    use follows it (s128), the trip index when it enters the unique empty sink (s318) or every use precedes it."""
    src_is_empty_start = (
        iv_edge.src is loop.start_block and isinstance(iv_edge.src, SDFGState) and not iv_edge.src.nodes()
    )
    sinks = [b for b in loop.nodes() if loop.out_degree(b) == 0]
    dst_is_unique_empty_sink = (
        isinstance(iv_edge.dst, SDFGState)
        and not iv_edge.dst.nodes()
        and len(sinks) == 1
        and sinks[0] is iv_edge.dst
        and loop.in_degree(iv_edge.dst) == 1
    )

    if src_is_empty_start:
        body_offset = norm_iter + 1  # update-at-top: body is post-increment
    elif dst_is_unique_empty_sink:
        body_offset = norm_iter  # update-at-bottom: body is pre-increment
    else:
        # Between content blocks: substitutable iff every use is on one side (s128).
        side = _consistent_use_side(loop, iv_edge, sym_name)
        if side == "before":
            body_offset = norm_iter
        elif side == "after":
            body_offset = norm_iter + 1
        elif side == "unused":
            # No body reads: any offset works; only the post-loop value matters.
            body_offset = norm_iter
        else:
            return None
    return body_offset


def _try_substitute_iedge_iv(
    parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: set[str]
) -> bool:
    """Substitute an interstate-edge IV ``sym := sym +/- step`` in the loop body by its closed form."""
    if not loop.loop_variable:
        return False
    iv_candidate = iedge_iv(loop, sdfg, sdfg_free_symbols)
    if iv_candidate is None:
        return False
    iv_edge, sym_name, step = iv_candidate
    if sym_name in loop_analysis.loop_header_names(loop):
        return False  # the trip count reads the counter the body changes

    # Bounds parsed only once an IV iedge exists -- three sympy round trips no shape without one ever needs.
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride == 0:
        return False

    # The IV symbol must be a counter PRIVATE to this loop.
    if _symbol_updated_in_other_loop(sdfg, loop, sym_name):
        return False

    # 2. Offset in TRIPS (``norm_iter``): an increment on the edge out of the empty start block (TOP)
    #    is seen by the whole body (``norm_iter + 1``); one into the unique empty sink (BOTTOM, the
    #    frontend's shape, s318) is not (``norm_iter``).
    sym_sym = symbolic.pystr_to_symbolic(sym_name)
    norm_iter = trip_index(loop, start, stride)

    body_offset = iedge_body_offset(loop, iv_edge, sym_name, norm_iter)
    if body_offset is None:
        return False

    # 3. Closed form: with the increment stripped, ``sym`` is ``sym_init``.
    post_iedge_expr = symbolic.simplify(as_expr(sym_sym) + body_offset * step)

    # 4. Clear the IV iedge first so the substitution does not rewrite it onto itself.
    iv_edge.data.assignments = {}

    loop.replace_dict({sym_name: symbolic.symstr(post_iedge_expr)})

    # 5. Materialize ``sym + trip_count * step`` for later readers (and the next outer iteration) on
    #    the iedge into a spliced ``iv_post`` state; existing exit edges are rerouted from it.
    post_loop_expr = symbolic.simplify(as_expr(sym_sym) + trip_count(loop, start, end, stride, sdfg) * step)
    post_loop_value = symbolic.symstr(post_loop_expr)

    # Two-level counter (s126): if the lone unconditional exit edge only assigns ``sym``, compose into it instead
    exit_edges = list(parent.out_edges(loop))
    if (
        len(exit_edges) == 1
        and exit_edges[0].data.is_unconditional()
        and list((exit_edges[0].data.assignments or {}).keys()) == [sym_name]
    ):
        exit_data = exit_edges[0].data
        composed = symbolic.pystr_to_symbolic(exit_data.assignments[sym_name]).subs(sym_sym, post_loop_expr)
        exit_data.assignments[sym_name] = symbolic.symstr(symbolic.simplify(composed))
        return True

    iv_post = parent.add_state(loop.label + "_iv_post")
    existing_out = list(parent.out_edges(loop))
    for oe in existing_out:
        parent.add_edge(iv_post, oe.dst, oe.data)
        parent.remove_edge(oe)
    iv_edge_out = InterstateEdge(assignments={sym_name: post_loop_value})
    parent.add_edge(loop, iv_post, iv_edge_out)

    return True


# Loop-carried ROTATION substitution: a delay line -> a shifted array read (TSVC s254):: x = b[N-1] for i in range(N):
# a[i] = (b[i] + x) * 0.5 # x holds b[i-1] here x = b[i] becomes a peeled first iteration plus a DOALL loop reading
# ``b[i-1]``.

#: How far the update's stored value may be chased back through staging transients before giving up.
ROTATION_CHASE_LIMIT = 4


class RematInput(NamedTuple):
    """One input connector of a cloned producer: a shifted read, or another clone."""

    conn: str  #: connector on the producer this input feeds
    container: str | None  #: array to read, when the value comes straight from memory
    subset: subsets.Subset | None  #: the element to read, ALREADY shifted back by one stride
    source: "RematSource | None"  #: producer to clone instead, when the body writes the container
    moved: bool  #: whether the shift actually changed the subset


class RematSource(NamedTuple):
    """A pure producer tasklet to re-evaluate at ``i - stride`` in place of the carried read."""

    producer: nodes.Tasklet  #: tasklet that computed the value the update stored
    out_conn: str  #: its single output connector
    inputs: list[RematInput]  #: one entry per input connector, in producer edge order
    dtype: dtypes.typeclass  #: element type the clone's result is stored as
    stage: nodes.AccessNode | None  #: transient version the update read from, when it dies with the update


class RotationPlan(NamedTuple):
    """The validated rewrite for one loop-carried rotation (delay-line) scalar."""

    accum: str  #: the carried scalar container
    src_data: str | None  #: array the rotated value comes from (``None`` when rematerializing)
    src_subset: subsets.Subset | None  #: the element to read, ALREADY shifted back by one stride
    post_node: nodes.AccessNode  #: the accumulator version the update writes
    write_state: SDFGState  #: body state holding that write
    chase: list[tuple[SDFGState, nodes.AccessNode]]  #: nodes only the update keeps alive
    reads: list[tuple[SDFGState, nodes.AccessNode]]  #: entry versions whose reads shift back
    touched: list[str]  #: containers the rewrite may leave dangling
    remat: RematSource | None = None  #: producer to clone, when the carried value is computed


def _subset_at(edge: MultiConnectorEdge[mm.Memlet], node: nodes.AccessNode) -> subsets.Subset | None:
    """The subset ``edge``'s memlet addresses in ``node``'s container (``node`` is one endpoint)."""
    memlet = edge.data
    if memlet is None:
        return None
    if not isinstance(node, nodes.AccessNode) or memlet.other_subset is None:
        return memlet.subset
    return memlet.subset if memlet.data == node.data else memlet.other_subset


def rotation_body_chain(loop: LoopRegion) -> list[SDFGState] | None:
    """The loop body as a straight-line list of states, or ``None`` if it is not one."""
    blocks = loop.nodes()
    if not blocks or any(not isinstance(b, SDFGState) for b in blocks):
        return None
    if len(loop.edges()) != len(blocks) - 1:
        return None
    for e in loop.edges():
        if e.data.assignments or e.data.condition.as_string not in ("1", "True", "(1)"):
            return None
    if any(loop.in_degree(b) > 1 or loop.out_degree(b) > 1 for b in blocks):
        return None
    chain: list[SDFGState] = []
    seen: dict[ControlFlowBlock, None] = {}
    cur: ControlFlowBlock | None = loop.start_block
    while cur is not None:
        if cur in seen:
            return None
        seen[cur] = None
        chain.append(as_state(cur))
        out = loop.out_edges(cur)
        cur = out[0].dst if out else None
    if len(chain) != len(blocks):
        return None
    for st in chain:
        if any(not isinstance(n, (nodes.Tasklet, nodes.AccessNode)) for n in st.nodes()):
            return None
    return chain


def _read_after_loop(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, container: str) -> bool:
    """Whether ``container`` can be READ on any path leaving ``loop``."""
    if parent is not sdfg:
        return True
    # An interstate edge can read a container in a condition or an assignment RHS, where "before" and "after" the loop
    # are not distinguishable by block reachability alone.
    for e in sdfg.all_interstate_edges():
        if any(m.data == container for m in e.data.get_read_memlets(sdfg.arrays)):
            return True
    for blk in sdutil.dfs_conditional(sdfg, sources=[loop]):
        if blk is loop:
            continue
        for st in [blk] if isinstance(blk, SDFGState) else list(blk.states()):
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == container and value_edges(st.out_edges(n)):
                    return True
    return False


def _body_nodes(chain: list[SDFGState], container: str) -> list[tuple[int, SDFGState, nodes.AccessNode]]:
    """Every access node for ``container`` in the body, as ``(chain index, state, node)``."""
    return [
        (si, st, n)
        for si, st in enumerate(chain)
        for n in st.nodes()
        if isinstance(n, nodes.AccessNode) and n.data == container
    ]


def _body_writes(chain: list[SDFGState], container: str) -> list[tuple[int, SDFGState, nodes.AccessNode, Any]]:
    """Every DATA write to ``container`` in the body, as ``(chain index, state, node, edge)``."""
    return [(si, st, n, e) for si, st, n in _body_nodes(chain, container) for e in value_edges(st.in_edges(n))]


def pure_producer(sdfg: SDFG, tasklet: nodes.Tasklet, out_conn: str | None, loop_var: symbolic.SymbolicType) -> bool:
    """Whether ``tasklet`` may be EVALUATED A SECOND TIME, on another iteration's inputs."""
    if len(tasklet.out_connectors) != 1 or out_conn is None or out_conn not in tasklet.out_connectors:
        return False
    if tasklet.code.language != dtypes.Language.Python:
        return False
    if str(loop_var) in tasklet.free_symbols:
        return False  # the code itself moves with the iteration; the clone would need it rewritten
    if tasklet.has_side_effects(sdfg):
        return False
    try:
        tree = ast.parse((tasklet.code.as_string or "").strip())
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    assign = tree.body[0]
    target = assign.targets[0] if len(assign.targets) == 1 else None
    if not isinstance(target, ast.Name) or target.id != out_conn:
        return False
    for node in ast.walk(assign.value):
        if isinstance(node, ast.Call):
            return False
        if isinstance(node, ast.Name) and node.id not in tasklet.in_connectors and node.id not in sdfg.symbols:
            return False  # a free name that is neither an input nor a symbol is unmodelled state
    return True


def staged_write(
    sdfg: SDFG, chain: list[SDFGState], reader_si: int, reader_node: nodes.AccessNode, reader_sub: subsets.Subset
) -> tuple[int, SDFGState, nodes.AccessNode, MultiConnectorEdge[mm.Memlet]] | None:
    """The body write whose value the read of ``reader_node[reader_sub]`` provably sees."""
    desc = sdfg.arrays.get(reader_node.data)
    if desc is None or not desc.transient:
        return None  # a non-transient may also be written from outside the body's dataflow
    staged = _body_writes(chain, reader_node.data)
    if len(staged) != 1:
        return None
    ssi, sstate, snode, sedge = staged[0]
    if ssi > reader_si:
        return None  # produced LATER in the body -> the read took a CARRIED value, not this one
    if ssi == reader_si and snode is not reader_node and not reaches(sstate, snode, reader_node):
        return None  # two versions in one state, and no dataflow path proves the producer's write ran
    write_sub = _subset_at(sedge, snode)
    if write_sub is None or _one_elem(write_sub, sdfg.facts()) != 1 or str(write_sub) != str(reader_sub):
        return None  # a different element -> that write says nothing about the value read here
    if not isinstance(sedge.src, nodes.Tasklet):
        return None
    return ssi, sstate, snode, sedge


def remat_source(
    sdfg: SDFG,
    chain: list[SDFGState],
    loop_var: symbolic.SymbolicType,
    stride: symbolic.SymbolicType,
    reader_si: int,
    reader_node: nodes.AccessNode,
    reader_sub: subsets.Subset,
    depth: int,
) -> RematSource | None:
    """The clone chain that recomputes, at ``i - stride``, the value read at ``reader_node``."""
    if depth >= ROTATION_CHASE_LIMIT:
        return None
    found = staged_write(sdfg, chain, reader_si, reader_node, reader_sub)
    if found is None:
        return None
    ssi, sstate, snode, sedge = found
    producer = sedge.src
    if not pure_producer(sdfg, producer, sedge.src_conn, loop_var):
        return None

    inputs: list[RematInput] = []
    for e in sstate.in_edges(producer):
        if e.data is None or e.data.is_empty():
            return None  # an ordering edge into the producer has no home on the clone
        src = e.src
        if not isinstance(src, nodes.AccessNode) or e.dst_conn is None:
            return None
        sub = _subset_at(e, src)
        if sub is None or _one_elem(sub, sdfg.facts()) != 1:
            return None
        if _body_writes(chain, src.data):
            # Re-reading this container late would read the overwrite, so recompute it instead.
            nested = remat_source(sdfg, chain, loop_var, stride, ssi, src, sub, depth + 1)
            if nested is None:
                return None
            inputs.append(RematInput(e.dst_conn, None, None, nested, False))
            continue
        shifted: subsets.Subset = copy.deepcopy(sub)
        shifted.replace({loop_var: loop_var - stride})
        inputs.append(RematInput(e.dst_conn, src.data, shifted, None, str(shifted) != str(sub)))
    if sorted(inp.conn for inp in inputs) != sorted(producer.in_connectors):
        return None  # a connector reads nothing, so the clone would evaluate an undefined name
    return RematSource(producer, sedge.src_conn, inputs, sdfg.arrays[reader_node.data].dtype, None)


def remat_shifts(source: RematSource) -> int:
    """Leaf reads in the clone chain that actually move with the iteration variable."""
    return sum(int(inp.moved) + (remat_shifts(inp.source) if inp.source is not None else 0) for inp in source.inputs)


def rematerializable_producer(
    sdfg: SDFG,
    chain: list[SDFGState],
    accum: str,
    loop_var: symbolic.SymbolicType,
    stride: symbolic.SymbolicType,
    wsi: int,
    write_edge: MultiConnectorEdge[mm.Memlet],
) -> RematSource | None:
    """The producer whose re-evaluation at ``i - stride`` equals the value ``write_edge`` stores."""
    # Cheap structural refusals first; the dataflow walk and the symbolic shift come last.
    stage = write_edge.src
    if not isinstance(stage, nodes.AccessNode):
        return None
    sdesc = sdfg.arrays.get(stage.data)
    stage_sub = _subset_at(write_edge, stage)
    if sdesc is None or stage_sub is None or _one_elem(stage_sub, sdfg.facts()) != 1:
        return None
    if not sdesc.transient or _uses(stage_sub, loop_var):
        return None  # a subset moving with i is the DIRECT shifted read, which the chase handles
    if sdesc.dtype != sdfg.arrays[accum].dtype:
        return None  # the copy into the carry CONVERTS; a clone feeding the reads would skip that
    source = remat_source(sdfg, chain, loop_var, stride, wsi, stage, stage_sub, 0)
    if source is None or not remat_shifts(source):
        return None  # nothing in the chain moves with i, so there is no delay to remove

    # The staging version dies with the update only when the update is its sole consumer and it holds
    # no value of its own; otherwise it stays and the update's write edge alone goes.
    write_state = chain[wsi]
    dies = write_state.out_degree(stage) == 1 and not value_edges(write_state.in_edges(stage))
    return source._replace(stage=stage if dies else None)


def shifted_source(
    sdfg: SDFG,
    chain: list[SDFGState],
    desc: data.Data,
    loop_var: Any,
    stride: symbolic.SymbolicType,
    wsi: int,
    write_edge: Any,
    chase: list[tuple[SDFGState, nodes.AccessNode]],
    touched: dict[str, None],
) -> tuple[str, Any] | None:
    """Chase the stored value back to an array element indexed by the iteration variable, looking only through a staging
    transient whose own single write already ran: an accumulation and a value written later in the body dead-end here.
    """
    cur_si, cur_node, cur_sub = wsi, write_edge.src, _subset_at(write_edge, write_edge.src)
    src_data = src_subset = None
    for _ in range(ROTATION_CHASE_LIMIT):
        if not isinstance(cur_node, nodes.AccessNode) or cur_sub is None:
            return None
        cdesc = sdfg.arrays.get(cur_node.data)
        if cdesc is None:
            return None
        if _uses(cur_sub, loop_var):
            src_data, src_subset = cur_node.data, cur_sub
            break
        if not cdesc.transient or _one_elem(cur_sub, sdfg.facts()) != 1:
            return None
        staged = _body_writes(chain, cur_node.data)
        if len(staged) != 1:
            return None
        ssi, _, snode, sedge = staged[0]
        if ssi > cur_si or (ssi == cur_si and snode is not cur_node):
            return None  # written LATER in the body -> a carried value, not this iteration's
        if _one_elem(_subset_at(sedge, snode), sdfg.facts()) != 1:
            return None
        # The whole staging container dies with the update; (6) proves nothing else reads it.
        chase.extend((st, n) for _, st, n in _body_nodes(chain, cur_node.data))
        touched[cur_node.data] = None
        cur_si, cur_node, cur_sub = ssi, sedge.src, _subset_at(sedge, sedge.src)
    if src_data is None:
        return None
    if sdfg.arrays[src_data].dtype != desc.dtype:
        return None  # the copy into the scalar converts; reading the source direct would skip that
    touched[src_data] = None
    # (3) An in-loop write to the source would make ``i - stride`` read the post-write version.
    if _body_writes(chain, src_data):
        return None
    shifted = copy.deepcopy(src_subset)
    required(shifted).replace({loop_var: loop_var - stride})
    if str(shifted) == str(src_subset):
        return None  # not shifted: a same-iteration copy, no delay to remove
    return src_data, shifted


def entry_value_reads(
    chain: list[SDFGState], accum: str, wsi: int, write_state: SDFGState, post_node: nodes.AccessNode, write_edge: Any
) -> list[tuple[SDFGState, nodes.AccessNode]] | None:
    """(4) The reads of ``accum`` in the body when each sees the entry value, the slot the update writes, by dataflow
    and block order: none reads the updated version or is ordered after it."""
    reads: list[tuple[SDFGState, nodes.AccessNode]] = []
    reachable_from_write = dict.fromkeys(sdutil.dfs_conditional(write_state, sources=[post_node]))
    written_slot = str(_subset_at(write_edge, post_node))
    for si, st, n in _body_nodes(chain, accum):
        if n is post_node:
            continue
        outs = st.out_edges(n)
        if not outs:
            continue
        if any(e.data is None or e.data.is_empty() for e in outs):
            return None  # an ordering edge OUT of a version the rewrite deletes loses its order
        if si > wsi or (si == wsi and n in reachable_from_write):
            return None  # this read observes the POST-update value: b[i], not b[i - stride]
        if any(str(_subset_at(e, n)) != written_slot for e in outs):
            return None  # reads a different slot than the update writes
        reads.append((st, n))
    if write_state.out_edges(post_node):
        return None  # read AFTER the write (value is b[i]), or an ordering edge we would drop
    return reads


def plan_rotation(
    parent: ControlFlowRegion,
    loop: LoopRegion,
    sdfg: SDFG,
    chain: list[SDFGState],
    accum: str,
    stride: symbolic.SymbolicType,
) -> RotationPlan | None:
    """Validate that ``accum`` is a delay line and enumerate the rewrite."""
    desc = sdfg.arrays.get(accum)
    if desc is None or not desc.transient or _one_elem(subsets.Range.from_array(desc), sdfg.facts()) != 1:
        return None

    # (1) written EXACTLY ONCE per iteration -- else a read's value is not this one update's.
    writes = _body_writes(chain, accum)
    if len(writes) != 1:
        return None
    wsi, write_state, post_node, write_edge = writes[0]
    if _one_elem(_subset_at(write_edge, post_node), sdfg.facts()) != 1:
        return None

    loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
    chase: list[tuple[SDFGState, nodes.AccessNode]] = [(write_state, post_node)]
    touched: dict[str, None] = {accum: None}
    src_data = shifted = None

    # (2r) the stored value may be a COMPUTED transient instead of an array element, in which case its closed form is
    # the producer re-evaluated one iteration back.
    remat = rematerializable_producer(sdfg, chain, accum, loop_var, stride, wsi, write_edge)
    if remat is not None:
        if remat.stage is not None:
            chase.append((write_state, remat.stage))
    else:
        traced = shifted_source(sdfg, chain, desc, loop_var, stride, wsi, write_edge, chase, touched)
        if traced is None:
            return None
        src_data, shifted = traced

    reads = entry_value_reads(chain, accum, wsi, write_state, post_node, write_edge)
    if reads is None:
        return None
    if remat is not None and any(st.in_edges(n) for st, n in reads):
        return None  # an ordering edge INTO the read has no home on the cloned producer

    # (5) nothing after the loop may read the value the deleted update used to leave behind.
    if _read_after_loop(parent, loop, sdfg, accum):
        return None

    # (6) the staging transients the update alone keeps alive must have no other consumer.
    for container in touched:
        if container in (accum, src_data):
            continue
        if sum(len(value_edges(st.out_edges(n))) for _, st, n in _body_nodes(chain, container)) != 1:
            return None
        if _read_after_loop(parent, loop, sdfg, container):
            return None
    return RotationPlan(accum, src_data, shifted, post_node, write_state, chase, reads, list(touched), remat)


def delete_rotation_update(chain: list[SDFGState], plan: RotationPlan) -> None:
    """Drop the delay line's update -- and the staging nodes only it kept alive -- from the body."""
    for st, n in plan.chase:
        if n in dict.fromkeys(st.nodes()):
            st.remove_node(n)
    # What fed the update is now an access node holding only ordering edges to a write that no longer exists
    for st in chain:
        for n in list(st.nodes()):
            if not isinstance(n, nodes.AccessNode) or n.data not in plan.touched:
                continue
            if st.out_degree(n) == 0 and not value_edges(st.in_edges(n)):
                st.remove_node(n)


def shift_rotation_reads(plan: RotationPlan) -> None:
    """Rewire every read of the carried scalar to the source element one stride back."""

    src_data, src_subset = plan.src_data, plan.src_subset
    assert src_data is not None and src_subset is not None  # the caller dispatches on plan.remat
    for st, n in plan.reads:
        new_src = st.add_access(src_data)
        for e in list(st.out_edges(n)):
            if isinstance(e.dst, nodes.AccessNode):
                memlet = mm.Memlet(
                    data=src_data, subset=copy.deepcopy(src_subset), other_subset=copy.deepcopy(_subset_at(e, e.dst))
                )
            else:
                memlet = mm.Memlet(data=src_data, subset=copy.deepcopy(src_subset))
            st.remove_edge(e)
            st.add_edge(new_src, None, e.dst, e.dst_conn, memlet)
        for e in list(st.in_edges(n)):
            st.remove_edge(e)
            st.add_edge(e.src, e.src_conn, new_src, None, copy.deepcopy(e.data))
        st.remove_node(n)


def emit_remat_clone(sdfg: SDFG, st: SDFGState, source: RematSource, hint: str) -> nodes.AccessNode:
    """Build one fresh evaluation of ``source`` in ``st`` and return the access node holding its result."""

    producer = source.producer
    clone = nodes.Tasklet(
        f"{producer.label}_remat",
        copy.deepcopy(producer.in_connectors),
        copy.deepcopy(producer.out_connectors),
        producer.code.as_string,
        producer.code.language,
    )
    st.add_node(clone)
    for inp in source.inputs:
        if inp.source is None:
            assert inp.container is not None and inp.subset is not None  # the matcher fills one or the other
            st.add_edge(
                st.add_access(inp.container),
                None,
                clone,
                inp.conn,
                mm.Memlet(data=inp.container, subset=copy.deepcopy(inp.subset)),
            )
        else:
            nested = emit_remat_clone(sdfg, st, inp.source, hint)
            st.add_edge(nested, None, clone, inp.conn, mm.Memlet(data=nested.data, subset="0"))
    name, _ = sdfg.add_scalar(f"{hint}_remat", source.dtype, transient=True, find_new_name=True)
    value = st.add_access(name)
    st.add_edge(clone, source.out_conn, value, None, mm.Memlet(data=name, subset="0"))
    return value


def rematerialize_rotation_reads(sdfg: SDFG, plan: RotationPlan) -> None:
    """Replace every read of the carried scalar by a fresh evaluation of its producer at ``i - stride``."""

    remat = plan.remat
    assert remat is not None  # the caller dispatches on plan.remat
    for st, n in plan.reads:
        value = emit_remat_clone(sdfg, st, remat, plan.accum)
        for e in list(st.out_edges(n)):
            if isinstance(e.dst, nodes.AccessNode):
                memlet = mm.Memlet(data=value.data, subset="0", other_subset=copy.deepcopy(_subset_at(e, e.dst)))
            else:
                memlet = mm.Memlet(data=value.data, subset="0")
            st.remove_edge(e)
            st.add_edge(value, None, e.dst, e.dst_conn, memlet)
        st.remove_node(n)


def try_substitute_rotation(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, budget: dict[Any, int]) -> bool:
    """Peel one iteration off ``loop`` and close one carried delay line."""
    from dace.transformation.interstate.loop_peeling import LoopPeeling
    from dace.transformation.passes.parallelization_prep import _constant_trip_count, _unique_block_label

    if not loop.loop_variable or loop.inverted or budget.get(loop, 0) <= 0:
        return False
    # Body shape first: a scan of the loop's own blocks, against three sympy round trips.
    chain = rotation_body_chain(loop)
    if chain is None:
        return False
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    # The peeled iteration's range check reads the stride's sign.
    if start is None or end is None or stride is None or stride == 0 or not stride.is_number:
        return False

    # Deterministic candidate order: body-block order, then node order inside each state.
    candidates = dict.fromkeys(n.data for st in chain for n in st.nodes() if isinstance(n, nodes.AccessNode))
    for accum in candidates:
        plan = plan_rotation(parent, loop, sdfg, chain, accum, stride)
        if plan is None:
            continue
        # Where the trip count is known, the peeled iteration needs no guard: refuse a loop that never runs.
        trip = _constant_trip_count(loop, sdfg)
        if trip is not None and trip < 1:
            return False
        # The peel is the only step that can still fail, and it runs BETWEEN the two halves of the
        # rewrite -- so probe its one raising step (the new loop start) while the SDFG is still
        # untouched, rather than discovering it half-applied.
        try:
            symbolic.evaluate(start + stride, sdfg.constants)
        except Exception:
            return False
        # ``LoopPeeling`` names the peeled iteration after the loop, so a second peel on the same loop
        peel_prefix = f"{loop.label}_{loop.loop_variable}"
        if any(b.label.startswith(peel_prefix) for b in loop.sdfg.all_control_flow_blocks()):
            loop.label = _unique_block_label(loop.sdfg, loop.label)
        # Order matters: the update is dead everywhere once the reads are closed, so it goes before
        # the peel CLONES the body; the reads shift only in the loop, which the peel has narrowed
        # to the iterations where the shifted read is in range.
        delete_rotation_update(chain, plan)
        LoopPeeling().apply_to(
            sdfg=loop.sdfg, loop=loop, verify=False, options={"count": 1, "begin": True, "guarded": True}
        )
        if plan.remat is None:
            shift_rotation_reads(plan)
        else:
            rematerialize_rotation_reads(sdfg, plan)
        budget[loop] -= 1
        return True
    return False


@properties.make_properties
@xf.explicit_cf_compatible
class LoopCarriedRotationSubstitution(ppl.Pass):
    """Replace a loop-carried delay-line scalar by its closed form one iteration back."""

    CATEGORY: str = "Optimization Preparation"

    peel_limit = properties.Property(
        dtype=int,
        default=4,
        desc="Iterations this pass may peel off one loop, which also bounds the delay "
        "depth a rotation may have (0 disables the pass). Peeling is what makes the "
        "shifted read valid on the first iteration.",
    )

    def __init__(self, peel_limit: int = 4) -> None:
        super().__init__()
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        if self.peel_limit <= 0:
            return None
        # Fixed point: a multi-stage delay line only reveals its outer stage once the inner one is gone (``y = x``
        # becomes ``y = b[i-1]``), so re-scan until nothing matches.
        budget: dict[Any, int] = {}
        count = 0
        while True:
            for node, _parent in list(sdfg.all_nodes_recursive()):
                if not isinstance(node, LoopRegion) or loop_analysis.loop_jumps(node):
                    continue
                budget.setdefault(node, self.peel_limit)
                if try_substitute_rotation(node.parent_graph, node, sdfg, budget):
                    count += 1
                    break  # SDFG mutated -> restart the scan on a fresh node list
            else:
                break
        return count or None
