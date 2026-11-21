# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower a guarded reduction ``if cond(i): acc = acc OP x`` (TSVC ``s3111``) to an unconditional masked one, ``masked =
x if cond else IDENTITY(OP); acc = acc OP masked``, so the reduction stages lift it to a tree reduction instead of one
atomic per passing iteration. The update stays its own tasklet: ``LoopToReduce`` keys off a fused tasklet's top-level
``OP`` and would drop the mask."""

import ast
import copy
from typing import Any, NamedTuple

import numpy as np

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_range
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.subsets import Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.split_statements import value_edges

#: AST binop class -> associative reduction operator string.
BINOP_TO_OP: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
}

#: Reduction operator string -> WCR lambda, to take the identity from :func:`~dace.dtypes.reduction_identity`.
OP_TO_WCR: dict[str, str] = {
    "+": "lambda a, b: a + b",
    "-": "lambda a, b: a + b",
    "*": "lambda a, b: a * b",
}


def _identity_value(op_str: str, dtype: dtypes.typeclass) -> Any:
    """The neutral element of ``op_str`` in ``dtype`` as a plain Python scalar, or ``None``."""
    redtype = operations.detect_reduction_type(OP_TO_WCR[op_str])
    ident = dtypes.reduction_identity(dtype, redtype)
    if ident is None:
        return None
    return ident.item() if isinstance(ident, np.generic) else ident


def binary_update(tasklet: nodes.Tasklet) -> tuple[str, str, str, str] | None:
    """``(out, op, lhs, rhs)`` of a Python tasklet ``out = lhs OP rhs`` over two connectors, ``OP`` in
    :data:`BINOP_TO_OP`."""
    if tasklet.code.language != dtypes.Language.Python:
        return None
    try:
        tree = ast.parse((tasklet.code.as_string or "").strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    rhs = assign.value
    if not isinstance(rhs, ast.BinOp):
        return None
    op_str = BINOP_TO_OP.get(type(rhs.op))
    if op_str is None or not (isinstance(rhs.left, ast.Name) and isinstance(rhs.right, ast.Name)):
        return None
    return assign.targets[0].id, op_str, rhs.left.id, rhs.right.id


class _Match(NamedTuple):
    loop: LoopRegion
    cond_block: ConditionalBlock
    true_branch: ControlFlowRegion
    true_state: SDFGState
    cond_codeblock: properties.CodeBlock
    acc_name: str
    upd_tasklet: nodes.Tasklet
    acc_in_conn: str
    addend_in_conn: str
    out_conn: str
    op_str: str
    identity_value: object


def single_element(desc: data.Data | None) -> bool:
    """Whether ``desc`` is a scalar or a one-element array."""
    return isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1,))


def names_seen_outside(state: SDFGState, sdfg: SDFG) -> dict[str, None]:
    """The data and symbols ``sdfg`` names outside ``state``'s writes: other states' access nodes, interstate edges,
    region conditions, and ``state``'s own sources."""
    seen = [dn.data for st in sdfg.states() if st is not state for dn in st.data_nodes()]
    seen += [s for e in sdfg.all_interstate_edges(recursive=True) for s in e.data.free_symbols]
    seen += [
        s
        for r in sdfg.all_control_flow_regions(recursive=True)
        for c in r.get_meta_codeblocks()
        for s in c.get_free_symbols()
    ]
    seen += [dn.data for dn in state.data_nodes() if state.in_degree(dn) == 0]
    return dict.fromkeys(seen)


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToConditionalReduce(ppl.Pass):
    """Rewrite ``if cond: acc = acc OP expr`` to an unconditional masked reduction (module docstring)."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self, only: list | None = None):
        super().__init__()
        # When given, only these loops are candidates (the segments a split left behind).
        self._only = only

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        rewritten = 0
        only = None if self._only is None else {id(loop) for loop in self._only}
        if only is not None and not only:
            return None
        for sd in sdfg.all_sdfgs_recursive():
            for region in list(sd.all_control_flow_regions()):
                if not (isinstance(region, LoopRegion) and region.loop_variable):
                    continue
                if only is not None and id(region) not in only:
                    continue
                # Stale-snapshot guard.
                if region.parent_graph is None or region not in region.parent_graph.nodes():
                    continue
                m = self._match(region, sd)
                if m is None or not self.hoist_is_trap_free(region, m.true_state, sd):
                    continue
                # The rewrite strips the body's interstate assignments, so the hoisted state must not read one
                assigned = {name for edge in region.all_interstate_edges() for name in edge.data.assignments}
                if assigned & m.true_state.free_symbols or self.read_elsewhere(sd, assigned, m.cond_block):
                    continue
                # An assignment past the block reaches the next iteration's guard, which the rewrite cannot carry
                if any(edge.data.assignments for edge in region.dfs_edges(m.cond_block)):
                    continue
                if self._rewrite(m, sd):
                    rewritten += 1
        return rewritten or None

    # match

    def _match(self, loop: LoopRegion, sdfg: SDFG) -> _Match | None:
        guarded = self.guarded_state(loop)
        if guarded is None:
            return None
        cb, cond_codeblock, true_branch, true_state = guarded
        acc = self.accumulator(true_state, sdfg)
        if acc is None:
            return None
        acc_name, sink = acc
        upd_tasklet = self._walk_back_to_update_tasklet(true_state, sink)
        if upd_tasklet is None:
            return None
        update = binary_update(upd_tasklet)
        if update is None:
            return None
        out_conn, op_str, lhs_name, rhs_name = update
        identity_value = _identity_value(op_str, sdfg.arrays[acc_name].dtype)
        if identity_value is None:
            return None

        # Which tasklet input is the accumulator and which the addend, by the source each traces back to.
        in_edges = value_edges(true_state.in_edges(upd_tasklet))
        if len(in_edges) != 2:
            return None
        acc_in_conn = None
        addend_in_conn = None
        for e in in_edges:
            src_an = self._trace_back_to_source_an(true_state, e.src)
            if src_an is not None and src_an.data == acc_name:
                acc_in_conn = e.dst_conn
            else:
                addend_in_conn = e.dst_conn
        if acc_in_conn is None or addend_in_conn is None or acc_in_conn == addend_in_conn:
            return None
        if dict.fromkeys([acc_in_conn, addend_in_conn]) != dict.fromkeys([lhs_name, rhs_name]):
            return None
        # ``acc - x1 - x2`` is order-independent, ``x1 - acc - x2`` is not.
        if op_str == "-" and acc_in_conn != lhs_name:
            return None
        # The mask scalar has the accumulator's type: a narrowing addend would be cut before the update, not after it.
        addend_edge = next(e for e in in_edges if e.dst_conn == addend_in_conn)
        addend_type = sdfg.arrays[addend_edge.data.data].dtype.type
        if not np.can_cast(addend_type, sdfg.arrays[acc_name].dtype.type, "safe"):
            return None

        return _Match(
            loop=loop,
            cond_block=cb,
            true_branch=true_branch,
            true_state=true_state,
            cond_codeblock=cond_codeblock,
            acc_name=acc_name,
            upd_tasklet=upd_tasklet,
            acc_in_conn=acc_in_conn,
            addend_in_conn=addend_in_conn,
            out_conn=out_conn,
            op_str=op_str,
            identity_value=identity_value,
        )

    def guarded_state(
        self, loop: LoopRegion
    ) -> tuple[ConditionalBlock, properties.CodeBlock, ControlFlowRegion, SDFGState] | None:
        """The loop body's one conditional block (beside empty states), its one guarded branch with no else content,
        and that branch's one non-empty state."""
        cond_blocks = []
        for b in loop.nodes():
            if isinstance(b, ConditionalBlock):
                cond_blocks.append(b)
            elif not isinstance(b, SDFGState) or len(b.nodes()) != 0:
                return None
        if len(cond_blocks) != 1:
            return None
        cb = cond_blocks[0]
        non_else = [(c, br) for c, br in cb.branches if c is not None]
        if len(non_else) != 1:
            return None
        if any(self._branch_has_content(br) for c, br in cb.branches if c is None):
            return None
        cond_codeblock, true_branch = non_else[0]
        if any(not isinstance(n, SDFGState) for n in true_branch.nodes()):
            return None
        content_states = [n for n in true_branch.nodes() if len(n.nodes()) > 0]
        if len(content_states) != 1:
            return None
        return cb, cond_codeblock, true_branch, content_states[0]

    @staticmethod
    def accumulator(true_state: SDFGState, sdfg: SDFG) -> tuple[str, nodes.AccessNode] | None:
        """The one scalar read and written back in ``true_state`` and its sink: the state's only sink, every other
        write going to scratch nothing else sees, as the hoist makes every write unconditional."""
        sources = {
            n.data: n for n in true_state.data_nodes() if true_state.in_degree(n) == 0 and true_state.out_degree(n) > 0
        }
        sinks = {
            n.data: n for n in true_state.data_nodes() if true_state.in_degree(n) > 0 and true_state.out_degree(n) == 0
        }
        acc_candidates = [name for name in sources if name in sinks and single_element(sdfg.arrays.get(name))]
        if len(acc_candidates) != 1:
            return None
        acc_name = acc_candidates[0]
        sink_ans = [n for n in true_state.data_nodes() if true_state.in_degree(n) > 0 and true_state.out_degree(n) == 0]
        if len(sink_ans) != 1 or sink_ans[0].data != acc_name:
            return None
        seen_elsewhere = names_seen_outside(true_state, sdfg)
        for n in true_state.data_nodes():
            if (
                n is not sink_ans[0]
                and true_state.in_degree(n) > 0
                and (not sdfg.arrays[n.data].transient or n.data in seen_elsewhere)
            ):
                return None
        return acc_name, sinks[acc_name]

    # match helpers

    @staticmethod
    def read_elsewhere(sdfg: SDFG, names: set[str], cond_block: ConditionalBlock) -> bool:
        """Whether any of ``names`` is read outside ``cond_block``'s own conditions: by a state, an interstate edge or
        another region's condition."""
        if not names:
            return False
        if any(names & edge.data.read_symbols() for edge in sdfg.all_interstate_edges(recursive=True)):
            return True
        if any(names & {str(sym) for sym in st.free_symbols} for st in sdfg.states()):
            return True
        return any(
            names & {str(sym) for sym in code.get_free_symbols()}
            for region in sdfg.all_control_flow_regions(recursive=True)
            if region is not cond_block
            for code in region.get_meta_codeblocks()
        )

    def hoist_is_trap_free(self, loop: LoopRegion, state: SDFGState, sdfg: SDFG) -> bool:
        """Whether running ``state`` on every iteration of ``loop`` is safe: the rewrite drops the guard, so a
        computation the guard protected would run where the source never ran it."""
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet):
                if not self.tasklet_is_trap_free(node, state, sdfg):
                    return False
            elif not isinstance(node, nodes.AccessNode):
                return False
        bounds = self.iterator_bounds(loop)
        if bounds is None:
            return False
        for edge in state.edges():
            if isinstance(edge.src, nodes.AccessNode) and not edge.data.is_empty():
                if not self.read_in_bounds(edge, state, sdfg, bounds):
                    return False
        return True

    def tasklet_is_trap_free(self, tasklet: nodes.Tasklet, state: SDFGState, sdfg: SDFG) -> bool:
        """A Python tasklet with no subscript (an indirection the memlets do not show) and no integer division
        or modulo by anything but a nonzero constant."""
        if tasklet.code.language != dtypes.Language.Python:
            return False
        try:
            tree = ast.parse((tasklet.code.as_string or "").strip())
        except SyntaxError:
            return False
        floating = {
            e.dst_conn
            for e in state.in_edges(tasklet)
            if e.data.data is not None and sdfg.arrays[e.data.data].dtype.type in (np.float16, np.float32, np.float64)
        }
        for expr in ast.walk(tree):
            if isinstance(expr, ast.Subscript):
                return False
            if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if not self.divisor_is_safe(expr.right, floating):
                    return False
            # C's modulo spelled as a call traps on an integer zero like ``%``; the floored helpers return 0 there
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name)
                and expr.func.id in ("CMod", "FtnMod", "c_mod", "ftn_mod")
                and len(expr.args) == 2
                and not self.divisor_is_safe(expr.args[1], floating)
            ):
                return False
        return True

    def divisor_is_safe(self, divisor: ast.expr, floating: set[str]) -> bool:
        """A nonzero numeric literal, or a floating-point input, whose zero gives an infinity rather than a trap."""
        if isinstance(divisor, ast.Constant):
            return isinstance(divisor.value, (int, float)) and divisor.value != 0
        return isinstance(divisor, ast.Name) and divisor.id in floating

    def iterator_bounds(self, loop: LoopRegion) -> tuple[str, Any, Any] | None:
        """``(iterator name, lowest, highest)`` value of ``loop``'s iterator, or ``None`` when an end or the
        direction is unknown."""
        first = loop_analysis.get_init_assignment(loop)
        last = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if first is None or last is None or stride is None:
            return None
        facts = loop.sdfg.facts()
        if loop_analysis._provably_le(1, stride, facts):
            return loop.loop_variable, first, last
        if loop_analysis._provably_le(stride, -1, facts):
            return loop.loop_variable, last, first
        return None

    def read_in_bounds(self, edge: Any, state: SDFGState, sdfg: SDFG, bounds: tuple[str, Any, Any]) -> bool:
        """Whether the read on ``edge`` stays inside its container on every iteration."""
        if edge.data.dynamic:
            return False
        subset = edge.data.get_src_subset(edge, state)
        return subset is not None and self.subset_in_bounds(subset, sdfg.arrays[edge.src.data], sdfg, bounds)

    def subset_in_bounds(self, subset: Subset, desc: data.Data, sdfg: SDFG, bounds: tuple[str, Any, Any]) -> bool:
        """Whether ``subset`` of ``desc`` stays inside it on every iteration (affine bounds, see :meth:`read_in_bounds`)."""
        if len(subset) != len(desc.shape):
            return False
        for (lo, hi, _), extent in zip(subset.ndrange(), desc.shape):
            smallest = self.affine_extreme(lo, bounds, True, sdfg.facts())
            largest = self.affine_extreme(hi, bounds, False, sdfg.facts())
            if smallest is None or largest is None:
                return False
            if not (
                loop_analysis._provably_le(0, smallest, sdfg.facts())
                and loop_analysis._provably_le(largest, extent - 1, sdfg.facts())
            ):
                return False
        return True

    def affine_extreme(
        self, bound: Any, bounds: tuple[str, Any, Any], lowest: bool, facts: symbolic.Facts
    ) -> Any | None:
        """The smallest (``lowest``) or largest value ``bound`` takes over the loop, or ``None`` when ``bound`` is
        not affine in the iterator or the sign of its slope is unknown."""
        name, low, high = bounds
        bound = symbolic.pystr_to_symbolic(str(bound))
        iterator = next((s for s in bound.free_symbols if str(s) == name), None)
        if iterator is None:
            return bound
        slope = symbolic.simplify(bound.diff(iterator), facts)
        if slope.has(iterator):
            return None
        if loop_analysis._provably_le(0, slope, facts):
            rising = True
        elif loop_analysis._provably_le(slope, 0, facts):
            rising = False
        else:
            return None
        return bound.subs(iterator, low if rising == lowest else high)

    def _branch_has_content(self, branch: ControlFlowBlock) -> bool:
        if not isinstance(branch, ControlFlowRegion):
            return False
        for n in branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return True
            if not isinstance(n, SDFGState):
                return True
        return False

    def _walk_back_to_update_tasklet(self, state: SDFGState, sink_an: nodes.AccessNode) -> nodes.Tasklet | None:
        """The tasklet writing ``sink_an``, through access nodes and identity tasklets."""
        cur = sink_an
        while True:
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            upstream = ins[0].src
            if isinstance(upstream, nodes.AccessNode):
                cur = upstream
                continue
            if not isinstance(upstream, nodes.Tasklet):
                return None
            # Is this an identity passthrough? Walk further back.
            try:
                tree = ast.parse((upstream.code.as_string or "").strip())
            except SyntaxError:
                return upstream  # treat as the update; downstream will refuse
            if (
                len(tree.body) == 1
                and isinstance(tree.body[0], ast.Assign)
                and isinstance(tree.body[0].value, ast.Name)
            ):
                # Identity tasklet -> step over.
                t_ins = list(state.in_edges(upstream))
                if len(t_ins) != 1:
                    return None
                cur = t_ins[0].src
                if not isinstance(cur, nodes.AccessNode):
                    return None
                continue
            return upstream

    def _trace_back_to_source_an(self, state: SDFGState, start: nodes.Node) -> nodes.AccessNode | None:
        """The source access node ``start`` reads through a chain of access nodes, or ``None``."""
        cur = start
        while True:
            if not isinstance(cur, nodes.AccessNode):
                return None
            if state.in_degree(cur) == 0:
                return cur
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            cur = ins[0].src

    def _addend_gather(self, m: _Match) -> tuple[str, Subset | None] | None:
        """``(array, subset)`` of the addend read, through one transient hop or directly, or ``None``."""
        addend_edge = next((e for e in m.true_state.in_edges(m.upd_tasklet) if e.dst_conn == m.addend_in_conn), None)
        if addend_edge is None or not isinstance(addend_edge.src, nodes.AccessNode):
            return None
        src = addend_edge.src
        if m.true_state.in_degree(src) == 1:
            pred = m.true_state.in_edges(src)[0]
            if isinstance(pred.src, nodes.AccessNode) and pred.data is not None:
                return pred.src.data, pred.data.get_src_subset(pred, m.true_state)
        return src.data, addend_edge.data.subset

    # rewrite

    def _rewrite(self, m: _Match, sdfg: SDFG) -> bool:
        """Splice ``masked = __addend if cond else IDENTITY`` before the update and hoist the branch state, its addend
        producer with it."""
        loop = m.loop
        true_state = m.true_state
        # The guard's addend read becomes ``__addend``, every other array read a wired ``__guardN`` input.
        guard_inputs: dict[str, tuple[str, str]] = {}
        cond_expr_resolved = self._resolve_cond(m, sdfg, addend_conn_name="__addend", guard_inputs=guard_inputs)
        if cond_expr_resolved is None:
            return False

        acc_desc = sdfg.arrays[m.acc_name]
        masked_val, _ = sdfg.add_scalar(f"{m.acc_name}_masked_val", acc_desc.dtype, transient=True, find_new_name=True)
        mask_body = self._build_mask_body(cond_expr_resolved, m.identity_value)
        # dict.fromkeys keeps the guard connectors in AST order; a set would reorder them per process.
        mask_tasklet = true_state.add_tasklet(
            name=f"{m.acc_name}_mask",
            inputs=dict.fromkeys(["__addend", *guard_inputs]),
            outputs=dict.fromkeys(["__out"]),
            code=mask_body,
            language=dtypes.Language.Python,
        )

        # ``producer -> update`` becomes ``producer -> mask -> masked_val -> update``.
        addend_edge = next(e for e in true_state.in_edges(m.upd_tasklet) if e.dst_conn == m.addend_in_conn)
        acc_subset = "0"
        masked_an = true_state.add_access(masked_val)
        true_state.add_edge(
            addend_edge.src, addend_edge.src_conn, mask_tasklet, "__addend", copy.deepcopy(addend_edge.data)
        )
        true_state.remove_edge(addend_edge)
        true_state.add_edge(mask_tasklet, "__out", masked_an, None, mm.Memlet(data=masked_val, subset=acc_subset))
        true_state.add_edge(
            masked_an, None, m.upd_tasklet, m.addend_in_conn, mm.Memlet(data=masked_val, subset=acc_subset)
        )

        # The guard's own array reads, as mask inputs.
        for conn, (arr_name, idx_str) in guard_inputs.items():
            true_state.add_edge(
                true_state.add_read(arr_name), None, mask_tasklet, conn, mm.Memlet(data=arr_name, subset=idx_str)
            )

        # 4. Hoist the true-branch state into the loop in the ConditionalBlock's place, stripping dead iedge assignments
        # (the resolved cond no longer references the gather symbols).
        was_start = m.cond_block is loop.start_block
        m.true_branch.remove_node(true_state)
        loop.add_node(true_state, is_start_block=was_start, ensure_unique_name=True)
        for ie in list(loop.in_edges(m.cond_block)):
            stripped = InterstateEdge(condition=ie.data.condition, assignments={})
            loop.add_edge(ie.src, true_state, stripped)
            loop.remove_edge(ie)
        for oe in list(loop.out_edges(m.cond_block)):
            loop.add_edge(true_state, oe.dst, oe.data)
            loop.remove_edge(oe)

        # 5. Drop the now-disconnected ConditionalBlock.
        loop.remove_node(m.cond_block)
        # 6. Collapse empty wrapper states
        self._collapse_empty_wrappers(loop)
        return True

    def _build_mask_body(self, cond_expr: str, identity_value: object) -> str:
        """Return Python source for the elementwise mask tasklet
        ``__out = (__addend if (cond) else IDENTITY)``, built as an AST so the
        resolved cond sub-expression is spliced structurally (no string surgery)."""
        cond_ast = ast.parse(cond_expr, mode="eval").body
        masked = ast.IfExp(
            test=cond_ast, body=ast.Name(id="__addend", ctx=ast.Load()), orelse=ast.Constant(value=identity_value)
        )
        assign = ast.Assign(targets=[ast.Name(id="__out", ctx=ast.Store())], value=masked)
        module = ast.Module(body=[assign], type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)

    def _resolve_cond(
        self,
        m: _Match,
        sdfg: SDFG,
        addend_conn_name: str = "__addend",
        guard_inputs: dict[str, tuple[str, str]] | None = None,
    ) -> str | None:
        """Rewrite the cond expression over tasklet input connectors only."""
        cond_text = m.cond_codeblock.as_string.strip()
        if guard_inputs is None:
            guard_inputs = {}

        # The interstate bindings, parsed once.
        binding_asts: dict[str, ast.AST] = {}
        for e in m.loop.all_interstate_edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                try:
                    binding_asts[lhs] = ast.parse(str(rhs), mode="eval").body
                except SyntaxError:
                    continue

        # A guard read of the addend's own element (``a[i] > K``) resolves to ``addend_conn_name``.
        connector_for_access: dict[tuple, str] = {}
        gather = self._addend_gather(m)
        if gather is not None:
            arr_name, sub = gather
            if sub is not None:
                try:
                    key = tuple(str(lo) for lo, hi, step in as_range(sub).ranges)
                    connector_for_access[(arr_name, key)] = addend_conn_name
                except Exception:
                    pass

        # A read that is neither the addend nor a wireable ``__guardN`` input lands in ``unwireable``: a refusal.
        unwireable: list = []
        wireable_guard_read = self._wireable_guard_read

        class _Subst(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                # An interstate-bound gather symbol inlines (``a_index`` -> ``a[i]``), its subscripts visited too.
                if node.id in binding_asts:
                    # A fresh copy per use: shared nodes break ``fix_missing_locations``.
                    sub = ast.copy_location(copy.deepcopy(binding_asts[node.id]), node)
                    return self.visit(sub)
                # A container read by name (a scalar guard, the running sum itself) would reach the mask unwired
                if node.id in sdfg.arrays:
                    unwireable.append(node)
                return node

            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                node.slice = self.visit(node.slice)  # the subscripted name itself is no bare read
                if not isinstance(node.value, ast.Name):
                    unwireable.append(node)  # subscript of a non-name (e.g. a call result)
                    return node
                arr_name = node.value.id
                idx = node.slice
                try:
                    idx_str = ast.unparse(idx)
                except Exception:
                    unwireable.append(node)
                    return node
                key = (arr_name, (idx_str,))
                conn = connector_for_access.get(key)
                if conn is None:
                    # Not the addend's element: wire this read as its own mask input.
                    if not wireable_guard_read(m, sdfg, arr_name, idx):
                        unwireable.append(node)
                        return node
                    conn = f"__guard{len(guard_inputs)}"
                    guard_inputs[conn] = (arr_name, idx_str)
                    connector_for_access[key] = conn  # reuse one connector per distinct element
                return ast.copy_location(ast.Name(id=conn, ctx=ast.Load()), node)

        try:
            cond_ast = ast.parse(cond_text, mode="eval").body
        except SyntaxError:
            return cond_text
        new_ast = _Subst().visit(cond_ast)
        if unwireable:
            return None
        ast.fix_missing_locations(new_ast)
        try:
            return ast.unparse(new_ast)
        except Exception:
            return cond_text

    def _wireable_guard_read(self, m: _Match, sdfg: SDFG, arr_name: str, idx: ast.AST) -> bool:
        """Can ``arr_name[idx]`` be handed to the mask tasklet as a plain input edge?"""
        desc = sdfg.arrays.get(arr_name)
        if desc is None:
            return False
        available = self._available_symbols(m, sdfg)
        for sub in ast.walk(idx):
            if isinstance(sub, (ast.Subscript, ast.Call)):
                return False  # data-dependent (indirect) index -- not a static memlet subset
            if isinstance(sub, ast.Name) and sub.id not in available:
                return False  # not a symbol we can put in a subset
        written = (n.data for n in m.true_state.data_nodes() if m.true_state.in_degree(n) > 0)
        if arr_name in written:
            return False
        # The mask reads it on every iteration, where the source's ``and`` / ``if`` may have skipped it
        bounds = self.iterator_bounds(m.loop)
        try:
            dims = idx.elts if isinstance(idx, ast.Tuple) else [idx]
            subset = subsets.Range.from_string(", ".join(ast.unparse(d) for d in dims))
        except (SyntaxError, TypeError, ValueError):
            return False
        # A symbol one interstate edge assigns takes that value at the guard (``off = 1`` on the edge into the loop)
        assigned: dict[str, list[str]] = {}
        for edge in sdfg.all_interstate_edges(recursive=True):
            for lhs, rhs in edge.data.assignments.items():
                assigned.setdefault(lhs, []).append(rhs)
        values = {
            symbolic.symbol(lhs): symbolic.pystr_to_symbolic(rhs[0]) for lhs, rhs in assigned.items() if len(rhs) == 1
        }
        if values:
            subset.replace(values)
        return bounds is not None and self.subset_in_bounds(subset, desc, sdfg, bounds)

    def _available_symbols(self, m: _Match, sdfg: SDFG) -> dict[str, None]:
        """Names usable in a memlet subset at the update tasklet, taken from the DEFINED-symbol API rather than
        ``sdfg.symbols`` membership."""
        return dict.fromkeys(
            [*m.true_state.symbols_defined_at(m.upd_tasklet), *m.true_state.defined_symbols(), *sdfg.constants]
        )

    def _collapse_empty_wrappers(self, loop: LoopRegion) -> None:
        """Eliminate empty SDFGState blocks in the loop body whose only role is to host an iedge -- after we stripped
        the dead iedge assignments in step 4 above, the wrapper is structurally empty AND its outgoing iedge carries
        nothing."""
        changed = True
        while changed:
            changed = False
            for blk in list(loop.nodes()):
                if not isinstance(blk, SDFGState) or len(blk.nodes()) > 0:
                    continue
                in_es = list(loop.in_edges(blk))
                out_es = list(loop.out_edges(blk))
                # Only collapse states with a single successor + no iedge assignments on the outgoing edge.
                if len(out_es) != 1:
                    continue
                oe = out_es[0]
                if oe.data.assignments:
                    continue
                # The entry is spliced out only when its successor becomes the new source.
                if blk is loop.start_block and (in_es or loop.in_degree(oe.dst) != 1):
                    continue
                for ie in in_es:
                    loop.add_edge(ie.src, oe.dst, ie.data)
                    loop.remove_edge(ie)
                loop.remove_edge(oe)
                loop.remove_node(blk)
                changed = True
                break


__all__ = ["LoopToConditionalReduce"]
