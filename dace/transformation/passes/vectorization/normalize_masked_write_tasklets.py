# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize a masked-write bare-if tasklet to the first-class ``IT`` function form.

Python frontend lowers a boolean-masked assignment ``A[mask] = value``
(``dace/frontend/python/newast.py``: ``tasklet_code += 'if __in_cond:\\n    '``) to a tasklet
whose body is a bare ``if`` with no else::

    if __in_cond:
        __out = value            # __out writes A[i]; when !cond, A[i] left as-is

A bare ``if`` STATEMENT is not straight-line -> ``SplitTasklets`` mis-splits it and every op
detector in ``ConvertTaskletsToTileOps`` (expect ``lhs = rhs`` bodies) skips it. Rewrite to a
first-class conditional-write function: single straight-line expression analysed uniformly like
``ITE``::

    __out = IT(__in_cond, value)

``IT(cond, e)`` = write-only ternary: write ``e`` where ``cond``, else leave destination
unchanged. Unlike ``ITE(cond, t, e)`` it has NO else arm -> never reads the old value;
``ConvertTaskletsToTileOps`` lowers it to a masked ``TileStore`` (``cond`` mask gates the store,
inactive lanes untouched). Runs in vectorize prep, before ``SplitTasklets``.

``IT`` is a first-class primitive, not a tile-only marker: :mod:`dace.codegen.cppunparse` unparses
``out = IT(c, v)`` to the guarded statement ``if (c) { out = v; }``, so it lowers correctly in a
scalar / remainder-tail scope too (where the tile passes never run). It has to be a statement
rather than a function like ``ITE`` precisely because it has no else arm -- there is no value to
return when the predicate is false.
"""
import ast
from typing import List, Optional

import dace
from dace.sdfg import SDFG, SDFGState, nodes as nd
from dace.sdfg.nodes import CodeBlock
from dace.transformation import pass_pipeline as ppl
# The name is owned by the unparser that gives it meaning, so the producer here and the C++
# lowering can never drift apart.
from dace.codegen.cppunparse import CONDITIONAL_WRITE_FUNC


class NormalizeMaskedWriteTasklets(ppl.Pass):
    """Rewrite every masked-write bare-if tasklet (``if cond: out = e``) into the
    ``out = IT(cond, e)`` function form. See the module docstring."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for state in sdfg.all_states():
            for tasklet in list(state.nodes()):
                if not isinstance(tasklet, nd.Tasklet):
                    continue
                if tasklet.code.language != dace.dtypes.Language.Python:
                    continue
                if self._normalize(tasklet) or self._demote_self_blend(state, tasklet):
                    self._mark_conditional_writes_dynamic(state, tasklet)
                    count += 1
        return count or None

    def _mark_conditional_writes_dynamic(self, state: SDFGState, tasklet: nd.Tasklet) -> None:
        """Mark every out-edge whose connector is written through ``IT`` as a DYNAMIC memlet.

        ``IT(cond, value)`` lowers to ``if (cond) { out = value; }``, so the connector is not
        assigned on the false path. That is only sound if the WRITE is skipped there too, which in
        DaCe means a dynamic memlet: a static one stores the connector unconditionally after the
        tasklet, so the false path would write whatever uninitialised value it happened to hold
        instead of leaving the destination alone (TSVC s277's guarded ``a[i] += c[i]*d[i]``).

        Enforced here, once, for EVERY producer of the ``IT`` form rather than at each rewrite
        site. :meth:`_normalize` happens to inherit a dynamic memlet from the frontend (it lowers
        an already-conditional ``A[mask] = value``), but :meth:`_demote_self_blend` turns an
        UNCONDITIONAL blend into a conditional write and must convert the memlet with it. Making
        the invariant explicit keeps the next producer from having to rediscover it.

        :param state: The state holding ``tasklet``.
        :param tasklet: A tasklet whose body was just rewritten to the ``IT`` form.
        """
        guarded = {
            node.targets[0].id
            for node in ast.walk(ast.parse(tasklet.code.as_string)) if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == CONDITIONAL_WRITE_FUNC
        }
        for edge in state.out_edges(tasklet):
            if edge.src_conn in guarded and edge.data is not None:
                edge.data.dynamic = True

    def _demote_self_blend(self, state: SDFGState, tasklet: nd.Tasklet) -> bool:
        """Rewrite a SELF-BLEND ``out = ITE(cond, value, out)`` into the masked write
        ``out = IT(cond, value)``, dropping the now-dead read of the destination.

        ``BranchNormalization`` lowers a single-arm ``if cond: arr[i] = value`` to
        ``arr[i] = ITE(cond, value, arr[i])`` -- correct, but it reads the destination back and
        blends it, which :class:`ConvertTaskletsToTileOps` lowers to a ``TileITE`` (load old tile,
        select, store all lanes). A one-armed ``if`` has no else VALUE at all: the inactive lanes
        must simply not be written. That is ``IT``, which lowers to a masked ``TileStore`` -- the
        instruction this pattern is asking for -- and it drops a tile load plus a select.

        Only fires when the else operand is a bare in-connector reading the SAME data and subset
        the output writes (that is what makes it a no-op arm rather than a real blend), and when
        that connector is not otherwise referenced by the body.

        :param state: The state holding ``tasklet`` (for its edges).
        :param tasklet: The candidate tasklet.
        :returns: ``True`` if the tasklet was rewritten.
        """
        parsed = self._parse_self_blend(tasklet)
        if parsed is None:
            return False
        out_conn, else_conn, cond_src, value_src = parsed

        in_edges = [e for e in state.in_edges(tasklet) if e.dst_conn == else_conn]
        out_edges = [e for e in state.out_edges(tasklet) if e.src_conn == out_conn]
        if len(in_edges) != 1 or len(out_edges) != 1:
            return False
        read, write = in_edges[0].data, out_edges[0].data
        # Same element, or the demotion would drop a genuine read of OTHER data.
        if read.data != write.data or str(read.subset) != str(write.subset):
            return False

        tasklet.code = CodeBlock(f"{out_conn} = {CONDITIONAL_WRITE_FUNC}({cond_src}, {value_src})",
                                 language=dace.dtypes.Language.Python)
        # Drop the WHOLE read path, not just the tasklet's edge: the read enters through the
        # enclosing map's ``IN_``/``OUT_`` connector pair, and removing only the inner edge would
        # strand them (a dangling out-connector the validator rejects). It also drops the tasklet's
        # own ``else_conn`` in-connector, so no separate removal is needed.
        state.remove_memlet_path(in_edges[0], remove_orphans=True)
        return True

    def _parse_self_blend(self, tasklet: nd.Tasklet):
        """Match a lone ``<out> = ITE(<cond>, <value>, <in_conn>)`` body.

        :param tasklet: The candidate tasklet.
        :returns: ``(out_conn, else_conn, cond_src, value_src)``, or ``None`` if it does not match.
        """
        try:
            tree = ast.parse((tasklet.code.as_string or "").strip())
        except SyntaxError:
            return None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return None
        assign = tree.body[0]
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
            return None
        out_conn = assign.targets[0].id
        if out_conn not in tasklet.out_connectors:
            return None
        call = assign.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "ITE"):
            return None
        if len(call.args) != 3:
            return None
        cond, value, else_arm = call.args
        if not (isinstance(else_arm, ast.Name) and else_arm.id in tasklet.in_connectors):
            return None
        # The else connector must feed ONLY this arm; otherwise dropping it changes the other use.
        others = [
            n.id for n in ast.walk(ast.Module(body=[ast.Expr(cond), ast.Expr(value)], type_ignores=[]))
            if isinstance(n, ast.Name)
        ]
        if else_arm.id in others:
            return None
        return out_conn, else_arm.id, ast.unparse(cond), ast.unparse(value)

    def _normalize(self, tasklet: nd.Tasklet) -> bool:
        """Rewrite one masked-write bare-if tasklet's body in place. Returns True if changed."""
        try:
            body = ast.parse(tasklet.code.as_string).body
        except (SyntaxError, ValueError):
            return False
        # Exactly one bare ``if`` with no else, whose body is straight-line assignments.
        if len(body) != 1 or not isinstance(body[0], ast.If) or body[0].orelse:
            return False
        ifnode = body[0]
        if not all(
                isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                for s in ifnode.body):
            return False
        out_conns = set(tasklet.out_connectors)
        if not any(s.targets[0].id in out_conns for s in ifnode.body):
            return False

        cond_src = ast.unparse(ifnode.test)
        new_lines: List[str] = []
        for s in ifnode.body:
            lhs = s.targets[0].id
            rhs = ast.unparse(s.value)
            if lhs in out_conns:
                # First-class conditional write ``out = IT(cond, value)`` (see module docstring).
                # ``cond`` (e.g. ``__in_cond``) stays a live input via the call.
                new_lines.append("%s = %s(%s, %s)" % (lhs, CONDITIONAL_WRITE_FUNC, cond_src, rhs))
            else:  # intermediate local: compute unconditionally (defined for every lane)
                new_lines.append("%s = %s" % (lhs, rhs))
        tasklet.code = CodeBlock("\n".join(new_lines), language=dace.dtypes.Language.Python)
        return True


class IfExpToITE(ast.NodeTransformer):
    """Rewrite every ``ast.IfExp`` (``t if cond else e``) to an ``ITE(cond, t, e)`` call,
    innermost first, so a nested ternary composes into a nested ``ITE`` call."""

    def visit_IfExp(self, node: ast.IfExp) -> ast.Call:
        self.generic_visit(node)  # nested IfExp arms first -- ITE(c, ITE(...), ...) composes
        return ast.Call(func=ast.Name(id="ITE", ctx=ast.Load()), args=[node.test, node.body, node.orelse], keywords=[])


class NormalizeTernaryTasklets(ppl.Pass):
    """Rewrite a tasklet-body Python ternary (``out = t if cond else e``) into the first-class
    ``out = ITE(cond, t, e)`` function form.

    ``SplitTasklets`` (:mod:`dace.transformation.passes.split_tasklets`, ``ASTSplitter``) already
    knows how to lower a ternary to ``ITE`` -- but only as a BY-PRODUCT of splitting a
    multi-operation body into single-op SSA statements. When the ternary IS the whole body,
    ``to_ssa`` collapses it to exactly one SSA line, and ``SplitTasklets`` treats that as
    "already split" (its ``len(ssa_statements) > 1`` gate) and leaves the ORIGINAL raw ternary in
    place. A flat ``_o = _t if _cond else 0.0`` (or a Symbol arm, or any parenthesisation /
    whitespace variant) therefore reaches ``ConvertTaskletsToTileOps`` unrewritten, where
    ``_detect_ite``'s raw-ternary branch only fires for an EXACT permutation of three
    IN-CONNECTOR names -- a literal or Symbol arm (n_in in (1, 2)) is missed entirely, even
    though the equivalent ``ITE(...)`` call spelling is already handled for those arities.

    This pass closes the gap at the body-text level: the tasklet's sole assignment is parsed,
    and if its RHS is a top-level ``ast.IfExp`` (nested arms rewritten too, innermost first), it is
    rewritten to the composed ``ITE(cond, t, e)`` call. Purely a text rewrite -- each arm
    (in-connector, free symbol, or literal/constant expression) passes through ``ast.unparse``
    verbatim, so no connector is invented, renamed, or dropped, and no symbol is touched. Runs in
    vectorize prep alongside :class:`NormalizeMaskedWriteTasklets` (same IT/ITE family, same
    prep slot) -- but unlike ``IT``, ``ITE`` always writes one of its two arms (no masking
    semantics), so this pass needs no tiled-vs-scalar-tail distinction.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for state in sdfg.all_states():
            for tasklet in list(state.nodes()):
                if not isinstance(tasklet, nd.Tasklet):
                    continue
                if tasklet.code.language != dace.dtypes.Language.Python:
                    continue
                if self._normalize_ternary(tasklet):
                    count += 1
        return count or None

    def _normalize_ternary(self, tasklet: nd.Tasklet) -> bool:
        """Rewrite one tasklet's sole assignment in place if its RHS is a ternary.
        Returns True if changed."""
        try:
            body = ast.parse(tasklet.code.as_string).body
        except (SyntaxError, ValueError):
            return False
        if len(body) != 1 or not isinstance(body[0], ast.Assign):
            return False
        assign = body[0]
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
            return False
        if assign.targets[0].id not in tasklet.out_connectors:
            return False
        if not isinstance(assign.value, ast.IfExp):
            return False
        assign.value = IfExpToITE().visit(assign.value)
        tasklet.code = CodeBlock(ast.unparse(assign), language=dace.dtypes.Language.Python)
        return True
