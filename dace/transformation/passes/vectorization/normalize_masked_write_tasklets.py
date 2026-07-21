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
                if self._normalize(tasklet):
                    count += 1
                elif self._demote_self_blend(state, tasklet):
                    count += 1
        return count or None

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
