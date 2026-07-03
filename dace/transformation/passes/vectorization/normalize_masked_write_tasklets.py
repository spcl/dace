# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize a masked-write bare-if tasklet to the first-class ``IT`` function form.

The Python frontend lowers a boolean-masked assignment ``A[mask] = value``
(``dace/frontend/python/newast.py``: ``tasklet_code += 'if __in_cond:\\n    '``)
to a tasklet whose body is a bare ``if`` statement with no else::

    if __in_cond:
        __out = value            # __out writes A[i]; when !cond, A[i] is left as-is

A bare ``if`` STATEMENT in a Python tasklet is not straight-line, so
``SplitTasklets`` mis-splits it and every op detector in
``ConvertTaskletsToTileOps`` (which expect ``lhs = rhs`` bodies) skips it. This
pass rewrites it to a first-class conditional-write function -- a single,
straight-line expression analysed uniformly like ``ITE``::

    __out = IT(__in_cond, value)

``IT(cond, e)`` is the write-only ternary: *write ``e`` where ``cond``, else leave
the destination unchanged*. Unlike ``ITE(cond, t, e)`` it has NO else arm, so it
never reads the old value -- ``ConvertTaskletsToTileOps`` lowers it to a **masked
``TileStore``** (the ``cond`` mask gates the store, inactive lanes untouched). Runs
in the vectorize prep, before ``SplitTasklets``.
"""
import ast
from typing import List, Optional

import dace
from dace.sdfg import SDFG, SDFGState, nodes as nd
from dace.sdfg.nodes import CodeBlock
from dace.transformation import pass_pipeline as ppl

#: The first-class conditional-write function emitted for a masked write. Two args
#: ``IT(cond, value)``: write ``value`` where ``cond``, else leave unchanged. Distinct
#: from ``ITE(cond, t, e)`` (which reads both arms); the ``(`` immediately after ``IT``
#: keeps it unambiguous from the ``ITE(`` prefix.
CONDITIONAL_WRITE_FUNC = "IT"


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
            scope = state.scope_dict()
            for tasklet in list(state.nodes()):
                if not isinstance(tasklet, nd.Tasklet):
                    continue
                if tasklet.code.language != dace.dtypes.Language.Python:
                    continue
                # Leave a masked-write tasklet in a SCALAR / tile-K1 remainder tail as the
                # original bare-``if``: those tails are plain step-1 scalar loops that the
                # tile passes (incl. ``ConvertTaskletsToTileOps``) skip, so a first-class
                # ``IT(...)`` there would survive to codegen undefined. Only tiled bodies
                # -- the divisible interior and the masked-tail slab -- get the ``IT`` form.
                if self._in_tail_scope(scope, tasklet):
                    continue
                if self._normalize(tasklet):
                    count += 1
        return count or None

    def _in_tail_scope(self, scope: dict, tasklet: nd.Tasklet) -> bool:
        """True iff ``tasklet`` sits inside a SCALAR / tile-K1 remainder-tail map scope
        (the ones ``ConvertTaskletsToTileOps._body_nsdfgs`` skips), so it must keep its
        bare-``if`` form rather than the tile-only ``IT`` function."""
        from dace.sdfg.nodes import MapEntry
        from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                           TILE_K1_TAIL_MARKER)
        entry = scope.get(tasklet)
        while entry is not None:
            if isinstance(entry, MapEntry) and (entry.map.label.endswith(SCALAR_TAIL_MARKER)
                                                or entry.map.label.endswith(TILE_K1_TAIL_MARKER)):
                return True
            entry = scope.get(entry)
        return False

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
                # First-class conditional write: ``out = IT(cond, value)`` -- write value
                # where cond, else leave unchanged. No old-value read; lowered to a masked
                # TileStore. ``cond`` (e.g. ``__in_cond``) stays a live input via the call.
                new_lines.append("%s = %s(%s, %s)" % (lhs, CONDITIONAL_WRITE_FUNC, cond_src, rhs))
            else:  # intermediate local: compute unconditionally (defined for every lane)
                new_lines.append("%s = %s" % (lhs, rhs))
        tasklet.code = CodeBlock("\n".join(new_lines), language=dace.dtypes.Language.Python)
        return True
