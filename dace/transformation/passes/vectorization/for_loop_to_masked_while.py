# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Finalize an SVE-style per-core for-loop into a masked while-loop.

Last step of the SVE-style chain. The earlier passes tile the innermost
data-parallel map into ``num_cores`` *clean, divisible* per-core blocks
(``divides_evenly=True``, block ``B = roundup(ceil(N/P), W)``), attach a
**global-keyed** ``_iter_mask`` (``mask[l] = (i + l < N)`` — see
:class:`GenerateIterationMask` ``mode='global'``), W-vectorize the
divisible block, then :class:`MapToForLoop` turns the W-strided per-core
block into a :class:`LoopRegion`.

Tiling ``divides_evenly=True`` keeps every map range affine so the
divisibility / subset analysis never sees a ``Min`` (a ``Min`` in a map
range breaks ``symbolic.simplify`` — ``SympifyError: cannot sympify
SymExpr``). The over-covering ragged last block is harmless during
analysis because every body memory op is ``_iter_mask``-gated.

This pass performs the *only* two rewrites that must happen after all
map-range analysis is done, and it writes them into the
:class:`LoopRegion` **condition / update CodeBlocks only** — never a map
range, so no ``Min`` ever reaches ``symbolic.simplify``:

1. **Min-swap** ``loop_condition``: ``i < B_end`` becomes
   ``i < Min(global_ub, B_end)`` so the loop stops at the global trip
   instead of running fully-dead blocks past it.
2. **W-stride normalize** ``update_statement`` to ``i = i + W``.

Legality (semantics-preserving): the pass only fires on a
``_iter_mask``-gated SVE loop enclosed by a ``core`` map, so ``Min``
merely drops iterations the mask had already made entirely inactive —
end-to-end output is identical to the untiled SDFG. ``global_ub`` is
optional: when unset it is auto-derived from the enclosing ``core``
map's range (symbolic-safe). Inert when no such guarded loop is found;
idempotent.
"""
import ast
from typing import Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl

_CORE_PREFIX = "core"


@properties.make_properties
class ForLoopToMaskedWhile(ppl.Pass):
    """Min-swap + W-stride normalize an SVE-style per-core for-loop."""

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    global_ub = properties.Property(dtype=str,
                                    default=None,
                                    allow_none=True,
                                    desc="The original (pre-tile) *exclusive* upper bound of the "
                                    "innermost trip (e.g. ``\"N\"``). The loop condition is clamped "
                                    "to ``i < Min(global_ub, <block-end>)``. Optional: when ``None`` "
                                    "it is auto-derived from the enclosing ``core`` map's range "
                                    "(``range[-1][1] + 1``), which is symbolic-safe.")

    def __init__(self, vector_width: int = 8, global_ub: Optional[str] = None):
        super().__init__()
        self.vector_width = vector_width
        self.global_ub = global_ub

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _loop_body_has_iter_mask(lr: LoopRegion) -> bool:
        """Whether ``lr``'s body holds a ``_iter_mask`` transient.

        The mask is added by :class:`GenerateIterationMask` to the
        innermost body NestedSDFG, which :class:`MapToForLoop` then nests
        inside this :class:`LoopRegion`. Gating on its presence is what
        makes the Min-swap a provable no-op (only mask-dead blocks are
        dropped).

        :param lr: The loop region to inspect.
        :returns: ``True`` if any NestedSDFG inside ``lr`` declares an
            ``_iter_mask`` array.
        """
        for n, _ in lr.all_nodes_recursive():
            if isinstance(n, dace.nodes.NestedSDFG):
                if any(a.startswith("_iter_mask") for a in n.sdfg.arrays):
                    return True
            elif isinstance(n, dace.nodes.AccessNode) and n.data.startswith("_iter_mask"):
                return True
        return False

    @staticmethod
    def _enclosing_core_map(lr: LoopRegion) -> Optional[dace.nodes.MapEntry]:
        """The ``core``-prefixed map enclosing ``lr``, or ``None``.

        The SVE-style chain tiles the innermost map into a ``core``-
        prefixed outer block-distribution map; the per-core for-loop is
        nested under it. Walking parents to that map both confirms this
        is the SVE loop (not an unrelated user :class:`LoopRegion`) and
        yields the map whose ``range[-1][1] + 1`` is the original
        pre-tile exclusive trip bound — the ``global_ub`` for the
        Min-swap when not given explicitly.

        :param lr: The loop region to inspect.
        :returns: The enclosing ``core`` :class:`MapEntry`, or ``None``.
        """
        sdfg = lr.sdfg
        while sdfg is not None:
            pnode = sdfg.parent_nsdfg_node
            pstate = sdfg.parent
            if pnode is not None and pstate is not None:
                scope = pstate.scope_dict()
                node = scope.get(pnode)
                while node is not None:
                    if (isinstance(node, dace.nodes.MapEntry)
                            and any(p.startswith(_CORE_PREFIX) for p in node.map.params)):
                        return node
                    node = scope.get(node)
            sdfg = pstate.sdfg if pstate is not None else None
        return None

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Min-swap + W-stride every SVE-style per-core for-loop.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of loops rewritten, or ``None`` if none.
        """
        W = self.vector_width
        rewritten = 0
        for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
            if not isinstance(cfg, LoopRegion):
                continue
            if not self._loop_body_has_iter_mask(cfg):
                continue
            core = self._enclosing_core_map(cfg)
            if core is None:
                continue
            # The global exclusive trip bound. Authoritative source is an
            # explicit ``global_ub`` captured at tile time by the SVE
            # orchestrator (immune to any intervening pass). The core map
            # re-derivation is the standalone/unit fallback when unset.
            derived = core.map.range[-1][1] + 1
            if self.global_ub is not None:
                gub = str(self.global_ub)
                # Consistency assert: the late core-map re-derivation MUST
                # agree with the value captured at tile time. A mismatch
                # means some pass between tiling and here perturbed the
                # ``core`` map range — fail loud rather than emit a wrong
                # Min clamp (silent OOB / lost iterations).
                if dace.symbolic.simplify(dace.symbolic.pystr_to_symbolic(gub) - derived) != 0:
                    raise RuntimeError(f"ForLoopToMaskedWhile: global_ub mismatch for loop {cfg.label!r} — "
                                       f"captured-at-tile {gub!r} != core-map re-derived {str(derived)!r}. "
                                       f"A pass between SVE tiling and here mutated the 'core' map range; "
                                       f"the Min clamp would be wrong (silent OOB or dropped iterations).")
            else:
                gub = str(derived)
            loop_var = cfg.loop_variable
            if cfg.loop_condition is None:
                continue
            # MapToForLoop emits ``<var> < <end>`` (fully parenthesized).
            # Extract the end expression via the AST — string splitting
            # mishandles the wrapping parens.
            try:
                expr = cfg.loop_condition.code[0].value
            except (AttributeError, IndexError, TypeError):
                continue
            if not (isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Lt)
                    and isinstance(expr.left, ast.Name) and expr.left.id == loop_var):
                continue
            end_node = expr.comparators[0]
            # Already Min-swapped (idempotent).
            if (isinstance(end_node, ast.Call) and isinstance(end_node.func, ast.Name)
                    and end_node.func.id in ("Min", "min")):
                continue
            end_expr = ast.unparse(end_node)
            cfg.loop_condition = CodeBlock(f"{loop_var} < Min({gub}, {end_expr})")
            # W-stride normalize (idempotent: identical string if already +W).
            cfg.update_statement = CodeBlock(f"{loop_var} = {loop_var} + {W}")
            rewritten += 1
        return rewritten or None
