# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give every ``LoopRegion`` loop variable a globally-unique name.

Independent source loops can share an iterator name (``for i`` in
both); once their regions merge into one SDFG the shared name aliases
and downstream passes mix the two iterator values.  This pass renames
each loop variable to a unique ``_loop_it_<N>`` symbol and cascades the
rename through the region's memlets, interstate edges, tasklet bodies,
and any nested-SDFG symbol mapping that imports it.

``assign_loop_iterator_post_value`` additionally materializes
``<loop_var> = <post_value>`` in a state after the loop, so downstream
reads of the original name see the counted-DO exit value Fortran programs.
ON by default as it will not affect Python/SDFG API inputs.
"""

from typing import Optional, Union

import sympy as sp

import dace
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

# Prefix for renamed iterators; self-identifying in codegen / dumped SDFGs.
_LOOP_ITER_NAME_PREFIX = "_loop_it"
# State-label prefix for the optional post-value assignment state.
_POST_VALUE_STATE_PREFIX = "loop_iter_post_value"


@dace.properties.make_properties
@explicit_cf_compatible
class UniqueLoopIterators(ppl.Pass):
    """Rename every LoopRegion's loop variable to a unique ``_loop_it_<N>``.

    The ``<N>`` counter is per-call and seeded just past any iterator already
    in ``_loop_it_<N>`` form anywhere in the SDFG tree, so renames never
    collide with existing unique names and the numbering is deterministic for
    a given SDFG regardless of how many times the pass (or other SDFGs) ran
    before -- no cross-call global state.
    """

    assign_loop_iterator_post_value = dace.properties.Property(
        dtype=bool,
        default=True,
        desc=(
            "If True, emit a post-loop state assigning the original loop variable its counted "
            "exit value so downstream reads see the iterator-after-loop value. Required especially for Fortran inputs"),
    )

    def __init__(self, assign_loop_iterator_post_value: bool = True):
        super().__init__()
        self.assign_loop_iterator_post_value = assign_loop_iterator_post_value

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rename_one_loop_var(self, cfg: Union[ControlFlowRegion, dace.SDFG], old_name: str, new_name: str) -> None:
        """Rename ``old_name`` to ``new_name`` inside ``cfg``.

        ``replace_dict`` cascades the rename through the region's states,
        edges, memlets, tasklets, nested-SDFG bodies, and nested-SDFG
        ``symbol_mapping`` *values* -- recursively, at every depth. It does
        NOT rename ``symbol_mapping`` *keys* (the inner symbol names, owned
        by each nested SDFG). Those keys must be re-keyed here for EVERY
        nested SDFG at any depth that imports the renamed symbol; this
        re-keying must NOT be gated on ``old_name in node.sdfg.symbols``,
        because ``replace_dict`` has already renamed that declaration away,
        so the gate would be false for every nested SDFG and a grandchild's
        ``symbol_mapping`` key (e.g. ``{i: _loop_it_0}`` whose body now reads
        ``_loop_it_0``) would be left stale -> "Missing symbols on nested
        SDFG: ['_loop_it_0']".

        :param cfg: Region (or SDFG) to rename within.
        :param old_name: Current iterator symbol name.
        :param new_name: Unique replacement name.
        """
        repl = {old_name: new_name}
        cfg.replace_meta_accesses(repl)
        cfg.replace_dict(repl)

        for state in cfg.all_states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.NestedSDFG):
                    continue
                if old_name in node.symbol_mapping:
                    # ``replace_dict`` already rewrote the mapped value;
                    # only the key (inner symbol name) needs re-keying.
                    node.symbol_mapping[new_name] = node.symbol_mapping.pop(old_name)
                if old_name in node.sdfg.symbols:
                    self._rename_one_loop_var(node.sdfg, old_name, new_name)

    def _compute_post_value(self, loop: LoopRegion) -> Optional[sp.Basic]:
        """Counted-DO exit value: one stride past the last attained value.

        ``post = init + int_floor(diff, step) * step`` where
        ``diff = loop_end - init + step``; ``int_floor`` stays
        integer-typed so codegen emits exact integer division.  E.g.
        ``DO i = 1, N`` -> ``N + 1``; ``DO i = N, 1, -1`` -> ``0``;
        ``DO i = 1, 10, 2`` -> ``11``.

        :param loop: Loop region to analyse.
        :returns: Post-loop iterator value.
        """
        loop_end = loop_analysis.get_loop_end(loop)
        if loop_end is None:
            return None
        stride = loop_analysis.get_loop_stride(loop)
        init = loop_analysis.get_init_assignment(loop)
        if stride is not None and init is not None:
            diff = loop_end - init + stride
            return init + dace.symbolic.int_floor(diff, stride) * stride
        if stride is not None:
            # Init unknown: last-attained + step (exact when step == 1).
            return loop_end + stride
        # Stride unknown: fall back to last-attained value.
        return loop_end

    def _apply_recursive(self, sdfg: dace.SDFG) -> None:
        array_names = frozenset(sdfg.arrays.keys())
        # Loop-variable names that more than one LoopRegion in THIS SDFG shares.
        # LoopFission clones a loop into siblings that keep the same
        # ``_loop_it_<N>`` name, which then aliases (e.g. LoopToMap refuses to
        # parallelize one sibling because the other "reads" the shared iterator
        # after it). Such duplicates must be re-disambiguated even though they
        # are already in ``_loop_it_*`` form.
        loop_vars = [
            r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable
        ]
        duplicated = {v for v in loop_vars if loop_vars.count(v) > 1}
        for cfg in sdfg.all_control_flow_regions():
            if not isinstance(cfg, LoopRegion):
                continue
            old_name = cfg.loop_variable
            if not old_name:
                # while / do-while loops have no induction variable.
                continue
            if old_name.startswith(f"{_LOOP_ITER_NAME_PREFIX}_") and old_name not in duplicated:
                # Already a unique ``_loop_it_<N>`` (e.g. this pass ran earlier
                # in the pipeline). Re-renaming a unique iterator is pointless
                # and, worse, the nested-SDFG re-key does not survive a second
                # rename of an already-imported iterator -> a deeply-nested SDFG
                # ends up referencing the new name without a ``symbol_mapping``
                # import ("Missing symbols on nested SDFG: ['_loop_it_<N>']").
                # Skipping unique names keeps the pass idempotent; duplicated
                # names (from fission) still fall through to be disambiguated.
                continue
            new_name = f"{_LOOP_ITER_NAME_PREFIX}_{self._next_id}"
            self._rename_one_loop_var(cfg, old_name, new_name)

            if self.assign_loop_iterator_post_value:
                post_value = self._compute_post_value(cfg)
                if post_value is not None:
                    post_value_str = dace.symbolic.symstr(post_value, arrayexprs=array_names).strip()
                    if post_value_str:
                        cfg.parent_graph.add_state_after(cfg,
                                                         f"{_POST_VALUE_STATE_PREFIX}_{self._next_id}",
                                                         assignments={old_name: f"({post_value_str})"})
            elif old_name in sdfg.symbols and old_name not in sdfg.used_symbols(all_symbols=False):
                # The rename was scoped to ``cfg`` so the LoopRegion's
                # body, init/condition/update no longer reference
                # ``old_name``. Without the post-value epilogue there is
                # also no surviving inter-state assignment using it, so the
                # SDFG-level declaration left behind by the frontend leaks
                # as a phantom free symbol on the enclosing NestedSDFG
                # boundary ("Missing symbols on nested SDFG: ['i']"). Drop
                # the dead declaration so the symbol table reflects actual
                # usage. The check uses ``used_symbols(all_symbols=False)``
                # rather than ``sdfg.free_symbols`` because the latter
                # unconditionally re-folds ``sdfg.symbols.keys()`` back in
                # (see ``ControlFlowRegion._used_symbols_internal``), which
                # would make the declaration appear "used" by virtue of
                # being declared and prevent its own removal -- circular.
                sdfg.remove_symbol(old_name)

            self._next_id += 1

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)

    @staticmethod
    def _first_free_id(sdfg: dace.SDFG) -> int:
        """Lowest ``<N>`` that no existing ``_loop_it_<N>`` iterator uses.

        Scans every ``LoopRegion`` in the whole SDFG tree (including nested
        SDFGs) so a fresh rename never collides with an iterator a previous run
        already produced.

        :param sdfg: The root SDFG.
        :returns: ``max(existing <N>) + 1``, or ``0`` if there are none.
        """
        prefix = f"{_LOOP_ITER_NAME_PREFIX}_"
        max_id = -1
        stack = [sdfg]
        while stack:
            graph = stack.pop()
            for cfg in graph.all_control_flow_regions():
                if isinstance(cfg, LoopRegion) and cfg.loop_variable and cfg.loop_variable.startswith(prefix):
                    suffix = cfg.loop_variable[len(prefix):]
                    if suffix.isdigit():
                        max_id = max(max_id, int(suffix))
            for state in graph.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        stack.append(node.sdfg)
        return max_id + 1

    def apply_pass(self, sdfg: dace.SDFG, _):
        """Rename every ``LoopRegion`` iterator in ``sdfg`` and its nested SDFGs.

        :param sdfg: SDFG to mutate in place.
        """
        self._next_id = self._first_free_id(sdfg)
        self._apply_recursive(sdfg)
