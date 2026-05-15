# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give every ``LoopRegion`` 's loop variable a globally-unique name.

Independent source loops can share an iterator name ( ``for i`` in
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
    """Rename every LoopRegion's loop variable to a unique ``_loop_it_<N>`` ."""

    _loop_var_counter = 0

    assign_loop_iterator_post_value = dace.properties.Property(
        dtype=bool,
        default=True,
        desc=(
            "If True, emit a post-loop state assigning the original loop variable its counted "
            "exit value so downstream reads see the iterator-after-loop value. Required especially for Fortran inputs"),
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rename_one_loop_var(self, cfg: Union[ControlFlowRegion, dace.SDFG], old_name: str, new_name: str) -> None:
        """Rename ``old_name`` to ``new_name`` inside ``cfg`` .

        ``replace_dict`` cascades the rename through the region's
        states, edges, memlets, tasklets, and nested-SDFG
        ``symbol_mapping`` *values*.  It does not rename
        ``symbol_mapping`` *keys* (the inner symbol names, owned by the
        inner SDFG), so those are re-keyed here and the inner SDFG is
        recursed into when its symbol table declares the name.

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
        ``diff = loop_end - init + step`` ; ``int_floor`` stays
        integer-typed so codegen emits exact integer division.  E.g.
        ``DO i = 1, N`` -> ``N + 1`` ; ``DO i = N, 1, -1`` -> ``0`` ;
        ``DO i = 1, 10, 2`` -> ``11`` .

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
        for cfg in sdfg.all_control_flow_regions():
            if not isinstance(cfg, LoopRegion):
                continue
            old_name = cfg.loop_variable
            if not old_name:
                # while / do-while loops have no induction variable.
                continue
            new_name = f"{_LOOP_ITER_NAME_PREFIX}_{UniqueLoopIterators._loop_var_counter}"
            self._rename_one_loop_var(cfg, old_name, new_name)

            if self.assign_loop_iterator_post_value:
                post_value = self._compute_post_value(cfg)
                if post_value is not None:
                    post_value_str = dace.symbolic.symstr(post_value, arrayexprs=array_names).strip()
                    if post_value_str:
                        cfg.parent_graph.add_state_after(
                            cfg,
                            f"{_POST_VALUE_STATE_PREFIX}_{UniqueLoopIterators._loop_var_counter}",
                            assignments={old_name: f"({post_value_str})"})

            UniqueLoopIterators._loop_var_counter += 1

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rename every ``LoopRegion`` iterator in ``sdfg`` and its nested SDFGs.

        :param sdfg: SDFG to mutate in place.
        """
        self._apply_recursive(sdfg)
