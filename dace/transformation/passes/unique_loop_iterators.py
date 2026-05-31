# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rename every ``LoopRegion`` loop variable to a globally-unique
``_loop_it_<N>`` symbol so independent source loops that share a name
no longer alias once their regions merge into one SDFG. When
``assign_loop_iterator_post_value`` is set, the original name is
materialised in a post-loop state with the counted-DO exit value
(Fortran semantics) so downstream reads see the iterator-after-loop
value.
"""

from typing import Optional, Union

import dace
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

_LOOP_ITER_NAME_PREFIX = "_loop_it"
_POST_VALUE_STATE_PREFIX = "loop_iter_post_value"


@dace.properties.make_properties
@explicit_cf_compatible
class UniqueLoopIterators(ppl.Pass):
    """Rename every LoopRegion loop variable to a unique ``_loop_it_<N>``.

    The ``<N>`` counter is per-call and seeded past any iterator already in
    ``_loop_it_<N>`` form (on loops AND on map parameters), so renames never
    collide with existing unique names and the numbering is deterministic
    for a given SDFG.
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

    def _rename_one_loop_var(self, cfg: Union[ControlFlowRegion, dace.SDFG], old_name: str, new_name: str):
        """Rename ``old_name`` to ``new_name`` inside ``cfg`` and re-key every
        nested-SDFG ``symbol_mapping`` at any depth that imports the symbol.

        ``replace_dict`` rewrites ``symbol_mapping`` *values* but not the *keys*
        (inner symbol names owned by each nested SDFG), so the re-keying must
        be done here -- and unconditionally, because ``replace_dict`` has
        already removed ``old_name`` from ``node.sdfg.symbols``.
        """
        repl = {old_name: new_name}
        cfg.replace_meta_accesses(repl)
        cfg.replace_dict(repl)

        for state in cfg.all_states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.NestedSDFG):
                    continue
                if old_name in node.symbol_mapping:
                    node.symbol_mapping[new_name] = node.symbol_mapping.pop(old_name)
                if old_name in node.sdfg.symbols:
                    self._rename_one_loop_var(node.sdfg, old_name, new_name)

    def _compute_post_value(self, loop: LoopRegion) -> Optional[dace.symbolic.SymbolicType]:
        """Counted-DO exit value: ``init + int_floor(diff, step) * step`` where
        ``diff = loop_end - init + step``. ``int_floor`` keeps the result
        integer-typed for exact codegen division.

        :param loop: Loop region to analyse.
        :returns: Post-loop iterator value, or ``None`` if the loop bound is
                  not statically known.
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
            return loop_end + stride
        return loop_end

    def _apply_recursive(self, sdfg: dace.SDFG):
        array_names = frozenset(sdfg.arrays.keys())
        # Names shared by more than one LoopRegion in this SDFG: cloning passes
        # (LoopFission) leave siblings carrying the same ``_loop_it_<N>``, so
        # duplicated names must be re-disambiguated even if already in the
        # unique-prefix form.
        loop_vars = [
            r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable
        ]
        duplicated = {v for v in loop_vars if loop_vars.count(v) > 1}
        for cfg in sdfg.all_control_flow_regions():
            if not isinstance(cfg, LoopRegion):
                continue
            old_name = cfg.loop_variable
            if not old_name:
                continue
            if old_name.startswith(f"{_LOOP_ITER_NAME_PREFIX}_") and old_name not in duplicated:
                # Already unique -- skip for idempotency. Re-renaming a unique
                # iterator would also strand nested-SDFG symbol_mapping keys
                # because the second rename can't see the prior import.
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
                # Drop the dead declaration the frontend left behind so it
                # doesn't leak as a phantom free symbol on the enclosing
                # NestedSDFG boundary. Use ``used_symbols(all_symbols=False)``;
                # ``free_symbols`` re-folds ``sdfg.symbols.keys()`` and would
                # always report the declaration as "used".
                sdfg.remove_symbol(old_name)

            self._next_id += 1

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)

    @staticmethod
    def _first_free_id(sdfg: dace.SDFG) -> int:
        """Lowest ``<N>`` that no existing ``_loop_it_<N>`` iterator uses.

        Scans both ``LoopRegion`` loop variables and ``MapEntry`` parameters
        across the SDFG tree, because ``LoopToMap`` carries the loop's
        ``_loop_it_<N>`` name onto the map parameter; reusing that ``<N>``
        for a fresh loop would alias the two iteration variables.

        :param sdfg: The root SDFG.
        :returns: ``max(existing <N>) + 1``, or ``0`` if there are none.
        """
        prefix = f"{_LOOP_ITER_NAME_PREFIX}_"

        def _id_of(name: str) -> int:
            suffix = name[len(prefix):]
            return int(suffix) if name.startswith(prefix) and suffix.isdigit() else -1

        max_id = -1
        stack = [sdfg]
        while stack:
            graph = stack.pop()
            for cfg in graph.all_control_flow_regions():
                if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                    max_id = max(max_id, _id_of(cfg.loop_variable))
            for state in graph.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        stack.append(node.sdfg)
                    elif isinstance(node, dace.nodes.MapEntry):
                        for param in node.map.params:
                            max_id = max(max_id, _id_of(param))
        return max_id + 1

    def apply_pass(self, sdfg: dace.SDFG, _):
        """Rename every ``LoopRegion`` iterator in ``sdfg`` and its nested SDFGs.

        :param sdfg: SDFG to mutate in place.
        :param _: Pipeline results (unused).
        """
        self._next_id = self._first_free_id(sdfg)
        self._apply_recursive(sdfg)
