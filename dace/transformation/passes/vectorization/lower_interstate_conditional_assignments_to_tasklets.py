# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Demote free symbols used in conditional-assignment tasklets to scalars."""
from typing import Any, Dict, List, Optional
import dace
from dace import SDFG, properties, SDFGState, symbolic
from dace.sdfg import ControlFlowRegion, nodes
from dace.sdfg.state import BreakBlock, ConditionalBlock, LoopRegion
from dace.transformation.passes.vectorization.utils.tasklets import is_python_tasklet
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerInterstateConditionalAssignmentsToTasklets(ppl.Pass):
    """Demote free symbols of conditional-assignment tasklets to fp64 scalars.

    Tested as part of the vectorization pipeline.
    """

    CATEGORY: str = 'Vectorization'

    conditional_assignment_tasklet_prefix = properties.Property(dtype=str,
                                                                default="condition_symbol_to_scalar",
                                                                allow_none=False)
    also_demote = properties.ListProperty(element_type=str, default=[])
    apply_once = properties.Property(dtype=bool, default=False)

    def __init__(self, also_demote: Optional[List[str]] = None) -> None:
        super().__init__()
        # A Property default is stored on the instance without copying, so every default-built
        # pass would otherwise share the one list object.
        self.also_demote = list(also_demote or [])

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, cfg: ControlFlowRegion) -> bool:
        """Recursively demote conditional-assignment free symbols within a control-flow region.

        :param cfg: The control-flow region (or SDFG) to process.
        :returns: ``True`` if a demotion was applied and ``apply_once`` requests early exit.
        :raises Exception: If an unsupported control-flow node type is encountered.
        """
        if self._applied > 0 and self.apply_once:
            return False

        if all(isinstance(n, SDFGState) for n in cfg.nodes()):
            # Ordered: this drives demote_symbol_to_scalar, which adds arrays to the SDFG.
            free_conditional_symbols: Dict[str, None] = {}
            for state in cfg.nodes():
                for node in state.nodes():
                    # Python-bodied only -- the expression parse below is undefined otherwise.
                    # NOT the lane-level guard: this demotes a symbol SDFG-wide and the conditional
                    # arm it reads is lowered before any map scope exists around it.
                    if (isinstance(node, nodes.Tasklet) and is_python_tasklet(node)
                            and node.label.startswith(self.conditional_assignment_tasklet_prefix)):
                        expr = symbolic.SymExpr(node.code.as_string.split(" = ")[-1])
                        syms = expr.free_symbols
                        # If not in inconnectors then it is a symbol
                        all_free_syms = {str(s) for s in syms if str(s) not in node.in_connectors}
                        # Should be empty
                        # Remove python boolean operators
                        # Remove array names
                        # Remove symbols coming from parent sdfg can't be demoted
                        # => Exclude them
                        func_calls = symbolic.arrays(expr)
                        boolean_func_calls = {
                            "OR", "Or", "or", "AND", "And", "and", "not", "Not", "NOT", "False", "True", "false",
                            "true", "FALSE", "TRUE"
                        }
                        arr_names = {str(k) for k in cfg.sdfg.arrays.keys()}
                        parent_symbol_name = {str(k)
                                              for k in cfg.sdfg.parent_nsdfg_node.symbol_mapping.keys()
                                              } if cfg.sdfg.parent_nsdfg_node is not None else {}
                        no_access_free_syms = all_free_syms - func_calls.union(boolean_func_calls).union(
                            arr_names).union(parent_symbol_name)
                        free_conditional_symbols.update(dict.fromkeys(sorted(no_access_free_syms)))

            for additional_demote_sym in self.also_demote:
                if additional_demote_sym in cfg.sdfg.symbols:
                    free_conditional_symbols[additional_demote_sym] = None

            # We should demote all the free conditional symbols
            for conditional_sym in free_conditional_symbols:
                sdfg = cfg.sdfg if not isinstance(cfg, SDFG) else cfg
                # An SDFG argument has no definition here to rewrite, and a symbol the graph
                # evaluates (subset, map range, loop variable) stops being expressible as a scalar.
                # Both are uniform across lanes, so the condition holds with them left symbols.
                if (not sdutil.symbol_demotes_to_transient_scalar(sdfg, conditional_sym)
                        or sdutil.symbol_carries_graph_structure(sdfg, conditional_sym)):
                    continue
                # Cast all symbols to fp64
                sdfg.symbols[conditional_sym] = dace.float64
                sdutil.demote_symbol_to_scalar(sdfg, conditional_sym, dace.float64, None)
                # Set-zero all of them
                assert conditional_sym not in sdfg.symbols
                self._applied += 1
                if self._applied > 0 and self.apply_once:
                    return True

        for n in cfg.nodes():
            if isinstance(n, SDFGState):
                for sn in n.nodes():
                    if isinstance(sn, nodes.NestedSDFG):
                        if self._apply(sn.sdfg) and self.apply_once:
                            return True
            elif isinstance(n, ConditionalBlock):
                for _, branch in n.branches:
                    if self._apply(branch) and self.apply_once:
                        return True
            elif isinstance(n, LoopRegion):
                for ln in n.nodes():
                    if not isinstance(ln, SDFGState):
                        if self._apply(ln) and self.apply_once:
                            return True
                    else:
                        for sn in ln.nodes():
                            if isinstance(sn, nodes.NestedSDFG):
                                if self._apply(sn.sdfg) and self.apply_once:
                                    return True
            elif isinstance(n, ControlFlowRegion):
                for ln in n.nodes():
                    if not isinstance(ln, SDFGState):
                        if self._apply(ln) and self.apply_once:
                            return True
                    else:
                        for sn in ln.nodes():
                            if isinstance(sn, nodes.NestedSDFG):
                                if self._apply(sn.sdfg) and self.apply_once:
                                    return True
            else:
                # Ok if a break block just connintue
                if isinstance(n, BreakBlock):
                    continue
                else:
                    raise Exception(f"Unsupported node type for pass node {n} type {type(n)}")

        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> bool:
        """Demote conditional-assignment free symbols to scalars across the SDFG.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results from previously run passes (unused).
        :returns: ``True`` if any symbol was demoted.
        """
        self._applied = 0
        has_applied = self._apply(sdfg)

        return has_applied
