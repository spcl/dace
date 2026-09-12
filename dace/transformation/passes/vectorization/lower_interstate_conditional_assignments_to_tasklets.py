# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Demote free symbols used in conditional-assignment tasklets to scalars."""
from typing import Any
import dace
from dace import dtypes, SDFG, properties, SDFGState, symbolic
from dace.sdfg import ControlFlowRegion, nodes
from dace.sdfg.state import BreakBlock, ConditionalBlock, LoopRegion
from dace.transformation.passes.vectorization.utils.tasklets import is_python_tasklet
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerInterstateConditionalAssignmentsToTasklets(ppl.Pass):
    """Demote to scalars the symbols a conditional binds, so branch lowering can predicate them.

    Two sources, both of which leave a symbol whose value depends on which branch ran:

    * the free symbols of a ``condition_symbol_to_scalar`` tasklet (the fp-factor path's own
      lowering of a guard);
    * a symbol an arm of a ``ConditionalBlock`` binds on one of its OWN interstate edges --
      ``if mask[j]: count = count + 1``.

    The second is a correctness requirement, not a tidy-up. A symbol holds ONE value for a whole
    tile, so a per-lane guard deciding an interstate assignment cannot be represented after
    widening: codegen emits ``if (<bool[W]>)``, an array decaying to a never-null pointer, and
    every lane takes the branch. ``BranchNormalization`` will not touch such an arm either -- it
    refuses any arm binding a symbol read outside it, on the reasoning that a loop carrying such a
    recurrence is sequential and therefore never tiled. That holds for a loop the frontend wrote as
    a loop; it does NOT hold for a ``dc.map`` whose body carries a masked counter, which arrives
    already parallel and is tiled with the scalar guard still in it. Demoting the symbol turns the
    binding into a dataflow write, which the ITE rewrite then gates per lane like any other.

    Refusals are inherited from :mod:`dace.sdfg.utils`: an SDFG argument has no definition here to
    rewrite, and a symbol the graph evaluates (a subset, a map range, a loop variable) stops being
    expressible as a scalar. Both are uniform across lanes, so the guard holds with them left as
    symbols.

    Tested as part of the vectorization pipeline.
    """

    CATEGORY: str = 'Vectorization'

    conditional_assignment_tasklet_prefix = properties.Property(dtype=str,
                                                                default="condition_symbol_to_scalar",
                                                                allow_none=False)
    also_demote = properties.ListProperty(element_type=str, default=[])
    apply_once = properties.Property(dtype=bool, default=False)

    def __init__(self, also_demote: list[str] | None = None) -> None:
        super().__init__()
        # A Property default is stored on the instance without copying, so every default-built
        # pass would otherwise share the one list object.
        self.also_demote = list(also_demote or [])
        #: Demotions performed by the current ``apply_pass`` run; reset at its top.
        self._applied: int = 0

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> dict[type[ppl.Pass] | ppl.Pass, None]:
        return {}

    @staticmethod
    def arm_bound_symbols(sd: SDFG) -> dict[str, 'dtypes.typeclass']:
        """Symbols an arm of a ``ConditionalBlock`` binds on one of its own interstate edges.

        Only ``sd``'s own regions are walked -- a nested SDFG binds symbols in its own scope and is
        reached separately, by the caller's walk over :meth:`~dace.SDFG.all_sdfgs_recursive`.

        :param sd: the SDFG whose conditional blocks are inspected.
        :returns: the bound names mapped to the dtype to demote each at (inferred from the
            assignment if undeclared, absent if neither), insertion-ordered.
        """
        bound: dict[str, dtypes.typeclass] = {}
        for block in sd.all_control_flow_blocks():
            if not isinstance(block, ConditionalBlock):
                continue
            for _condition, region in block.branches:
                for r in region.all_control_flow_regions():
                    for e in r.edges():
                        # An assignment DEFINES its symbol -- CloudSC's zlcrit is bound but never
                        # declared, so a ``sd.symbols`` miss must fall back to ``new_symbols``.
                        inferred = None
                        for name in sorted(e.data.assignments):
                            if name in bound:
                                continue
                            if name in sd.symbols:
                                bound[name] = sd.symbols[name]
                                continue
                            if inferred is None:
                                inferred = e.data.new_symbols(sd, sd.symbols)
                            if name in inferred:
                                bound[name] = inferred[name]
        return bound

    def demote_arm_bound_symbols(self, sdfg: SDFG) -> int:
        """Demote every conditionally-bound symbol that CAN be demoted, in ``sdfg`` and its nests.

        :param sdfg: the SDFG to transform in place.
        :returns: how many symbols were demoted.
        """
        demoted = 0
        for sd in sdfg.all_sdfgs_recursive():
            # Both gates below answer from a whole-SDFG scan that depends on ``sd`` alone, so ask
            # each ONCE per unmutated span rather than once per candidate symbol -- the two ran to
            # 8% of the vectorizer on CloudSC purely by re-walking. Only a demotion here can
            # invalidate them, and candidates far outnumber demotions, so drop them on a demotion
            # and rebuild lazily at the next ask.
            structural: set[str] | None = None
            free_syms: set[str] | None = None
            for name, dtype in self.arm_bound_symbols(sd).items():
                if name in sd.arrays:
                    continue  # already a scalar -- demoted by an earlier round or by another pass
                if structural is None:
                    structural = sdutil.structural_symbols(sd)
                    # Consulted only for a top-level SDFG; a nested one reads its parent's mapping.
                    free_syms = sd.free_symbols if sd.parent_nsdfg_node is None else set()
                if (not sdutil.symbol_demotes_to_transient_scalar(sd, name, free_symbols=free_syms)
                        or sdutil.symbol_carries_graph_structure(sd, name, structural=structural)):
                    continue
                # The symbol's OWN dtype. Overwriting it with fp64 first -- which is what
                # ``demote_symbol_to_scalar`` reads -- turned an integer accumulator into a
                # double, and every body that shifts or masks it stopped compiling:
                # ``invalid operands of types 'double' and 'int' to binary 'operator>>'``.
                # Nothing downstream wants a float: the ITE rewrite gates the write it finds,
                # whatever its dtype, and the rest of the graph already read this symbol at the
                # dtype it was declared with.
                sdutil.demote_symbol_to_scalar(sd, name, dtype, None)
                demoted += 1
                structural = free_syms = None  # the demotion rewrote ``sd``; rebuild before the next ask
        return demoted

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
            free_conditional_symbols: dict[str, None] = {}
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
                # Declared dtype where there is one, for the reason above. A symbol read only
                # by a lifted guard need not be declared at all, and fp64 stays the fallback for
                # that case -- it is the one the condition tasklet's operands promote to.
                sym_dtype = sdfg.symbols[conditional_sym] if conditional_sym in sdfg.symbols else dace.float64
                sdutil.demote_symbol_to_scalar(sdfg, conditional_sym, sym_dtype, None)
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

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> bool:
        """Demote conditional-assignment free symbols to scalars across the SDFG.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results from previously run passes (unused).
        :returns: ``True`` if any symbol was demoted.
        """
        self._applied = 0
        # Before the tasklet scan: that one only sees guards the fp-factor path already lowered
        # into ``condition_symbol_to_scalar`` tasklets, while an arm's own interstate binding is
        # still a symbol on an edge and is the shape that miscompiles when tiled.
        self._applied += self.demote_arm_bound_symbols(sdfg)
        has_applied = self._apply(sdfg)

        return has_applied or self._applied > 0
