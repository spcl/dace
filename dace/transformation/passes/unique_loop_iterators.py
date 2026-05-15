# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give every ``LoopRegion``'s loop variable a globally-unique name.

Two ``LoopRegion``s synthesised from independent source-level loops can
share the same Python-level iterator name (``for i in ...`` in both).
Once they merge into a single SDFG -- e.g. through inlining -- the
shared name turns into an alias and downstream passes (symbol
propagation, transient promotion, codegen) silently mix the two
iterator values.  This pass walks every region, renames each loop
variable to a unique ``_loop_it_<N>`` symbol, and propagates the
rename through:

* the cfg's memlets and meta accesses,
* every interstate-edge assignment / condition inside the region,
* every tasklet body that mentions the iterator,
* the symbol mapping of any nested SDFG that imports the iterator.

The ``assign_loop_iterator_post_value`` property opts into a second
behaviour: a postfix assignment ``<loop_var> = <post_value>`` is
materialised in a state immediately after the loop region, so any
downstream reads of the original loop-variable name see the
"iterator-after-loop" value (gfortran / ifort / flang all leave a
counted-DO iterator at one stride past the last attained value when
the loop exits normally; the post-value formula below produces the
same result).  This is OFF by default because the convention is
Fortran-specific -- pure Python / dace-frontend callers leave the
iterator semantically undefined after a loop, so a postfix assignment
would only add dead state.
"""

from typing import Optional, Union

import sympy as sp

import dace
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

# Symbol-name prefix the pass uses for the renamed iterators.  Includes
# ``loop_it`` so the source of the name is unambiguous in any artifact
# that surfaces it (codegen output, dumped SDFGs, error messages).
_LOOP_ITER_NAME_PREFIX = "_loop_it"
# State-label prefix for the optional postfix-assignment state added
# after a LoopRegion when ``assign_loop_iterator_post_value`` is set.
_POST_VALUE_STATE_PREFIX = "loop_iter_post_value"


def _rename_symbol_by_name(expr: Union[sp.Basic, str], old_name: str, new_name: str) -> sp.Basic:
    """Replace every sympy ``Symbol`` with ``.name == old_name`` in
    ``expr`` with a fresh ``Symbol(new_name)`` -- regardless of the
    original symbol's assumptions.

    ``sp.subs(old_sym, new_sym)`` and ``sp.replace`` both match by
    full symbol identity (name + assumptions).  DaCe and the HLFIR
    bridge can leave several symbols sharing one name but carrying
    different assumption flags (integer-vs-default), so a strict
    identity match would miss aliases.  Filtering ``free_symbols`` by
    name and feeding the matches to sympy's ``xreplace`` (a literal
    structural substitution) sidesteps that.
    """
    if isinstance(expr, str):
        expr = dace.symbolic.pystr_to_symbolic(expr)
    if not isinstance(expr, sp.Basic):
        return expr
    new_sym = sp.Symbol(new_name)
    matches = {s: new_sym for s in expr.free_symbols if getattr(s, 'name', None) == old_name}
    if not matches:
        return expr
    return expr.xreplace(matches)


@dace.properties.make_properties
@explicit_cf_compatible
class UniqueLoopIterators(ppl.Pass):
    """Rename every LoopRegion's loop variable to a unique ``_loop_it_<N>``."""

    # Module-wide counter -- one pass instance reused across nested SDFGs
    # must not reset midway.  Class-level so two top-level apply_pass
    # calls keep producing fresh names across the whole compilation
    # (matters when an HLFIR translation generates many small SDFGs that
    # later compose into one).
    _loop_var_counter = 0

    assign_loop_iterator_post_value = dace.properties.Property(
        dtype=bool,
        default=False,
        desc=("If True, emit a post-loop assignment state that sets the original loop variable to its "
              "post-loop value (``init + diff - (diff mod step)``) so downstream reads see the "
              "iterator-after-loop value.  Required by Fortran callers; pure Python / DaCe semantics "
              "leave the iterator undefined after a loop and don't need it."),
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rename_one_loop_var(self, cfg: Union[ControlFlowRegion, dace.SDFG], old_name: str, new_name: str) -> None:
        """Rename ``old_name`` to ``new_name`` inside ``cfg``.  The
        ControlFlowRegion's own ``replace_dict`` already cascades the
        rename through every state, interstate edge, memlet, tasklet,
        and nested region inside the region -- selectively scoped to
        the region.  Nested SDFG ``symbol_mapping`` *keys* (the inner
        symbol names) are not handled by that cascade -- the key is
        owned by the inner SDFG -- so we re-key explicitly and recurse
        into the inner SDFG when its own symbol table declares the name.
        """
        repl = {old_name: new_name}
        cfg.replace_meta_accesses(repl)
        cfg.replace_dict(repl)

        for state in cfg.all_states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.NestedSDFG):
                    continue
                if old_name in node.symbol_mapping:
                    rhs = node.symbol_mapping.pop(old_name)
                    node.symbol_mapping[new_name] = _rename_symbol_by_name(rhs, old_name, new_name)
                if old_name in node.sdfg.symbols:
                    self._rename_one_loop_var(node.sdfg, old_name, new_name)

    def _compute_post_value(self, loop: LoopRegion) -> Optional[sp.Basic]:
        """Return the iterator value matching the convention every
        mainstream Fortran compiler uses for counted ``DO`` loops:
        one stride past the last attained value.  Returns ``None`` when
        the loop's end / stride / init can't be statically recovered.

        Formula: ``post = init + diff - (diff mod step)`` with
        ``diff = loop_end - init + step``.  Stays integer-typed end-to-
        end (the floor/integer-division variant ``init + floor((end -
        init) / step) * step + step`` triggers a sympy / int_floor
        codegen interaction that drops the trailing ``+ step``).

        Worked cases:
        * ``DO i = 1, N``        (step = 1):  diff = N,    mod = 0,  post = N + 1
        * ``DO i = N, 1, -1``    (step = -1): diff = -N,   mod = 0,  post = 0
        * ``DO i = 1, 10, 2``    (step = 2):  diff = 11,   mod = 1,  post = 11
        * ``DO i = 10, 2, -2``   (step = -2): diff = -10,  mod = 0,  post = 0
        """
        loop_end = loop_analysis.get_loop_end(loop)
        if loop_end is None:
            return None
        stride = loop_analysis.get_loop_stride(loop)
        init = loop_analysis.get_init_assignment(loop)
        if stride is not None and init is not None:
            diff = loop_end - init + stride
            return init + diff - sp.Mod(diff, stride)
        if stride is not None:
            # Init unknown -- fall back to last-attained + step
            # (correct when step == 1 since loop_end already equals
            # the last attained value).
            return loop_end + stride
        # Stride unknown -- fall back to last-attained value.
        return loop_end

    def _apply_recursive(self, sdfg: dace.SDFG) -> None:
        # Names DaCe knows are arrays -- ``symstr`` uses this set to
        # render array subscripts as ``arr[idx]`` (Python syntax for
        # interstate-edge assignments) rather than ``arr(idx)`` (sympy
        # default function-call form, which C++ codegen later rejects
        # since ``arr`` is a pointer, not callable).
        array_names = frozenset(sdfg.arrays.keys())
        for cfg in sdfg.all_control_flow_regions():
            if not isinstance(cfg, LoopRegion):
                continue
            old_name = cfg.loop_variable
            if not old_name:
                # ``while``/``do-while`` loops with no explicit
                # induction variable have nothing to rename.
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
        self._apply_recursive(sdfg)
