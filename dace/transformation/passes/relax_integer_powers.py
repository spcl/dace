# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RelaxIntegerPowers`` -- lower ``base ** exp`` to ``ipow`` where the exponent
is a non-negative integer.

DaCe's symbolic C++ printer emits ``dace::math::pow`` (libm, ``double``) for a
non-constant exponent, which is illegal where an integer is required -- an array
size, a subscript, or a loop bound.

``Pow(base, exp) -> ipow(base, exp)`` whenever ``exp`` is a provable non-negative
integer: a non-negative integer constant, an integer-valued float literal, or a
symbolic integer proven ``>= 0`` by interval analysis over the enclosing iterator
ranges (``K - i - 1`` with ``for i in range(K)``.
"""
from typing import Any, Dict, FrozenSet, Optional, Tuple

import numpy
from dace.ordered import OrderedSet

from dace import SDFG, data, subsets, symbolic, symbolic_engine
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis

#: A live iteration range ``symbol name -> (low, high)`` (inclusive).
_Ranges = Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]

#: Per-symbol-name facts an enclosing scope declares (``'integer'``/``'positive'``/``'nonnegative'``).
_Facts = Dict[str, FrozenSet[str]]

#: The power head, in whichever backend built the expression.
_POW = symbolic_engine.Pow

#: Build modes tried when rebuilding a node, in order: evaluating first, then unevaluated.
_BUILD_MODES = ({}, {'evaluate': False})


def _rebuild(expr, args, mapped):
    """``expr``'s head applied to ``mapped``, built the way ``expr`` itself was, or ``None``.

    An engine may hold several ops under one head (real division and a product both read ``Mul``), and
    a structural head is not a constructor at all -- so the head is trusted only where it reproduces
    ``expr`` from ``expr``'s OWN arguments. DaCe mints packed shapes and strides unevaluated, and their
    argument order is exactly what the evaluating constructor canonicalizes away: reproducing those
    needs the unevaluated build, which is then also what the rewritten node is built with, so the pack
    survives. Checked per node, so an unmodelled kind costs a missed relaxation rather than a changed
    value.
    """
    for kwargs in _BUILD_MODES:
        try:
            if expr.func(*args, **kwargs) == expr:
                return expr.func(*mapped, **kwargs)
        except (TypeError, ValueError):
            continue
    return None


def _map_pows(expr, rewrite):
    """``expr`` with every ``Pow`` passed through ``rewrite(base, exp)``, bottom-up.

    Replaces ``expr.replace(Pow, ...)``, which is sympy ``Wild`` machinery: a head walk is all this
    needs and it works on any backend's values.
    """
    args = expr.args
    if not args:
        return expr
    mapped = [_map_pows(a, rewrite) for a in args]
    if isinstance(expr, _POW):
        return rewrite(mapped[0], mapped[1])
    if all(m is a for m, a in zip(mapped, args)):
        return expr
    rebuilt = _rebuild(expr, args, mapped)
    return expr if rebuilt is None else rebuilt


def _affine_coeff(exp, sym):
    """``exp``'s coefficient in ``sym`` if affine in it, else ``None``.

    Two substitutions and a linearity check rather than ``diff`` (sympy-island calculus) or
    ``coeff(sym, 1)`` -- the latter answers 0 for ``sym**2``, a constant, so the corner minimum below
    would read a quadratic as independent of the iterator and "prove" it non-negative.
    """
    at0 = exp.subs({sym: 0})
    slope = exp.subs({sym: 1}) - at0
    if exp.subs({sym: 2}) - at0 != 2 * slope:
        return None
    return slope


def _ordered_range(
        begin: symbolic.SymbolicType, end: symbolic.SymbolicType,
        step: Optional[symbolic.SymbolicType]) -> Optional[Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]:
    """Inclusive ``(low, high)`` for an iterator ``begin..end`` stepping by ``step``.

    Direction needs the *provable* sign of ``step``. Unknown sign (``0:K:s``) -> which end is
    smaller is unknown -> ``None`` (no trusted range). Guessing ascending would "prove" a
    negative exponent non-negative and relax it to an out-of-range ``ipow``.
    """
    if step is None:
        return None
    step = symbolic_engine.sympify(step)  # a range step may be a raw Python int, with no is_positive
    if step.is_positive:
        return (begin, end)
    if step.is_negative:
        return (end, begin)
    return None


def _loop_range(loop: LoopRegion) -> Optional[Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]:
    """Loop iterator's inclusive ``(low, high)``, or ``None`` if bounds or stride sign unknown."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    if start is None or end is None:
        return None
    return _ordered_range(start, end, loop_analysis.get_loop_stride(loop))


def _symbol_facts(sdfg: SDFG) -> _Facts:
    """Sign / integrality facts per symbol name: integrality from each SDFG's declared symbol
    dtypes, sign off the symbol objects stored in array descriptors (recursively -- a size symbol
    may appear only in a nested SDFG's shapes)."""
    facts: Dict[str, OrderedSet] = {}
    for g in sdfg.all_sdfgs_recursive():
        for name, dtype in g.symbols.items():
            # ``issubclass``, not ``numpy.issubdtype``: the latter converts through ``numpy.dtype``,
            # which raises on a symbol carrying a non-numeric typeclass (``dace.callback``, whose
            # ``.type`` is an instance rather than a scalar class).
            if isinstance(dtype.type, type) and issubclass(dtype.type, numpy.integer):
                facts.setdefault(name, OrderedSet()).add('integer')
        for desc in g.arrays.values():
            if not isinstance(desc, data.Array):
                continue
            for sym in desc.free_symbols:
                declared = facts.setdefault(sym.name, OrderedSet())
                if sym.is_integer:
                    declared.add('integer')
                if sym.is_positive:
                    declared.add('positive')
                elif sym.is_nonnegative:
                    declared.add('nonnegative')
    return {name: frozenset(declared) for name, declared in facts.items()}


def _merged_facts(own: _Facts, inherited: Optional[_Facts]) -> _Facts:
    """``own`` widened by ``inherited`` -- a name known in both keeps every fact either states."""
    if not inherited:
        return own
    merged = dict(own)
    for name, declared in inherited.items():
        merged[name] = merged.get(name, frozenset()) | declared
    return merged


def exponent_relaxes_to_ipow(exp: symbolic.SymbolicType, sdfg: SDFG, ranges: Optional[_Ranges] = None) -> bool:
    """Whether ``exp`` is a provable non-negative integer under ``sdfg``'s declared symbol
    assumptions -- the SAME proof :class:`RelaxIntegerPowers` uses to lower a ``Pow`` to
    ``ipow``. The tile-op emitter calls this to choose ``ipow`` vs ``pow`` for a ``**`` /
    ``pow`` operand, so the ``**`` lowering and the pow->ipow relaxation agree on which
    powers are integer.

    :param exp: the exponent expression.
    :param sdfg: the SDFG whose symbol sign/integrality assumptions govern the proof.
    :param ranges: optional live iterator ranges to minimise an affine exponent over.
    :returns: ``True`` iff ``exp`` provably ``>= 0`` and integer.
    """
    relaxer = RelaxIntegerPowers()
    relaxer._relaxed = 0
    relaxer._facts = _symbol_facts(sdfg)
    return relaxer._relaxed_exponent(exp, ranges or {}) is not None


@transformation.explicit_cf_compatible
class RelaxIntegerPowers(ppl.Pass):
    """Lower non-negative-integer ``Pow`` to ``ipow`` across the SDFG's size,
    subscript and bound expressions."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        self._relaxed = 0
        self._facts: _Facts = {}
        self._visit_sdfg(sdfg, {})
        return self._relaxed or None

    def _proven_nonnegative(self, exp: symbolic.SymbolicType, ranges: _Ranges) -> bool:
        """Is ``exp`` provably ``>= 0`` -- minimised over the live iterator ranges?"""
        corners = {}
        for sym in exp.free_symbols:
            if sym.name not in ranges:
                continue
            coeff = _affine_coeff(exp, sym)
            if coeff is None or not coeff.is_number:
                return False  # non-affine in an iterator -> no simple corner minimum
            low, high = ranges[sym.name]
            corners[sym] = high if coeff.is_negative else low
        residual = exp.subs(corners) if corners else exp
        return symbolic.ask('nonnegative', residual, self._facts) is True

    def _relaxed_exponent(self, exp: symbolic.SymbolicType, ranges: _Ranges) -> Optional[symbolic.SymbolicType]:
        """The integer exponent to feed ``ipow``, or ``None`` to keep ``pow``."""
        if exp.is_Number:
            if exp.is_integer:
                value = int(exp)
            elif exp.is_real:
                literal = float(exp)
                if literal != int(literal):
                    return None  # genuinely fractional (0.5 -> sqrt)
                value = int(literal)  # integer-valued float literal (2.0 -> 2)
            else:
                return None
            return symbolic_engine.Integer(value) if value >= 0 else None  # negative -> reciprocal
        if symbolic.ask('integer', exp, self._facts) is not True:
            return None
        return exp if self._proven_nonnegative(exp, ranges) else None

    def _relax(self, expr, ranges: _Ranges):
        """Rewrite each provable ``Pow`` in ``expr`` to ``ipow``."""
        core = expr.expr if isinstance(expr, symbolic.SymExpr) else expr
        if not isinstance(core, symbolic.SymbolicBasic):
            return expr

        def to_ipow(base, exp):
            result = self._relaxed_exponent(exp, ranges)
            if result is None:
                return base**exp
            self._relaxed += 1
            # `apply_head`, not the `ipow` class: calling it directly would sympify a native operand
            # and pull the whole expression back onto the sympy island.
            return symbolic.apply_head('ipow', base, result if result.is_number else exp)

        relaxed = _map_pows(core, to_ipow)
        return expr if relaxed is core else relaxed

    def _relax_subset(self, sub, ranges: _Ranges) -> None:
        if isinstance(sub, subsets.Range):
            sub.ranges = [tuple(self._relax(component, ranges) for component in rng) for rng in sub.ranges]
        elif isinstance(sub, subsets.Indices):
            sub.indices = [self._relax(idx, ranges) for idx in sub.indices]

    def _relax_descriptor(self, desc: data.Array, ranges: _Ranges) -> None:
        desc.shape = tuple(self._relax(item, ranges) for item in desc.shape)
        desc.strides = tuple(self._relax(item, ranges) for item in desc.strides)
        desc.offset = tuple(self._relax(item, ranges) for item in desc.offset)
        desc.total_size = self._relax(desc.total_size, ranges)

    def _relax_text(self, text: str, ranges: _Ranges) -> Optional[str]:
        """Relax provable ``Pow`` in a Python-expression string; return the rewritten
        text, or ``None`` if unparseable or unchanged. Powers carry Python ``**``, so a
        string without ``**`` needs no work."""
        if not text or '**' not in text:
            return None
        try:
            expr = symbolic.pystr_to_symbolic(text)
        except Exception:  # noqa: BLE001 -- a non-symbolic statement (e.g. a call) is left as-is
            return None
        if not isinstance(expr, symbolic.SymbolicBasic):
            return None
        relaxed = self._relax(expr, ranges)
        if relaxed is expr:
            return None
        out = str(relaxed)
        return out if out != text else None

    def _relax_code(self, code, ranges: _Ranges) -> None:
        """Relax a :class:`~dace.properties.CodeBlock` (a loop bound / condition or a
        branch predicate) in place. These codegen through the interstate-edge unparser,
        NOT the descriptor path -- an un-relaxed ``R**e`` there becomes a ``dace::math::pow``
        (``double``) loop bound that can round to an extra iteration."""
        if code is None:
            return
        relaxed = self._relax_text(code.as_string, ranges)
        if relaxed is not None:
            code.as_string = relaxed

    def _relax_assignments(self, assignments: Dict[str, str], ranges: _Ranges) -> None:
        """Relax the RHS of each interstate-edge assignment in place."""
        for var, value in list(assignments.items()):
            if isinstance(value, str):
                relaxed = self._relax_text(value, ranges)
                if relaxed is not None:
                    assignments[var] = relaxed

    def _relax_symbol_mapping(self, nsdfg: nodes.NestedSDFG, ranges: _Ranges) -> None:
        """Relax each nested-SDFG symbol-mapping value (an outer-scope expression) in place."""
        for name, value in list(nsdfg.symbol_mapping.items()):
            core = value.expr if isinstance(value, symbolic.SymExpr) else value
            if not isinstance(core, symbolic.SymbolicBasic):
                continue
            relaxed = self._relax(core, ranges)
            if relaxed is not core:
                nsdfg.symbol_mapping[name] = relaxed

    def _nested_facts(self, nsdfg: nodes.NestedSDFG) -> _Facts:
        """What the enclosing scope proves about each symbol mapped into ``nsdfg``.

        A nested SDFG a library expansion mints declares symbol dtypes only -- the sign registry stays
        on the SDFG the frontend built -- so an exponent that is a provable non-negative integer
        outside would decline inside and emit a ``double`` ``pow`` in an integer bound. The inner name
        IS the mapped outer expression, so what holds of that expression holds of the name.
        """
        inner: _Facts = {}
        for name, value in nsdfg.symbol_mapping.items():
            value = symbolic.pystr_to_symbolic(value) if isinstance(value, str) else value
            if not isinstance(value, symbolic.SymbolicBasic):
                continue
            proven = OrderedSet()
            if symbolic.ask('integer', value, self._facts) is True:
                proven.add('integer')
            if symbolic.ask('positive', value, self._facts) is True:
                proven.add('positive')
            elif symbolic.ask('nonnegative', value, self._facts) is True:
                proven.add('nonnegative')
            if proven:
                inner[str(name)] = frozenset(proven)
        return inner

    def _nested_ranges(self, nsdfg: nodes.NestedSDFG, ranges: _Ranges) -> _Ranges:
        """Carry outer ranges through a nested SDFG's symbol mapping."""
        inner: _Ranges = {}
        for name, value in nsdfg.symbol_mapping.items():
            value = symbolic.pystr_to_symbolic(value) if isinstance(value, str) else value
            if isinstance(value, symbolic_engine.Symbol) and value.name in ranges:
                inner[str(name)] = ranges[value.name]
        return inner

    def _visit_sdfg(self, sdfg: SDFG, ranges: _Ranges, inherited: Optional[_Facts] = None) -> None:
        # ``sdfg.free_symbols`` yields names; the sign / integrality assumptions
        # live on the symbol objects in the array descriptors, so collect those.
        saved = self._facts
        self._facts = _merged_facts(_symbol_facts(sdfg), inherited)
        self._visit_region(sdfg, ranges, OrderedSet())
        self._facts = saved

    def _visit_region(self, region, ranges: _Ranges, relaxed_arrays: OrderedSet) -> None:
        # Interstate edges carry symbol assignments + branch conditions that codegen
        # through the interstate-edge unparser (``x = R**k`` -> ``dace::math::pow``).
        for iedge in region.edges():
            if iedge.data is None:
                continue
            self._relax_assignments(iedge.data.assignments, ranges)
            self._relax_code(iedge.data.condition, ranges)
        for block in region.nodes():
            if isinstance(block, LoopRegion):
                inner = dict(ranges)
                var = block.loop_variable
                if var:
                    rng = _loop_range(block)
                    if rng is not None:
                        inner[str(var)] = rng
                    else:
                        inner.pop(str(var), None)  # rebound to an unknown range
                # condition + init see the iterator OUTSIDE its body range (condition fails at
                # ``i = end + step``; init runs pre-bind) -> relax under enclosing ranges, no own
                # iterator. update runs with in-body values -> keep them.
                self._relax_code(block.loop_condition, ranges)
                self._relax_code(block.init_statement, ranges)
                self._relax_code(block.update_statement, inner)
                self._visit_region(block, inner, relaxed_arrays)
            elif isinstance(block, SDFGState):
                self._visit_state(block, ranges, relaxed_arrays)
            elif isinstance(block, ConditionalBlock):
                for condition, branch in block.branches:
                    self._relax_code(condition, ranges)
                    self._visit_region(branch, ranges, relaxed_arrays)
            elif isinstance(block, ControlFlowRegion):
                self._visit_region(block, ranges, relaxed_arrays)

    def _visit_state(self, state: SDFGState, ranges: _Ranges, relaxed_arrays: OrderedSet) -> None:
        sdfg = state.sdfg
        children = state.scope_children()
        scope_ranges = {}  # scope entry (or None) -> live ranges there

        def descend(entry, live: _Ranges) -> None:
            scope_ranges[entry] = live
            for node in children[entry]:
                if isinstance(node, nodes.MapEntry):
                    self._relax_subset(node.map.range, live)
                    inner = dict(live)
                    for conn in node.in_connectors:
                        if not conn.startswith('IN_'):
                            inner.pop(conn, None)
                    for param, rng in zip(node.map.params, node.map.range.ranges):
                        prng = _ordered_range(rng[0], rng[1], rng[2])  # (begin, end, step)
                        if prng is not None:
                            inner[str(param)] = prng
                        else:
                            inner.pop(str(param), None)  # unknown-sign step: direction unknown
                    descend(node, inner)
                elif isinstance(node, nodes.NestedSDFG):
                    self._relax_symbol_mapping(node, live)
                    self._visit_sdfg(node.sdfg, self._nested_ranges(node, live), self._nested_facts(node))
                elif isinstance(node, nodes.AccessNode) and node.data not in relaxed_arrays:
                    relaxed_arrays.add(node.data)
                    desc = sdfg.arrays.get(node.data)
                    if isinstance(desc, data.Array):
                        self._relax_descriptor(desc, live)

        descend(None, ranges)

        scope = state.scope_dict()
        for edge in state.edges():
            if edge.data is None:
                continue
            live = scope_ranges.get(scope.get(edge.dst), ranges)
            for sub in (edge.data.subset, edge.data.other_subset):
                if sub is not None:
                    self._relax_subset(sub, live)
