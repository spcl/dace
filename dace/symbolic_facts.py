# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Facts about symbols (sign predicates and relations) and the prover that answers questions under them. """
import enum
from collections.abc import Iterable
from typing import NamedTuple, cast

import sympy

from dace import symbolic


class Predicate(enum.Enum):
    """ A sign fact about one symbol. """
    POSITIVE = enum.auto()
    NONNEGATIVE = enum.auto()
    NEGATIVE = enum.auto()
    NONPOSITIVE = enum.auto()
    NONZERO = enum.auto()


class RelationKind(enum.Enum):
    LT = enum.auto()
    LE = enum.auto()
    EQ = enum.auto()
    NE = enum.auto()


class Relation(NamedTuple):
    """ ``lhs <kind> rhs`` between two symbolic expressions. """
    kind: RelationKind
    lhs: sympy.Expr
    rhs: sympy.Expr


class Truth(enum.Enum):
    TRUE = enum.auto()
    FALSE = enum.auto()
    UNKNOWN = enum.auto()


class InconsistentAssumptionsError(ValueError):
    """ Raised when registered facts contradict each other. """

    def __init__(self, subject: str, facts: list[str]) -> None:
        super().__init__(f'Inconsistent facts about {subject}: {", ".join(facts)}')


def predicate_relation(predicate: Predicate, expr: sympy.Expr) -> Relation:
    """ The relation saying ``expr`` satisfies ``predicate``. """
    zero = sympy.Integer(0)
    if predicate is Predicate.POSITIVE:
        return Relation(RelationKind.LT, zero, expr)
    if predicate is Predicate.NONNEGATIVE:
        return Relation(RelationKind.LE, zero, expr)
    if predicate is Predicate.NEGATIVE:
        return Relation(RelationKind.LT, expr, zero)
    if predicate is Predicate.NONPOSITIVE:
        return Relation(RelationKind.LE, expr, zero)
    return Relation(RelationKind.NE, expr, zero)


class Facts:
    """
    Relations that may be assumed, and which symbol names are integers (from their dtypes). Passed explicitly to every
    proof; ``Facts.none()`` assumes nothing. The relations are solved once, on construction, into ``substitution``,
    which rewrites a symbol as an expression of nonnegative slack variables; ``nonzero`` keeps the ``!=`` facts.
    """
    __slots__ = ('relations', 'integers', 'substitution', 'nonzero')

    relations: tuple[Relation, ...]
    integers: frozenset[str]
    substitution: dict[sympy.Symbol, sympy.Expr]
    nonzero: tuple[sympy.Expr, ...]

    def __init__(self, relations: Iterable[Relation], integers: frozenset[str]) -> None:
        # A canonical order, so that what is proven does not depend on the order the facts were given in
        self.relations = tuple(
            sorted(dict.fromkeys(relations),
                   key=lambda relation: (relation.kind.name, str(relation.lhs), str(relation.rhs))))
        self.integers = integers
        self.substitution = eliminate(self.relations, integers)
        self.nonzero = tuple(
            reduced(relation, integers, self.substitution) for relation in self.relations
            if relation.kind is RelationKind.NE)

    @classmethod
    def none(cls) -> 'Facts':
        return cls((), frozenset())


def relation_names(relation: Relation) -> set[str]:
    return {str(free) for side in (relation.lhs, relation.rhs) for free in sympy.sympify(side).free_symbols}


def with_integers(expr: sympy.Expr, integers: frozenset[str]) -> sympy.Expr:
    """ ``expr`` over plain SymPy symbols, flagged integer where the dtype says so. """
    return cast(
        sympy.Expr,
        expr.xreplace({
            free: sympy.Symbol(free.name, integer=free.name in integers or None)
            for free in expr.free_symbols if isinstance(free, sympy.Symbol)
        }))


def linear_symbols(difference: sympy.Expr) -> list[sympy.Symbol]:
    """ The non-slack symbols ``difference`` is linear in, with a numeric coefficient, by name. """
    symbols = [
        free for free in difference.free_symbols
        if isinstance(free, sympy.Symbol) and not isinstance(free, sympy.Dummy)
    ]
    return sorted((free for free in symbols if cast(sympy.Expr, difference.coeff(free)).is_number and (
        poly := difference.as_poly(free)) is not None and poly.degree() == 1),
                  key=lambda free: free.name)


def split_extrema(relation: Relation) -> list[Relation]:
    """
    An ordering whose difference ``rhs - lhs`` adds a ``Min`` (or subtracts a ``Max``) as one relation per argument,
    since it holds exactly when it holds for every one; ``a <= Min(b, c) - 1`` is ``a <= b - 1`` and ``a <= c - 1``.
    """
    if relation.kind not in (RelationKind.LT, RelationKind.LE):
        return [relation]
    difference = cast(sympy.Expr, sympy.sympify(relation.rhs - relation.lhs))
    for extremum in sorted(difference.atoms(sympy.Min, sympy.Max), key=str):
        coefficient = cast(sympy.Expr, difference.coeff(extremum))
        rest = sympy.expand(difference - coefficient * extremum)
        if not coefficient.is_number or extremum in rest.atoms(sympy.Min, sympy.Max):
            continue
        if coefficient.is_positive if isinstance(extremum, sympy.Min) else coefficient.is_negative:
            return [
                split for arg in extremum.args
                for split in split_extrema(Relation(relation.kind, sympy.Integer(0), rest + coefficient * arg))
            ]
    return [relation]


def eliminate(relations: Iterable[Relation], integers: frozenset[str]) -> dict[sympy.Symbol, sympy.Expr]:
    """
    Solves each relation, rewritten as ``e >= 0`` (or ``e == 0``), for one of its symbols as ``e = slack`` with a
    nonnegative slack (zero for an equality), and substitutes the result into the rest.
    """
    substitution: dict[sympy.Symbol, sympy.Expr] = {}
    for relation in (split for relation in relations for split in split_extrema(relation)):
        # SymPy rewrites a ``Mod`` it is substituted into by Python's rounding (``Mod(x + 2, 2)`` is ``Mod(x, 2)``),
        # which C's ``%`` does not share; leaving such a fact out only proves less
        if relation.kind is RelationKind.NE or any(
                sympy.sympify(side).atoms(sympy.Mod) for side in (relation.lhs, relation.rhs)):
            continue
        difference = reduced(relation, integers, substitution)
        if relation.kind is RelationKind.LT and difference.is_integer:
            difference -= 1
        if difference.is_negative or (relation.kind is RelationKind.EQ and difference.is_nonzero):
            raise InconsistentAssumptionsError('relations', [f'{relation.lhs} {relation.kind.name} {relation.rhs}'])
        linear = linear_symbols(difference)
        if not linear:
            continue
        target = linear[-1]
        coefficient = cast(sympy.Expr, difference.coeff(target))  # numeric, checked by linear_symbols
        slack = sympy.Integer(0) if relation.kind is RelationKind.EQ else sympy.Dummy(
            'slack', integer=difference.is_integer, nonnegative=True)
        solved = sympy.expand((slack - (difference - coefficient * target)) / coefficient)
        substitution = {name: sympy.expand(value.xreplace({target: solved})) for name, value in substitution.items()}
        substitution[target] = solved
    return substitution


def reduced(relation: Relation, integers: frozenset[str], substitution: dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    """ ``rhs - lhs`` of the relation, rewritten through the substitution. """
    difference = with_integers(sympy.sympify(relation.rhs - relation.lhs), integers)
    return sympy.expand(difference.xreplace(substitution))


class DivisionKind(enum.Enum):
    FLOOR = enum.auto()
    CEIL = enum.auto()
    MOD = enum.auto()


class Division(NamedTuple):
    """
    An ``int_floor``, ``int_ceil`` or ``Mod`` in a query. C++ divides with truncation (``/``, ``%``, and
    ``dace::math::int_ceil`` is ``(x + y - 1) / y``) while Python and SymPy round down; the two agree when the numerator
    is nonnegative and the denominator positive, so only there are the bounds of a division known.
    """
    kind: DivisionKind
    numerator: sympy.Expr
    denominator: sympy.Expr

    def bounds(self, lower: bool) -> tuple[sympy.Expr, ...]:
        """ The lower (or upper) bounds of the quotient (or remainder), for a nonnegative numerator over a positive
        denominator. """
        x, y = self.numerator, self.denominator
        if self.kind is DivisionKind.FLOOR:
            return (sympy.Integer(0), (x - y + 1) / y) if lower else (x / y, x)
        if self.kind is DivisionKind.CEIL:
            return (sympy.Integer(0), x / y) if lower else ((x + y - 1) / y, x)
        return (sympy.Integer(0), ) if lower else (y - 1, x)


def division_kind(atom: sympy.Expr) -> DivisionKind:
    if isinstance(atom, symbolic.int_floor):
        return DivisionKind.FLOOR
    if isinstance(atom, symbolic.int_ceil):
        return DivisionKind.CEIL
    return DivisionKind.MOD


def lowered(expr: sympy.Expr, integers: frozenset[str], substitution: dict[sympy.Symbol, sympy.Expr],
            divisions: dict[sympy.Symbol, Division]) -> sympy.Expr:
    """
    ``expr`` with every division replaced, innermost first, by an integer placeholder recorded in ``divisions`` (the
    same division by the same placeholder), whose operands are rewritten through ``substitution``. This comes before
    anything rebuilds ``expr``, which would have SymPy rewrite a ``Mod`` by Python's rounding.
    """
    division_types = (symbolic.int_floor, symbolic.int_ceil, sympy.Mod)
    while innermost := [
            atom for atom in expr.atoms(*division_types) if not any(arg.atoms(*division_types) for arg in atom.args)
    ]:
        replacements: dict[sympy.Expr, sympy.Symbol] = {}
        for atom in sorted(innermost, key=str):
            # Placeholders of inner divisions are integers too
            known = integers | {placeholder.name for placeholder in divisions}
            numerator, denominator = (lowered(
                sympy.expand(with_integers(cast(sympy.Expr, arg), known).xreplace(substitution)), integers, {},
                divisions) for arg in atom.args)
            division = Division(division_kind(atom), numerator, denominator)
            placeholder = next((known for known, existing in divisions.items() if existing == division), None)
            if placeholder is None:
                # The name cannot be a program symbol, so the placeholder never meets the substitution
                placeholder = sympy.Symbol(f'{division.kind.name.lower()}:{len(divisions)}', integer=True)
                divisions[placeholder] = division
            replacements[atom] = placeholder
        expr = expr.xreplace(replacements)
    return expr


def proves_sign(expr: sympy.Expr, divisions: dict[sympy.Symbol, Division], strict: bool) -> bool:
    """
    Whether ``expr`` is provably nonnegative (positive if ``strict``). A placeholder it is linear in is replaced by
    each bound of its division in turn, the lower ones if its coefficient is nonnegative and the upper ones if it is
    nonpositive, outermost first; for an integer ``expr``, a bound above -1 suffices.
    """
    expr = sympy.expand(expr)
    if strict and expr.is_integer:
        expr, strict = expr - 1, False
    placeholder = next((known for known in reversed(divisions) if known in expr.free_symbols), None)
    if placeholder is None:
        return (expr.is_positive if strict else expr.is_nonnegative) is True
    division = divisions[placeholder]
    poly = expr.as_poly(placeholder)
    if poly is None or poly.degree() != 1 or not (proves_sign(division.numerator, divisions, False)
                                                  and proves_sign(division.denominator, divisions, True)):
        return False
    coefficient = cast(sympy.Expr, poly.coeff_monomial(placeholder))
    if proves_sign(coefficient, divisions, False):
        bounds = division.bounds(lower=True)
    elif proves_sign(-coefficient, divisions, False):
        bounds = division.bounds(lower=False)
    else:
        return False
    integer = not strict and expr.is_integer
    return any(
        proves_sign(expr.xreplace({placeholder: bound}) + 1, divisions, True
                    ) if integer else proves_sign(expr.xreplace({placeholder: bound}), divisions, strict)
        for bound in bounds)


def by_cases(expr: sympy.Expr, kind: RelationKind, extremum: sympy.Expr, facts: Facts) -> Truth:
    """ ``0 <kind> expr`` asked once for each argument the ``Min`` or ``Max`` can equal, assuming it does. """
    truths: set[Truth] = set()
    for arg in cast(tuple[sympy.Expr, ...], extremum.args):
        others = [other for other in cast(tuple[sympy.Expr, ...], extremum.args) if other != arg]
        case = [
            Relation(RelationKind.LE, arg, other) if isinstance(extremum, sympy.Min) else Relation(
                RelationKind.LE, other, arg) for other in others
        ]
        try:
            case_facts = Facts(facts.relations + tuple(case), facts.integers)
        except InconsistentAssumptionsError:
            continue  # The argument is never the extremum
        truths.add(ask(Relation(kind, sympy.Integer(0), cast(sympy.Expr, expr.xreplace({extremum: arg}))), case_facts))
    return truths.pop() if len(truths) == 1 else Truth.UNKNOWN


def ask(query: Relation, facts: Facts) -> Truth:
    expr = cast(sympy.Expr, sympy.sympify(query.rhs - query.lhs))
    extrema = sorted(expr.atoms(sympy.Min, sympy.Max), key=str)
    if extrema:
        return by_cases(expr, query.kind, extrema[0], facts)
    divisions: dict[sympy.Symbol, Division] = {}
    difference = lowered(expr, facts.integers, facts.substitution, divisions)
    difference = with_integers(difference, facts.integers | {placeholder.name for placeholder in divisions})
    # The substitution may bring in divisions of its own
    difference = lowered(sympy.expand(difference.xreplace(facts.substitution)), facts.integers, {}, divisions)
    if query.kind is RelationKind.EQ:
        answer = difference.is_zero
    elif query.kind is RelationKind.NE:
        answer = difference.is_nonzero or (difference in facts.nonzero or -difference in facts.nonzero or None)
    else:
        strict = query.kind is RelationKind.LT
        if proves_sign(difference, divisions, strict):
            return Truth.TRUE
        return Truth.FALSE if proves_sign(-difference, divisions, not strict) else Truth.UNKNOWN
    if answer is None:
        return Truth.UNKNOWN
    return Truth.TRUE if answer else Truth.FALSE


def provably_nonnegative(expr: sympy.Expr, facts: Facts) -> bool:
    return ask(Relation(RelationKind.LE, sympy.Integer(0), expr), facts) is Truth.TRUE


def provably_le(lhs: sympy.Expr, rhs: sympy.Expr, facts: Facts) -> bool:
    return ask(Relation(RelationKind.LE, lhs, rhs), facts) is Truth.TRUE
