# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Facts about symbols (sign predicates and relations) and the prover that answers questions under them. """
import enum
from collections.abc import Iterable
from typing import NamedTuple, cast

import sympy


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

    @staticmethod
    def none() -> 'Facts':
        return Facts((), frozenset())


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
    return sorted(
        (free for free in symbols
         if cast(sympy.Expr, difference.coeff(free)).is_number and sympy.Poly(difference, free).degree() == 1),
        key=lambda free: free.name)


def eliminate(relations: Iterable[Relation], integers: frozenset[str]) -> dict[sympy.Symbol, sympy.Expr]:
    """
    Solves each relation, rewritten as ``e >= 0`` (or ``e == 0``), for one of its symbols as ``e = slack`` with a
    nonnegative slack (zero for an equality), and substitutes the result into the rest.
    """
    substitution: dict[sympy.Symbol, sympy.Expr] = {}
    for relation in relations:
        if relation.kind is RelationKind.NE:
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


def ask(query: Relation, facts: Facts) -> Truth:
    difference = reduced(query, facts.integers, facts.substitution)
    if query.kind is RelationKind.EQ:
        answer = difference.is_zero
    elif query.kind is RelationKind.LE:
        answer = difference.is_nonnegative
    elif query.kind is RelationKind.LT:
        answer = difference.is_positive
    else:
        answer = difference.is_nonzero or (difference in facts.nonzero or -difference in facts.nonzero or None)
    if answer is None:
        return Truth.UNKNOWN
    return Truth.TRUE if answer else Truth.FALSE


def provably_nonnegative(expr: sympy.Expr, facts: Facts) -> bool:
    return ask(Relation(RelationKind.LE, sympy.Integer(0), expr), facts) is Truth.TRUE


def provably_le(lhs: sympy.Expr, rhs: sympy.Expr, facts: Facts) -> bool:
    return ask(Relation(RelationKind.LE, lhs, rhs), facts) is Truth.TRUE
