# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Facts about symbols (sign predicates and relations) and the prover that answers questions under them. """
import enum
import dataclasses
from typing import NamedTuple, cast
from collections.abc import Callable, Mapping

import sympy
import sympy.core.facts


class Predicate(enum.Enum):
    """ A sign fact about one symbol; the value is the SymPy assumption it sets. """
    POSITIVE = 'positive'
    NONNEGATIVE = 'nonnegative'
    NEGATIVE = 'negative'
    NONPOSITIVE = 'nonpositive'
    NONZERO = 'nonzero'


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


PREDICATE_TESTS: dict[Predicate, Callable[[sympy.Expr], bool | None]] = {
    Predicate.POSITIVE: lambda expr: expr.is_positive,
    Predicate.NONNEGATIVE: lambda expr: expr.is_nonnegative,
    Predicate.NEGATIVE: lambda expr: expr.is_negative,
    Predicate.NONPOSITIVE: lambda expr: expr.is_nonpositive,
    Predicate.NONZERO: lambda expr: expr.is_nonzero,
}


def truth_of(answer: bool | None) -> Truth:
    if answer is None:
        return Truth.UNKNOWN
    return Truth.TRUE if answer else Truth.FALSE


class Elimination(NamedTuple):
    """ The symbols facts are proven against, and the substitution that folds every relation into a slack. """
    twins: dict[str, sympy.Symbol]
    substitution: dict[sympy.Symbol, sympy.Expr]


@dataclasses.dataclass(frozen=True, slots=True)
class Facts:
    """
    What may be assumed about symbols, by name: sign predicates, relations, and which names are integers (from their
    dtypes). Passed explicitly to every proof; ``Facts.none()`` assumes nothing.
    """
    predicates: Mapping[str, frozenset[Predicate]]
    relations: tuple[Relation, ...]
    integers: frozenset[str]
    elimination: Elimination = dataclasses.field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'elimination', eliminate(self))

    @staticmethod
    def none() -> 'Facts':
        return Facts({}, (), frozenset())

    def merged(self, other: 'Facts') -> 'Facts':
        names = dict.fromkeys([*self.predicates, *other.predicates])
        predicates = {
            name: self.predicates.get(name, frozenset()) | other.predicates.get(name, frozenset())
            for name in names
        }
        return Facts(predicates, tuple(dict.fromkeys(self.relations + other.relations)), self.integers | other.integers)


def twin_of(name: str, facts: Facts) -> sympy.Symbol:
    flags = {predicate.value: True for predicate in facts.predicates.get(name, frozenset())}
    if name in facts.integers:
        flags['integer'] = True
    try:
        return sympy.Symbol(name, **flags)
    except sympy.core.facts.InconsistentAssumptions as error:
        raise InconsistentAssumptionsError(name, sorted(flags)) from error


def as_twins(expr: sympy.Expr, twins: dict[str, sympy.Symbol], facts: Facts) -> sympy.Expr:
    replacements: dict[sympy.Basic, sympy.Basic] = {}
    for free in expr.free_symbols:
        if not isinstance(free, sympy.Symbol):
            continue
        if free.name not in twins:
            twins[free.name] = twin_of(free.name, facts)
        replacements[free] = twins[free.name]
    return cast(sympy.Expr, expr.xreplace(replacements))


class Form(NamedTuple):
    """ ``expr >= 0``, or ``expr == 0`` when ``equality``. """
    expr: sympy.Expr
    equality: bool


def relation_forms(difference: sympy.Expr, kind: RelationKind) -> tuple[Form, ...]:
    if kind is RelationKind.LT:
        return (Form(difference - 1, False), ) if difference.is_integer else (Form(difference, False), )
    if kind is RelationKind.LE:
        return (Form(difference, False), )
    if kind is RelationKind.EQ:
        return (Form(difference, True), )
    return ()


def predicate_forms(twin: sympy.Symbol) -> tuple[Form, ...]:
    if twin.is_positive and twin.is_integer:
        return (Form(twin - 1, False), )
    if twin.is_nonnegative:
        return (Form(twin, False), )
    if twin.is_negative and twin.is_integer:
        return (Form(-twin - 1, False), )
    if twin.is_nonpositive:
        return (Form(-twin, False), )
    return ()


def linear_in(form: sympy.Expr, twin: sympy.Symbol) -> bool:
    coefficient = form.coeff(twin)
    return coefficient is not None and bool(coefficient.is_number) and sympy.Poly(form, twin).degree() == 1


def elimination_target(form: sympy.Expr, twins: dict[str, sympy.Symbol]) -> sympy.Symbol | None:
    """ The symbol to solve ``form`` for: linear, a twin, preferring one without sign predicates. """
    linear = [twin for twin in twins.values() if twin in form.free_symbols and linear_in(form, twin)]
    unflagged = [twin for twin in linear if not predicate_forms(twin)]
    candidates = sorted(unflagged or linear, key=lambda twin: twin.name)
    return candidates[-1] if candidates else None


def contradicts(form: Form) -> bool:
    if form.equality:
        return bool(form.expr.is_nonzero)
    return bool(form.expr.is_negative)


def eliminate(facts: Facts) -> Elimination:
    twins = {name: twin_of(name, facts) for name in facts.predicates}
    substitution: dict[sympy.Symbol, sympy.Expr] = {}
    pending = [
        form for relation in facts.relations
        for form in relation_forms(sympy.expand(as_twins(sympy.sympify(relation.rhs -
                                                                       relation.lhs), twins, facts)), relation.kind)
    ]
    while pending:
        form = pending.pop(0)
        form = Form(sympy.expand(form.expr.xreplace(substitution)), form.equality)
        if contradicts(form):
            raise InconsistentAssumptionsError('relations', [f'{form.expr} {"==" if form.equality else ">="} 0'])
        target = elimination_target(form.expr, twins)
        if target is None:
            continue
        coefficient = form.expr.coeff(target)
        rest = form.expr - coefficient * target
        slack = sympy.Integer(0) if form.equality else sympy.Dummy(
            'slack', integer=form.expr.is_integer, nonnegative=True)
        solved = sympy.expand((slack - rest) / coefficient)
        substitution = {name: sympy.expand(value.xreplace({target: solved})) for name, value in substitution.items()}
        substitution[target] = solved
        pending.extend(predicate_forms(target))
    return Elimination(twins, substitution)


def reduced(expr: sympy.Expr, facts: Facts) -> sympy.Expr:
    twins = dict(facts.elimination.twins)
    return sympy.expand(as_twins(sympy.sympify(expr), twins, facts).xreplace(facts.elimination.substitution))


def ask_predicate(predicate: Predicate, expr: sympy.Expr, facts: Facts) -> Truth:
    return truth_of(PREDICATE_TESTS[predicate](reduced(expr, facts)))


def ask(query: Relation, facts: Facts) -> Truth:
    difference = sympy.sympify(query.rhs) - sympy.sympify(query.lhs)
    if query.kind is RelationKind.LT:
        return ask_predicate(Predicate.POSITIVE, difference, facts)
    if query.kind is RelationKind.LE:
        return ask_predicate(Predicate.NONNEGATIVE, difference, facts)
    if query.kind is RelationKind.EQ:
        return truth_of(reduced(difference, facts).is_zero)
    return ask_predicate(Predicate.NONZERO, difference, facts)


def provably_nonnegative(expr: sympy.Expr, facts: Facts) -> bool:
    return ask_predicate(Predicate.NONNEGATIVE, expr, facts) is Truth.TRUE


def provably_le(lhs: sympy.Expr, rhs: sympy.Expr, facts: Facts) -> bool:
    return ask(Relation(RelationKind.LE, lhs, rhs), facts) is Truth.TRUE
