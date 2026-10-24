# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Facts about symbols (sign predicates and relations), facts about the contents of data containers, and the prover
that answers questions under the symbol facts."""

import copy
import enum
import functools
from collections.abc import Callable, Iterable
from typing import Generic, NamedTuple, TypeVar, cast

import sympy
from ordered_set import OrderedSet

from dace import symbolic


class Predicate(enum.Enum):
    """A sign fact about one symbol."""

    POSITIVE = enum.auto()
    NONNEGATIVE = enum.auto()
    NEGATIVE = enum.auto()
    NONPOSITIVE = enum.auto()
    NONZERO = enum.auto()


class Scale(enum.Enum):
    """How large a symbol's values are. Every ``SMALL`` symbol is much smaller than every ``DEFAULT`` one; a proof that
    relies on this records the relation, and the compiled program checks it on every call."""

    SMALL = enum.auto()
    DEFAULT = enum.auto()


class RelationKind(enum.Enum):
    LT = enum.auto()
    LE = enum.auto()
    EQ = enum.auto()
    NE = enum.auto()
    #: ``lhs`` divides ``rhs``: ``rhs`` is ``lhs`` times an integer
    DIVIDES = enum.auto()
    #: ``lhs << rhs``, which implies ``lhs < rhs``; ``lhs >> rhs`` is ``rhs << lhs``
    MUCH_LT = enum.auto()


class Relation(NamedTuple):
    """``lhs <kind> rhs`` between two symbolic expressions."""

    kind: RelationKind
    lhs: sympy.Expr
    rhs: sympy.Expr

    def replaced(self, mapping) -> "Relation":
        """This relation with the symbols of both sides replaced, as ``symbolic.replace_symbols`` does."""
        return Relation(
            self.kind, symbolic.replace_symbols(self.lhs, mapping), symbolic.replace_symbols(self.rhs, mapping)
        )


class DataFactKind(enum.Enum):
    """A property of the contents of a one-dimensional container ``A``."""

    #: The entries other than ``sentinel`` are pairwise distinct and lie in ``[lo, hi]``; ``permutation(0..N-1)`` is
    #: ``INJECTIVE`` with ``lo = 0``, ``hi = N - 1`` and no sentinel over an array of extent ``N``.
    INJECTIVE = enum.auto()
    #: ``A[x] <= A[y]`` for ``lo <= x <= y <= hi`` (CSR row pointers, prefix sums).
    NONDECREASING = enum.auto()
    #: Every entry is ``>= 0``.
    NONNEGATIVE = enum.auto()
    #: The entries are pairwise distinct within each segment ``[segments[r], segments[r + 1])`` (unique column
    #: indices in every row of a CSR matrix whose row pointers are ``segments``).
    SEGMENT_INJECTIVE = enum.auto()


class DataFact(NamedTuple):
    """A fact about the contents of one container; ``lo``, ``hi`` are symbolic, ``segments`` names a container."""

    kind: DataFactKind
    lo: sympy.Expr | None = None
    hi: sympy.Expr | None = None
    sentinel: int | None = None
    segments: str | None = None

    def replaced(self, mapping, renames: dict[str, str] | None = None) -> "DataFact":
        """This fact with its bounds' symbols replaced, as ``symbolic.replace_symbols`` does, and ``segments``
        renamed by ``renames``."""
        bound = [None if b is None else symbolic.replace_symbols(b, mapping) for b in (self.lo, self.hi)]
        segments = (
            self.segments if renames is None or self.segments is None else renames.get(self.segments, self.segments)
        )
        return DataFact(self.kind, bound[0], bound[1], self.sentinel, segments)

    def names(self) -> set[str]:
        """The symbols the bounds name."""
        return {str(s) for b in (self.lo, self.hi) if b is not None for s in b.free_symbols}

    def text(self, array: str) -> str:
        if self.kind is DataFactKind.INJECTIVE:
            values = f" with values in [{self.lo}, {self.hi}]" if self.lo is not None else ""
            off = f" apart from {self.sentinel}" if self.sentinel is not None else ""
            return f"{array} injective{off}{values}"
        if self.kind is DataFactKind.NONDECREASING:
            return f"{array} nondecreasing on [{self.lo}, {self.hi}]"
        if self.kind is DataFactKind.SEGMENT_INJECTIVE:
            return f"{array} injective within each segment of {self.segments}"
        return f"{array} nonnegative"


def permutation(lo: sympy.Expr, hi: sympy.Expr) -> DataFact:
    """``permutation(lo..hi)``: every value in ``[lo, hi]`` once (over an array of extent ``hi - lo + 1``)."""
    return DataFact(DataFactKind.INJECTIVE, sympy.sympify(lo), sympy.sympify(hi))


class Truth(enum.Enum):
    TRUE = enum.auto()
    FALSE = enum.auto()
    UNKNOWN = enum.auto()


class InconsistentAssumptionsError(ValueError):
    """Raised when registered facts contradict each other."""

    def __init__(self, subject: str, facts: list[str]) -> None:
        super().__init__(f"Inconsistent facts about {subject}: {', '.join(facts)}")


def predicate_relation(predicate: Predicate, expr: sympy.Expr) -> Relation:
    """The relation saying ``expr`` satisfies ``predicate``."""
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


def comparison_relation(condition: sympy.Basic) -> Relation | None:
    """The relation an order comparison (``<``, ``<=``, ``>``, ``>=``) states, or None for any other expression."""
    if isinstance(condition, (sympy.StrictLessThan, sympy.LessThan)):
        kind = RelationKind.LT if isinstance(condition, sympy.StrictLessThan) else RelationKind.LE
        return Relation(kind, condition.lhs, condition.rhs)
    if isinstance(condition, (sympy.StrictGreaterThan, sympy.GreaterThan)):
        kind = RelationKind.LT if isinstance(condition, sympy.StrictGreaterThan) else RelationKind.LE
        return Relation(kind, condition.rhs, condition.lhs)
    return None


@functools.lru_cache(maxsize=1 << 16, typed=True)
def _relation_order_text(kind: RelationKind, lhs: sympy.Expr, rhs: sympy.Expr) -> tuple[str, str, str]:
    """``relation_order`` of the relation ``lhs <kind> rhs``: its text, a pure function of the (immutable) sides."""
    return kind.name, str(lhs), str(rhs)


def _relation_order(relation: Relation) -> tuple[str, str, str]:
    """The canonical sort key of a relation. Printing the sides is the cost of building ``Facts``, and the same
    relations are given again and again (every derivation of the facts of a scope), so the text is cached."""
    return _relation_order_text(relation.kind, relation.lhs, relation.rhs)


class Facts:
    """
    Relations that may be assumed, and which symbol names are integers (from their dtypes). Passed explicitly to every
    proof; ``Facts.none()`` assumes nothing. The relations are solved once, on construction, into ``substitution``,
    which rewrites a symbol as an expression of nonnegative slack variables; ``slacks`` gives each slack as the difference
    of its relation (and each divisibility quotient as the quotient of its sides), ``residuals`` the bounds (expressions ``>= 0``) of slacks solved away, and ``nonzero`` keeps the
    ``!=`` facts.

    ``small`` names the ``SMALL`` symbols and ``parameters`` the symbols a check at program entry can read; an order
    proven only from their scales is added to ``assumed``, the runtime assumptions of the SDFG the facts belong to.
    Without ``assumed``, scales prove nothing.
    """

    __slots__ = (
        "relations",
        "integers",
        "small",
        "parameters",
        "assumed",
        "solution",
    )

    relations: tuple[Relation, ...]
    integers: frozenset[str]
    small: frozenset[str]
    parameters: frozenset[str]
    assumed: OrderedSet[Relation] | None

    def __init__(
        self,
        relations: Iterable[Relation],
        integers: frozenset[str],
        small: frozenset[str] = frozenset(),
        parameters: frozenset[str] = frozenset(),
        assumed: OrderedSet[Relation] | None = None,
        check: bool = True,
    ) -> None:
        """:param check: Solve the relations now, raising ``InconsistentAssumptionsError`` if they contradict; without,
        they are solved by the first proof that needs them."""
        # A canonical order, so that what is proven does not depend on the order the facts were given in
        self.relations = tuple(sorted(dict.fromkeys(relations), key=_relation_order))
        self.integers = integers
        self.small = small
        self.parameters = parameters
        self.assumed = assumed
        self.solution = solve(self.relations, integers) if check else None

    def solved(
        self,
    ) -> tuple[
        dict[sympy.Symbol, sympy.Expr], dict[sympy.Dummy, sympy.Expr], tuple[sympy.Expr, ...], tuple[sympy.Expr, ...]
    ]:
        if self.solution is None:
            self.solution = solve(self.relations, self.integers)
        return self.solution

    @property
    def substitution(self) -> dict[sympy.Symbol, sympy.Expr]:
        return self.solved()[0]

    @property
    def slacks(self) -> dict[sympy.Dummy, sympy.Expr]:
        return self.solved()[1]

    @property
    def residuals(self) -> tuple[sympy.Expr, ...]:
        return self.solved()[2]

    @property
    def nonzero(self) -> tuple[sympy.Expr, ...]:
        return self.solved()[3]

    @classmethod
    def none(cls) -> "Facts":
        return cls((), frozenset())

    def with_relations(self, relations: Iterable[Relation]) -> "Facts":
        """These facts with more relations, the same scales, recording into the same assumptions."""
        return Facts((*self.relations, *relations), self.integers, self.small, self.parameters, self.assumed)

    def about(self, names: set[str]) -> "Facts":
        """These facts without the relations sharing no symbol with ``names``, directly or through other relations:
        what they prove about an expression over ``names`` is the same."""
        names, kept, rest = set(names), [], list(self.relations)
        grown = True
        while grown:
            grown = False
            for relation in list(rest):
                if relation_names(relation) & names:
                    names |= relation_names(relation)
                    kept.append(relation)
                    rest.remove(relation)
                    grown = True
        if not rest and names >= self.integers | self.small | self.parameters:
            return self
        # A part of facts is as consistent as the whole
        return Facts(
            kept, self.integers & names, self.small & names, self.parameters & names, self.assumed, check=False
        )

    def recording(self, assumed: OrderedSet[Relation] | None) -> "Facts":
        """These facts, recording into ``assumed`` instead."""
        recording = copy.copy(self)
        recording.assumed = assumed
        return recording

    # Determined by what it proves, so equal facts share cached proofs; where proofs are recorded is no part of it,
    # but whether they may be (scales prove nothing without) is
    def key(self) -> tuple[object, ...]:
        return (self.relations, self.integers, self.small, self.parameters, self.assumed is not None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Facts) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def assumptions(self, name: str) -> dict[str, bool]:
        """The SymPy assumptions the facts prove for the symbol ``name``: integrality from its dtype, and a sign, or at
        least that it is nonzero."""
        assumed = {"integer": True} if name in self.integers else {"real": True}
        free, zero = sympy.Symbol(name), sympy.Integer(0)
        for kind, lhs, rhs, sign in (
            (RelationKind.LT, zero, free, "positive"),
            (RelationKind.LE, zero, free, "nonnegative"),
            (RelationKind.LT, free, zero, "negative"),
            (RelationKind.LE, free, zero, "nonpositive"),
        ):
            if ask(Relation(kind, lhs, rhs), self) is Truth.TRUE:
                assumed[sign] = True
                break
        else:
            if ask(Relation(RelationKind.NE, free, zero), self) is Truth.TRUE:
                assumed["nonzero"] = True
        return assumed


Result = TypeVar("Result")


class cached_proof(Generic[Result]):
    """``lru_cache`` for ``function(expr, facts)`` that keeps what the proofs assumed with the result, and records it
    into ``facts.assumed`` on every call, a cache hit included."""

    def __init__(self, function: Callable[[sympy.Expr, Facts], Result]) -> None:
        functools.update_wrapper(self, function)

        @functools.lru_cache(maxsize=2048, typed=True)
        def cached(expr: sympy.Expr, facts: Facts) -> tuple[Result, tuple[Relation, ...]]:
            assumed: OrderedSet[Relation] | None = None if facts.assumed is None else OrderedSet()
            return function(expr, facts.recording(assumed)), tuple(assumed or ())

        self.cached = cached
        self.cache_info = cached.cache_info
        self.cache_clear = cached.cache_clear

    def __call__(self, expr: sympy.Expr, facts: Facts) -> Result:
        # Only the relations connected to the symbols of ``expr`` decide it, so facts that differ elsewhere (the
        # iterators of other loops) share the cached result
        names = {str(free) for free in sympy.sympify(expr).free_symbols}
        result, assumed = self.cached(expr, facts.about(names))
        if facts.assumed is not None:
            facts.assumed.update(assumed)
        return result


@functools.lru_cache(maxsize=4096)
def relation_names(relation: Relation) -> frozenset[str]:
    return frozenset(str(free) for side in (relation.lhs, relation.rhs) for free in sympy.sympify(side).free_symbols)


def with_integers(expr: sympy.Expr, integers: frozenset[str]) -> sympy.Expr:
    """``expr`` over plain SymPy symbols, flagged integer where the dtype says so."""
    return cast(
        sympy.Expr,
        expr.xreplace(
            {
                free: sympy.Symbol(free.name, integer=free.name in integers or None)
                for free in expr.free_symbols
                if isinstance(free, sympy.Symbol) and not isinstance(free, sympy.Dummy)
            }
        ),
    )


def linear_slacks(difference: sympy.Expr) -> list[sympy.Dummy]:
    """The slacks ``difference`` is linear in, with a numeric coefficient, in creation order."""
    return sorted(
        (
            free
            for free in difference.free_symbols
            if isinstance(free, sympy.Dummy)
            and free.is_nonnegative  # A slack, not the quotient of a divisibility fact
            and cast(sympy.Expr, difference.coeff(free)).is_number
            and (poly := difference.as_poly(free)) is not None
            and poly.degree() == 1
        ),
        key=lambda free: free.dummy_index,
    )


def linear_symbols(difference: sympy.Expr) -> list[sympy.Symbol]:
    """The non-slack symbols ``difference`` is linear in, with a numeric coefficient, by name."""
    symbols = [
        free for free in difference.free_symbols if isinstance(free, sympy.Symbol) and not isinstance(free, sympy.Dummy)
    ]
    return sorted(
        (
            free
            for free in symbols
            if cast(sympy.Expr, difference.coeff(free)).is_number
            and (poly := difference.as_poly(free)) is not None
            and poly.degree() == 1
        ),
        key=lambda free: free.name,
    )


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
                split
                for arg in extremum.args
                for split in split_extrema(Relation(relation.kind, sympy.Integer(0), rest + coefficient * arg))
            ]
    return [relation]


@functools.lru_cache(maxsize=1024)
def solve(
    relations: tuple[Relation, ...], integers: frozenset[str]
) -> tuple[
    dict[sympy.Symbol, sympy.Expr], dict[sympy.Dummy, sympy.Expr], tuple[sympy.Expr, ...], tuple[sympy.Expr, ...]
]:
    """What ``Facts`` solves its relations into, shared by equal facts and never modified: the substitution, slacks and
    residuals ``eliminate`` gives, and the ``!=`` facts reduced through the substitution."""
    substitution, slacks, residuals = eliminate(relations, integers)
    nonzero = tuple(
        reduced(relation, integers, substitution) for relation in relations if relation.kind is RelationKind.NE
    )
    return substitution, slacks, residuals, nonzero


def eliminate(
    relations: Iterable[Relation], integers: frozenset[str]
) -> tuple[dict[sympy.Symbol, sympy.Expr], dict[sympy.Dummy, sympy.Expr], tuple[sympy.Expr, ...]]:
    """
    Solves each relation, rewritten as ``e >= 0`` (or ``e == 0``), for one of its symbols as ``e = slack`` with a
    nonnegative slack (zero for an equality), and substitutes the result into the rest. A relation over slacks only
    is solved for a slack, whose own bound is kept as a residual. Also returns each slack as ``e`` over the original
    symbols, and the residuals (expressions ``>= 0``).
    """
    substitution: dict[sympy.Symbol, sympy.Expr] = {}
    slacks: dict[sympy.Dummy, sympy.Expr] = {}
    residuals: list[sympy.Expr] = []
    splits = [split for relation in relations for split in split_extrema(relation)]
    # A divisibility fact last: solved first, it rewrites its multiple in a quotient no later order on it can bound
    for relation in sorted(splits, key=lambda relation: relation.kind is RelationKind.DIVIDES):
        # SymPy rewrites a ``Mod`` it is substituted into by Python's rounding (``Mod(x + 2, 2)`` is ``Mod(x, 2)``),
        # which C's ``%`` does not share; leaving such a fact out only proves less
        if relation.kind is RelationKind.NE or any(
            sympy.sympify(side).atoms(sympy.Mod) for side in (relation.lhs, relation.rhs)
        ):
            continue
        kind = RelationKind.LT if relation.kind is RelationKind.MUCH_LT else relation.kind
        if kind is RelationKind.DIVIDES:
            # The multiple equals the divisor times an integer quotient
            divisor, multiple = (rewritten(side, integers, substitution) for side in (relation.lhs, relation.rhs))
            if divisor.is_number and multiple.is_number:
                if not (multiple.is_zero if divisor.is_zero else (multiple / divisor).is_integer):
                    raise InconsistentAssumptionsError("relations", [f"{relation.lhs} DIVIDES {relation.rhs}"])
                continue
            quotient = sympy.Dummy("quotient", integer=True)
            difference = sympy.expand(multiple - divisor * quotient)
            kind = RelationKind.EQ
            # Written back like a slack, as the quotient of the original sides
            slacks[quotient] = with_integers(cast(sympy.Expr, sympy.sympify(relation.rhs / relation.lhs)), integers)
        else:
            difference = reduced(relation, integers, substitution)
        original = with_integers(cast(sympy.Expr, sympy.sympify(relation.rhs - relation.lhs)), integers)
        if kind is RelationKind.LT and difference.is_integer:
            difference -= 1
            original -= 1
        if difference.is_negative or (kind is RelationKind.EQ and difference.is_nonzero):
            raise InconsistentAssumptionsError("relations", [f"{relation.lhs} {relation.kind.name} {relation.rhs}"])
        # A relation over slacks only constrains earlier relations; solving it for a slack keeps it, and forgetting
        # that slack's own bound only proves less
        linear = linear_symbols(difference) or linear_slacks(difference)
        if not linear:
            continue
        target = linear[-1]
        coefficient = cast(sympy.Expr, difference.coeff(target))  # numeric, checked by linear_symbols
        slack = (
            sympy.Integer(0)
            if kind is RelationKind.EQ
            else sympy.Dummy("slack", integer=difference.is_integer, nonnegative=True)
        )
        if isinstance(slack, sympy.Dummy):
            slacks[slack] = original
        solved = sympy.expand((slack - (difference - coefficient * target)) / coefficient)
        substitution = {name: sympy.expand(value.xreplace({target: solved})) for name, value in substitution.items()}
        residuals = [sympy.expand(residual.xreplace({target: solved})) for residual in residuals]
        if isinstance(target, sympy.Dummy):
            residuals.append(solved)
        substitution[target] = solved
    return substitution, slacks, tuple(residuals)


def rewritten(expr: sympy.Expr, integers: frozenset[str], substitution: dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    """``expr`` rewritten through the substitution."""
    return sympy.expand(with_integers(sympy.sympify(expr), integers).xreplace(substitution))


def reduced(relation: Relation, integers: frozenset[str], substitution: dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    """``rhs - lhs`` of the relation, rewritten through the substitution."""
    return rewritten(relation.rhs - relation.lhs, integers, substitution)


class DivisionKind(enum.Enum):
    FLOOR = enum.auto()
    CEIL = enum.auto()
    MOD = enum.auto()
    #: SymPy's ``floor`` and ``ceiling`` of a real: exact, whatever the sign of their argument
    REAL_FLOOR = enum.auto()
    REAL_CEIL = enum.auto()
    #: ``int_sqrt``: the integer square root, -1 below zero
    SQRT = enum.auto()

    @property
    def exact(self) -> bool:
        """Whether the bounds hold for every argument, not only for a nonnegative numerator over a positive
        denominator."""
        return self in (DivisionKind.REAL_FLOOR, DivisionKind.REAL_CEIL, DivisionKind.SQRT)


class Division(NamedTuple):
    """
    An ``int_floor``, ``int_ceil`` or ``Mod`` in a query. ``int_floor`` floors like Python (``py_floor`` unless its
    operands are provably nonnegative), so its bounds hold for every numerator over a positive denominator. ``%`` and
    ``int_ceil`` divide like C++, with truncation (``dace::math::int_ceil`` is ``(x + y - 1) / y``), which agrees with
    rounding down only for a nonnegative numerator over a positive denominator, so only there are their bounds known.
    """

    kind: DivisionKind
    numerator: sympy.Expr
    denominator: sympy.Expr

    def bounds(self, lower: bool) -> tuple[sympy.Expr, ...]:
        """The lower (or upper) bounds of the quotient (or remainder), for a nonnegative numerator over a positive
        denominator."""
        x, y = self.numerator, self.denominator
        if self.kind is DivisionKind.FLOOR:
            return ((x - y + 1) / y,) if lower else (x / y,)
        if self.kind is DivisionKind.CEIL:
            return (sympy.Integer(0), x / y) if lower else ((x + y - 1) / y, x)
        if self.kind is DivisionKind.REAL_FLOOR:
            return (x - 1,) if lower else (x,)
        if self.kind is DivisionKind.REAL_CEIL:
            return (x,) if lower else (x + 1,)
        if self.kind is DivisionKind.SQRT:
            return (sympy.Integer(-1),) if lower else ()
        return (sympy.Integer(0),) if lower else (y - 1, x)


def division_kind(atom: sympy.Expr) -> DivisionKind:
    if isinstance(atom, sympy.floor):
        return DivisionKind.REAL_FLOOR
    if isinstance(atom, sympy.ceiling):
        return DivisionKind.REAL_CEIL
    if isinstance(atom, symbolic.int_sqrt):
        return DivisionKind.SQRT
    if isinstance(atom, symbolic.int_floor):
        return DivisionKind.FLOOR
    if isinstance(atom, symbolic.int_ceil):
        return DivisionKind.CEIL
    return DivisionKind.MOD


def lowered(
    expr: sympy.Expr,
    integers: frozenset[str],
    substitution: dict[sympy.Symbol, sympy.Expr],
    divisions: dict[sympy.Symbol, Division],
) -> sympy.Expr:
    """
    ``expr`` with every division replaced, innermost first, by an integer placeholder recorded in ``divisions`` (the
    same division by the same placeholder), whose operands are rewritten through ``substitution``. This comes before
    anything rebuilds ``expr``, which would have SymPy rewrite a ``Mod`` by Python's rounding.
    """
    division_types = (symbolic.int_floor, symbolic.int_ceil, sympy.Mod, sympy.floor, sympy.ceiling, symbolic.int_sqrt)
    while innermost := [
        atom for atom in expr.atoms(*division_types) if not any(arg.atoms(*division_types) for arg in atom.args)
    ]:
        replacements: dict[sympy.Expr, sympy.Symbol] = {}
        for atom in sorted(innermost, key=str):
            # Placeholders of inner divisions are integers too
            known = integers | {placeholder.name for placeholder in divisions}
            numerator, denominator = (
                lowered(
                    sympy.expand(with_integers(cast(sympy.Expr, arg), known).xreplace(substitution)),
                    integers,
                    {},
                    divisions,
                )
                for arg in (*atom.args, sympy.Integer(1))[:2]
            )
            division = Division(division_kind(atom), numerator, denominator)
            # An exact quotient (and its zero remainder) is the same under C's truncation and Python's rounding; an
            # exact int_ceil is not, as C computes it as ``(x + y - 1) / y``
            quotient = cast(sympy.Expr, sympy.cancel(numerator / denominator))
            if division.kind in (
                DivisionKind.FLOOR,
                DivisionKind.MOD,
                DivisionKind.REAL_FLOOR,
                DivisionKind.REAL_CEIL,
            ) and (quotient.is_integer):
                replacements[atom] = sympy.Integer(0) if division.kind is DivisionKind.MOD else quotient
                continue
            placeholder = next((known for known, existing in divisions.items() if existing == division), None)
            if placeholder is None:
                # The name cannot be a program symbol, so the placeholder never meets the substitution
                placeholder = sympy.Symbol(f"{division.kind.name.lower()}:{len(divisions)}", integer=True)
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
    if (
        poly is None
        or poly.degree() != 1
        or not (
            division.kind.exact
            or (
                (division.kind is DivisionKind.FLOOR or proves_sign(division.numerator, divisions, False))
                and proves_sign(division.denominator, divisions, True)
            )
        )
    ):
        return False
    coefficient = cast(sympy.Expr, poly.coeff_monomial(placeholder))
    if proves_sign(coefficient, divisions, False):
        bounds = division.bounds(lower=True)
        # Over a symbolic denominator ``(x - y + 1) / y`` is no linear bound; a floored quotient of a nonnegative
        # numerator by a positive denominator is nonnegative outright
        if division.kind is DivisionKind.FLOOR and proves_sign(division.numerator, divisions, False):
            bounds = (*bounds, sympy.Integer(0))
    elif proves_sign(-coefficient, divisions, False):
        bounds = division.bounds(lower=False)
    else:
        return False
    integer = not strict and expr.is_integer
    return any(
        proves_sign(expr.xreplace({placeholder: bound}) + 1, divisions, True)
        if integer
        else proves_sign(expr.xreplace({placeholder: bound}), divisions, strict)
        for bound in bounds
    )


def by_cases(expr: sympy.Expr, kind: RelationKind, extremum: sympy.Expr, facts: Facts) -> Truth:
    """``0 <kind> expr`` asked once for each argument the ``Min`` or ``Max`` can equal, assuming it does."""
    truths: set[Truth] = set()
    for arg in cast(tuple[sympy.Expr, ...], extremum.args):
        others = [other for other in cast(tuple[sympy.Expr, ...], extremum.args) if other != arg]
        case = [
            Relation(RelationKind.LE, arg, other)
            if isinstance(extremum, sympy.Min)
            else Relation(RelationKind.LE, other, arg)
            for other in others
        ]
        try:
            case_facts = facts.with_relations(case)
        except InconsistentAssumptionsError:
            continue  # The argument is never the extremum
        truths.add(ask(Relation(kind, sympy.Integer(0), cast(sympy.Expr, expr.xreplace({extremum: arg}))), case_facts))
    return truths.pop() if len(truths) == 1 else Truth.UNKNOWN


def generalized_divisions(expr: sympy.Expr) -> sympy.Expr:
    """``expr`` with every division (and rounding) atom replaced by a fresh symbol, the same atom by the same one."""
    division_types = (symbolic.int_floor, symbolic.int_ceil, sympy.Mod, sympy.floor, sympy.ceiling, symbolic.int_sqrt)
    atoms = sorted(expr.atoms(*division_types), key=str)
    if not atoms:
        return expr
    # Plain symbols: the proof treats a Dummy as one of its own slacks. The names cannot be program symbols.
    taken = {str(free) for free in expr.free_symbols}
    names = (f"__dace_division{index}" for index in range(len(atoms) + len(taken)))
    fresh = [name for name in names if name not in taken][: len(atoms)]
    return cast(sympy.Expr, expr.xreplace({atom: sympy.Symbol(name) for atom, name in zip(atoms, fresh)}))


def divides(query: Relation, facts: Facts) -> Truth:
    """Whether ``lhs`` divides ``rhs``: true when their quotient is an integer under the facts."""
    divisor, multiple = (rewritten(side, facts.integers, facts.substitution) for side in (query.lhs, query.rhs))
    if divisor.is_zero:
        return ask(Relation(RelationKind.EQ, query.rhs, sympy.Integer(0)), facts)
    if cast(sympy.Expr, sympy.cancel(multiple / divisor)).is_integer:
        return Truth.TRUE
    return Truth.FALSE if divisor.is_number and multiple.is_number else Truth.UNKNOWN


def is_small(expr: sympy.Expr, facts: Facts) -> bool:
    """Whether every symbol of ``expr`` is ``SMALL``; an expression of literals only has the default scale."""
    names = {str(free) for free in sympy.sympify(expr).free_symbols}
    return bool(names) and names <= facts.small


#: A literal extent below this is as small as a ``SMALL`` symbol for a schedule: too few iterations to share out.
SMALL_CONSTANT_EXTENT = 64


def is_small_extent(expr: sympy.Expr, facts: Facts) -> bool:
    """Whether a scheduled extent is too small to share out: ``SMALL`` (:func:`is_small`), or a literal below
    :data:`SMALL_CONSTANT_EXTENT`. For schedules only: the canonical form stays parallel over literals."""
    expr = sympy.sympify(expr)
    if expr.is_number:
        return bool(expr < SMALL_CONSTANT_EXTENT)
    return is_small(expr, facts)


def is_large(expr: sympy.Expr, facts: Facts) -> bool:
    """Whether ``expr`` grows with a ``DEFAULT`` parameter: every term naming one has a positive coefficient, and it
    names parameters only."""
    expr = sympy.expand(sympy.sympify(expr))
    names = {str(free) for free in expr.free_symbols}
    large = names - facts.small
    if not large or not names <= facts.parameters:
        return False
    terms = [term for term in sympy.Add.make_args(expr) if {str(free) for free in term.free_symbols} & large]
    return all(cast(sympy.Expr, term.as_coeff_Mul()[0]).is_positive for term in terms)


def much_less(lhs: sympy.Expr, rhs: sympy.Expr, facts: Facts) -> bool:
    """Whether ``lhs << rhs`` follows from the scales: ``lhs`` is small and ``rhs`` large, both over parameters."""
    return (
        facts.assumed is not None
        and is_small(lhs, facts)
        and {str(free) for free in sympy.sympify(lhs).free_symbols} <= facts.parameters
        and is_large(rhs, facts)
    )


def ask(query: Relation, facts: Facts) -> Truth:
    """Whether the query holds under the facts. An order proven only from scales (``lhs << rhs``) is TRUE, and is
    recorded in ``facts.assumed``, so that the program checks it before it runs."""
    if query.kind is RelationKind.MUCH_LT:
        if query in facts.relations:
            return Truth.TRUE
        if prove(Relation(RelationKind.LE, query.rhs, query.lhs), facts) is Truth.TRUE:
            return Truth.FALSE
        return assume(Relation(RelationKind.LT, query.lhs, query.rhs), facts)
    answer = prove(query, facts)
    if answer is not Truth.UNKNOWN or query.kind not in (RelationKind.LT, RelationKind.LE):
        return answer
    if (truth := assume(Relation(RelationKind.LT, query.lhs, query.rhs), facts)) is Truth.TRUE:
        return truth
    if assume(Relation(RelationKind.LT, query.rhs, query.lhs), facts) is Truth.TRUE:
        return Truth.FALSE
    return Truth.UNKNOWN


def assume(order: Relation, facts: Facts) -> Truth:
    """TRUE, recording ``order`` (``lhs < rhs``), when the scales make ``lhs << rhs``; UNKNOWN otherwise."""
    if not much_less(order.lhs, order.rhs, facts):
        return Truth.UNKNOWN
    assert facts.assumed is not None  # checked by much_less
    facts.assumed.add(order)
    return Truth.TRUE


def prove(query: Relation, facts: Facts) -> Truth:
    """Whether the query follows from the relations alone."""
    if query.kind is RelationKind.DIVIDES:
        return divides(query, facts)
    expr = cast(sympy.Expr, sympy.sympify(query.rhs - query.lhs))
    if expr.is_number and query.kind is not RelationKind.MUCH_LT:
        answer = {
            RelationKind.EQ: expr.is_zero,
            RelationKind.NE: expr.is_nonzero,
            RelationKind.LT: expr.is_positive,
            RelationKind.LE: expr.is_nonnegative,
        }[query.kind]
        return Truth.UNKNOWN if answer is None else Truth.TRUE if answer else Truth.FALSE
    facts = facts.about({str(free) for free in expr.free_symbols})
    extrema = sorted(expr.atoms(sympy.Min, sympy.Max), key=str)
    if extrema:
        answer = by_cases(expr, query.kind, extrema[0], facts)
        if answer is Truth.UNKNOWN:
            # A case relation over a division (``int_floor(N, 2) <= N - 1``) is not one the solver uses. What holds
            # for every integer in place of each division holds for the division too, so ask that instead.
            # The generalized query is asked whole: an extremum that occurred only inside a division is gone with
            # it (``int_floor(Min(N, i + 13) - i - 1, 7)``), so there may be no extremum left to split on.
            generalized = generalized_divisions(expr)
            if generalized is not expr:
                answer = prove(Relation(query.kind, sympy.Integer(0), generalized), facts)
        return answer
    divisions: dict[sympy.Symbol, Division] = {}
    difference = lowered(expr, facts.integers, facts.substitution, divisions)
    difference = with_integers(difference, facts.integers | {placeholder.name for placeholder in divisions})
    # The substitution may bring in divisions of its own
    difference = lowered(sympy.expand(difference.xreplace(facts.substitution)), facts.integers, {}, divisions)
    if query.kind is RelationKind.EQ:
        answer = difference.is_zero
    elif query.kind is RelationKind.NE:
        answer = difference.is_nonzero
        if answer is None and (difference in facts.nonzero or -difference in facts.nonzero):
            answer = True
    else:
        strict = query.kind is RelationKind.LT
        # A difference that is at least a nonnegative multiple of a residual (or nothing) has the residual's sign
        if any(proves_sign(difference - bound, divisions, strict) for bound in residual_bounds(difference, facts)):
            return Truth.TRUE
        if any(
            proves_sign(-difference - bound, divisions, not strict) for bound in residual_bounds(-difference, facts)
        ):
            return Truth.FALSE
        return Truth.UNKNOWN
    if answer is None:
        return Truth.UNKNOWN
    return Truth.TRUE if answer else Truth.FALSE


def residual_bounds(difference: sympy.Expr, facts: Facts) -> list[sympy.Expr]:
    """Zero, and each residual (``>= 0``) scaled by 1 and by every positive factor that cancels one of its slacks in
    ``difference``: the slack ``N - 1 - i`` of ``i <= N - 1`` bounds ``2 * N - 2`` only twice over."""
    bounds: list[sympy.Expr] = [sympy.Integer(0)]
    for residual in facts.residuals:
        slacks = [s for s in residual.free_symbols if isinstance(s, sympy.Dummy)]
        factors = {difference.coeff(s) / residual.coeff(s) for s in slacks if residual.coeff(s) != 0}
        bounds += [factor * residual for factor in [sympy.Integer(1), *factors] if factor.is_positive]
    return list(dict.fromkeys(bounds))


def holds(query: Relation, facts: Facts) -> bool:
    """Whether the query is TRUE under the facts. The orders its proof took from the scales are recorded only then: a
    caller that acts on TRUE alone does not rely on an order that only disproved the query."""
    trial: OrderedSet[Relation] | None = None if facts.assumed is None else OrderedSet()
    if ask(query, facts.recording(trial)) is not Truth.TRUE:
        return False
    if facts.assumed is not None and trial:
        facts.assumed.update(trial)
    return True


def provably_nonnegative(expr: sympy.Expr, facts: Facts) -> bool:
    return holds(Relation(RelationKind.LE, sympy.Integer(0), expr), facts)


def provably_le(lhs: sympy.Expr, rhs: sympy.Expr, facts: Facts) -> bool:
    return holds(Relation(RelationKind.LE, lhs, rhs), facts)


def provably_integer(expr: sympy.Expr, facts: Facts) -> bool:
    """Whether ``expr`` is an integer for every value of its symbols, those the facts type as integers flagged so."""
    return with_integers(cast(sympy.Expr, sympy.sympify(expr)), facts.integers).is_integer is True


def provably_positive(expr: sympy.Expr, facts: Facts) -> bool:
    return holds(Relation(RelationKind.LT, sympy.Integer(0), cast(sympy.Expr, sympy.sympify(expr))), facts)
