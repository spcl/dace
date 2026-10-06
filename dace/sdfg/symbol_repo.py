# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" The symbol repository of an SDFG: the dtypes and facts of its parameters and of the names its scopes open. """
import enum
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Union

import sympy
from ordered_set import OrderedSet

from dace import dtypes, symbolic
from dace.dtypes import validate_name
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.validation import InvalidSDFGError
from dace.symbolic_facts import (Facts, InconsistentAssumptionsError, Predicate, Relation, RelationKind,
                                 predicate_relation, relation_names)

if TYPE_CHECKING:
    from dace.sdfg.sdfg import SDFG, InterstateEdge

#: What opens a scope: a loop (its iterator), a map or consume entry (its parameters and dynamic inputs), or an
#: interstate edge (its assignments).
ScopeOwner = Union[LoopRegion, nodes.EntryNode, 'InterstateEdge']

UNSIGNED_TYPES = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)


class SymbolInfo(NamedTuple):
    dtype: dtypes.typeclass
    predicates: frozenset[Predicate]


@dataclass(slots=True)
class Scope:
    """ The dtypes and sign predicates declared for names a scope opens, and the relations assumed in it. Which scopes
    enclose which comes from the graph (``SymbolResolver.facts_at``), not from the repository. """
    types: dict[str, dtypes.typeclass] = field(default_factory=dict)
    predicates: dict[str, frozenset[Predicate]] = field(default_factory=dict)
    relations: OrderedSet[Relation] = field(default_factory=lambda: OrderedSet[Relation]([]))


class OwnerKind(enum.Enum):
    LOOP = enum.auto()
    ENTRY = enum.auto()
    EDGE = enum.auto()


class ScopeKey(NamedTuple):
    """ Where a scope owner sits in its SDFG: the block indices down to the region (or state) holding it, and its
    position among the blocks (``LOOP``), nodes (``ENTRY``) or edges (``EDGE``) there. """
    kind: OwnerKind
    graph: tuple[int, ...]
    position: int

    def to_json(self) -> dict[str, Any]:
        return {'kind': self.kind.name, 'graph': list(self.graph), 'position': self.position}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> 'ScopeKey':
        return cls(OwnerKind[obj['kind']], tuple(obj['graph']), obj['position'])


def scope_owners(graph: AbstractControlFlowRegion, path: tuple[int, ...] = ()) -> Iterator[tuple[ScopeKey, ScopeOwner]]:
    """ Every loop, scope entry and interstate edge of an SDFG (not of its nested SDFGs), with its key. """
    for index, edge in enumerate(graph.edges()):
        yield ScopeKey(OwnerKind.EDGE, path, index), edge.data
    for index, block in enumerate(graph.nodes()):
        if isinstance(block, LoopRegion):
            yield ScopeKey(OwnerKind.LOOP, path, index), block
        if isinstance(block, SDFGState):
            for position, node in enumerate(block.nodes()):
                if isinstance(node, nodes.EntryNode):
                    yield ScopeKey(OwnerKind.ENTRY, (*path, index), position), node
        elif isinstance(block, AbstractControlFlowRegion):
            yield from scope_owners(block, (*path, index))


def bound_names(owner: ScopeOwner) -> set[str]:
    """ The names a scope owner binds, which are the only ones its scope may open. """
    if isinstance(owner, LoopRegion):
        return {owner.loop_variable} if owner.loop_variable else set()
    if isinstance(owner, nodes.MapEntry):
        params = owner.map.params
    elif isinstance(owner, nodes.ConsumeEntry):
        params = [owner.consume.pe_index]
    elif isinstance(owner, nodes.EntryNode):
        raise TypeError(f'Unknown scope entry {owner}')
    else:
        return set(owner.assignments)
    return {*params, *(connector for connector in owner.in_connectors if not connector.startswith('IN_'))}


def symbol_facts(types: dict[str, dtypes.typeclass], predicates: dict[str, frozenset[Predicate]],
                 relations: Iterable[Relation]) -> Facts:
    """ The facts of a symbol table; raises ``InconsistentAssumptionsError`` if they contradict. """
    integers = frozenset(name for name, dtype in types.items()
                         if dtype in dtypes.INTEGER_TYPES and dtype != dtypes.bool_)
    # An unsigned type is a sign fact of its own
    unsigned = {name: frozenset({Predicate.NONNEGATIVE}) for name, dtype in types.items() if dtype in UNSIGNED_TYPES}
    signs = [
        predicate_relation(predicate, symbolic.symbol(name))
        for name, named in [*predicates.items(), *unsigned.items()] for predicate in named
    ]
    return Facts((*signs, *relations), integers)


def scope_facts(scope: Scope) -> Facts:
    """ The facts a scope declares on its own. """
    return symbol_facts(scope.types, scope.predicates, scope.relations)


def relation_to_json(relation: Relation) -> dict[str, str]:
    return {
        'kind': relation.kind.name,
        'lhs': symbolic.serialize_symbolic(relation.lhs),
        'rhs': symbolic.serialize_symbolic(relation.rhs)
    }


def as_expr(value: Any) -> sympy.Expr:
    if not isinstance(value, sympy.Expr):
        raise TypeError(f'{value} is not a symbolic expression')
    return value


def relation_from_json(obj: dict[str, str]) -> Relation:
    return Relation(RelationKind[obj['kind']], as_expr(symbolic.deserialize_symbolic(obj['lhs'])),
                    as_expr(symbolic.deserialize_symbolic(obj['rhs'])))


def scope_to_json(scope: Scope) -> dict[str, Any]:
    result: dict[str, Any] = {'types': {name: dtype.to_json() for name, dtype in sorted(scope.types.items())}}
    if scope.predicates:
        result['predicates'] = {
            name: sorted(predicate.name for predicate in named)
            for name, named in sorted(scope.predicates.items())
        }
    if scope.relations:
        result['relations'] = [relation_to_json(relation) for relation in scope.relations]
    return result


def scope_from_json(obj: dict[str, Any], context: dict[str, Any] | None) -> Scope:
    return Scope({
        name: dtypes.json_to_typeclass(dtype, context)
        for name, dtype in obj['types'].items()
    }, {
        name: frozenset(Predicate[predicate] for predicate in named)
        for name, named in obj.get('predicates', {}).items()
    }, OrderedSet(relation_from_json(relation) for relation in obj.get('relations', [])))


def inconsistent(name: str, declared: SymbolInfo, added: SymbolInfo) -> InconsistentAssumptionsError:
    return InconsistentAssumptionsError(name, [
        f'declared {declared.dtype} {sorted(p.name for p in declared.predicates)}',
        f're-added as {added.dtype} {sorted(p.name for p in added.predicates)}'
    ])


@dataclass(slots=True)
class SymbolRepo:
    """
    The symbols of one SDFG. ``params`` declares the names passed in from outside; ``scopes`` holds what is declared
    for the names a loop, scope entry or interstate edge opens, only for owners that declare something. Which scopes
    enclose a point, and the facts holding there, come from the graph (``SymbolResolver.facts_at``).
    """
    params: Scope = field(default_factory=Scope)
    scopes: dict[ScopeOwner, Scope] = field(default_factory=dict)

    def scope(self, at: ScopeOwner | None) -> Scope:
        """ The declarations of ``at`` (``None``: the parameters), created empty on first use. """
        if at is None:
            return self.params
        return self.scopes.setdefault(at, Scope())

    def facts(self) -> Facts:
        """ The facts of the parameters, for explicit use in proofs. """
        return scope_facts(self.params)

    def add(self,
            name: str,
            dtype: dtypes.typeclass,
            predicates: frozenset[Predicate] = frozenset(),
            at: ScopeOwner | None = None) -> None:
        """ Declares a name in the scope of ``at``. Declaring it again with the same dtype and predicates does nothing.

            :raise ValueError: If ``at`` does not bind the name.
            :raise InconsistentAssumptionsError: If the name is declared there with another dtype or other predicates,
                                                 or the predicates contradict the facts there.
        """
        if at is not None and name not in bound_names(at):
            raise ValueError(f'{at} does not bind "{name}"')
        if predicates and not isinstance(at, (LoopRegion, nodes.EntryNode, type(None))):
            raise NotImplementedError('Facts declared for the names an interstate edge assigns are not supported')
        scope = self.scope(at)
        added = SymbolInfo(dtype, predicates)
        if name in scope.types:
            declared = SymbolInfo(scope.types[name], scope.predicates.get(name, frozenset()))
            if declared != added:
                raise inconsistent(name, declared, added)
            return
        if predicates:
            scope_facts(Scope({**scope.types, name: dtype}, {**scope.predicates, name: predicates}, scope.relations))
            scope.predicates[name] = predicates
        scope.types[name] = dtype

    def set_type(self, name: str, dtype: dtypes.typeclass, at: ScopeOwner | None = None) -> None:
        """ Changes the dtype of a declared name, keeping its facts. """
        scope = self.scope(at)
        if name not in scope.types:
            raise KeyError(f'Symbol "{name}" is not declared in this scope')
        scope_facts(Scope({**scope.types, name: dtype}, scope.predicates, scope.relations))
        scope.types[name] = dtype

    def set_predicates(self, name: str, predicates: frozenset[Predicate], at: ScopeOwner | None = None) -> None:
        """ Replaces the sign predicates of a declared name. """
        scope = self.scope(at)
        if name not in scope.types:
            raise KeyError(f'Symbol "{name}" is not declared in this scope')
        candidate = {**scope.predicates, name: predicates}
        if not predicates:
            del candidate[name]
        scope_facts(Scope(scope.types, candidate, scope.relations))
        scope.predicates = candidate

    def add_relation(self, relation: Relation, at: ScopeOwner | None = None) -> None:
        """ Assumes a relation in the scope of ``at``; assuming a known relation again does nothing.

            :raise KeyError: If a relation of the parameters names an undeclared parameter.
        """
        if at is None:
            for name in sorted(relation_names(relation) - self.params.types.keys()):
                raise KeyError(f'Symbol "{name}" is not declared')
        if not isinstance(at, (LoopRegion, nodes.EntryNode, type(None))):
            raise NotImplementedError('Facts declared for the names an interstate edge assigns are not supported')
        scope = self.scope(at)
        if relation in scope.relations:
            return
        scope_facts(Scope(scope.types, scope.predicates, OrderedSet([*scope.relations, relation])))
        scope.relations.add(relation)

    def remove(self, name: str, at: ScopeOwner | None = None) -> None:
        """ Removes a declared name.

            :raise ValueError: If a relation of the same scope still names it.
        """
        scope = self.scope(at)
        if name not in scope.types:
            raise KeyError(f'Symbol "{name}" is not declared in this scope')
        mentioning = [relation for relation in scope.relations if name in relation_names(relation)]
        if mentioning:
            raise ValueError(f'Cannot remove symbol "{name}": the relations {mentioning} still name it')
        scope.predicates.pop(name, None)
        del scope.types[name]

    def replace(self, names: dict[str, str], replacements: Mapping[str, symbolic.SymbolicType]) -> None:
        """ Renames declared names (``names``) in every scope, as ``SDFG.replace_dict`` does in the whole graph, and
        rewrites the relations. A name replaced by an expression becomes relations over it; a parameter's free symbols
        are declared with its dtype. """
        for owner, scope in [(None, self.params), *self.scopes.items()]:
            scope.relations = OrderedSet(
                Relation(relation.kind, symbolic.replace_symbols(relation.lhs, replacements),
                         symbolic.replace_symbols(relation.rhs, replacements)) for relation in scope.relations)
            for name, new_name in names.items():
                if name not in scope.types:
                    continue
                dtype = scope.types.pop(name)
                predicates = scope.predicates.pop(name, frozenset())
                if validate_name(new_name):
                    scope.types[new_name] = dtype
                    if predicates:
                        scope.predicates[new_name] = predicates
                    continue
                replacement = replacements[name]
                if not isinstance(replacement, sympy.Basic):
                    raise TypeError(f'Cannot replace symbol "{name}" by {replacement}')
                if owner is None:
                    scope.types.update({str(free): dtype for free in replacement.free_symbols})
                scope.relations.update([
                    predicate_relation(predicate, as_expr(replacement))
                    for predicate in sorted(predicates, key=lambda p: p.name)
                ])

    def validate(self, sdfg: 'SDFG') -> None:
        """ Raises ``InvalidSDFGError`` if a declaring owner is not in ``sdfg`` or does not bind the names it declares,
        or if the parameters' facts contradict. The facts at each point are checked where they are derived. """
        owned = list(scope_owners(sdfg))
        owners = {owner for _, owner in owned}
        bound = {name for _, owner in owned for name in bound_names(owner)}
        for owner, scope in [(None, self.params), *self.scopes.items()]:
            if owner is not None and owner not in owners:
                raise InvalidSDFGError(f'The symbol scope of {owner} has no owner in the SDFG', sdfg, None)
            if owner is not None and not scope.types.keys() <= bound_names(owner):
                raise InvalidSDFGError(f'{owner} does not bind the symbols {sorted(scope.types)} its scope declares',
                                       sdfg, None)
            named = {*scope.predicates, *(name for relation in scope.relations for name in relation_names(relation))}
            visible = self.params.types.keys() | (bound if owner is not None else set())
            if not named <= visible:
                raise InvalidSDFGError(f'Symbol facts name undeclared symbols {sorted(named - visible)}', sdfg, None)
        try:
            self.facts()
        except InconsistentAssumptionsError as error:
            raise InvalidSDFGError(str(error), sdfg, None) from error

    def to_json(self, sdfg: 'SDFG') -> dict[str, Any]:
        """ Serializes the repository with its scopes keyed by the position of their owners in ``sdfg``. A scope whose
        owner is no longer in ``sdfg`` is left out; validation reports it. """
        result: dict[str, Any] = {'params': scope_to_json(self.params)}
        keys = {owner: key for key, owner in scope_owners(sdfg)} if self.scopes else {}
        live = [(keys[owner], scope) for owner, scope in self.scopes.items() if owner in keys]
        if live:
            result['scopes'] = [{'owner': key.to_json(), **scope_to_json(scope)} for key, scope in live]
        return result

    @classmethod
    def from_json(cls, obj: dict[str, Any], sdfg: 'SDFG', context: dict[str, Any] | None = None) -> 'SymbolRepo':
        """ Loads a repository whose scopes are keyed by the position of their owners in ``sdfg``. """
        owners = dict(scope_owners(sdfg)) if obj.get('scopes') else {}

        def owner_at(key: dict[str, Any]) -> ScopeOwner:
            scope_key = ScopeKey.from_json(key)
            if scope_key not in owners:
                raise KeyError(f'No scope owner at {scope_key} in SDFG "{sdfg.name}"')
            return owners[scope_key]

        return cls(scope_from_json(obj['params'], context),
                   {owner_at(scope['owner']): scope_from_json(scope, context)
                    for scope in obj.get('scopes', [])})
