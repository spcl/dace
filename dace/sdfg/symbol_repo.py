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
    """ The names a scope opens with their dtypes and sign predicates, the relations assumed in it, and the scope it is
    nested in (``None``: the SDFG parameters). """
    parent: ScopeOwner | None
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


def chain_facts(chain: list[Scope]) -> Facts:
    """ The facts holding in the innermost scope of ``chain`` (inner to outer); an inner scope shadows the facts of the
    outer names it reopens. """
    types: dict[str, dtypes.typeclass] = {}
    predicates: dict[str, frozenset[Predicate]] = {}
    relations: list[Relation] = []
    for scope in reversed(chain):
        relations = [relation for relation in relations if not relation_names(relation) & scope.types.keys()]
        predicates = {name: named for name, named in predicates.items() if name not in scope.types}
        types.update(scope.types)
        predicates.update(scope.predicates)
        relations.extend(scope.relations)
    return symbol_facts(types, predicates, relations)


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


def scope_from_json(obj: dict[str, Any], parent: ScopeOwner | None, context: dict[str, Any] | None) -> Scope:
    return Scope(parent, {
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
    The symbols of one SDFG. ``params`` declares the names passed in from outside; ``scopes`` maps every loop, scope
    entry and interstate edge that opens names to the scope it opens. A name is looked up from a scope outwards to the
    parameters, so an inner scope shadows an outer one.
    """
    params: Scope = field(default_factory=lambda: Scope(None))
    scopes: dict[ScopeOwner, Scope] = field(default_factory=dict)

    def scope(self, at: ScopeOwner | None) -> Scope:
        if at is None:
            return self.params
        if at not in self.scopes:
            raise KeyError(f'{at} opens no scope in this symbol repository')
        return self.scopes[at]

    def owners(self, at: ScopeOwner | None) -> list[ScopeOwner]:
        """ ``at`` and the owners of every scope enclosing it, inner to outer. """
        owners = []
        while at is not None:
            owners.append(at)
            at = self.scope(at).parent
        return owners

    def chain(self, at: ScopeOwner | None) -> list[Scope]:
        """ The scope of ``at`` and every scope enclosing it, inner to outer, ending with the parameters. """
        return [*(self.scopes[owner] for owner in self.owners(at)), self.params]

    def declaring(self, name: str, at: ScopeOwner | None) -> ScopeOwner | None:
        """ The owner of the innermost scope around ``at`` that opens ``name`` (``None``: a parameter). """
        while at is not None:
            scope = self.scope(at)
            if name in scope.types:
                return at
            at = scope.parent
        if name not in self.params.types:
            raise KeyError(f'Symbol "{name}" is not declared')
        return None

    def lookup(self, name: str, at: ScopeOwner | None = None) -> SymbolInfo:
        scope = self.scope(self.declaring(name, at))
        return SymbolInfo(scope.types[name], scope.predicates.get(name, frozenset()))

    def facts(self, at: ScopeOwner | None = None) -> Facts:
        """ The facts assumed inside the scope of ``at``, for explicit use in proofs. """
        return chain_facts(self.chain(at))

    def open_scope(self, owner: ScopeOwner, parent: ScopeOwner | None = None) -> None:
        """ Opens an empty scope for ``owner``, nested in the scope of ``parent``; ``add`` declares its names. """
        if owner in self.scopes:
            raise ValueError(f'{owner} already opens a scope')
        self.scope(parent)
        self.scopes[owner] = Scope(parent)

    def close_scope(self, owner: ScopeOwner) -> None:
        nested = [inner for inner, scope in self.scopes.items() if scope.parent is owner]
        if nested:
            raise ValueError(f'Cannot close the scope of {owner}: the scopes of {nested} are nested in it')
        del self.scopes[owner]

    def add(self,
            name: str,
            dtype: dtypes.typeclass,
            predicates: frozenset[Predicate] = frozenset(),
            at: ScopeOwner | None = None) -> None:
        """ Declares a name in the scope of ``at``. Declaring it again with the same dtype and predicates does nothing.

            :raise InconsistentAssumptionsError: If the name is declared there with another dtype or other predicates,
                                                 or the predicates contradict the facts there.
        """
        scope = self.scope(at)
        added = SymbolInfo(dtype, predicates)
        if name in scope.types:
            declared = SymbolInfo(scope.types[name], scope.predicates.get(name, frozenset()))
            if declared != added:
                raise inconsistent(name, declared, added)
            return
        if predicates:
            self.check(
                at,
                Scope(scope.parent, {
                    **scope.types, name: dtype
                }, {
                    **scope.predicates, name: predicates
                }, scope.relations))
            scope.predicates[name] = predicates
        scope.types[name] = dtype

    def set_type(self, name: str, dtype: dtypes.typeclass, at: ScopeOwner | None = None) -> None:
        """ Changes the dtype of a declared name, keeping its facts. """
        scope = self.scope(at)
        if name not in scope.types:
            raise KeyError(f'Symbol "{name}" is not declared in this scope')
        self.check(at, Scope(scope.parent, {**scope.types, name: dtype}, scope.predicates, scope.relations))
        scope.types[name] = dtype

    def set_predicates(self, name: str, predicates: frozenset[Predicate], at: ScopeOwner | None = None) -> None:
        """ Replaces the sign predicates of a declared name. """
        scope = self.scope(at)
        if name not in scope.types:
            raise KeyError(f'Symbol "{name}" is not declared in this scope')
        candidate = {**scope.predicates, name: predicates}
        if not predicates:
            del candidate[name]
        self.check(at, Scope(scope.parent, scope.types, candidate, scope.relations))
        scope.predicates = candidate

    def add_relation(self, relation: Relation, at: ScopeOwner | None = None) -> None:
        """ Assumes a relation in the scope of ``at``; assuming a known relation again does nothing.

            :raise KeyError: If the relation names a symbol that is not visible there.
        """
        scope = self.scope(at)
        for name in sorted(relation_names(relation)):
            self.declaring(name, at)
        if relation in scope.relations:
            return
        self.check(at, Scope(scope.parent, scope.types, scope.predicates, OrderedSet([*scope.relations, relation])))
        scope.relations.add(relation)

    def remove(self, name: str, at: ScopeOwner | None = None) -> None:
        """ Removes a declared name.

            :raise ValueError: If an assumed relation still names it.
        """
        scope = self.scope(at)
        if name not in scope.types:
            raise KeyError(f'Symbol "{name}" is not declared in this scope')
        mentioning = [
            relation for owner, other in [(None, self.params), *self.scopes.items()] for relation in other.relations
            if name in relation_names(relation) and self.declaring(name, owner) is at
        ]
        if mentioning:
            raise ValueError(f'Cannot remove symbol "{name}": the relations {mentioning} still name it')
        scope.predicates.pop(name, None)
        del scope.types[name]

    def check(self, at: ScopeOwner | None, candidate: Scope) -> None:
        """ Raises ``InconsistentAssumptionsError`` if the facts would contradict with ``candidate`` as the scope of
        ``at``. """
        chain_facts([candidate, *self.chain(at)[1:]])

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
        """ Raises ``InvalidSDFGError`` if a scope's owner is not in ``sdfg`` or opens a name it does not bind, if facts
        name a symbol that is not visible, or if the facts contradict. """
        owners = {owner for _, owner in scope_owners(sdfg)} if self.scopes else set()
        for owner, scope in [(None, self.params), *self.scopes.items()]:
            if owner is not None and owner not in owners:
                raise InvalidSDFGError(f'The symbol scope of {owner} has no owner in the SDFG', sdfg, None)
            if owner is not None and not scope.types.keys() <= bound_names(owner):
                raise InvalidSDFGError(f'{owner} does not bind the symbols {sorted(scope.types)} its scope opens', sdfg,
                                       None)
            visible = {name for inner in self.chain(owner) for name in inner.types}
            named = {*scope.predicates, *(name for relation in scope.relations for name in relation_names(relation))}
            if not named <= visible:
                raise InvalidSDFGError(f'Symbol facts name undeclared symbols {sorted(named - visible)}', sdfg, None)
            if owner is not None and not named:
                continue
            try:
                self.facts(owner)
            except InconsistentAssumptionsError as error:
                raise InvalidSDFGError(str(error), sdfg, None) from error

    def to_json(self, sdfg: 'SDFG') -> dict[str, Any]:
        """ Serializes the repository with its scopes keyed by the position of their owners in ``sdfg``. The scope of
        an owner no longer in ``sdfg``, and every scope nested in it, is left out; validation reports it. """
        result: dict[str, Any] = {'params': scope_to_json(self.params)}
        keys = {owner: key for key, owner in scope_owners(sdfg)} if self.scopes else {}
        live = {owner: scope for owner, scope in self.scopes.items() if all(at in keys for at in self.owners(owner))}
        if live:
            result['scopes'] = [{
                'owner': keys[owner].to_json(),
                'parent': None if scope.parent is None else keys[scope.parent].to_json(),
                **scope_to_json(scope)
            } for owner, scope in live.items()]
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

        return cls(
            scope_from_json(obj['params'], None, context), {
                owner_at(scope['owner']):
                scope_from_json(scope, None if scope['parent'] is None else owner_at(scope['parent']), context)
                for scope in obj.get('scopes', [])
            })
