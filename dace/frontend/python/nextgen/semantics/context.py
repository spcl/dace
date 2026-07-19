# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Semantic context for the next-generation Python frontend.

A single :class:`ProgramContext` instance threads through the entire lowering
stage. It owns the descriptor repository (which *is* the resulting tree's
``containers``/``symbols``/``constants`` — descriptors are registered once and
never cloned), the name-binding table, and the demand-driven inference service.
"""
import ast
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dace import data, dtypes, symbolic
from dace.sdfg.sdfg import NestedDict
from dace.frontend.python.nextgen.common import FrontendError
from dace.frontend.python.nextgen.lowering.parse_cache import CalleeParseCache
from dace.frontend.python.nextgen.semantics.values import StaticSequence
from dace.frontend.python.nextgen.semantics import structures as structure_support


@dataclass
class BindingSnapshot:
    """
    A restorable view of the name-binding state (bindings and compile-time
    static values). The container repository is deliberately *not* part of a
    snapshot: containers registered on a discarded path become orphans, which
    are harmless.
    """
    bindings: Dict[str, 'Binding']
    static_values: Dict[str, StaticSequence]
    constant_values: Dict[str, Any]


@dataclass
class Binding:
    """
    Associates a source-level name with its current meaning.

    :param kind: One of ``'container'``, ``'symbol'``, ``'constant'``,
                 ``'static'`` (compile-time Python value), ``'callback'``.
    :param container: For container bindings, the repository name the source
                      name currently refers to (may differ from the source
                      name after rebinding).
    :param version: Rebinding counter for SSA-lite versioning. A source name
                    that is assigned a second, differently-shaped value gets a
                    new repository container with a bumped version suffix
                    instead of mutating the original descriptor.
    """
    kind: str
    container: Optional[str] = None
    version: int = 0


class ProgramContext:
    """
    Mutable semantic state shared by all lowering rules.

    The repository dictionaries handed to this class are the *same objects*
    later attached to the ``ScheduleTreeRoot`` — registration is by reference,
    with no descriptor cloning.
    """

    def __init__(self, name: str, filename: str, argtypes: Dict[str, data.Data], global_vars: Dict[str, Any],
                 constants: Dict[str, Tuple[data.Data, Any]]):
        self.name = name
        self.filename = filename
        self.globals = global_vars

        #: Descriptor repository (attached directly to the tree root). A
        #: NestedDict so dotted structure-member paths (``tracers.data``)
        #: resolve through the base Structure, matching ``SDFG.arrays``.
        self.containers: Dict[str, data.Data] = NestedDict()
        self.symbols: Dict[str, Any] = {}
        #: Compile-time constants as (descriptor, value) tuples, shared with the tree root.
        self.constants: Dict[str, Tuple[data.Data, Any]] = dict(constants)

        #: Source-name binding table.
        self.bindings: Dict[str, Binding] = {}

        #: Compile-time Python sequence values for 'static' bindings.
        self.static_values: Dict[str, StaticSequence] = {}

        #: Compile-time Python values for 'constant' bindings (opaque
        #: compile-time objects such as enum classes that cannot materialize
        #: as containers).
        self.constant_values: Dict[str, Any] = {}

        #: Names of generated return containers, in return-value order.
        self.return_names: List[str] = []

        #: Closure-array containers by source qualified name, so an external
        #: array referenced from multiple (nested) programs maps to a single
        #: repository container.
        self.closure_containers: Dict[str, str] = {}

        #: Prefix applied to materialized return containers (empty at top level,
        #: set by :meth:`inline_scope` while lowering an inlined callee).
        self.return_prefix: str = ''

        #: Stack of function objects currently being inlined (recursion detection).
        self.inline_stack: List[Any] = []

        #: Cache of resolved structure-member descriptors by dotted repository
        #: path, so repeated accesses to the same member share one descriptor
        #: object (member resolution clones by contract).
        self._member_descriptors: Dict[str, data.Data] = {}

        #: Compile-time symbolic values of materialized single-assignment ANF
        #: scalar temporaries, keyed by repository container name. Lets
        #: registry-call arguments (e.g. computed symbolic shapes) pass the
        #: symbolic expression by value instead of rejecting the data
        #: container. Only ANF temps are recorded: they are written exactly
        #: once and used immediately, so the alias can never go stale.
        self.symbolic_scalar_values: Dict[str, Any] = {}

        #: Per-parse cache of preprocessed+canonicalized callees, shared by
        #: all call sites (including nested inline scopes, which reuse this
        #: context object).
        self.parse_cache = CalleeParseCache()

        self._name_counter = 0

        for argument_name, descriptor in argtypes.items():
            self.containers[argument_name] = descriptor
            self.bindings[argument_name] = Binding(kind='container', container=argument_name)
            for free_symbol in descriptor.free_symbols:
                self.symbols.setdefault(free_symbol.name, free_symbol)
        for global_name, value in global_vars.items():
            if isinstance(value, symbolic.symbol):
                self.symbols.setdefault(value.name, value)

    # ------------------------------------------------------------------ #
    # Name and descriptor management
    # ------------------------------------------------------------------ #

    def fresh_name(self, prefix: str = '__tmp') -> str:
        """Allocate a repository-unique container or symbol name."""
        while True:
            candidate = f'{prefix}{self._name_counter}'
            self._name_counter += 1
            if candidate not in self.containers and candidate not in self.symbols:
                return candidate

    def add_container(self, name: str, descriptor: data.Data, transient: bool = True) -> str:
        """
        Register a descriptor in the repository, uniquifying the name if
        necessary. The descriptor object is stored as-is (no cloning).

        :return: The actual repository name used.
        """
        descriptor.transient = transient
        actual_name = name
        if actual_name in self.containers or actual_name in self.symbols:
            actual_name = self.fresh_name(f'{name}_')
        self.containers[actual_name] = descriptor
        for free_symbol in descriptor.free_symbols:
            self.symbols.setdefault(free_symbol.name, free_symbol)
        return actual_name

    def bind(self, source_name: str, container_name: str) -> None:
        """Bind (or rebind) a source-level name to a repository container."""
        existing = self.bindings.get(source_name)
        version = existing.version + 1 if existing is not None else 0
        self.bindings[source_name] = Binding(kind='container', container=container_name, version=version)

    def bind_symbol(self, source_name: str, dtype: dtypes.typeclass = dtypes.int64) -> symbolic.symbol:
        """Bind a source-level name as a typed symbol (e.g., a loop index)."""
        symbol_value = symbolic.symbol(source_name, dtype)
        self.symbols[source_name] = symbol_value
        self.bindings[source_name] = Binding(kind='symbol')
        return symbol_value

    def bind_static(self, source_name: str, value: StaticSequence) -> None:
        """Bind a source-level name to a compile-time Python sequence value."""
        existing = self.bindings.get(source_name)
        version = existing.version + 1 if existing is not None else 0
        self.bindings[source_name] = Binding(kind='static', version=version)
        self.static_values[source_name] = value

    def bind_constant(self, source_name: str, value: Any) -> None:
        """Bind a source-level name to an arbitrary compile-time Python value
        (no container is materialized)."""
        existing = self.bindings.get(source_name)
        version = existing.version + 1 if existing is not None else 0
        self.bindings[source_name] = Binding(kind='constant', version=version)
        self.constant_values[source_name] = value

    def static_value_of(self, source_name: str) -> Optional[StaticSequence]:
        """Return the static value a name is bound to, if any."""
        binding = self.bindings.get(source_name)
        if binding is None or binding.kind != 'static':
            return None
        return self.static_values.get(source_name)

    def register_closure_array(self, name: str, qualified_name: str, descriptor: data.Data) -> str:
        """
        Register an external (closure) array as a non-transient container,
        deduplicated by its source qualified name so every reference to the
        same external array — including from inlined callees — shares one
        repository container.

        Preprocessing injects top-level closure arrays into the argument
        types, so the container may already exist under this exact
        descriptor; in that case it is adopted rather than re-registered.

        :return: The repository container name.
        """
        if qualified_name in self.closure_containers:
            return self.closure_containers[qualified_name]
        if self.containers.get(name) is descriptor:
            descriptor.transient = False
            self.closure_containers[qualified_name] = name
            return name
        if name in self.closure_containers.values():
            # The same external array reaches nested closures under different
            # qualified names; the mangled reference name encodes the source
            # expression and is stable across closures.
            self.closure_containers[qualified_name] = name
            return name
        actual_name = self.add_container(name, descriptor, transient=False)
        self.closure_containers[qualified_name] = actual_name
        return actual_name

    def add_constant_container(self, name: str, descriptor: data.Data, value: Any) -> str:
        """
        Register a compile-time constant with an accompanying (transient)
        container descriptor, for materialized static values.

        :return: The actual repository name used.
        """
        actual_name = self.add_container(name, descriptor, transient=True)
        self.constants[actual_name] = (descriptor, value)
        return actual_name

    # ------------------------------------------------------------------ #
    # Branch-scoped binding state
    # ------------------------------------------------------------------ #

    def snapshot(self) -> BindingSnapshot:
        """Capture the current binding state (shallow copies)."""
        return BindingSnapshot(bindings=dict(self.bindings),
                               static_values=dict(self.static_values),
                               constant_values=dict(self.constant_values))

    def restore(self, saved: BindingSnapshot) -> None:
        """Restore a previously captured binding state. The snapshot itself
        stays intact, so it can be restored multiple times."""
        self.bindings = dict(saved.bindings)
        self.static_values = dict(saved.static_values)
        self.constant_values = dict(saved.constant_values)

    # ------------------------------------------------------------------ #
    # Nested-program inlining support
    # ------------------------------------------------------------------ #

    @contextmanager
    def inline_scope(self, function: Any, parameter_bindings: Dict[str, str], callee_globals: Dict[str, Any],
                     return_prefix: str) -> Iterator[List[str]]:
        """
        Establish a fresh binding scope for lowering an inlined callee into
        the shared repository. Saves and restores the caller's bindings,
        static values, globals, and return handling; seeds the callee's
        parameter bindings; and tracks the inline stack for recursion
        detection.

        :param function: The callee function object (for recursion detection).
        :param parameter_bindings: Mapping from callee parameter names to
                                   repository container names.
        :param callee_globals: The callee's resolved global variables.
        :param return_prefix: Prefix for materialized callee return containers.
        :yield: The callee's ``return_names`` list, populated as return
                statements are lowered (read it before the scope exits).
        """
        saved = (self.bindings, self.static_values, self.constant_values, self.globals, self.return_prefix,
                 self.return_names)
        self.inline_stack.append(function)
        self.bindings = {
            name: Binding(kind='container', container=container)
            for name, container in parameter_bindings.items()
        }
        self.static_values = {}
        self.constant_values = {}
        self.globals = callee_globals
        self.return_prefix = return_prefix
        self.return_names = []
        try:
            yield self.return_names
        finally:
            self.inline_stack.pop()
            (self.bindings, self.static_values, self.constant_values, self.globals, self.return_prefix,
             self.return_names) = saved

    def resolve(self, source_name: str) -> Optional[Binding]:
        """Look up the current binding of a source-level name."""
        return self.bindings.get(source_name)

    def member_access_of(self, source_name: str, member: str) -> Optional[Tuple[str, data.Data]]:
        """
        Resolve a structure member access (``source_name.member``) when the
        source name is bound to a container with members (a
        :class:`~dace.data.Structure`).

        :return: A 2-tuple of (dotted repository data path, member descriptor),
                 or None if the name is not bound to a structure or the member
                 does not exist. The member descriptor is cached per path so
                 repeated accesses share one object.
        """
        binding = self.bindings.get(source_name)
        if binding is None or binding.kind != 'container':
            return None
        # NOTE: binding.container may itself be a dotted structure-member path
        # (a name aliased to a nested structure member); NestedDict.get() does
        # not resolve dotted keys (only __getitem__/__contains__ do), so use
        # 'in'/'[]' here rather than dict.get().
        if binding.container not in self.containers:
            return None
        base_descriptor = self.containers[binding.container]
        path = structure_support.structure_member_path(binding.container, member)
        cached = self._member_descriptors.get(path)
        if cached is not None:
            return path, cached
        resolved = structure_support.resolve_member_access(binding.container, base_descriptor, member)
        if resolved is None:
            return None
        self._member_descriptors[path] = resolved.descriptor
        return resolved.data_name, resolved.descriptor

    def descriptor_of(self, source_name: str, node: Optional[ast.AST] = None) -> data.Data:
        """
        Return the descriptor a source-level name currently refers to.

        :raises FrontendError: If the name is not bound to a container.
        """
        binding = self.bindings.get(source_name)
        if binding is None or binding.kind != 'container' or binding.container not in self.containers:
            raise FrontendError(f'Name "{source_name}" is not bound to a data container', self.filename, node)
        return self.containers[binding.container]

    def container_of(self, source_name: str, node: Optional[ast.AST] = None) -> str:
        """
        Return the repository container name a source-level name refers to.

        :raises FrontendError: If the name is not bound to a container.
        """
        binding = self.bindings.get(source_name)
        if binding is None or binding.kind != 'container' or binding.container not in self.containers:
            raise FrontendError(f'Name "{source_name}" is not bound to a data container', self.filename, node)
        return binding.container

    def defined_view(self) -> Dict[str, Any]:
        """
        A flat name-to-value view of everything visible for expression parsing:
        container descriptors under their *source* names, symbols, and
        symbolic globals. Structure members appear under their dotted source
        names (``tracers.data``) so the shared memlet parser can parse member
        subscripts. Used by the shared memlet parser.
        """
        result: Dict[str, Any] = {}
        for source_name, binding in self.bindings.items():
            if binding.kind == 'container' and binding.container in self.containers:
                descriptor = self.containers[binding.container]
                result[source_name] = descriptor
                members = structure_support.descriptor_members(descriptor)
                if members:
                    for member_name in members:
                        member = self.member_access_of(source_name, member_name)
                        if member is not None:
                            result[f'{source_name}.{member_name}'] = member[1]
        result.update(self.symbols)
        return result
