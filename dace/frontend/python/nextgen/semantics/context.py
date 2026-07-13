# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Semantic context for the next-generation Python frontend.

A single :class:`ProgramContext` instance threads through the entire lowering
stage. It owns the descriptor repository (which *is* the resulting tree's
``containers``/``symbols``/``constants`` — descriptors are registered once and
never cloned), the name-binding table, and the demand-driven inference service.
"""
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dace import data, dtypes, symbolic
from dace.frontend.python.nextgen.common import FrontendError
from dace.frontend.python.nextgen.semantics.values import StaticSequence


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

        #: Descriptor repository (attached directly to the tree root).
        self.containers: Dict[str, data.Data] = {}
        self.symbols: Dict[str, Any] = {}
        #: Compile-time constants as (descriptor, value) tuples, shared with the tree root.
        self.constants: Dict[str, Tuple[data.Data, Any]] = dict(constants)

        #: Source-name binding table.
        self.bindings: Dict[str, Binding] = {}

        #: Compile-time Python sequence values for 'static' bindings.
        self.static_values: Dict[str, StaticSequence] = {}

        #: Names of generated return containers, in return-value order.
        self.return_names: List[str] = []

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

    def static_value_of(self, source_name: str) -> Optional[StaticSequence]:
        """Return the static value a name is bound to, if any."""
        binding = self.bindings.get(source_name)
        if binding is None or binding.kind != 'static':
            return None
        return self.static_values.get(source_name)

    def add_constant_container(self, name: str, descriptor: data.Data, value: Any) -> str:
        """
        Register a compile-time constant with an accompanying (transient)
        container descriptor, for materialized static values.

        :return: The actual repository name used.
        """
        actual_name = self.add_container(name, descriptor, transient=True)
        self.constants[actual_name] = (descriptor, value)
        return actual_name

    def resolve(self, source_name: str) -> Optional[Binding]:
        """Look up the current binding of a source-level name."""
        return self.bindings.get(source_name)

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
        symbolic globals. Used by the shared memlet parser.
        """
        result: Dict[str, Any] = {}
        for source_name, binding in self.bindings.items():
            if binding.kind == 'container' and binding.container in self.containers:
                result[source_name] = self.containers[binding.container]
        result.update(self.symbols)
        return result
