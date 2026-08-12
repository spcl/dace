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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from dace import data, dtypes, symbolic
from dace.sdfg.sdfg import NestedDict
from dace.frontend.python.nextgen import provenance
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
    :param declared: Whether a type annotation (``b: dace.float64``) fixed this
                     container's descriptor. Declared names are not re-typed by
                     the values later assigned to them: the declaration is the
                     type, and assignments convert into it.
    """
    kind: str
    container: Optional[str] = None
    version: int = 0
    declared: bool = False


def foldable_scalar_names(body: List[ast.stmt]) -> Set[str]:
    """
    The source-level names of a canonical body whose compile-time value may be
    recorded alongside the scalar container they materialize into.

    Recording a value is only sound while it cannot go stale, which needs two
    properties that a plain ``name = <symbolic expression>`` does not have on
    its own -- ANF temporaries have both by construction, source names have to
    be checked:

    - the name is assigned exactly ONCE, so no later assignment can invalidate
      the recorded expression. A loop-carried ``x = x + 1`` is the case that
      matters: the loop body is lowered once, so a value recorded from ``x =
      0`` would be substituted into every iteration's reads;
    - every read sits inside the block holding the assignment, so an expression
      over a loop or map parameter cannot outlive the scope that defines the
      parameter (``for i in ...: y = i * 2`` with ``y`` read after the loop).

    :param body: The canonical statements to analyze.
    :return: The set of names that satisfy both properties.
    """
    assignment_blocks: Dict[str, Tuple[int, ...]] = {}
    reassigned: Set[str] = set()
    read_blocks: Dict[str, List[Tuple[int, ...]]] = {}

    def record(node: ast.AST, block: Tuple[int, ...]) -> None:
        for descendant in ast.walk(node):
            if not isinstance(descendant, ast.Name):
                continue
            if isinstance(descendant.ctx, ast.Load):
                read_blocks.setdefault(descendant.id, []).append(block)
            elif descendant.id in assignment_blocks:
                reassigned.add(descendant.id)
            else:
                assignment_blocks[descendant.id] = block

    def visit(statements: List[ast.stmt], block: Tuple[int, ...]) -> None:
        for statement in statements:
            for _, value in ast.iter_fields(statement):
                if isinstance(value, list) and any(isinstance(item, ast.stmt) for item in value):
                    visit([item for item in value if isinstance(item, ast.stmt)], block + (id(statement), ))
                elif isinstance(value, ast.AST):
                    record(value, block)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            record(item, block)

    visit(body, ())
    return {
        name
        for name, assigned_in in assignment_blocks.items()
        if name not in reassigned and all(read_in[:len(assigned_in)] == assigned_in
                                          for read_in in read_blocks.get(name, ()))
    }


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
        #: Symbols whose value is only computed while the program RUNS (the
        #: element count of a boolean-mask gather, see
        #: ``mechanisms.advanced_indexing._resolve_symbol_from_scalar``). A
        #: container sized by one of these is perfectly usable inside the
        #: program and unusable at its boundary, where the caller has to
        #: allocate before the call; :mod:`~...lowering.rules.returns` checks
        #: this to refuse such a return with the real reason instead of letting
        #: it fail as an unevaluatable shape at call time. Each symbol maps to
        #: a short description of the quantity it holds ("the number of
        #: elements ``A > 15`` selects"), which is what a diagnostic shows: the
        #: symbol's own generated name says nothing to the reader.
        self.deferred_symbols: Dict[str, str] = {}
        #: Source name -> repository symbol name, for the rare symbol whose
        #: repository name differs (see :meth:`bind_symbol`). The repository
        #: itself is keyed by the symbol's own name, since it IS the tree's
        #: symbol table.
        self.symbol_aliases: Dict[str, str] = {}
        #: Compile-time constants as (descriptor, value) tuples, shared with the tree root.
        self.constants: Dict[str, Tuple[data.Data, Any]] = dict(constants)

        #: Names among :attr:`constants` whose value inference substitutes at
        #: every use, so generated code never mentions them. They still reach
        #: the tree root -- a callback's namespace is built from ALL constants,
        #: and a statement running in the interpreter may well name one -- but
        #: carrying them into the SDFG's constant repository would emit a dead
        #: ``constexpr`` for a ``dace.compiletime`` argument that was supposed
        #: to fold away. Kept separate from the values because a tree produced
        #: from an existing SDFG (``sdfg_to_tree``) has genuine scalar
        #: constants its tasklets DO reference by name, and those must survive.
        self.folded_constants: Set[str] = set(constants)

        #: Python callables of the callbacks preprocessing DETECTED, keyed by
        #: the sanitized name it rewrote the call to (``LOG.append`` becomes
        #: ``LOG_append``). Statements that end up in the interpreter reference
        #: these names, so a callback node's execution namespace has to contain
        #: them; they are not program globals under that name, and no other
        #: stage can recover the callable from the name alone.
        self.callback_callables: Dict[str, Any] = {}

        #: Containers holding the result of a CALL whose return type could not
        #: be inferred and which no annotation declared. They are ``pyobject``
        #: like any other opaque value, but unlike a dict literal or a binding
        #: the join merge gave up on, the user can fix them -- by annotating
        #: the binding or hinting the callee's return type -- so a use that
        #: needs a real type is reported against them rather than deferred to
        #: the interpreter. See
        #: :mod:`~dace.frontend.python.nextgen.lowering.mechanisms.opaque_values`.
        self.untyped_call_results: Set[str] = set()

        #: Source-name binding table.
        self.bindings: Dict[str, Binding] = {}

        #: Compile-time Python sequence values for 'static' bindings.
        self.static_values: Dict[str, StaticSequence] = {}

        #: Compile-time Python values for 'constant' bindings (opaque
        #: compile-time objects such as enum classes that cannot materialize
        #: as containers).
        self.constant_values: Dict[str, Any] = {}

        #: The expression each generated temporary replaced, by temporary name.
        #: Filled from the records canonicalization left on the AST as each
        #: statement is lowered (see
        #: :mod:`dace.frontend.python.nextgen.provenance`), so it always holds
        #: the temporaries of the body being lowered. Read through
        #: :meth:`describe_expression`, never directly: a diagnostic that names
        #: a container would otherwise name the temporary an expression was
        #: hoisted into rather than what the user wrote.
        self.expression_sources: Dict[str, ast.AST] = {}

        #: Names of generated return containers, in return-value order.
        self.return_names: List[str] = []

        #: Closure-array containers by source qualified name, so an external
        #: array referenced from multiple (nested) programs maps to a single
        #: repository container.
        self.closure_containers: Dict[str, str] = {}

        #: Prefix applied to materialized return containers (empty at top level,
        #: set by :meth:`inline_scope` while lowering an inlined callee).
        self.return_prefix: str = ''

        #: Element type declared by the callee's ``-> dtype`` return annotation,
        #: while lowering an inlined callee that has one (None otherwise). The
        #: top-level program's annotation arrives through ``argtypes`` instead,
        #: as a pre-registered ``__return`` container.
        self.return_dtype: Optional[dtypes.typeclass] = None

        #: Stack of function objects currently being inlined (recursion detection).
        self.inline_stack: List[Any] = []

        #: Cache of resolved structure-member descriptors by dotted repository
        #: path, so repeated accesses to the same member share one descriptor
        #: object (member resolution clones by contract).
        self._member_descriptors: Dict[str, data.Data] = {}

        #: Compile-time symbolic values of materialized scalars, keyed by
        #: repository container name. Lets registry-call arguments (e.g.
        #: computed symbolic shapes) pass the symbolic expression by value
        #: instead of rejecting the data container, and lets a subset carry the
        #: arithmetic instead of an indirection. Recorded for ANF temps (which
        #: are written exactly once and used immediately) and for the source
        #: names :func:`foldable_scalar_names` clears.
        self.symbolic_scalar_values: Dict[str, Any] = {}

        #: Source-level names whose compile-time value may be recorded in
        #: :attr:`symbolic_scalar_values`, for the body currently being lowered
        #: (see :func:`foldable_scalar_names`).
        self.foldable_scalar_names: Set[str] = set()

        #: Containers holding an opaque HANDLE a registry replacement produced
        #: (an MPI subarray from ``dace.comm.Subarray``, a redistribution from
        #: ``dace.comm.Redistribute``). These are Python-object-typed like an
        #: interpreter fallback's result, but unlike one they are exactly what
        #: other replacements consume by name, so argument resolution passes
        #: them through instead of refusing them. Keyed by repository container
        #: name (the value records how to replay the producing call); never
        #: rolled back, since a container name is never reused.
        self.replacement_handles: Dict[str, Any] = {}

        #: Per-parse cache of preprocessed+canonicalized callees, shared by
        #: all call sites (including nested inline scopes, which reuse this
        #: context object).
        self.parse_cache = CalleeParseCache()

        self._name_counter = 0
        self._map_label_counters: Dict[str, int] = {}

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

    def fresh_map_label(self, line: int) -> str:
        """
        Allocate a distinct label for a map scope originating at a source line.

        The line alone does not identify a map: one statement can expand to
        several map scopes (``b[:] = a * 2 + 1`` is two), and every one of them
        then carried the same label. Labels need not be unique, but identical
        ones make a dump unreadable and leave no way -- for a test or a person
        -- to say which map is meant. So the first map on a line keeps the bare
        ``map_<line>`` and each further one gets a counting suffix, the same
        idiom SSA-split containers already use (``pc_0``, ``pc_1``).

        :param line: Source line the map originates at.
        :return: The label to give the map node.
        """
        base = f'map_{line}'
        index = self._map_label_counters.get(base)
        if index is None:
            self._map_label_counters[base] = 0
            return base
        self._map_label_counters[base] = index + 1
        return f'{base}_{index}'

    def describe_expression(self, value: Union[ast.AST, str]) -> str:
        """
        Render an expression node (or a container name) as the source text the
        user wrote, resolving the temporaries canonicalization hoisted it into.

        Every diagnostic that names an expression should go through this rather
        than unparsing the canonical node directly: after A-normal form that
        node is often a generated name (``__anf1``), which no reader can trace
        back to ``A[A > 15]``. Falls back to the plain unparse when nothing was
        recorded, so it is always safe to call in an error path.

        :param value: An expression node, or the name of a container.
        :return: Single-line source text describing the expression.
        """
        return provenance.describe_expression(value, self.expression_sources)

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

    def retype_container(self, name: str, descriptor: data.Data) -> None:
        """
        Replace an existing container's descriptor in place, keeping the name,
        every binding to it, and its ``transient`` flag.

        Used for a method replacement that changes what its receiver IS rather
        than producing a new container (``x.requires_grad_()`` turns ``x`` into
        a :class:`~dace.data.ml.ParameterArray`), which the registry reports
        through ``infers_method_self_descriptor``.

        The ``transient`` flag is deliberately taken from the container that is
        already registered rather than from the new descriptor: whether a
        container is program-visible storage or a temporary is the frontend's
        own bookkeeping — an argument that gains a gradient is still an
        argument — and the classic frontend's in-place conversion
        (``ParameterArray.make_parameter``) leaves it alone for the same reason.
        """
        existing = self.containers[name]
        descriptor.transient = existing.transient
        self.containers[name] = descriptor
        for free_symbol in descriptor.free_symbols:
            self.symbols.setdefault(free_symbol.name, free_symbol)

    def bind(self, source_name: str, container_name: str, declared: bool = False) -> None:
        """Bind (or rebind) a source-level name to a repository container.

        :param declared: Set when a type annotation fixes the descriptor of the
                         bound container (see :class:`Binding`).
        """
        existing = self.bindings.get(source_name)
        version = existing.version + 1 if existing is not None else 0
        self.bindings[source_name] = Binding(kind='container',
                                             container=container_name,
                                             version=version,
                                             declared=declared)

    def bind_symbol(self,
                    source_name: str,
                    dtype: dtypes.typeclass = dtypes.int64,
                    symbol_name: Optional[str] = None) -> symbolic.symbol:
        """
        Bind a source-level name as a typed symbol (e.g., a loop index).

        :param symbol_name: The repository name of the symbol, when it has to
            differ from the source name -- a loop variable that shadows an
            existing CONTAINER of the same name needs its own (an SDFG rejects
            a symbol and a data descriptor sharing a name). Every consumer
            reaches the symbol through this binding, so the indirection is
            invisible: :meth:`defined_view` hands the memlet parser the symbol
            OBJECT under the source name, and ``resolve_symbol_names`` rewrites
            emitted code to the symbol's own name.
        """
        symbol_value = symbolic.symbol(symbol_name or source_name, dtype)
        self.symbols[symbol_value.name] = symbol_value
        if symbol_value.name != source_name:
            self.symbol_aliases[source_name] = symbol_value.name
        else:
            self.symbol_aliases.pop(source_name, None)
        self.bindings[source_name] = Binding(kind='symbol')
        return symbol_value

    def symbol_of(self, source_name: str) -> Any:
        """The symbol object a symbol-bound source name refers to."""
        return self.symbols[self.symbol_aliases.get(source_name, source_name)]

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
        Register an external (closure) array, deduplicated by its source
        qualified name so every reference to the same external array — including
        from inlined callees — shares one repository container.

        .. note::
           The qualified name does not always identify an array uniquely: two
           different objects reached through the same field expression share it
           (``self.q`` in both ``ObjA`` and ``ObjB``) and, once inlined, also
           share the mangled reference name ``__g_self_q``, so the two collapse
           onto one container. See ``test_nested_objects`` and
           ``preparse_test::test_nested_objects_same_name``.

        Preprocessing injects top-level closure arrays into the argument
        types, so the container may already exist under this exact
        descriptor; in that case it is adopted rather than re-registered.

        A closure array is normally non-transient: it is storage the caller owns
        and passes in. An object may nevertheless declare its field transient
        through ``__descriptor__``, meaning the array is the program's own
        scratch space and must not become a program argument, so the
        descriptor's own flag is honoured rather than overwritten.

        :return: The repository container name.
        """
        transient = bool(getattr(descriptor, 'transient', False))

        if qualified_name in self.closure_containers:
            return self.closure_containers[qualified_name]
        if self.containers.get(name) is descriptor:
            self.closure_containers[qualified_name] = name
            return name
        if name in self.closure_containers.values():
            # The same external array reaches nested closures under different
            # qualified names; the mangled reference name encodes the source
            # expression and is stable across closures.
            self.closure_containers[qualified_name] = name
            return name
        actual_name = self.add_container(name, descriptor, transient=transient)
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
    def inline_scope(self,
                     function: Any,
                     parameter_bindings: Dict[str, str],
                     callee_globals: Dict[str, Any],
                     return_prefix: str,
                     return_dtype: Optional[dtypes.typeclass] = None,
                     body: Optional[List[ast.stmt]] = None) -> Iterator[List[str]]:
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
        :param return_dtype: Element type declared by the callee's return
                             annotation, if it has one.
        :param body: The callee's canonical body, analyzed for the names whose
                     compile-time value may be recorded.
        :yield: The callee's ``return_names`` list, populated as return
                statements are lowered (read it before the scope exits).
        """
        saved = (self.bindings, self.static_values, self.constant_values, self.globals, self.return_prefix,
                 self.return_names, self.expression_sources, self.return_dtype, self.foldable_scalar_names)
        self.inline_stack.append(function)
        # A callee was canonicalized by its own pipeline run, whose temporary
        # names restart from zero: the caller's map would answer a callee's
        # ``__anf0`` with the caller's unrelated expression. The callee refills
        # it from its own body as that body is lowered.
        self.expression_sources = {}
        self.bindings = {
            name: Binding(kind='container', container=container)
            for name, container in parameter_bindings.items()
        }
        self.static_values = {}
        self.constant_values = {}
        self.globals = callee_globals
        self.return_prefix = return_prefix
        self.return_names = []
        self.return_dtype = return_dtype
        self.foldable_scalar_names = foldable_scalar_names(body) if body is not None else set()
        try:
            yield self.return_names
        finally:
            self.inline_stack.pop()
            (self.bindings, self.static_values, self.constant_values, self.globals, self.return_prefix,
             self.return_names, self.expression_sources, self.return_dtype, self.foldable_scalar_names) = saved

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
        for source_name, value in self.constant_values.items():
            # Names bound to a compile-time SYMBOLIC value (an index hoisted
            # out of a subscript, see ``rules.assign._bind_index_symbol``)
            # resolve to the expression itself.
            if symbolic.issymbolic(value) or isinstance(value, int):
                result[source_name] = value
        result.update(self.symbols)
        for source_name, symbol_name in self.symbol_aliases.items():
            if symbol_name in self.symbols:
                result[source_name] = self.symbols[symbol_name]
        return result
