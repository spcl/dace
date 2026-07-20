# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Demand-driven inference for canonical (post-ANF) expressions.

Because lowering only ever sees depth-1 ("flat") expressions, inference here
is intentionally small: it classifies an expression as a container access, a
symbolic expression, or a compile-time constant, and computes the result
descriptor for flat operator expressions. There is no separate whole-program
inference pass — rules ask on demand.

Descriptor inference for library calls (NumPy and friends) is added by the
call-lowering rules through the replacement registry; this module only covers
the operator core.
"""
import ast
import copy
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy
from dace import data, dtypes, symbolic
from dace.frontend.python import astutils
from dace.frontend.python.memlet_parser import ParseMemlet, MemletExpr
from dace.frontend.python.nextgen.common import SUPPORTED_DATA_ATTRIBUTES, UnsupportedFeatureError
from dace.frontend.python.nextgen.semantics import values
from dace.frontend.python.nextgen.semantics.context import ProgramContext
from dace.frontend.python.nextgen.semantics.values import StaticSequence

#: Comparison and boolean operators always produce booleans.
_BOOLEAN_OPS = (ast.Compare, ast.BoolOp)


def _apply_unary_operator(operator: ast.unaryop, value: Any) -> Any:
    """Apply a unary AST operator to a compile-time constant."""
    if isinstance(operator, ast.USub):
        return -value
    if isinstance(operator, ast.UAdd):
        return +value
    if isinstance(operator, ast.Invert):
        return ~value
    raise TypeError(f'Cannot constant-fold unary operator {type(operator).__name__}')


@dataclass
class Inferred:
    """
    Classification of a canonical expression.

    :param kind: ``'data'`` (container access), ``'symbolic'`` (symbol
                 expression), ``'constant'`` (compile-time value), or
                 ``'static'`` (compile-time Python sequence, see
                 :class:`~dace.frontend.python.nextgen.semantics.values.StaticSequence`).
    :param descriptor: Result data descriptor for ``'data'`` expressions.
    :param value: The symbolic expression, constant, or static sequence otherwise.
    """
    kind: str
    descriptor: Optional[data.Data] = None
    value: Any = None

    @property
    def is_data(self) -> bool:
        return self.kind == 'data'

    @property
    def is_pyobject(self) -> bool:
        return self.kind == 'data' and isinstance(self.descriptor.dtype, dtypes.pyobject)

    @property
    def dtype(self) -> Optional[dtypes.typeclass]:
        if self.descriptor is not None:
            return self.descriptor.dtype
        if self.kind == 'symbolic':
            try:
                return symbolic.symtype(self.value)
            except (TypeError, AttributeError) as error:
                # Mixed or missing symbol dtypes in the expression
                raise UnsupportedFeatureError(f'Cannot infer symbolic expression type of "{self.value}": {error}',
                                              category='type-inference')
        if self.kind == 'constant' and isinstance(self.value, tuple(dtypes.dtype_to_typeclass().keys())):
            return dtypes.dtype_to_typeclass(type(self.value))
        return None


class _LocationShim:
    """Minimal visitor stand-in for the shared memlet parser's error reports."""

    def __init__(self, filename: str):
        self.filename = filename


def is_literal_constant(value: Any) -> bool:
    """Whether a constant node's value is a plain Python literal (as opposed
    to an arbitrary object embedded by preprocessing's global resolution)."""
    if value is None or value is Ellipsis:
        return True
    if isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(is_literal_constant(element) for element in value)
    return False


def _qualified_object_name(obj: Any, fallback: Optional[str]) -> Optional[str]:
    """The registry-facing qualified name of a resolved Python object."""
    module_name = getattr(obj, '__module__', None)
    object_name = getattr(obj, '__name__', None)
    if module_name and object_name and module_name != 'builtins':
        return f'{module_name}.{object_name}'
    if object_name:
        return object_name
    return fallback


def broadcast_shapes(first: Sequence[Any], second: Sequence[Any]) -> Tuple[Any, ...]:
    """
    NumPy-style shape broadcasting for symbolic shapes.

    :raises UnsupportedFeatureError: If the shapes cannot be broadcast.
    """
    result: List[Any] = []
    for dim_a, dim_b in zip(_padded(first, second), _padded(second, first)):
        if dim_a is None:
            result.append(dim_b)
        elif dim_b is None:
            result.append(dim_a)
        elif dim_a == dim_b or dim_b == 1:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        else:
            # Symbolically unequal dimensions: assume equality (matches the
            # stable frontend, which defers mismatches to runtime).
            result.append(dim_a)
    return tuple(result)


def _padded(shape: Sequence[Any], other: Sequence[Any]) -> List[Any]:
    pad = max(len(other) - len(shape), 0)
    return [None] * pad + list(shape)


class InferenceService:
    """Classifies canonical expressions against a :class:`ProgramContext`."""

    def __init__(self, context: ProgramContext):
        self.context = context
        self._shim = _LocationShim(context.filename)

    def infer(self, node: ast.expr) -> Inferred:
        """
        Infer the classification and result descriptor of a canonical
        expression.

        :raises UnsupportedFeatureError: If the expression cannot be inferred.
        """
        if isinstance(node, ast.Constant):
            return Inferred(kind='constant', value=node.value)
        if isinstance(node, (ast.List, ast.Tuple)):
            sequence_kind = 'list' if isinstance(node, ast.List) else 'tuple'
            return Inferred(kind='static', value=StaticSequence(elements=list(node.elts), kind=sequence_kind))
        if isinstance(node, ast.Name):
            return self._infer_name(node)
        if isinstance(node, ast.Attribute):
            return self._infer_attribute(node)
        if isinstance(node, ast.UnaryOp):
            operand = self.infer(node.operand)
            if isinstance(node.op, ast.Not):
                return self._demote_to_bool(operand)
            if operand.kind == 'constant':
                try:
                    return Inferred(kind='constant', value=_apply_unary_operator(node.op, operand.value))
                except TypeError:
                    pass
            if operand.kind == 'symbolic':
                return Inferred(kind='symbolic', value=self._symbolic_expression(node))
            # Data operands: sign/inversion preserves descriptor shape and dtype
            return operand
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)
        if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp)):
            return self._infer_operator(node)
        if isinstance(node, ast.Call):
            inferred = self.infer_call(node)
            if inferred is not None:
                return inferred
        raise UnsupportedFeatureError(f'Cannot infer type of expression: {astutils.unparse(node)}',
                                      self.context.filename,
                                      node,
                                      category='type-inference')

    def resolve_callee(self, func: ast.expr) -> Tuple[str, Optional[Any]]:
        """
        Resolve a canonical callee expression (a name or attribute chain) to a
        qualified name and, when possible, the Python object it refers to.

        The qualified name is normalized to ``module.__name__``-based form
        (e.g. ``numpy.zeros`` even for ``np.zeros``), matching the keys of the
        replacement registry.

        :return: A 2-tuple of (qualified name, resolved object or None).
        """
        # Preprocessing embeds resolved global objects (dace programs, SDFGs,
        # constants) directly into the AST as constant nodes with a qualname.
        if isinstance(func, ast.Constant) and not is_literal_constant(func.value):
            resolved = func.value
            return _qualified_object_name(resolved, getattr(func, 'qualname', None)), resolved

        parts: List[str] = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return astutils.rname(func), None
        parts.append(node.id)
        parts.reverse()

        root = self._global_value(parts[0])
        resolved = root
        for attribute in parts[1:]:
            if resolved is None:
                break
            resolved = getattr(resolved, attribute, None)

        if resolved is not None:
            qualified = _qualified_object_name(resolved, None)
            if qualified is not None:
                return qualified, resolved
        if len(parts) > 1 and isinstance(root, types.ModuleType):
            return f'{root.__name__}.{".".join(parts[1:])}', resolved
        return '.'.join(parts), resolved

    #: Ufunc methods with dedicated registry entries (``get_ufunc(method)``).
    _UFUNC_METHODS = ('reduce', 'accumulate', 'outer')

    def resolve_ufunc_call(self, node: ast.Call) -> Optional[Tuple[Any, Optional[str]]]:
        """
        Resolve a canonical call as a NumPy universal function invocation,
        either direct (``numpy.add(...)``) or through one of its
        ``reduce``/``accumulate``/``outer`` methods (``numpy.add.reduce(...)``
        — an ``ast.Attribute`` call whose *base* resolves to a
        ``numpy.ufunc``, not the callee itself, which resolves to a bound
        method object instead).

        Shared by inference (:meth:`infer_call`) and lowering
        (``dispatch._lower_registry_call``) so both stages agree on which
        calls are ufunc invocations and which method they use.

        :return: A 2-tuple of (ufunc object, method name, or ``None`` for a
                 direct call), or ``None`` if the call is not a ufunc
                 invocation.
        """
        _, callee = self.resolve_callee(node.func)
        if isinstance(callee, numpy.ufunc):
            return callee, None
        if isinstance(node.func, ast.Attribute) and node.func.attr in self._UFUNC_METHODS:
            _, base = self.resolve_callee(node.func.value)
            if isinstance(base, numpy.ufunc):
                return base, node.func.attr
        return None

    def infer_call(self, node: ast.Call) -> Optional[Inferred]:
        """
        Descriptor inference for a canonical flat call through the
        descriptor-inference families of the replacement registry
        (:class:`dace.frontend.common.op_repository.Replacements`).

        Queried in order: method inference for calls on data-bound objects,
        ufunc inference for NumPy universal functions, then free-function
        inference by qualified name.

        :return: The inferred result, or None if no registry entry matched
                 (the caller decides how to fall back).
        """
        from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
        arguments = self.call_arguments(node)
        if arguments is None:
            return None
        input_descs, args, kwargs = arguments

        # Method calls on data containers (a.sum(), a.copy(), ...)
        if isinstance(node.func, ast.Attribute):
            base = self._bound_descriptor_of(node.func.value)
            if base is not None:
                infer_fn = oprepo.Replacements.get_method_descriptor_inference(type(base), node.func.attr)
                if infer_fn is not None:
                    return self._registry_inference(infer_fn, base, *args, **kwargs)
                # No method-family inference for this (type, method) pair:
                # fall through to ufunc/free-function inference below rather
                # than hard-aborting the whole call (the base being data
                # doesn't rule out `node.func` resolving some other way, e.g.
                # a qualified free function embedded via a constant callee).

        qualname, callee = self.resolve_callee(node.func)

        # NumPy universal functions (np.add, np.sin, ...), direct or through
        # one of their reduce/accumulate/outer methods (np.add.reduce(...)).
        ufunc_form = self.resolve_ufunc_call(node)
        if ufunc_form is not None:
            ufunc, ufunc_method = ufunc_form
            infer_fn = oprepo.Replacements.get_ufunc_descriptor_inference(ufunc_method)
            if infer_fn is None:
                return None
            return self._registry_inference(infer_fn, input_descs, ufunc.__name__, *args, **kwargs)

        # Free functions by qualified name (numpy.zeros, numpy.sum, ...).
        # Fall back to the source-level name: the qualname preprocessing
        # attaches to embedded callee constants, or the textual call name.
        infer_fn = oprepo.Replacements.get_descriptor_inference(qualname)
        if infer_fn is None:
            textual_name = getattr(node.func, 'qualname', None) or astutils.rname(node.func)
            if textual_name != qualname:
                infer_fn = oprepo.Replacements.get_descriptor_inference(textual_name)
        if infer_fn is None:
            return None
        return self._registry_inference(infer_fn, input_descs, *args, **kwargs)

    def call_arguments(self, node: ast.Call) -> Optional[Tuple[Dict[str, data.Data], List[Any], Dict[str, Any]]]:
        """
        Resolve canonical call arguments to the replacement registry's
        inference convention: data operands are passed by name with their
        descriptors collected separately; constants, symbols, and static
        sequences are passed by value.

        :return: A 3-tuple of (input descriptors by name, positional argument
                 values, keyword argument values), or None if any argument
                 cannot be represented (e.g., references an opaque object).
        """
        input_descs: Dict[str, data.Data] = {}

        def convert(argument: ast.expr) -> Tuple[bool, Any]:
            try:
                inferred = self.infer(argument)
            except UnsupportedFeatureError:
                return False, None
            if inferred.is_pyobject:
                return False, None
            if inferred.is_data:
                try:
                    name = astutils.rname(argument)
                except TypeError:
                    # Data-valued compound expression (e.g. UnaryOp over an
                    # array) — cannot be passed to the registry by name
                    return False, None
                input_descs[name] = inferred.descriptor
                return True, name
            if inferred.kind in ('constant', 'symbolic'):
                return True, inferred.value
            if inferred.kind == 'static':
                # A static sequence's elements may themselves be data
                # containers (e.g. the ``(A, B)`` in ``numpy.concatenate((A,
                # B))``): each is passed through by name, exactly like a
                # top-level data argument, matching the classic frontend's
                # convention of passing lists of container names here.
                elements = []
                for element in inferred.value.elements:
                    ok, value = convert(element)
                    if not ok:
                        return False, None
                    elements.append(value)
                return True, tuple(elements) if inferred.value.kind == 'tuple' else elements
            return False, None

        args: List[Any] = []
        for argument in node.args:
            ok, value = convert(argument)
            if not ok:
                return None
            args.append(value)
        kwargs: Dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            ok, value = convert(keyword.value)
            if not ok:
                return None
            kwargs[keyword.arg] = value
        return input_descs, args, kwargs

    def _global_value(self, name: str) -> Optional[Any]:
        """Resolve a root name against the program globals, tolerating the
        module-name rewriting done by preprocessing (aliased imports appear
        under their real module names in the AST)."""
        if name in self.context.globals:
            return self.context.globals[name]
        for value in self.context.globals.values():
            if isinstance(value, types.ModuleType) and value.__name__ == name:
                return value
        return None

    def _bound_descriptor_of(self, node: ast.expr) -> Optional[data.Data]:
        """
        The container descriptor a canonical expression is bound to, if any.

        Beyond a bare ``Name`` (``a.sum()``), one more level is admitted --
        an indexed or attribute expression that itself infers to a data
        result, e.g. ``A[0].sum()`` or a structure member access -- so a
        method call on such an expression is at least *typed* (whether it
        can subsequently be *lowered* through the method-replacement family
        is a separate, stricter question decided at dispatch time, which
        only accepts a whole-container ``Name``/member receiver).
        """
        if isinstance(node, ast.Name):
            binding = self.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                return self.context.containers[binding.container]
            return None
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            try:
                inferred = self.infer(node)
            except UnsupportedFeatureError:
                return None
            if inferred.is_data:
                return inferred.descriptor
        return None

    def _registry_inference(self, infer_fn: Any, *args: Any, **kwargs: Any) -> Optional[Inferred]:
        """Invoke a registry inference function defensively and normalize its
        result. Only single-descriptor results are supported; multi-output
        inference results fall back."""
        try:
            result = infer_fn(*args, **kwargs)
        except Exception:
            return None
        if isinstance(result, data.Data):
            return Inferred(kind='data', descriptor=result)
        if isinstance(result, (tuple, list)) and len(result) == 1 and isinstance(result[0], data.Data):
            return Inferred(kind='data', descriptor=result[0])
        return None

    def parse_access(self, node: Union[ast.Name, ast.Subscript]) -> MemletExpr:
        """
        Parse a canonical data access (name or subscript of a name) into a
        memlet expression with an explicit subset, using the shared memlet
        parser.

        :raises UnsupportedFeatureError: If the shared parser cannot handle
            the access form. The parser fails in assorted ways on exotic
            indexing (advanced indexing, ``.flat``, long index tuples, ...);
            every failure at this boundary is a feature gap, not a crash.
        """
        try:
            return ParseMemlet(self._shim, self.context.defined_view(), self._restore_index_sequences(node))
        except UnsupportedFeatureError:
            raise
        except Exception as error:
            raise UnsupportedFeatureError(f'Unsupported access expression "{astutils.unparse(node)}": {error}',
                                          self.context.filename,
                                          node,
                                          category='memlet-parse')

    def _restore_index_sequences(self, node: Union[ast.Name, ast.Subscript]) -> Union[ast.Name, ast.Subscript]:
        """
        Put literal index sequences back into a subscript before parsing it.

        ``A[:, (1, 2, 3)]`` is an advanced-indexing access, but ANF hoists the
        tuple into a temporary bound to a compile-time sequence, leaving
        ``A[:, __anf0]``. The shared memlet parser recognizes an array index
        only from a literal or a registered container, so it would classify the
        temporary as a scalar symbol and silently produce a subset referring to
        a name that has no runtime value. Substituting the literal back keeps
        the two spellings equivalent, and is a no-op for every other access.

        The literal is re-encoded the way the parser expects it
        (``memlet_parser.py::_fill_missing_slices``): an ``ast.Name`` whose
        ``id`` is the Python list itself, which is what the classic frontend's
        global resolver leaves in the AST.
        """
        if not isinstance(node, ast.Subscript):
            return node
        replacements: Dict[str, ast.expr] = {}
        for name in {n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name)}:
            binding = self.context.resolve(name)
            if binding is None or binding.kind != 'static':
                continue
            sequence = self.context.static_values.get(name)
            if sequence is None:
                continue
            try:
                values = self.sequence_constants(sequence)
            except UnsupportedFeatureError:
                continue
            if not values or not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                continue
            replacements[name] = list(values)
        if not replacements:
            return node

        class _Substituter(ast.NodeTransformer):

            def visit_Name(self, name_node: ast.Name) -> ast.AST:
                if name_node.id not in replacements:
                    return name_node
                return ast.Name(id=list(replacements[name_node.id]), ctx=ast.Load())

        restored = copy.deepcopy(node)
        restored.slice = _Substituter().visit(restored.slice)
        return ast.copy_location(restored, node)

    def constant_int(self, node: ast.expr) -> Optional[int]:
        """Resolve a canonical atom to a compile-time integer, or None."""
        value = self.constant_value(node)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def constant_value(self, node: ast.expr) -> Any:
        """Resolve a canonical atom to a compile-time value, or None."""
        try:
            inferred = self.infer(node)
        except UnsupportedFeatureError:
            return None
        if inferred.kind == 'constant':
            return inferred.value
        return None

    def sequence_constants(self, sequence: StaticSequence) -> List[Any]:
        """
        Resolve all elements of a static sequence to compile-time values.

        :raises UnsupportedFeatureError: If any element is not a compile-time
            constant (e.g., references runtime data).
        """
        result = []
        for element in sequence.elements:
            value = self.constant_value(element)
            if value is None:
                raise UnsupportedFeatureError(
                    f'Python sequence element "{astutils.unparse(element)}" is not a compile-time constant',
                    self.context.filename,
                    element,
                    category='static-sequence')
            result.append(value)
        return result

    def sequence_descriptor(self, sequence: StaticSequence) -> data.Array:
        """The constant-array descriptor a static sequence materializes to."""
        array = numpy.array(self.sequence_constants(sequence))
        return data.Array(dtypes.dtype_to_typeclass(array.dtype.type), list(array.shape))

    # ------------------------------------------------------------------ #

    def _infer_name(self, node: ast.Name) -> Inferred:
        binding = self.context.resolve(node.id)
        if binding is not None:
            if binding.kind == 'container':
                # Materialized ANF scalar temps with a known pure-symbolic
                # value stay symbolic to inference, so computed shape
                # expressions and derived temps keep compile-time values.
                symbolic_value = self.context.symbolic_scalar_values.get(binding.container)
                if symbolic_value is not None:
                    return Inferred(kind='symbolic', value=symbolic_value)
                return Inferred(kind='data', descriptor=self.context.containers[binding.container])
            if binding.kind == 'symbol':
                return Inferred(kind='symbolic', value=self.context.symbols[node.id])
            if binding.kind == 'static':
                return Inferred(kind='static', value=self.context.static_values[node.id])
            if binding.kind == 'constant':
                return Inferred(kind='constant', value=self.context.constant_values[node.id])
        if node.id in self.context.symbols:
            return Inferred(kind='symbolic', value=self.context.symbols[node.id])
        if node.id in self.context.constants:
            return Inferred(kind='constant', value=self.context.constants[node.id][1])
        if node.id in self.context.globals:
            value = self.context.globals[node.id]
            if isinstance(value, symbolic.symbol):
                return Inferred(kind='symbolic', value=value)
            return Inferred(kind='constant', value=value)
        raise UnsupportedFeatureError(f'Use of undefined name "{node.id}"',
                                      self.context.filename,
                                      node,
                                      category='undefined-name')

    def _infer_attribute(self, node: ast.Attribute) -> Inferred:
        """Infer a structure member access (``tracers.data``) or an attribute
        chain resolving through the program globals to a compile-time value
        (``dace.int32``). Any other attribute read is a feature gap that
        degrades to the interpreter."""
        if isinstance(node.value, ast.Name):
            member = self.context.member_access_of(node.value.id, node.attr)
            if member is not None:
                return Inferred(kind='data', descriptor=member[1])
            binding = self.context.resolve(node.value.id)
            # Attribute on a compile-time constant value (e.g. an enum member
            # of a constant-bound enum class)
            if binding is not None and binding.kind == 'constant':
                base_value = self.context.constant_values[node.value.id]
                if hasattr(base_value, node.attr):
                    return Inferred(kind='constant', value=getattr(base_value, node.attr))
            # Compile-time descriptor properties of data-bound names
            # (``A.dtype``, ``A.shape``, ``A.ndim``)
            if binding is not None and binding.kind == 'container':
                descriptor = self.context.containers[binding.container]
                if node.attr == 'dtype':
                    return Inferred(kind='constant', value=descriptor.dtype)
                if node.attr == 'shape' and isinstance(descriptor, data.Array):
                    return Inferred(kind='constant', value=tuple(descriptor.shape))
                if node.attr == 'ndim' and isinstance(descriptor, data.Array):
                    return Inferred(kind='constant', value=len(descriptor.shape))
                # Registry-backed attributes needing an actual data operation
                # (``.T``/``.real``/``.imag``/``.flat``): typed through the
                # ATTRIBUTE family of the replacement registry so
                # ``lowering.dispatch``'s dedicated frontend paths (mirroring
                # the reshape view precedent, ``dispatch._lower_reshape_call``)
                # can materialize them. Scoped to ``SUPPORTED_DATA_ATTRIBUTES``
                # -- see its docstring for why a registered attribute outside
                # that set must keep failing inference here rather than typing
                # successfully with no matching lowering path.
                if node.attr in SUPPORTED_DATA_ATTRIBUTES:
                    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population
                    infer_fn = oprepo.Replacements.get_attribute_descriptor_inference(type(descriptor), node.attr)
                    if infer_fn is not None:
                        try:
                            result = infer_fn(descriptor)
                        except Exception:
                            result = None
                        if isinstance(result, (tuple, list)) and len(result) == 1:
                            result = result[0]
                        if isinstance(result, data.Data):
                            return Inferred(kind='data', descriptor=result)
        _, resolved = self.resolve_callee(node)
        if resolved is not None:
            if isinstance(resolved, symbolic.symbol):
                return Inferred(kind='symbolic', value=resolved)
            return Inferred(kind='constant', value=resolved)
        raise UnsupportedFeatureError(f'Cannot infer type of expression: {astutils.unparse(node)}',
                                      self.context.filename,
                                      node,
                                      category='type-inference')

    def _infer_subscript(self, node: ast.Subscript) -> Inferred:
        base = self.infer(node.value)
        if base.kind == 'static':
            element = values.fold_subscript(base.value, node, self.constant_int)
            if isinstance(element, ast.expr):
                return self.infer(element)
            return Inferred(kind='static', value=element)
        if not base.is_data:
            raise UnsupportedFeatureError('Subscript of a non-container value',
                                          self.context.filename,
                                          node,
                                          category='type-inference')
        expr = self.parse_access(node)
        # Array-read indices (``x[A_col[j]]``) are NOT rejected here: the
        # shared memlet parser represents them as applied sympy functions,
        # whose ``.size()`` still computes a definite shape, and the
        # elementwise computation mechanism now lowers this pattern as
        # indirection (see ``lowering.access.indirect_index_reads``). A
        # consumer that cannot handle indirection (e.g. an assignment target,
        # or a ufunc/creation-call argument) re-resolves the same expression
        # through ``resolve_access``, which keeps rejecting it there.
        if expr.arrdims:
            # Advanced (array-valued) indexing: the result shape follows NumPy's
            # own rules, not the subset's -- index arrays broadcast together and
            # collapse the indexed dimensions into one chunk.
            from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing
            shape = [s for s in advanced_indexing.output_shape(expr, self.context, self, node) if s != 1]
            if not shape:
                return Inferred(kind='data', descriptor=data.Scalar(base.descriptor.dtype))
            return Inferred(kind='data', descriptor=data.Array(base.descriptor.dtype, shape))
        try:
            shape = [s for s in expr.subset.size() if s != 1]
            if not shape:
                return Inferred(kind='data', descriptor=data.Scalar(base.descriptor.dtype))
            return Inferred(kind='data', descriptor=data.Array(base.descriptor.dtype, shape))
        except UnsupportedFeatureError:
            raise
        except Exception as error:
            # Exotic subsets produce non-real symbolic sizes (e.g. sympy zoo)
            # that crash descriptor validation; treat them as feature gaps.
            raise UnsupportedFeatureError(f'Cannot infer subscript shape of "{astutils.unparse(node)}": {error}',
                                          self.context.filename,
                                          node,
                                          category='data-dependent-subscript')

    def dtype_of(self, inferred: Inferred) -> Optional[dtypes.typeclass]:
        """
        Context-aware dtype of an inferred value. Symbolic expressions resolve
        their symbols' dtypes by name through the context: sympy symbol
        identity ignores the dace dtype attribute, so the objects embedded in
        an expression may carry stale defaults from the process-wide cache.
        """
        if inferred.kind == 'symbolic':
            return self.symbolic_dtype(inferred.value)
        return inferred.dtype

    def symbolic_dtype(self, expression: Any) -> dtypes.typeclass:
        """The result dtype of a pure symbolic expression, resolving each free
        symbol by name through the context (with NumPy promotion when the
        symbol dtypes differ)."""
        found: List[dtypes.typeclass] = []
        for free_symbol in getattr(expression, 'free_symbols', ()):
            name = str(free_symbol)
            registered = self.context.symbols.get(name)
            if registered is None:
                global_value = self.context.globals.get(name)
                if isinstance(global_value, symbolic.symbol):
                    registered = global_value
            candidate = registered if registered is not None else free_symbol
            found.append(getattr(candidate, 'dtype', symbolic.DEFAULT_SYMBOL_TYPE))
        if not found:
            return symbolic.DEFAULT_SYMBOL_TYPE
        return dtypes.result_type_of(found[0], *found[1:]) if len(found) > 1 else found[0]

    def _symbolic_expression(self, node: ast.expr) -> Any:
        """
        Build the symbolic value of a purely symbolic/constant expression.

        Parsing the source text mints fresh default-typed symbols for every
        name, so free symbols are substituted with their known context values:
        the recorded symbolic values of materialized ANF scalar temps, and the
        registered (correctly typed) symbol objects for program symbols.
        """
        expr = symbolic.pystr_to_symbolic(astutils.unparse(node))
        if not hasattr(expr, 'free_symbols'):
            return expr
        replacements = {}
        for free_symbol in expr.free_symbols:
            name = str(free_symbol)
            binding = self.context.resolve(name)
            if binding is not None and binding.kind == 'container':
                value = self.context.symbolic_scalar_values.get(binding.container)
                if value is not None:
                    replacements[free_symbol] = value
                    continue
            registered = self.context.symbols.get(name)
            if registered is None:
                # Symbols of inlined callees resolve through their globals
                global_value = self.context.globals.get(name)
                if isinstance(global_value, symbolic.symbol):
                    registered = global_value
            if registered is not None and registered is not free_symbol:
                replacements[free_symbol] = registered
        return expr.subs(replacements) if replacements else expr

    def _infer_operator(self, node: ast.expr) -> Inferred:
        if isinstance(node, ast.BinOp):
            operands = [self.infer(node.left), self.infer(node.right)]
        elif isinstance(node, ast.Compare):
            operands = [self.infer(node.left)] + [self.infer(c) for c in node.comparators]
        else:  # BoolOp
            operands = [self.infer(v) for v in node.values]

        # Operators over Python sequences follow Python semantics at compile
        # time; sequences mixed with data operands materialize as constant
        # arrays and participate in broadcasting instead.
        if isinstance(node, ast.BinOp) and any(op.kind == 'static' for op in operands):
            if not any(op.is_data for op in operands):
                left_sequence = operands[0].value if operands[0].kind == 'static' else None
                right_sequence = operands[1].value if operands[1].kind == 'static' else None
                return Inferred(kind='static',
                                value=values.fold_binop(node, left_sequence, right_sequence, self.constant_int))
            operands = [
                Inferred(kind='data', descriptor=self.sequence_descriptor(op.value)) if op.kind == 'static' else op
                for op in operands
            ]

        # Opaque Python objects poison the expression: consuming them requires
        # the interpreter, which the dispatch seam turns into a callback.
        if any(op.is_pyobject for op in operands):
            return Inferred(kind='data', descriptor=data.Scalar(dtypes.pyobject()))

        boolean_result = isinstance(node, _BOOLEAN_OPS)
        data_operands = [op for op in operands if op.is_data]
        if not data_operands:
            # Purely symbolic/constant expression
            return Inferred(kind='symbolic', value=self._symbolic_expression(node))

        result_dtype = self._result_dtype(operands, boolean_result)
        shape: Tuple[Any, ...] = ()
        for operand in data_operands:
            operand_shape = tuple(operand.descriptor.shape) if isinstance(operand.descriptor, data.Array) else ()
            shape = broadcast_shapes(shape, operand_shape)
        if not shape or all(s == 1 for s in shape):
            return Inferred(kind='data', descriptor=data.Scalar(result_dtype))
        return Inferred(kind='data', descriptor=data.Array(result_dtype, list(shape)))

    def _result_dtype(self, operands: List[Inferred], boolean_result: bool) -> dtypes.typeclass:
        if boolean_result:
            return dtypes.bool_
        known = [dtype for dtype in (self.dtype_of(op) for op in operands) if dtype is not None]
        if not known:
            raise UnsupportedFeatureError('Cannot determine operator result type',
                                          self.context.filename,
                                          category='type-inference')
        return dtypes.result_type_of(known[0], *known[1:]) if len(known) > 1 else known[0]

    def _demote_to_bool(self, operand: Inferred) -> Inferred:
        if operand.is_data:
            descriptor = operand.descriptor
            if isinstance(descriptor, data.Array):
                return Inferred(kind='data', descriptor=data.Array(dtypes.bool_, list(descriptor.shape)))
            return Inferred(kind='data', descriptor=data.Scalar(dtypes.bool_))
        return operand
