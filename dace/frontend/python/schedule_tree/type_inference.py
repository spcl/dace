# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Static type inference helpers for the direct Python schedule-tree frontend."""

import ast
import collections.abc as cabc
import copy
import inspect
import numbers
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Iterable as TypingIterable, Iterator as TypingIterator, List, Optional, Sequence, Tuple, \
    get_args, get_origin

from dace import data, dtypes, symbolic, subsets
from dace.data.pydata import PythonClass, PythonDict, PythonList, PythonTuple
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils, memlet_parser
from dace.frontend.python.schedule_tree.array_literal_support import infer_array_literal_descriptor
from dace.frontend.python.schedule_tree.dict_support import DictSupportContext, DictSupportLibrary, StaticDictBinding
from dace.frontend.python.schedule_tree.match_support import UnsupportedMatchPatternError, lower_match_to_statements
from dace.frontend.python.schedule_tree.structure_support import bind_target_structure, descriptor_from_structure, \
    direct_class_annotation_type, member_descriptor, nested_direct_class_owner, \
    python_class_requirement_for_member_assignment
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.sdfg.type_inference import infer_expr_type


@dataclass
class _Binding:
    descriptor: Optional[data.Data]
    kind: str = 'value'
    structure: Optional[Any] = None


def _clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


def _clone_binding(binding: _Binding) -> _Binding:
    descriptor = _clone_descriptor(binding.descriptor) if binding.descriptor is not None else None
    return _Binding(descriptor=descriptor, kind=binding.kind, structure=copy.deepcopy(binding.structure))


def _normalize_inferred_structure(result: Any) -> Optional[Any]:
    if isinstance(result, data.Data):
        descriptor = _clone_descriptor(result)
        descriptor.transient = True
        return descriptor
    if isinstance(result, tuple):
        elements = []
        for element in result:
            normalized = _normalize_inferred_structure(element)
            if normalized is None and element is not None:
                return None
            elements.append(normalized)
        return tuple(elements)
    if isinstance(result, list):
        elements = []
        for element in result:
            normalized = _normalize_inferred_structure(element)
            if normalized is None and element is not None:
                return None
            elements.append(normalized)
        return elements
    return None


def _binding_from_inference_result(result: Any) -> Optional[_Binding]:
    if result is None:
        return None

    if isinstance(result, data.Data):
        descriptor = _clone_descriptor(result)
        descriptor.transient = True
        kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
        structure = descriptor if isinstance(descriptor, data.Scalar) else None
        return _Binding(descriptor=descriptor, kind=kind, structure=structure)

    structure = _normalize_inferred_structure(result)
    if structure is None:
        return None
    if isinstance(structure, (tuple, list)) and len(structure) == 0:
        return _Binding(descriptor=None, kind='value', structure=structure)

    descriptor = descriptor_from_structure(structure)
    if descriptor is None:
        return None
    descriptor.transient = True
    return _Binding(descriptor=descriptor, kind='container', structure=structure)


def _resolve_ufunc_inference_target(node: ast.Call, env: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    func_value = try_resolve_static_value(node.func, env)
    if isinstance(func_value, np.ufunc):
        return 'ufunc', func_value.__name__

    if not isinstance(node.func, ast.Attribute):
        return None

    owner_value = try_resolve_static_value(node.func.value, env)
    if isinstance(owner_value, np.ufunc) and node.func.attr in {'reduce', 'accumulate', 'outer'}:
        return node.func.attr, owner_value.__name__

    return None


def _unparse(node: ast.AST) -> str:
    return astutils.unparse(node)


def _normalize_dtype(dtype: Any) -> Optional[dtypes.typeclass]:
    if isinstance(dtype, dtypes.typeclass):
        return dtype
    if isinstance(dtype, data.Data):
        return dtype.dtype
    if dtype in (int, float, complex, bool):
        return dtypes.typeclass(dtype)
    if dtype is str:
        return dtypes.string
    try:
        return dtypes.typeclass(dtype)
    except (KeyError, TypeError, ValueError):
        return None


def _pyobject_scalar_descriptor() -> data.Scalar:
    return data.Scalar(dtypes.pyobject(), transient=True)


def _string_scalar_descriptor() -> data.Scalar:
    return data.Scalar(dtypes.string, transient=True)


def _is_scalar_subscript(node: ast.Subscript, subset: subsets.Range, new_axes: Sequence[int],
                         arrdims: Dict[int, str]) -> bool:
    if new_axes or arrdims:
        return False
    if isinstance(node.slice, ast.Slice):
        return False
    if isinstance(node.slice, ast.Tuple):
        for element in node.slice.elts:
            if isinstance(element, ast.Slice):
                return False
            if isinstance(element, ast.Constant) and (element.value is None or element.value is Ellipsis):
                return False
    for (start, end, step), tile in zip(subset.ranges, subset.tile_sizes):
        if tile != 1 or step != 1 or start != end:
            return False
    return True


def _infer_static_subscript_descriptor(descriptor: data.Data, node: ast.Subscript,
                                       evaluation_context: Dict[str, Any]) -> Optional[data.Data]:
    if not hasattr(descriptor, 'shape') or not hasattr(descriptor, 'dtype'):
        return None

    index_value = try_resolve_static_value(node.slice, evaluation_context)
    if index_value is UNRESOLVED:
        return None

    result_shape = _infer_static_subscript_shape(tuple(descriptor.shape), index_value)
    if result_shape is None:
        return None
    if not result_shape:
        return data.Scalar(descriptor.dtype, transient=True)
    return data.Array(descriptor.dtype, list(result_shape), transient=True)


def _infer_static_subscript_shape(array_shape: Tuple[Any, ...], index_value: Any) -> Optional[Tuple[Any, ...]]:
    expanded = _expand_static_indices(index_value, len(array_shape))
    if expanded is None:
        return None

    chunks: List[Any] = []
    advanced_shapes: List[Tuple[int, ...]] = []
    advanced_groups = 0
    in_advanced_group = False
    array_dim = 0

    for index in expanded:
        if index is None:
            chunks.append((1, ))
            in_advanced_group = False
            continue

        if array_dim >= len(array_shape):
            return None

        if _is_static_integer_index(index):
            array_dim += 1
            in_advanced_group = False
            continue

        advanced_shape = _static_advanced_index_shape(index)
        if advanced_shape is not None:
            advanced_shapes.append(advanced_shape)
            if not in_advanced_group:
                chunks.append('ADV')
                advanced_groups += 1
                in_advanced_group = True
            array_dim += 1
            continue

        if not isinstance(index, slice):
            return None

        slice_dim = _static_slice_result_dim(array_shape[array_dim], index)
        if slice_dim is None:
            return None
        chunks.append((slice_dim, ))
        array_dim += 1
        in_advanced_group = False

    while array_dim < len(array_shape):
        chunks.append((array_shape[array_dim], ))
        array_dim += 1

    if not advanced_shapes:
        return tuple(dim for chunk in chunks for dim in chunk)

    broadcast_shape = _broadcast_static_shapes(advanced_shapes)
    if broadcast_shape is None:
        return None

    if advanced_groups == 1:
        output_shape: List[Any] = []
        inserted = False
        for chunk in chunks:
            if chunk == 'ADV':
                if not inserted:
                    output_shape.extend(broadcast_shape)
                    inserted = True
                continue
            output_shape.extend(chunk)
        return tuple(output_shape)

    output_shape = list(broadcast_shape)
    for chunk in chunks:
        if chunk == 'ADV':
            continue
        output_shape.extend(chunk)
    return tuple(output_shape)


def _expand_static_indices(index_value: Any, rank: int) -> Optional[List[Any]]:
    indices = list(index_value) if isinstance(index_value, tuple) else [index_value]
    if sum(1 for index in indices if index is Ellipsis) > 1:
        return None

    consumed = sum(1 for index in indices if index is not None and index is not Ellipsis)
    expanded: List[Any] = []
    ellipsis_seen = False
    for index in indices:
        if index is Ellipsis:
            ellipsis_seen = True
            expanded.extend([slice(None)] * max(rank - consumed, 0))
            continue
        expanded.append(index)

    if not ellipsis_seen:
        expanded.extend([slice(None)] * max(rank - consumed, 0))

    return expanded


def _is_static_integer_index(index: Any) -> bool:
    return isinstance(index, numbers.Integral) and not isinstance(index, bool)


def _static_advanced_index_shape(index: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(index, np.ndarray):
        if index.ndim == 0 or index.dtype == bool:
            return None
        return tuple(index.shape)

    if isinstance(index, list):
        return _static_nested_sequence_shape(index)

    if isinstance(index, tuple):
        nested_shape = _static_nested_sequence_shape(list(index))
        if nested_shape is None:
            return None
        return nested_shape

    return None


def _static_nested_sequence_shape(value: List[Any]) -> Optional[Tuple[int, ...]]:
    if not value:
        return (0, )
    first = value[0]
    if isinstance(first, (list, tuple)):
        inner_shape = _static_nested_sequence_shape(list(first))
        if inner_shape is None:
            return None
        for element in value[1:]:
            if not isinstance(element, (list, tuple)):
                return None
            if _static_nested_sequence_shape(list(element)) != inner_shape:
                return None
        return (len(value), ) + inner_shape

    if any(isinstance(element, (list, tuple)) for element in value[1:]):
        return None
    if any(not _is_static_integer_index(element) for element in value):
        return None
    return (len(value), )


def _static_slice_result_dim(dim_size: Any, index: slice) -> Optional[Any]:
    if index == slice(None):
        return dim_size

    step = 1 if index.step is None else index.step
    try:
        if step == 0:
            return None
    except TypeError:
        pass

    step_is_negative = (step < 0) == True
    step_is_positive = (step > 0) == True
    if not step_is_negative and not step_is_positive:
        return None

    if index.start is None:
        start = dim_size - 1 if step_is_negative else 0
    else:
        start = index.start

    if index.stop is None:
        stop = -1 if step_is_negative else dim_size
    else:
        stop = index.stop

    try:
        if (start < 0) == True:
            start += dim_size
    except TypeError:
        pass
    try:
        if (stop < 0) == True:
            stop += dim_size
    except TypeError:
        pass

    end = stop + 1 if step_is_negative else stop - 1
    return subsets.Range([(start, end, step)]).size()[0]


def _broadcast_static_shapes(shapes: Sequence[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    result: List[int] = []
    max_rank = max(len(shape) for shape in shapes)
    for axis in range(max_rank):
        axis_sizes = []
        for shape in shapes:
            offset = axis - (max_rank - len(shape))
            axis_sizes.append(1 if offset < 0 else shape[offset])
        size = max(axis_sizes)
        if any(axis_size not in {1, size} for axis_size in axis_sizes):
            return None
        result.append(size)
    return tuple(result)


def _should_fallback_to_pyobject_scalar(node: ast.AST, value: Any = UNRESOLVED) -> bool:
    if value is None or isinstance(value, (str, bytes, numbers.Number, bool, type(Ellipsis))):
        return False
    return isinstance(node, (ast.Await, ast.Attribute, ast.BinOp, ast.BoolOp, ast.Call, ast.Compare, ast.FormattedValue,
                             ast.IfExp, ast.JoinedStr, ast.Name, ast.NamedExpr, ast.UnaryOp, ast.Yield, ast.YieldFrom))


class ScheduleTreeTypeInference(ast.NodeVisitor):
    """Conservative binding inference for the direct schedule-tree frontend."""

    def __init__(self, globals_env: Dict[str, Any], argtypes: Dict[str, data.Data]) -> None:
        self.globals = copy.copy(globals_env)
        self.dict_support = DictSupportLibrary()
        self.bindings: Dict[str, _Binding] = {
            name: _Binding(descriptor=_clone_descriptor(descriptor), kind='container')
            for name, descriptor in argtypes.items()
        }
        self.results: Dict[str, _Binding] = {}
        self.annotated_class_types: Dict[str, type[Any]] = {}

    def infer(self, program: ast.AST) -> Dict[str, _Binding]:
        if isinstance(program, ast.Module):
            program = program.body[0] if program.body else None
        if not isinstance(program, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {}
        self._initialize_direct_class_annotations(program)
        for stmt in program.body:
            self.visit(stmt)
        return {name: _clone_binding(binding) for name, binding in self.results.items()}

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._infer_assignment(target, node.value, None)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        annotated_descriptor = self._evaluate_descriptor(node.annotation)
        class_type = self._evaluate_annotation_class_type(node.annotation)
        if class_type is not None and isinstance(node.target, ast.Name):
            self.annotated_class_types[node.target.id] = class_type
        if node.value is None:
            if isinstance(node.target, ast.Name) and annotated_descriptor is not None:
                self._store_binding(node.target.id, annotated_descriptor)
            return
        self._infer_assignment(node.target, node.value, annotated_descriptor)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id not in self.bindings:
            scalar_descriptor = self._infer_scalar_descriptor(node.value, None)
            if scalar_descriptor is not None:
                self._store_binding(node.target.id, scalar_descriptor)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._apply_method_self_descriptor_side_effect(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._bind_loop_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        before = {name: _clone_binding(binding) for name, binding in self.bindings.items()}
        then_bindings = self._visit_branch(node.body, before)
        else_bindings = self._visit_branch(node.orelse, before)
        self._merge_branch_bindings(before, then_bindings, else_bindings)

    def visit_Match(self, node: ast.Match) -> None:
        try:
            lowered = lower_match_to_statements(node, astutils.copy_tree(node.subject))
        except UnsupportedMatchPatternError:
            return
        for stmt in lowered:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested function-local bindings must not leak into the enclosing
        # schedule-tree type-inference scope.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def _visit_branch(self, body: Sequence[ast.AST], initial: Dict[str, _Binding]) -> Dict[str, _Binding]:
        previous = self.bindings
        self.bindings = {name: _clone_binding(binding) for name, binding in initial.items()}
        try:
            for stmt in body:
                self.visit(stmt)
            return {name: _clone_binding(binding) for name, binding in self.bindings.items()}
        finally:
            self.bindings = previous

    def _merge_branch_bindings(self, before: Dict[str, _Binding], then_bindings: Dict[str, _Binding],
                               else_bindings: Dict[str, _Binding]) -> None:
        merged = {name: _clone_binding(binding) for name, binding in before.items()}
        candidate_names = (set(then_bindings.keys()) | set(else_bindings.keys())) - set(before.keys())
        for name in candidate_names:
            left = then_bindings.get(name)
            right = else_bindings.get(name)
            if left is None or right is None:
                continue
            if self._compatible_bindings(left, right):
                merged[name] = _clone_binding(left)
                self.results[name] = _clone_binding(left)
        self.bindings = merged

    def _compatible_bindings(self, left: _Binding, right: _Binding) -> bool:
        if left.kind != right.kind:
            return False
        if not self._compatible_descriptors(left.descriptor, right.descriptor):
            return False
        return self._compatible_structures(left.structure, right.structure)

    def _compatible_descriptors(self, left: Optional[data.Data], right: Optional[data.Data]) -> bool:
        if left is None or right is None:
            return left is right
        if type(left) is not type(right):
            return False
        if hasattr(left, 'is_equivalent'):
            return left.is_equivalent(right)
        return left == right

    def _compatible_structures(self, left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is right
        if isinstance(left, StaticDictBinding) and isinstance(right, StaticDictBinding):
            if set(left.entries.keys()) != set(right.entries.keys()):
                return False
            return all(self._compatible_descriptors(left.entries[key], right.entries[key]) for key in left.entries)
        if isinstance(left, data.Data) and isinstance(right, data.Data):
            return self._compatible_descriptors(left, right)
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            return all(self._compatible_structures(lval, rval) for lval, rval in zip(left, right))
        if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
            return all(self._compatible_structures(lval, rval) for lval, rval in zip(left, right))
        return False

    def _infer_assignment(self, target: ast.AST, value: ast.AST, annotated_descriptor: Optional[data.Data]) -> None:
        binding = self._infer_binding(value, annotated_descriptor)
        if binding is not None:
            if isinstance(target, ast.Name):
                self._store_binding(target.id, binding.descriptor, kind=binding.kind, structure=binding.structure)
                return
            if isinstance(target, (ast.Tuple, ast.List)) and binding.structure is not None:
                self._bind_target_structure(target, binding.structure)
        self._ensure_pythonclass_for_direct_class_annotation(target)
        self._update_dict_subscript_binding(target, value)

    def _ensure_pythonclass_for_direct_class_annotation(self, target: ast.AST) -> None:
        if not isinstance(target, ast.Attribute):
            return

        owner_binding = self._resolve_binding(target.value)
        if owner_binding is None or owner_binding.descriptor is None:
            return
        if python_class_requirement_for_member_assignment(owner_binding.descriptor, target.attr) is None:
            return

        root = self._attribute_root_and_members(target.value)
        if root is None:
            return
        root_name, member_names = root
        class_type = self.annotated_class_types.get(root_name)
        if class_type is None:
            return
        if nested_direct_class_owner(class_type, member_names) is None:
            return

        binding = self.bindings.get(root_name)
        if binding is None or binding.descriptor is None or isinstance(binding.descriptor, PythonClass):
            return

        try:
            python_class_descriptor = PythonClass.from_class(class_type)
        except (TypeError, ValueError):
            return

        self._store_binding(root_name, python_class_descriptor, kind=binding.kind)

    def _update_dict_subscript_binding(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            return
        binding = self.bindings.get(target.value.id)
        if binding is None or binding.descriptor is None:
            return
        dict_binding = binding.structure if isinstance(binding.structure, StaticDictBinding) else None
        updated = self.dict_support.infer_assignment_binding(self._dict_support_context(), binding.descriptor,
                                                             dict_binding, target.slice, value)
        if updated is None:
            return
        updated_descriptor, updated_binding = updated
        self._store_binding(target.value.id,
                            updated_descriptor,
                            kind=binding.kind,
                            structure=updated_binding if updated_binding is not None else None)

    def _infer_binding(self, value: ast.AST, annotated_descriptor: Optional[data.Data]) -> Optional[_Binding]:
        binding = self._infer_internal_iterator_binding(value)
        if binding is not None:
            return binding

        binding = self._resolve_binding(value)
        if binding is not None:
            return binding

        if isinstance(value, ast.Dict):
            descriptor = self.dict_support.infer_literal_descriptor(self._dict_support_context(), value)
            structure = self.dict_support.infer_literal_binding(self._dict_support_context(), value)
            return _Binding(descriptor=descriptor, kind='container', structure=structure)

        replacement_binding = self._try_replacement_binding_inference(value)
        if replacement_binding is not None:
            return replacement_binding

        inferred_descriptor = self._infer_descriptor(value)
        if inferred_descriptor is not None:
            kind = 'scalar' if isinstance(inferred_descriptor, data.Scalar) else 'container'
            structure = inferred_descriptor if isinstance(inferred_descriptor, data.Scalar) else None
            return _Binding(descriptor=inferred_descriptor, kind=kind, structure=structure)

        scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
        if scalar_descriptor is not None:
            return _Binding(descriptor=scalar_descriptor, kind='scalar', structure=scalar_descriptor)

        return None

    def _resolve_binding(self, value: ast.AST) -> Optional[_Binding]:
        if isinstance(value, ast.Name) and value.id in self.bindings:
            return _clone_binding(self.bindings[value.id])

        if isinstance(value, ast.Attribute):
            base_binding = self._resolve_binding(value.value)
            if base_binding is None or base_binding.descriptor is None:
                return None
            descriptor = member_descriptor(base_binding.descriptor, value.attr)
            if descriptor is None:
                return None
            kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
            structure = descriptor if isinstance(descriptor, data.Scalar) else None
            return _Binding(descriptor=descriptor, kind=kind, structure=structure)

        if isinstance(value, ast.Subscript):
            binding = self._resolve_binding(value.value)
            if binding is None or binding.descriptor is None:
                return None
            if isinstance(binding.descriptor, PythonDict):
                descriptor = self.dict_support.infer_subscript_descriptor(
                    self._dict_support_context(), binding.descriptor, value.slice,
                    binding.structure if isinstance(binding.structure, StaticDictBinding) else None)
                if descriptor is None:
                    return None
                kind = 'scalar' if isinstance(descriptor, data.Scalar) else binding.kind
                structure = descriptor if isinstance(descriptor, data.Scalar) else None
                return _Binding(descriptor=descriptor, kind=kind, structure=structure)
            structure = self._subscript_structure(binding, value.slice)
            descriptor = descriptor_from_structure(structure) if structure is not None else None
            if descriptor is None:
                descriptor = self._subscript_descriptor(binding.descriptor, value)
            if descriptor is None:
                return None
            kind = 'scalar' if isinstance(descriptor, data.Scalar) else binding.kind
            return _Binding(descriptor=descriptor, kind=kind, structure=structure)

        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'reshape':
            base_binding = self._resolve_binding(value.func.value)
            if base_binding is None or base_binding.descriptor is None:
                return None
            shape = self._parse_shape(value.args[0]) if value.args else list(base_binding.descriptor.shape)
            return _Binding(descriptor=self._make_view_descriptor(base_binding.descriptor, shape), kind='container')

        if isinstance(value, (ast.Tuple, ast.List)):
            structure = self._structure_from_expression(value)
            if structure is None:
                return None
            descriptor = descriptor_from_structure(structure)
            if descriptor is None:
                return None
            kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
            return _Binding(descriptor=descriptor, kind=kind, structure=structure)

        return None

    def _infer_internal_iterator_binding(self, value: ast.AST) -> Optional[_Binding]:
        if not isinstance(value, ast.Call):
            return None
        helper_name = astutils.rname(value.func)
        if helper_name == '__dace_iterator_init' and value.args:
            structure = self._infer_iterable_structure(value.args[0])
            if structure is None:
                return None
            return _Binding(descriptor=descriptor_from_structure(structure), kind='iterator', structure=structure)
        if helper_name == '__dace_iterator_next' and value.args and isinstance(value.args[0], ast.Name):
            iterator_binding = self.bindings.get(value.args[0].id)
            if iterator_binding is None or iterator_binding.structure is None:
                return None
            structure = (data.Scalar(dtypes.bool, transient=True), copy.deepcopy(iterator_binding.structure))
            return _Binding(descriptor=descriptor_from_structure(structure), kind='iterator-value', structure=structure)
        return None

    def _infer_iterable_structure(self, node: ast.AST, env: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        env = env or self._evaluation_context()

        if isinstance(node, ast.Call):
            call_name = astutils.rname(node.func)
            if call_name == 'dace.nounroll' and node.args:
                return self._infer_iterable_structure(node.args[0], env)
            if call_name == 'zip' and node.args:
                elements = [self._infer_iterable_structure(arg, env) for arg in node.args]
                if any(element is None for element in elements):
                    return None
                return tuple(elements)
            if call_name == 'enumerate' and node.args:
                inner = self._infer_iterable_structure(node.args[0], env)
                if inner is None:
                    return None
                return (data.Scalar(dtypes.int64, transient=True), inner)
            if call_name == 'iter' and node.args:
                return self._infer_iterable_structure(node.args[0], env)
            return None

        if isinstance(node, ast.Name) and node.id in self.bindings:
            return self._element_structure_from_binding(self.bindings[node.id])

        value = self._safe_eval(node, env)
        if value is None:
            return None
        return self._infer_iterable_structure_from_value(value)

    def _infer_iterable_structure_from_value(self, value: Any) -> Optional[Any]:
        if dtypes.is_array(value):
            descriptor = _clone_descriptor(data.create_datadescriptor(value))
            descriptor.transient = True
            return self._element_structure_from_descriptor(descriptor)

        if isinstance(value, (list, tuple)):
            if not value:
                return None
            structures = [self._structure_from_value(element) for element in value]
            return self._merge_structures(structures)

        structure = self._structure_from_iterator_annotation(value, '__iter__', returns_iterator=True)
        if structure is not None:
            return structure
        structure = self._structure_from_iterator_method(value, '__iter__', returns_iterator=True)
        if structure is not None:
            return structure
        structure = self._structure_from_iterator_annotation(value, '__next__', returns_iterator=False)
        if structure is not None:
            return structure
        return self._structure_from_iterator_method(value, '__next__', returns_iterator=False)

    def _structure_from_iterator_annotation(self, value: Any, method_name: str, *,
                                            returns_iterator: bool) -> Optional[Any]:
        method = getattr(type(value), method_name, None)
        if method is None:
            return None
        try:
            annotation = inspect.signature(method).return_annotation
        except (TypeError, ValueError):
            return None
        if annotation is inspect.Signature.empty:
            return None
        return self._structure_from_annotation(annotation, returns_iterator=returns_iterator)

    def _structure_from_iterator_method(self, value: Any, method_name: str, *, returns_iterator: bool) -> Optional[Any]:
        method = getattr(type(value), method_name, None)
        if method is None:
            return None

        try:
            method_ast, _, _, _ = astutils.function_to_ast(method)
        except TypeError:
            return None

        if not method_ast.body or not isinstance(method_ast.body[0], ast.FunctionDef):
            return None

        function_node = method_ast.body[0]
        env = copy.copy(getattr(method, '__globals__', {}))
        env.update(self.globals)
        if function_node.args.args:
            env[function_node.args.args[0].arg] = value

        local_values: Dict[str, Any] = {}
        for stmt in function_node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                evaluated = self._safe_eval(stmt.value, {**env, **local_values})
                if evaluated is not None:
                    local_values[stmt.targets[0].id] = evaluated
                continue
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                merged_env = {**env, **local_values}
                if returns_iterator:
                    return self._infer_iterable_structure(stmt.value, merged_env)
                return self._structure_from_expression(stmt.value, merged_env)

        yielded_structures: List[Any] = []
        for yielded in ast.walk(function_node):
            if isinstance(yielded, ast.Yield) and yielded.value is not None:
                structure = self._structure_from_expression(yielded.value, {**env, **local_values})
                yielded_structures.append(structure)
            elif isinstance(yielded, ast.YieldFrom):
                structure = self._infer_iterable_structure(yielded.value, {**env, **local_values})
                yielded_structures.append(structure)

        return self._merge_structures(yielded_structures)

    def _structure_from_annotation(self, annotation: Any, *, returns_iterator: bool) -> Optional[Any]:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if returns_iterator and origin in {TypingIterator, TypingIterable, cabc.Iterator, cabc.Iterable} and args:
            return self._structure_from_annotation(args[0], returns_iterator=False)

        if origin in {tuple, Tuple} and args:
            if len(args) == 2 and args[1] is Ellipsis:
                element = self._structure_from_annotation(args[0], returns_iterator=False)
                return (element, ) if element is not None else None
            elements = [self._structure_from_annotation(arg, returns_iterator=False) for arg in args]
            if any(element is None for element in elements):
                return None
            return tuple(elements)

        if origin in {list, List} and args:
            element = self._structure_from_annotation(args[0], returns_iterator=False)
            return [element] if element is not None else None

        try:
            descriptor = _clone_descriptor(data.create_datadescriptor(annotation))
        except Exception:
            descriptor = None

        if descriptor is None:
            return None
        descriptor.transient = True
        return descriptor

    def _merge_structures(self, structures: Sequence[Any]) -> Optional[Any]:
        filtered = [structure for structure in structures if structure is not None]
        if not filtered:
            return None

        if all(isinstance(structure, data.Scalar) for structure in filtered):
            dtype = filtered[0].dtype
            for structure in filtered[1:]:
                dtype = dtypes.result_type_of(dtype, structure.dtype)
            return data.Scalar(dtype, transient=True)

        first = filtered[0]
        if isinstance(first, tuple) and all(
                isinstance(structure, tuple) and len(structure) == len(first) for structure in filtered):
            elements = [
                self._merge_structures([structure[index] for structure in filtered]) for index in range(len(first))
            ]
            if any(element is None for element in elements):
                return None
            return tuple(elements)

        if isinstance(first, list) and all(
                isinstance(structure, list) and len(structure) == len(first) for structure in filtered):
            elements = [
                self._merge_structures([structure[index] for structure in filtered]) for index in range(len(first))
            ]
            if any(element is None for element in elements):
                return None
            return elements

        if all(
                isinstance(structure, data.Data) and self._compatible_descriptors(first, structure)
                for structure in filtered):
            return _clone_descriptor(first)

        return None

    def _element_structure_from_binding(self, binding: _Binding) -> Optional[Any]:
        if binding.structure is not None and isinstance(binding.structure, (list, tuple)):
            if not binding.structure:
                return None
            return copy.deepcopy(binding.structure[0])
        if binding.descriptor is None:
            return None
        return self._element_structure_from_descriptor(binding.descriptor)

    def _element_structure_from_descriptor(self, descriptor: data.Data) -> Optional[Any]:
        if isinstance(descriptor, data.Scalar):
            return None
        if isinstance(descriptor, (PythonList, PythonTuple)):
            if descriptor.dtype == dtypes.pyobject():
                return None
            return data.Scalar(descriptor.dtype, transient=True)
        if hasattr(descriptor, 'shape'):
            if len(descriptor.shape) <= 1:
                return data.Scalar(descriptor.dtype, transient=True)
            return self._make_view_descriptor(descriptor, descriptor.shape[1:])
        return None

    def _structure_from_value(self, value: Any) -> Optional[Any]:
        if isinstance(value, tuple):
            elements = [self._structure_from_value(element) for element in value]
            if any(element is None for element in elements):
                return None
            return tuple(elements)
        if isinstance(value, list):
            elements = [self._structure_from_value(element) for element in value]
            if any(element is None for element in elements):
                return None
            return elements
        try:
            descriptor = _clone_descriptor(data.create_datadescriptor(value))
        except Exception:
            return None
        descriptor.transient = True
        return descriptor

    def _structure_from_expression(self, node: ast.AST, env: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        env = env or self._evaluation_context()
        if isinstance(node, (ast.Tuple, ast.List)):
            elements = [self._structure_from_expression(element, env) for element in node.elts]
            if any(element is None for element in elements):
                return None
            return elements if isinstance(node, ast.List) else tuple(elements)

        if isinstance(node, ast.Name) and node.id in self.bindings:
            binding = self.bindings[node.id]
            if binding.structure is not None:
                return copy.deepcopy(binding.structure)
            if binding.descriptor is not None:
                return _clone_descriptor(binding.descriptor)

        if isinstance(node, ast.Call) and astutils.rname(node.func) not in {'tuple', 'list'}:
            return None

        value = self._safe_eval(node, env)
        if value is None:
            return None
        return self._structure_from_value(value)

    def _subscript_structure(self, binding: _Binding, slice_node: ast.AST) -> Optional[Any]:
        if binding.structure is None or not isinstance(binding.structure, (list, tuple)):
            return None
        index_value = self._safe_eval(slice_node, self._evaluation_context())
        if not isinstance(index_value, int):
            return None
        if index_value < 0 or index_value >= len(binding.structure):
            return None
        return copy.deepcopy(binding.structure[index_value])

    def _subscript_descriptor(self, descriptor: data.Data, node: ast.Subscript) -> Optional[data.Data]:
        if isinstance(descriptor, (PythonList, PythonTuple)):
            if descriptor.dtype == dtypes.pyobject():
                return None
            return data.Scalar(descriptor.dtype, transient=True)

        dict_descriptor = self.dict_support.infer_subscript_descriptor(self._dict_support_context(), descriptor,
                                                                       node.slice)
        if dict_descriptor is not None:
            return dict_descriptor

        static_descriptor = _infer_static_subscript_descriptor(descriptor, node, self._evaluation_context())

        try:
            subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(descriptor, node, self._evaluation_context())
        except Exception:
            return static_descriptor
        if _is_scalar_subscript(node, subset, new_axes, arrdims):
            return data.Scalar(descriptor.dtype, transient=True)

        if static_descriptor is not None:
            if isinstance(static_descriptor, data.Scalar):
                return static_descriptor
            return self._make_view_descriptor(descriptor, static_descriptor.shape)

        return self._make_view_descriptor(descriptor, subset.size(), new_axes)

    def _infer_known_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        binding = self._resolve_binding(node)
        if binding is not None and binding.descriptor is not None:
            return _clone_descriptor(binding.descriptor)
        return self._infer_descriptor(node)

    def _infer_operator_operand(self, node: ast.AST) -> Optional[Any]:
        binding = self._resolve_binding(node)
        if binding is not None and binding.descriptor is not None:
            return _clone_descriptor(binding.descriptor)

        descriptor = self._infer_descriptor(node)
        if descriptor is not None:
            return descriptor

        value = try_resolve_static_value(node, self._evaluation_context())
        if value is not UNRESOLVED:
            return value

        return self._infer_scalar_descriptor(node, None)

    def _infer_binop_descriptor(self, node: ast.BinOp) -> Optional[data.Data]:
        left_operand = self._infer_operator_operand(node.left)
        right_operand = self._infer_operator_operand(node.right)
        if left_operand is None or right_operand is None:
            return None

        infer_fn = oprepo.Replacements.get_operator_descriptor_inference(
            type(node.op).__name__, left_operand, right_operand)
        if infer_fn is None:
            return None
        try:
            return infer_fn(left_operand, right_operand)
        except Exception:
            return None

    def _infer_unaryop_descriptor(self, node: ast.UnaryOp) -> Optional[data.Data]:
        operand = self._infer_operator_operand(node.operand)
        if operand is None:
            return None

        infer_fn = oprepo.Replacements.get_operator_descriptor_inference(type(node.op).__name__, operand)
        if infer_fn is None:
            return None
        try:
            return infer_fn(operand)
        except Exception:
            return None

    def _infer_boolop_descriptor(self, node: ast.BoolOp) -> Optional[data.Data]:
        if len(node.values) == 0:
            return None

        current = self._infer_operator_operand(node.values[0])
        if current is None:
            return None

        for value in node.values[1:]:
            next_operand = self._infer_operator_operand(value)
            if next_operand is None:
                return None
            infer_fn = oprepo.Replacements.get_operator_descriptor_inference(
                type(node.op).__name__, current, next_operand)
            if infer_fn is None:
                return None
            try:
                current = infer_fn(current, next_operand)
            except Exception:
                return None
            if current is None:
                return None
        return current

    def _infer_compare_descriptor(self, node: ast.Compare) -> Optional[data.Data]:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return None

        left_operand = self._infer_operator_operand(node.left)
        right_operand = self._infer_operator_operand(node.comparators[0])
        if left_operand is None or right_operand is None:
            return None

        infer_fn = oprepo.Replacements.get_operator_descriptor_inference(
            type(node.ops[0]).__name__, left_operand, right_operand)
        if infer_fn is None:
            return None
        try:
            return infer_fn(left_operand, right_operand)
        except Exception:
            return None

    def _infer_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        if isinstance(node, ast.Dict):
            return self.dict_support.infer_literal_descriptor(self._dict_support_context(), node)

        if isinstance(node, ast.Call):
            inferred = infer_array_literal_descriptor(node, self._infer_descriptor, self._infer_scalar_descriptor,
                                                      self._evaluation_context)
            if inferred is not None:
                return inferred

        if isinstance(node, ast.Attribute):
            binding = self._resolve_binding(node)
            if binding is not None and binding.descriptor is not None:
                return _clone_descriptor(binding.descriptor)

        if isinstance(node, ast.BinOp):
            inferred = self._infer_binop_descriptor(node)
            if inferred is not None:
                return inferred

        if isinstance(node, ast.UnaryOp):
            inferred = self._infer_unaryop_descriptor(node)
            if inferred is not None:
                return inferred

        if isinstance(node, ast.BoolOp):
            inferred = self._infer_boolop_descriptor(node)
            if inferred is not None:
                return inferred

        if isinstance(node, ast.Compare):
            inferred = self._infer_compare_descriptor(node)
            if inferred is not None:
                return inferred

        if isinstance(node, ast.Call):
            # Try the method descriptor-inference registry first (a.sum(), etc.)
            if isinstance(node.func, ast.Attribute):
                inferred = self._try_method_descriptor_inference(node)
                if inferred is not None:
                    return inferred.descriptor

            inferred = self._try_ufunc_descriptor_inference(node)
            if inferred is not None:
                return inferred.descriptor

            # Try the free-function descriptor-inference registry (numpy.sum(), etc.)
            inferred = self._try_descriptor_inference(node)
            if inferred is not None:
                return inferred.descriptor

        # Attribute inference (a.T, a.flat, a.real, a.imag, etc.)
        if isinstance(node, ast.Attribute):
            inferred = self._try_attribute_descriptor_inference(node)
            if inferred is not None:
                return inferred.descriptor

        return None

    def _infer_scalar_descriptor(self, node: ast.AST, annotated_descriptor: Optional[data.Data]) -> Optional[data.Data]:
        if annotated_descriptor is not None and isinstance(annotated_descriptor, data.Scalar):
            return _clone_descriptor(annotated_descriptor)

        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            return _string_scalar_descriptor()

        scalar_types = {
            name: binding.descriptor.dtype
            for name, binding in self.bindings.items()
            if binding.descriptor is not None and isinstance(binding.descriptor, data.Scalar)
        }
        try:
            inferred_type = infer_expr_type(_unparse(node), scalar_types)
        except Exception:
            inferred_type = None
        if inferred_type is not None:
            return data.Scalar(inferred_type, transient=True)

        value = try_resolve_static_value(node, self._evaluation_context())
        if value is not UNRESOLVED and value is not None:
            try:
                descriptor = _clone_descriptor(data.create_datadescriptor(value))
            except Exception:
                descriptor = None
            if isinstance(descriptor, data.Scalar):
                descriptor.transient = True
                return descriptor

        if isinstance(value, numbers.Number) or isinstance(value, bool):
            dtype = _normalize_dtype(type(value))
            if dtype is not None:
                return data.Scalar(dtype, transient=True)

        if _should_fallback_to_pyobject_scalar(node, value):
            return _pyobject_scalar_descriptor()

        if value is UNRESOLVED:
            return None
        return None

    def _try_replacement_binding_inference(self, node: ast.AST) -> Optional[_Binding]:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                inferred = self._try_method_descriptor_inference(node)
                if inferred is not None:
                    return inferred
            inferred = self._try_ufunc_descriptor_inference(node)
            if inferred is not None:
                return inferred
            return self._try_descriptor_inference(node)

        if isinstance(node, ast.Attribute):
            return self._try_attribute_descriptor_inference(node)

        return None

    def _try_descriptor_inference(self, node: ast.Call) -> Optional[_Binding]:
        """Query the descriptor-inference registry for a call node."""
        call_name = self._resolved_callable_name(node.func)
        infer_fn = oprepo.Replacements.get_descriptor_inference(call_name)
        if infer_fn is None:
            textual_name = astutils.rname(node.func)
            if textual_name != call_name:
                infer_fn = oprepo.Replacements.get_descriptor_inference(textual_name)
        if infer_fn is None:
            return None
        input_descs, args, kwargs = self._resolve_call_inputs_for_inference(node)
        try:
            result = infer_fn(input_descs, *args, **kwargs)
        except Exception:
            return None
        return _binding_from_inference_result(result)

    def _resolved_callable_name(self, node: ast.AST) -> str:
        textual_name = astutils.rname(node)
        resolved = try_resolve_static_value(node, self._evaluation_context())
        if resolved is not UNRESOLVED:
            module_name = getattr(resolved, '__module__', None)
            callable_name = getattr(resolved, '__name__', None)
            if module_name and callable_name and module_name != 'builtins':
                return f'{module_name}.{callable_name}'

        if '.' in textual_name:
            root_name, suffix = textual_name.split('.', 1)
            root_value = try_resolve_static_value(ast.Name(id=root_name, ctx=ast.Load()), self._evaluation_context())
            module_name = getattr(root_value, '__name__', None) if root_value is not UNRESOLVED else None
            if module_name is not None:
                return f'{module_name}.{suffix}'

        return textual_name

    def _try_ufunc_descriptor_inference(self, node: ast.Call) -> Optional[_Binding]:
        """Query the descriptor-inference registry for a NumPy ufunc call or ufunc method."""
        target = _resolve_ufunc_inference_target(node, self._evaluation_context())
        if target is None:
            return None

        method_name, ufunc_name = target
        infer_fn = oprepo.Replacements.get_ufunc_descriptor_inference(method_name)
        if infer_fn is None:
            return None

        input_descs, args, kwargs = self._resolve_call_inputs_for_inference(node)
        try:
            result = infer_fn(input_descs, ufunc_name, *args, **kwargs)
        except Exception:
            return None
        return _binding_from_inference_result(result)

    def _try_method_descriptor_inference(self, node: ast.Call) -> Optional[_Binding]:
        """Query the method descriptor-inference registry for ``obj.method(...)`` calls."""
        if not isinstance(node.func, ast.Attribute):
            return None
        # Resolve the object (e.g. ``a`` in ``a.sum()``)
        obj_binding = self._resolve_binding(node.func.value)
        if obj_binding is not None and obj_binding.descriptor is not None:
            obj_desc = obj_binding.descriptor
        else:
            obj_desc = try_resolve_static_value(node.func.value, self._evaluation_context())
            if obj_desc is UNRESOLVED:
                return None
        method_name = node.func.attr
        infer_fn = oprepo.Replacements.get_method_descriptor_inference(type(obj_desc), method_name)
        if infer_fn is None:
            return None
        # Resolve the remaining arguments (skip 'self')
        _input_descs, args, kwargs = self._resolve_call_inputs_for_inference(node)
        try:
            result = infer_fn(obj_desc, *args, **kwargs)
        except Exception:
            return None
        return _binding_from_inference_result(result)

    def _apply_method_self_descriptor_side_effect(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return
        if not isinstance(node.func.value, ast.Name):
            return

        obj_name = node.func.value.id
        obj_binding = self.bindings.get(obj_name)
        if obj_binding is None or obj_binding.descriptor is None:
            return

        infer_fn = oprepo.Replacements.get_method_self_descriptor_inference(
            type(obj_binding.descriptor).__name__, node.func.attr)
        if infer_fn is None:
            return

        _input_descs, args, kwargs = self._resolve_call_inputs_for_inference(node)
        try:
            updated_self = infer_fn(obj_binding.descriptor, *args, **kwargs)
        except Exception:
            return
        if not isinstance(updated_self, data.Data):
            return

        self._store_binding(obj_name, updated_self, kind=obj_binding.kind)

    def _try_attribute_descriptor_inference(self, node: ast.Attribute) -> Optional[_Binding]:
        """Query the attribute descriptor-inference registry for ``obj.attr`` accesses."""
        obj_binding = self._resolve_binding(node.value)
        if obj_binding is None or obj_binding.descriptor is None:
            return None
        obj_desc = obj_binding.descriptor
        classname = type(obj_desc).__name__
        infer_fn = oprepo.Replacements.get_attribute_descriptor_inference(classname, node.attr)
        if infer_fn is None:
            return None
        try:
            result = infer_fn(obj_desc)
        except Exception:
            return None
        return _binding_from_inference_result(result)

    def _resolve_call_inputs_for_inference(self, call_node: ast.Call) -> tuple:
        """Resolve call arguments to ``(input_descriptors, args, kwargs)``."""
        input_descs = {}
        args = []
        for arg in call_node.args:
            binding = self._resolve_binding(arg)
            if binding is not None and binding.descriptor is not None:
                name = astutils.rname(arg) if isinstance(arg, (ast.Name, ast.Attribute)) else f'__arg{len(args)}'
                input_descs[name] = binding.descriptor
                args.append(name)
            else:
                val = self._safe_eval(arg, self._evaluation_context())
                args.append(val)
        kwargs = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                continue
            val = self._safe_eval(kw.value, self._evaluation_context())
            kwargs[kw.arg] = val
        return input_descs, args, kwargs

    def _bind_loop_target(self, target: ast.AST) -> None:
        loop_scalar = data.Scalar(dtypes.int64, transient=True)
        if isinstance(target, ast.Name):
            self._store_binding(target.id, loop_scalar)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_loop_target(element)

    def _bind_target_structure(self, target: ast.AST, structure: Any) -> None:

        def _bind(name: str, substructure: Any) -> None:
            descriptor = descriptor_from_structure(substructure)
            if descriptor is None:
                return
            kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
            self._store_binding(name, descriptor, kind=kind, structure=substructure)

        bind_target_structure(target, structure, _bind)

    def _store_binding(self,
                       name: str,
                       descriptor: data.Data,
                       *,
                       kind: Optional[str] = None,
                       structure: Optional[Any] = None) -> None:
        binding_kind = kind or ('scalar' if isinstance(descriptor, data.Scalar) else 'container')
        binding = _Binding(descriptor=_clone_descriptor(descriptor),
                           kind=binding_kind,
                           structure=copy.deepcopy(structure))
        self.bindings[name] = binding
        self.results[name] = _clone_binding(binding)

    def _safe_eval(self, node: ast.AST, env: Dict[str, Any]) -> Optional[Any]:
        if isinstance(node, ast.Call) and astutils.rname(node.func) not in {'tuple', 'list'}:
            return None
        value = try_resolve_static_value(node, env)
        if value is UNRESOLVED:
            return None
        return value

    def _evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(self.globals)
        context.update({
            name: binding.descriptor
            for name, binding in self.bindings.items() if binding.descriptor is not None
        })
        return context

    def _dict_support_context(self) -> DictSupportContext:
        return DictSupportContext(infer_descriptor=self._infer_known_descriptor,
                                  infer_scalar_descriptor=self._infer_scalar_descriptor,
                                  evaluation_context=self._evaluation_context)

    def _evaluate_annotation_class_type(self, node: Optional[ast.AST]) -> Optional[type[Any]]:
        if node is None:
            return None
        value = self._safe_eval(node, self._evaluation_context())
        return direct_class_annotation_type(value)

    def _evaluate_descriptor(self, node: Optional[ast.AST]) -> Optional[data.Data]:
        if node is None:
            return None
        class_type = self._evaluate_annotation_class_type(node)
        if class_type is not None:
            descriptor = data.Structure.from_class(class_type)
            descriptor.transient = True
            return descriptor
        value = self._safe_eval(node, self._evaluation_context())
        if isinstance(value, data.Data):
            descriptor = _clone_descriptor(value)
            descriptor.transient = True
            return descriptor
        dtype = _normalize_dtype(value)
        if dtype is not None:
            return data.Scalar(dtype, transient=True)
        return None

    def _initialize_direct_class_annotations(self, program: ast.AST) -> None:
        arguments = list(program.args.posonlyargs) + list(program.args.args) + list(program.args.kwonlyargs)
        if program.args.vararg is not None:
            arguments.append(program.args.vararg)
        if program.args.kwarg is not None:
            arguments.append(program.args.kwarg)

        for argument in arguments:
            class_type = self._evaluate_annotation_class_type(argument.annotation)
            if class_type is not None:
                self.annotated_class_types[argument.arg] = class_type

    def _attribute_root_name(self, node: ast.AST) -> Optional[str]:
        root = self._attribute_root_and_members(node)
        return root[0] if root is not None else None

    def _attribute_root_and_members(self, node: ast.AST) -> Optional[Tuple[str, List[str]]]:
        current = node
        members: List[str] = []
        while isinstance(current, ast.Attribute):
            members.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            members.reverse()
            return current.id, members
        return None

    def _parse_shape(self, node: ast.AST) -> List[Any]:
        value = self._safe_eval(node, self._evaluation_context())
        if isinstance(value, (list, tuple)):
            return [self._shape_dim(dim) for dim in value]
        if value is not None:
            return [self._shape_dim(value)]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._shape_dim(symbolic.pystr_to_symbolic(_unparse(elem))) for elem in node.elts]
        return [self._shape_dim(symbolic.pystr_to_symbolic(_unparse(node)))]

    def _parse_dtype(self, node: Optional[ast.AST]) -> Optional[dtypes.typeclass]:
        if node is None:
            return None
        value = self._safe_eval(node, self._evaluation_context())
        return _normalize_dtype(value)

    def _shape_dim(self, value: Any) -> Any:
        if isinstance(value, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic)):
            return value
        if isinstance(value, str):
            return symbolic.pystr_to_symbolic(value)
        return value

    def _call_argument(self, node: ast.Call, position: int, keyword: str) -> Optional[ast.AST]:
        if len(node.args) > position:
            return node.args[position]
        for kw in node.keywords:
            if kw.arg == keyword:
                return kw.value
        return None

    def _make_view_descriptor(self,
                              descriptor: data.Data,
                              shape: Optional[Sequence[Any]] = None,
                              new_axes: Optional[Sequence[int]] = None) -> data.Data:
        view_desc = data.View.view(descriptor)
        if shape is None:
            shape = descriptor.shape
        shape_list = list(shape)
        if new_axes:
            for axis in sorted(new_axes):
                shape_list.insert(axis, 1)
        if hasattr(view_desc, 'set_shape'):
            view_desc.set_shape(shape_list)
        return view_desc
