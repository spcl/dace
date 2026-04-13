# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python frontend entry point for building schedule trees directly from AST."""

import ast
import copy
import inspect
import numbers
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dace import data, dtypes, symbolic
from dace.data.pydata import PythonList, PythonTuple
from dace.frontend.python import astutils, memlet_parser, preprocessing
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.frontend.python.schedule_tree.match_support import UnsupportedMatchPatternError, lower_match_to_statements
from dace.frontend.python.schedule_tree import (ExpressionPlanningContext, GenericExpressionSupportLibrary,
                                                NumpyLoweringContext, NumpySupportLibrary, ScheduleTreeTypeInference,
                                                _Binding, resolve_function_calls)
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.type_inference import infer_expr_type

_INTERNAL_ITERATOR_HELPERS = {
    '__dace_iterator_init',
    '__dace_iterator_next',
}


def _clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


def _unparse(node: ast.AST) -> str:
    try:
        working_node = copy.deepcopy(node)
    except Exception:
        working_node = node
    sanitized = _sanitize_ast_for_unparse(working_node)
    return astutils.unparse(sanitized)


def _sanitize_ast_for_unparse(node: ast.AST) -> ast.AST:

    class _ConstantNormalizer(ast.NodeTransformer):

        def visit_Constant(self, constant: ast.Constant) -> ast.AST:
            replacement_name = _constant_source_name(constant.value)
            if replacement_name is None:
                return constant
            return ast.copy_location(ast.Name(id=replacement_name, ctx=ast.Load()), constant)

    sanitized = _ConstantNormalizer().visit(node)
    return ast.fix_missing_locations(sanitized)


def _constant_source_name(value: Any) -> Optional[str]:
    if isinstance(value, (str, bytes, numbers.Number, bool, type(None), type(Ellipsis))):
        return None

    candidate = None
    if hasattr(value, 'f') and hasattr(value.f, '__name__'):
        candidate = value.f.__name__
    if not isinstance(candidate, str) or not candidate:
        candidate = getattr(value, 'name', None)
    if not isinstance(candidate, str) or not candidate:
        candidate = getattr(value, '__name__', None)
    if not isinstance(candidate, str) or not candidate:
        candidate = getattr(value, '__qualname__', None)
        if isinstance(candidate, str) and candidate:
            candidate = candidate.split('.')[-1]

    if not isinstance(candidate, str) or not candidate:
        return None

    sanitized = re.sub(r'\W|^(?=\d)', '_', candidate)
    if not sanitized or sanitized in {'True', 'False', 'None'}:
        return None
    return sanitized


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
    except TypeError:
        return None


def _pyobject_scalar_descriptor() -> data.Scalar:
    return data.Scalar(dtypes.pyobject(), transient=True)


def _should_fallback_to_pyobject_scalar(node: ast.AST, value: Any = UNRESOLVED) -> bool:
    if value is None or isinstance(value, (str, bytes, numbers.Number, bool, type(Ellipsis))):
        return False
    return isinstance(node, (ast.Await, ast.Attribute, ast.BinOp, ast.BoolOp, ast.Call, ast.Compare, ast.FormattedValue,
                             ast.IfExp, ast.JoinedStr, ast.Name, ast.NamedExpr, ast.UnaryOp, ast.Yield, ast.YieldFrom))


def build_schedule_tree(name: str,
                        parsed_ast: preprocessing.PreprocessedAST,
                        argtypes: Dict[str, data.Data],
                        *,
                        constants: Optional[Dict[str, Tuple[data.Data, Any]]] = None,
                        callback_mapping: Optional[Dict[str, str]] = None,
                        arg_names: Optional[Sequence[str]] = None,
                        inline_calls: bool = True) -> tn.ScheduleTreeRoot:
    """
    Build a schedule tree directly from a preprocessed Python AST.

    :param name: Program name.
    :param parsed_ast: Preprocessed program AST and metadata.
    :param argtypes: Mapping from visible argument names to DaCe descriptors.
    :param inline_calls: If True (default), resolve and inline nested
        ``@dace.program`` calls after building the tree.
    :return: A schedule tree rooted at a top-level scope.
    """
    builder = PythonScheduleTreeBuilder(name,
                                        parsed_ast,
                                        argtypes,
                                        constants=constants,
                                        callback_mapping=callback_mapping,
                                        arg_names=arg_names)
    root = builder.build()
    if inline_calls:
        resolve_function_calls(root)
    return root


class PythonScheduleTreeBuilder(ast.NodeVisitor):
    """Builds schedule trees from preprocessed Python ASTs."""

    def __init__(self,
                 name: str,
                 parsed_ast: preprocessing.PreprocessedAST,
                 argtypes: Dict[str, data.Data],
                 *,
                 constants: Optional[Dict[str, Tuple[data.Data, Any]]] = None,
                 callback_mapping: Optional[Dict[str, str]] = None,
                 arg_names: Optional[Sequence[str]] = None) -> None:
        self.name = name
        self.parsed_ast = parsed_ast
        self.argtypes = {k: _clone_descriptor(v) for k, v in argtypes.items()}
        self.globals = copy.copy(parsed_ast.program_globals)
        self.root = tn.ScheduleTreeRoot(name=name,
                                        children=[],
                                        containers={},
                                        symbols={},
                                        constants=self._clone_constants(constants),
                                        callback_mapping=dict(callback_mapping or {}),
                                        arg_names=list(arg_names or []))
        self.scope_stack: List[tn.ScheduleTreeScope] = [self.root]
        self.bindings: Dict[str, _Binding] = {}
        self.annotated_descriptors: Dict[str, data.Data] = {}
        self.expression_support = GenericExpressionSupportLibrary()
        self.numpy_support = NumpySupportLibrary()

        self._initialize_root_scope()
        self.inferred_bindings = ScheduleTreeTypeInference(self.globals, self.argtypes).infer(self._program_node())
        for name, binding in self.inferred_bindings.items():
            if binding.descriptor is not None:
                self.root.containers.setdefault(name, _clone_descriptor(binding.descriptor))

    def build(self) -> tn.ScheduleTreeRoot:
        """Build the schedule tree for the program AST."""
        program = self._program_node()
        for stmt in program.body:
            self.visit(stmt)
        return self.root

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._handle_assignment(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        descriptor = self._evaluate_descriptor(node.annotation)
        if descriptor is not None and isinstance(node.target, ast.Name):
            self.annotated_descriptors[node.target.id] = descriptor
            if node.value is None:
                if not isinstance(descriptor, data.Reference):
                    self._register_binding(node.target.id, descriptor, kind='container')
                return
        if node.value is not None:
            self._handle_assignment(node.target, node.value, annotated_descriptor=descriptor)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        value = ast.BinOp(left=copy.deepcopy(node.target), op=node.op, right=copy.deepcopy(node.value))
        self._handle_assignment(node.target, value)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        if self._is_dace_program_call(node.value):
            self._materialize_call_args(node.value)
            self._emit_function_call(node.value)
            return
        planned_value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                                node.value,
                                                                materialize_root=False)
        if self._handle_expression(planned_value):
            return
        self._append_node(tn.StatementNode(code=CodeBlock(self._format_runtime_expression(planned_value))))

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            values: List[CodeBlock] = []
        elif self._is_dace_program_call(node.value):
            # Materialize array-valued arguments, emit the function call;
            # the inlining pass will propagate the callee's return value.
            self._materialize_call_args(node.value)
            tmp = self._fresh_transient_name('__stree_retval')
            self._emit_function_call(node.value, return_targets=[tmp])
            self._append_node(tn.ReturnNode(values=[CodeBlock(tmp)]))
            return
        elif isinstance(node.value, ast.Tuple):
            planned_values = [
                self.expression_support.plan_expression(self._expression_planning_context(),
                                                        value,
                                                        materialize_root=True) for value in node.value.elts
            ]
            values = [
                CodeBlock(self._format_runtime_expression(self._materialize_return_value(v))) for v in planned_values
            ]
        else:
            planned_value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                                    node.value,
                                                                    materialize_root=True)
            planned_value = self._materialize_return_value(planned_value)
            values = [CodeBlock(self._format_runtime_expression(planned_value))]
        self._append_node(tn.ReturnNode(values=values))

    def _materialize_return_value(self, value: ast.AST) -> ast.AST:
        """Try to lower a return-value expression into a named temporary.

        If the expression is a call whose descriptor can be inferred (e.g.
        ``numpy.sum(X)``), it is emitted as a proper computation node
        (LibraryCall / TaskletNode) writing to a fresh transient, and the
        transient's ``ast.Name`` is returned.  Otherwise *value* is returned
        unchanged so the caller can still emit it as opaque text.
        """
        if not isinstance(value, ast.Call):
            return value
        descriptor = self._infer_descriptor(value, '__probe')
        if descriptor is None:
            return value
        name = self._fresh_transient_name('__stree_retval')
        kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
        self._register_binding(name, descriptor, kind=kind)
        target = ast.Name(id=name, ctx=ast.Store())
        if self._emit_computed_assignment(target, value, descriptor):
            return ast.Name(id=name, ctx=ast.Load())
        # Fallback: emit as opaque assignment.
        self._append_node(tn.AssignNode(name=name, value=CodeBlock(_unparse(value))))
        return ast.Name(id=name, ctx=ast.Load())

    def visit_Pass(self, node: ast.Pass) -> None:
        del node

    def visit_Break(self, node: ast.Break) -> None:
        del node
        self._append_node(tn.BreakNode())

    def visit_Continue(self, node: ast.Continue) -> None:
        del node
        self._append_node(tn.ContinueNode())

    def visit_If(self, node: ast.If) -> None:
        self._emit_if_chain(node)

    def visit_For(self, node: ast.For) -> None:
        loop_indices = self._parse_for_indices(node.target)
        iterator_kind, iterator_ranges = self._parse_for_iterator(node.iter)

        if iterator_kind == 'dace.map':
            map_scope = tn.MapScope(node=tn.FrontendMap(params=loop_indices, ranges=iterator_ranges), children=[])
            for index_name in loop_indices:
                map_scope.symbols[index_name] = symbolic.symbol(index_name, dtypes.int64)
            self._append_node(map_scope)
            self._visit_body(map_scope, node.body)
        elif iterator_kind == 'range':
            index_name = loop_indices[0]
            start, stop, step = iterator_ranges[0]
            comparator = '>' if stop.startswith('-') or step.startswith('-') else '<'
            loop_scope = tn.LoopScope(loop=tn.FrontendLoop(
                loop_condition=CodeBlock(f'{index_name} {comparator} {stop}'),
                init_statement=CodeBlock(f'{index_name} = {start}'),
                update_statement=CodeBlock(f'{index_name} = {index_name} + {step}'),
                loop_variable=index_name),
                                      children=[])
            loop_scope.symbols[index_name] = symbolic.symbol(index_name, dtypes.int64)
            self._append_node(loop_scope)
            self._visit_body(loop_scope, node.body)
        else:
            self._append_node(tn.StatementNode(code=CodeBlock(_unparse(node))))

        if node.orelse:
            else_scope = tn.ElseScope(children=[])
            self._append_node(else_scope)
            self._visit_body(else_scope, node.orelse)

    def visit_While(self, node: ast.While) -> None:
        loop_scope = tn.LoopScope(
            loop=tn.FrontendLoop(loop_condition=CodeBlock(self._format_runtime_expression(node.test))), children=[])
        self._append_node(loop_scope)
        self._visit_body(loop_scope, node.body)
        if node.orelse:
            else_scope = tn.ElseScope(children=[])
            self._append_node(else_scope)
            self._visit_body(else_scope, node.orelse)

    def generic_visit(self, node: ast.AST) -> None:
        import warnings
        warnings.warn(
            f'Schedule tree frontend: unhandled AST node {type(node).__name__} '
            f'at line {getattr(node, "lineno", "?")} — wrapping as callback',
            stacklevel=2)
        self._wrap_as_callback(node, f'unhandled {type(node).__name__}')

    # ------------------------------------------------------------------ #
    #  PythonCallbackNode helpers                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_name_flow(node: ast.AST) -> Tuple[set, set]:
        """Walk AST subtree, return (names in Load ctx, names in Store ctx)."""
        inputs: set = set()
        outputs: set = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Store):
                    outputs.add(child.id)
                elif isinstance(child.ctx, ast.Load):
                    inputs.add(child.id)
            elif isinstance(child, ast.alias):
                if child.asname:
                    outputs.add(child.asname)
                else:
                    outputs.add(child.name.split('.')[0])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                outputs.add(child.name)
            elif isinstance(child, ast.ExceptHandler) and isinstance(child.name, str):
                outputs.add(child.name)
        return inputs, outputs

    def _callback_code_text(self, node: ast.AST) -> str:
        """Return parseable source code for a callback-wrapped AST node."""
        if isinstance(node, ast.stmt):
            try:
                return ast.unparse(copy.deepcopy(node))
            except Exception:
                try:
                    return astutils.unparse(copy.deepcopy(node))
                except Exception:
                    return 'pass'

        try:
            return self._format_runtime_expression(node)
        except Exception:
            try:
                return _unparse(node)
            except Exception:
                try:
                    return ast.unparse(copy.deepcopy(node))
                except Exception:
                    return 'None'

    def _wrap_as_callback(self, node: ast.AST, reason: str) -> None:
        """Emit a PythonCallbackNode for constructs that cannot be lowered."""
        code_text = self._callback_code_text(node)
        inputs, outputs = self._analyze_name_flow(node)
        known_inputs = sorted(inputs & set(self.bindings))
        for output_name in sorted(outputs):
            binding = self.bindings.get(output_name)
            if binding is None or binding.descriptor is None:
                self._register_binding(output_name, _pyobject_scalar_descriptor(), kind='scalar')
        try:
            code = CodeBlock(code_text)
        except Exception:
            code = CodeBlock('pass')
        self._append_node(
            tn.PythonCallbackNode(code=code, reason=reason, input_names=known_inputs, output_names=sorted(outputs)))

    # ------------------------------------------------------------------ #
    #  Category C visitors — always callback                              #
    # ------------------------------------------------------------------ #

    def visit_Try(self, node: ast.Try) -> None:
        self._wrap_as_callback(node, 'try/except')

    # Python 3.11+ except* (TryStar)
    if hasattr(ast, 'TryStar'):
        visit_TryStar = visit_Try

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._wrap_as_callback(node, 'class definition')

    def visit_Import(self, node: ast.Import) -> None:
        self._wrap_as_callback(node, 'import')

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._wrap_as_callback(node, 'import')

    def visit_Yield(self, node: ast.AST) -> None:
        self._wrap_as_callback(node, 'yield')

    def visit_YieldFrom(self, node: ast.AST) -> None:
        self._wrap_as_callback(node, 'yield from')

    def visit_Await(self, node: ast.AST) -> None:
        self._wrap_as_callback(node, 'await')

    def visit_Match(self, node: ast.AST) -> None:
        subject = self._match_subject_expression(node.subject)
        try:
            lowered = lower_match_to_statements(node, subject)
        except UnsupportedMatchPatternError:
            self._wrap_as_callback(node, 'match/case')
            return

        for stmt in lowered:
            self.visit(stmt)

    # ------------------------------------------------------------------ #
    #  Category B visitors — try to lower, fall back to callback          #
    # ------------------------------------------------------------------ #

    def visit_Global(self, node: ast.Global) -> None:
        # Try to trace each name to a known data container or closure entry
        for name in node.names:
            if name in self.globals and isinstance(self.globals[name], data.Data):
                descriptor = _clone_descriptor(self.globals[name])
                self._register_binding(name, descriptor, kind='container')
            elif name in self.globals:
                # Known global value but not a data container — bind as symbol if possible
                value = self.globals[name]
                if isinstance(value, (int, float)):
                    self._register_binding(name,
                                           data.Scalar(dtypes.typeclass(type(value)), transient=False),
                                           kind='scalar')
                else:
                    self._wrap_as_callback(node, 'global scope')
                    return
            else:
                self._wrap_as_callback(node, 'global scope')
                return

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        # Try to trace each name to a known binding (closure entry)
        for name in node.names:
            if name in self.bindings:
                pass  # Already bound — nonlocal just marks scope; binding exists
            elif name in self.globals:
                if isinstance(self.globals[name], data.Data):
                    descriptor = _clone_descriptor(self.globals[name])
                    self._register_binding(name, descriptor, kind='container')
                else:
                    self._wrap_as_callback(node, 'nonlocal scope')
                    return
            else:
                self._wrap_as_callback(node, 'nonlocal scope')
                return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle nested function definitions.

        If the function is only called statically within the parent scope,
        it could become a FunctionCallRegion (future work). For now, wrap
        as callback since we need the full inlining infrastructure.
        """
        self._wrap_as_callback(node, 'nested function')

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._wrap_as_callback(node, 'async function')

    def visit_Delete(self, node: ast.Delete) -> None:
        # del of DaCe arrays is a no-op (runtime manages memory)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.bindings:
                continue  # No-op for known containers
            else:
                self._wrap_as_callback(node, 'delete')
                return

    def visit_Raise(self, node: ast.Raise) -> None:
        # May become ThrowNode in the future; wrap as callback for now
        self._wrap_as_callback(node, 'raise')

    def _append_node(self, node: tn.ScheduleTreeNode) -> None:
        scope = self.scope_stack[-1]
        node.parent = scope
        scope.children.append(node)

    def _visit_body(self, scope: tn.ScheduleTreeScope, body: Sequence[ast.AST]) -> None:
        self.scope_stack.append(scope)
        try:
            for stmt in body:
                self.visit(stmt)
        finally:
            self.scope_stack.pop()

    def _program_node(self) -> ast.FunctionDef:
        program_ast = self.parsed_ast.preprocessed_ast
        if isinstance(program_ast, ast.Module):
            node = program_ast.body[0]
        else:
            node = program_ast
        if not isinstance(node, ast.FunctionDef):
            raise TypeError('Expected a preprocessed FunctionDef as schedule-tree frontend input')
        return node

    def _match_subject_expression(self, subject: ast.AST) -> ast.AST:
        planned = self.expression_support.plan_expression(self._expression_planning_context(),
                                                          subject,
                                                          materialize_root=False)
        if isinstance(planned, (ast.Name, ast.Constant)):
            return planned

        descriptor = self._infer_compute_descriptor(planned)
        if descriptor is None:
            descriptor = self._infer_scalar_descriptor(planned, None)
        if descriptor is None:
            descriptor = _pyobject_scalar_descriptor()
        return self._materialize_temporary_expression(planned, descriptor)

    def _initialize_root_scope(self) -> None:
        for name, descriptor in self.argtypes.items():
            self.root.containers[name] = _clone_descriptor(descriptor)
            self.bindings[name] = _Binding(descriptor=_clone_descriptor(descriptor), kind='container')
            self.globals[name] = descriptor
            for free_symbol in descriptor.free_symbols:
                self.root.symbols[free_symbol.name] = free_symbol

        for name, value in self.globals.items():
            if isinstance(value, symbolic.symbol):
                self.root.symbols[name] = value

    def _handle_assignment(self,
                           target: ast.AST,
                           value: ast.AST,
                           annotated_descriptor: Optional[data.Data] = None) -> None:
        # Intercept nested @dace.program calls — materialize array-valued
        # arguments into temporaries first, then emit FunctionCallScope.
        if self._is_dace_program_call(value):
            self._materialize_call_args(value)
            targets = [target.id] if isinstance(target, ast.Name) else [_unparse(target)]
            self._emit_function_call(value, return_targets=targets)
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                            value,
                                                            materialize_root=False)
            value = self._materialize_structured_assignment_value(value)
            self._bind_structured_assignment(target, value)
            self._append_node(tn.StatementNode(code=CodeBlock(self._format_assignment_statement(target, value))))
            return

        value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                        value,
                                                        materialize_root=False)

        source_access = self._resolve_data_access(value)
        if isinstance(target, ast.Name):
            self._handle_name_assignment(target.id, value, source_access, annotated_descriptor)
            return

        target_access = self._resolve_data_access(target)
        if source_access is not None and target_access is not None:
            _, source_memlet, _, _ = source_access
            target_name, target_memlet, _, _ = target_access
            memlet = copy.deepcopy(source_memlet)
            memlet.other_subset = copy.deepcopy(target_memlet.subset)
            self._append_node(tn.CopyNode(target=target_name, memlet=memlet))
            return

        if self._emit_computed_assignment(target, value, annotated_descriptor):
            return

        self._append_node(tn.StatementNode(code=CodeBlock(self._format_assignment_statement(target, value))))

    def _handle_name_assignment(self, name: str, value: ast.AST, source_access: Optional[Tuple[str, Memlet, data.Data,
                                                                                               Optional[data.Data]]],
                                annotated_descriptor: Optional[data.Data]) -> None:
        if self._is_internal_iterator_binding_name(name) or self._is_internal_iterator_helper_call(value):
            self._infer_internal_iterator_binding(name, value, annotated_descriptor)
            self._append_node(tn.AssignNode(name=name, value=CodeBlock(self._format_runtime_expression(value))))
            return

        existing = self.bindings.get(name)
        target_descriptor = annotated_descriptor or self.annotated_descriptors.get(name)

        if source_access is not None:
            source_name, memlet, source_desc, view_desc = source_access

            if target_descriptor is not None and isinstance(target_descriptor, data.Reference):
                ref_desc = self._ensure_reference_binding(name, target_descriptor)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

            if existing is not None and isinstance(existing.descriptor, data.Reference):
                ref_desc = self._ensure_reference_binding(name, existing.descriptor)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

            if existing is None and self._should_bind_as_reference(value, source_desc):
                ref_desc = self._ensure_reference_binding(name, source_desc)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

            if existing is None and self._is_aliasable_descriptor(source_desc):
                new_view_desc = view_desc or self._make_view_descriptor(source_desc)
                self._register_binding(name, new_view_desc, kind='view')
                self._append_node(
                    tn.ViewNode(target=name,
                                source=source_name,
                                memlet=memlet,
                                src_desc=source_desc,
                                view_desc=new_view_desc))
                return

            if existing is not None and existing.descriptor is not None and self._can_promote_to_reference(
                    existing.descriptor, source_desc):
                ref_desc = self._ensure_reference_binding(name, existing.descriptor)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

        inferred_descriptor = self._infer_descriptor(value, name)
        if inferred_descriptor is not None:
            if existing is None and self._should_bind_expression_as_reference(value, inferred_descriptor):
                ref_desc = self._ensure_reference_binding(name, inferred_descriptor)
                self._append_node(
                    tn.RefSetNode(target=name,
                                  memlet=None,
                                  src_desc=inferred_descriptor,
                                  ref_desc=ref_desc,
                                  source_expr=self._format_runtime_expression(value)))
                return
            kind = 'reference' if isinstance(inferred_descriptor, data.Reference) else 'container'
            self._register_binding(name, inferred_descriptor, kind=kind)
        else:
            scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
            if scalar_descriptor is not None:
                self._register_binding(name, scalar_descriptor, kind='scalar')

        if self._emit_computed_assignment(ast.Name(id=name, ctx=ast.Store()), value, annotated_descriptor):
            return

        self._append_node(tn.AssignNode(name=name, value=CodeBlock(self._format_runtime_expression(value))))

    def _handle_expression(self, value: ast.AST) -> bool:
        if not isinstance(value, ast.Call) or not self._should_lower_as_library_call(value):
            return False

        in_memlets = self._collect_input_memlets(value)
        if not in_memlets:
            return False

        self._append_node(
            tn.LibraryCall(node=tn.FrontendLibrary(name=astutils.rname(value.func),
                                                   properties=self._library_properties(value)),
                           in_memlets=in_memlets,
                           out_memlets=set()))
        return True

    def _emit_computed_assignment(self, target: ast.AST, value: ast.AST,
                                  annotated_descriptor: Optional[data.Data]) -> bool:
        lowered = self.expression_support.lower_assignment(self._expression_planning_context(), target, value,
                                                           annotated_descriptor)
        if lowered is not None:
            self._append_node(lowered)
            return True

        lowered = self.numpy_support.lower_assignment(self._numpy_lowering_context(), target, value,
                                                      annotated_descriptor)
        if lowered is not None:
            self._append_node(lowered)
            return True

        output = self._resolve_output_target(target, value, annotated_descriptor)
        if output is None:
            return False

        _, out_memlet, _ = output

        if isinstance(value, ast.Call) and self._should_lower_as_library_call(value):
            in_memlets = self._collect_input_memlets(value)
            if not in_memlets:
                return False
            self._append_node(
                tn.LibraryCall(node=tn.FrontendLibrary(name=astutils.rname(value.func),
                                                       properties=self._library_properties(value)),
                               in_memlets=in_memlets,
                               out_memlets={'out': out_memlet}))
            return True

        in_memlets = self._collect_input_memlets(value)
        if not in_memlets:
            return False

        tasklet = tn.FrontendTasklet(name=self._tasklet_name(target),
                                     code=CodeBlock(f'{_unparse(target)} = {_unparse(value)}'))
        self._append_node(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets={'out': out_memlet}))
        return True

    def _format_runtime_expression(self, node: ast.AST) -> str:
        return _unparse(self._rewrite_protocol_loads(node))

    def _format_assignment_statement(self, target: ast.AST, value: ast.AST) -> str:
        rewritten_call = self._rewrite_protocol_store(target, value)
        if rewritten_call is not None:
            return _unparse(rewritten_call)
        return f'{_unparse(target)} = {self._format_runtime_expression(value)}'

    def _rewrite_protocol_loads(self, node: ast.AST) -> ast.AST:

        class _ProtocolLoadRewriter(ast.NodeTransformer):

            def __init__(self, builder: 'PythonScheduleTreeBuilder') -> None:
                self.builder = builder

            def visit_Attribute(self, attr_node: ast.Attribute) -> ast.AST:
                attr_node.value = self.visit(attr_node.value)
                rewritten = self.builder._protocol_load_call(attr_node)
                if rewritten is None:
                    return attr_node
                return ast.copy_location(rewritten, attr_node)

        try:
            working = copy.deepcopy(node)
        except Exception:
            working = node
        rewritten = _ProtocolLoadRewriter(self).visit(working)
        return ast.fix_missing_locations(rewritten)

    def _protocol_load_call(self, node: ast.Attribute) -> Optional[ast.AST]:
        if not isinstance(node.ctx, ast.Load):
            return None

        base_value = try_resolve_static_value(node.value, self._evaluation_context())
        if base_value is UNRESOLVED or self._is_native_attribute_base(base_value):
            return None

        owner_expr = self._type_expr(copy.deepcopy(node.value))
        obj_expr = copy.deepcopy(node.value)
        objtype = type(base_value)

        try:
            static_attr = inspect.getattr_static(base_value, node.attr)
        except AttributeError:
            static_attr = None

        if static_attr is not None and self._is_descriptor_object(static_attr) and hasattr(static_attr, '__get__'):
            descriptor_expr = self._descriptor_expr(copy.deepcopy(node.value), node.attr)
            return ast.Call(func=ast.Attribute(value=descriptor_expr, attr='__get__', ctx=ast.Load()),
                            args=[obj_expr, copy.deepcopy(owner_expr)],
                            keywords=[])

        getattribute = objtype.__dict__.get('__getattribute__')
        if getattribute is not None and getattribute is not object.__getattribute__:
            return ast.Call(func=ast.Attribute(value=copy.deepcopy(owner_expr), attr='__getattribute__',
                                               ctx=ast.Load()),
                            args=[obj_expr, ast.Constant(node.attr)],
                            keywords=[])

        if static_attr is None and '__getattr__' in objtype.__dict__:
            return ast.Call(func=ast.Attribute(value=copy.deepcopy(owner_expr), attr='__getattr__', ctx=ast.Load()),
                            args=[obj_expr, ast.Constant(node.attr)],
                            keywords=[])

        return None

    def _rewrite_protocol_store(self, target: ast.AST, value: ast.AST) -> Optional[ast.AST]:
        if not isinstance(target, ast.Attribute):
            return None

        base_value = try_resolve_static_value(target.value, self._evaluation_context())
        if base_value is UNRESOLVED or self._is_native_attribute_base(base_value):
            return None

        owner_expr = self._type_expr(copy.deepcopy(target.value))
        obj_expr = copy.deepcopy(target.value)
        objtype = type(base_value)
        rewritten_value = self._rewrite_protocol_loads(value)

        try:
            static_attr = inspect.getattr_static(base_value, target.attr)
        except AttributeError:
            static_attr = None

        if static_attr is not None and self._is_descriptor_object(static_attr) and hasattr(static_attr, '__set__'):
            descriptor_expr = self._descriptor_expr(copy.deepcopy(target.value), target.attr)
            return ast.Call(func=ast.Attribute(value=descriptor_expr, attr='__set__', ctx=ast.Load()),
                            args=[obj_expr, rewritten_value],
                            keywords=[])

        setattr_method = objtype.__dict__.get('__setattr__')
        if setattr_method is not None and setattr_method is not object.__setattr__:
            return ast.Call(func=ast.Attribute(value=copy.deepcopy(owner_expr), attr='__setattr__', ctx=ast.Load()),
                            args=[obj_expr, ast.Constant(target.attr), rewritten_value],
                            keywords=[])

        return None

    def _type_expr(self, value_expr: ast.AST) -> ast.AST:
        return ast.Call(func=ast.Name(id='type', ctx=ast.Load()), args=[value_expr], keywords=[])

    def _descriptor_expr(self, value_expr: ast.AST, attr_name: str) -> ast.AST:
        return ast.Subscript(value=ast.Attribute(value=self._type_expr(value_expr), attr='__dict__', ctx=ast.Load()),
                             slice=ast.Constant(attr_name),
                             ctx=ast.Load())

    def _is_descriptor_object(self, value: Any) -> bool:
        return any(hasattr(value, attr) for attr in ('__get__', '__set__', '__delete__'))

    def _is_native_attribute_base(self, value: Any) -> bool:
        if dtypes.ismodule(value):
            return True
        if isinstance(value, (dtypes.typeclass, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic, data.Data)):
            return True
        module_name = getattr(type(value), '__module__', '')
        return module_name.startswith(('numpy', 'dace', 'sympy', 'builtins'))

    def _resolve_data_access(self, node: ast.AST) -> Optional[Tuple[str, Memlet, data.Data, Optional[data.Data]]]:
        if isinstance(node, ast.Name) and node.id in self.bindings and self.bindings[node.id].descriptor is not None:
            descriptor = _clone_descriptor(self.bindings[node.id].descriptor)
            return (node.id, Memlet.from_array(node.id, descriptor), descriptor, _clone_descriptor(descriptor))

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            base_name = node.value.id
            if base_name not in self.bindings or self.bindings[base_name].descriptor is None:
                return None
            descriptor = _clone_descriptor(self.bindings[base_name].descriptor)
            try:
                subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(descriptor, node,
                                                                              self._evaluation_context())
            except Exception:
                return None
            if arrdims:
                return None
            memlet = Memlet(data=base_name, subset=subset)
            return (base_name, memlet, descriptor, self._make_view_descriptor(descriptor, subset.size(), new_axes))

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'reshape':
            base_access = self._resolve_data_access(node.func.value)
            if base_access is None:
                return None
            source_name, memlet, descriptor, _ = base_access
            shape = self._parse_shape(node.args[0]) if node.args else list(descriptor.shape)
            view_desc = self._make_view_descriptor(descriptor, shape)
            return (source_name, memlet, descriptor, view_desc)

        return None

    def _resolve_output_target(self, target: ast.AST, value: ast.AST,
                               annotated_descriptor: Optional[data.Data]) -> Optional[Tuple[str, Memlet, data.Data]]:
        if isinstance(target, ast.Name):
            if annotated_descriptor is not None and isinstance(annotated_descriptor, data.Reference):
                return None

            existing = self.bindings.get(target.id)
            if existing is not None and existing.descriptor is not None and not isinstance(
                    existing.descriptor, data.Reference):
                descriptor = _clone_descriptor(existing.descriptor)
                return (target.id, Memlet.from_array(target.id, descriptor), descriptor)

            if annotated_descriptor is not None:
                kind = 'scalar' if isinstance(annotated_descriptor, data.Scalar) else 'container'
                self._register_binding(target.id, annotated_descriptor, kind=kind)
                descriptor = _clone_descriptor(self.bindings[target.id].descriptor)
                return (target.id, Memlet.from_array(target.id, descriptor), descriptor)

            inferred_descriptor = self._infer_compute_descriptor(value)
            if inferred_descriptor is None:
                return None
            kind = 'scalar' if isinstance(inferred_descriptor, data.Scalar) else 'container'
            self._register_binding(target.id, inferred_descriptor, kind=kind)
            descriptor = _clone_descriptor(self.bindings[target.id].descriptor)
            return (target.id, Memlet.from_array(target.id, descriptor), descriptor)

        target_access = self._resolve_data_access(target)
        if target_access is None:
            return None
        target_name, target_memlet, descriptor, _ = target_access
        return (target_name, target_memlet, descriptor)

    def _infer_compute_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        inferred_descriptor = self._infer_plannable_expression_descriptor(node)
        if inferred_descriptor is not None:
            return inferred_descriptor

        access = self._resolve_data_access(node)
        if access is not None:
            _, _, descriptor, view_descriptor = access
            result = _clone_descriptor(view_descriptor or descriptor)
            result.transient = True
            return result

        for _, _, descriptor, _ in self._collect_expression_accesses(node):
            result = _clone_descriptor(descriptor)
            result.transient = True
            return result
        return None

    def _collect_expression_accesses(self, node: ast.AST) -> List[Tuple[str, Memlet, data.Data, Optional[data.Data]]]:
        accesses: List[Tuple[str, Memlet, data.Data, Optional[data.Data]]] = []
        seen = set()

        def _visit(current: ast.AST) -> None:
            access = self._resolve_data_access(current)
            if access is not None:
                name, memlet, descriptor, view_descriptor = access
                key = (name, str(memlet.subset), str(memlet.other_subset) if memlet.other_subset is not None else '')
                if key not in seen:
                    seen.add(key)
                    accesses.append((name, memlet, descriptor, view_descriptor))
                return

            for child in ast.iter_child_nodes(current):
                _visit(child)

        _visit(node)
        return accesses

    def _collect_input_memlets(self, node: ast.AST) -> Dict[str, Memlet]:
        result: Dict[str, Memlet] = {}
        for index, (_, memlet, _, _) in enumerate(self._collect_expression_accesses(node)):
            result[f'in{index}'] = memlet
        return result

    def _infer_descriptor(self, node: ast.AST, target_name: str) -> Optional[data.Data]:
        if isinstance(node, ast.Call):
            call_name = astutils.rname(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'reshape':
                access = self._resolve_data_access(node)
                if access is not None:
                    return access[3]

            # Try the method descriptor-inference registry first (a.sum(), a.reshape(), etc.)
            if isinstance(node.func, ast.Attribute):
                inferred = self._try_method_descriptor_inference(node)
                if inferred is not None:
                    return inferred

            # Try the free-function descriptor-inference registry (numpy.sum(), etc.)
            inferred = self._try_descriptor_inference(node)
            if inferred is not None:
                return inferred

        # Attribute inference (a.T, a.flat, a.real, a.imag, etc.)
        if isinstance(node, ast.Attribute):
            inferred = self._try_attribute_descriptor_inference(node)
            if inferred is not None:
                return inferred

        value = try_resolve_static_value(node, self._evaluation_context())
        if value is not UNRESOLVED:
            try:
                descriptor = _clone_descriptor(data.create_datadescriptor(value))
                descriptor.transient = True
                return descriptor
            except Exception:
                pass

        return None

    def _try_descriptor_inference(self, node: ast.Call) -> Optional[data.Data]:
        """Query the descriptor-inference registry for a call node."""
        from dace.frontend.common.op_repository import Replacements
        call_name = astutils.rname(node.func)
        infer_fn = Replacements.get_descriptor_inference(call_name)
        if infer_fn is None:
            return None
        input_descs, args, kwargs = self._resolve_call_inputs(node)
        try:
            result = infer_fn(input_descs, *args, **kwargs)
        except Exception:
            return None
        if result is not None:
            result = _clone_descriptor(result)
            result.transient = True
        return result

    def _try_method_descriptor_inference(self, node: ast.Call) -> Optional[data.Data]:
        """Query the method descriptor-inference registry for ``obj.method(...)`` calls."""
        from dace.frontend.common.op_repository import Replacements
        if not isinstance(node.func, ast.Attribute):
            return None
        # Resolve the object (e.g. ``a`` in ``a.sum()``)
        obj_access = self._resolve_data_access(node.func.value)
        if obj_access is None:
            return None
        _, _, obj_desc, _ = obj_access
        classname = type(obj_desc).__name__  # 'Array', 'View', 'Scalar'
        method_name = node.func.attr
        infer_fn = Replacements.get_method_descriptor_inference(classname, method_name)
        if infer_fn is None:
            return None
        # Resolve the remaining arguments (skip 'self')
        _input_descs, args, kwargs = self._resolve_call_inputs(node)
        try:
            result = infer_fn(obj_desc, *args, **kwargs)
        except Exception:
            return None
        if result is not None:
            result = _clone_descriptor(result)
            result.transient = True
        return result

    def _try_attribute_descriptor_inference(self, node: ast.Attribute) -> Optional[data.Data]:
        """Query the attribute descriptor-inference registry for ``obj.attr`` accesses."""
        from dace.frontend.common.op_repository import Replacements
        obj_access = self._resolve_data_access(node.value)
        if obj_access is None:
            return None
        _, _, obj_desc, _ = obj_access
        classname = type(obj_desc).__name__
        infer_fn = Replacements.get_attribute_descriptor_inference(classname, node.attr)
        if infer_fn is None:
            return None
        try:
            result = infer_fn(obj_desc)
        except Exception:
            return None
        if result is not None:
            result = _clone_descriptor(result)
            result.transient = True
        return result

    def _resolve_call_inputs(self, call_node: ast.Call) -> tuple:
        """Resolve call arguments to ``(input_descriptors, args, kwargs)``."""
        input_descs: Dict[str, data.Data] = {}
        args: list = []
        for arg in call_node.args:
            access = self._resolve_data_access(arg)
            if access is not None:
                name, _, desc, _ = access
                input_descs[name] = desc
                args.append(name)
            else:
                val = try_resolve_static_value(arg, self._evaluation_context())
                args.append(val if val is not UNRESOLVED else _unparse(arg))
        kwargs: dict = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                continue
            val = try_resolve_static_value(kw.value, self._evaluation_context())
            kwargs[kw.arg] = val if val is not UNRESOLVED else _unparse(kw.value)
        return input_descs, args, kwargs

    def _infer_scalar_descriptor(self, node: ast.AST, annotated_descriptor: Optional[data.Data]) -> Optional[data.Data]:
        if annotated_descriptor is not None and isinstance(annotated_descriptor, data.Scalar):
            return _clone_descriptor(annotated_descriptor)

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
        if isinstance(value, numbers.Number) or isinstance(value, bool):
            dtype = _normalize_dtype(type(value))
            if dtype is not None:
                return data.Scalar(dtype, transient=True)

        if _should_fallback_to_pyobject_scalar(node, value):
            return _pyobject_scalar_descriptor()

        if value is UNRESOLVED:
            return None

        return None

    def _evaluate_descriptor(self, node: Optional[ast.AST]) -> Optional[data.Data]:
        if node is None:
            return None
        try:
            value = astutils.evalnode(node, self._evaluation_context())
        except Exception:
            return None
        if isinstance(value, data.Data):
            return _clone_descriptor(value)
        return None

    def _parse_shape(self, node: ast.AST) -> List[Any]:
        value = try_resolve_static_value(node, self._evaluation_context())
        if value is UNRESOLVED:
            value = None

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
        value = try_resolve_static_value(node, self._evaluation_context())
        if value is UNRESOLVED:
            return None
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

    def _library_properties(self, node: ast.Call) -> Dict[str, Any]:
        return {kw.arg: _unparse(kw.value) for kw in node.keywords if kw.arg is not None}

    def _tasklet_name(self, target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return f'{target.id}_tasklet'
        if isinstance(target, ast.Subscript):
            return f'{_unparse(target.value)}_tasklet'
        return 'tasklet'

    def _fresh_transient_name(self, prefix: str = '__stree_tmp') -> str:
        index = 0
        candidate = prefix
        while candidate in self.bindings or candidate in self.root.containers or candidate in self.globals:
            index += 1
            candidate = f'{prefix}{index}'
        return candidate

    def _materialize_temporary_expression(self, value: ast.AST, descriptor: data.Data) -> ast.AST:
        name = self._fresh_transient_name()
        kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
        self._register_binding(name, descriptor, kind=kind)
        target = ast.Name(id=name, ctx=ast.Store())

        if self._emit_computed_assignment(target, value, descriptor):
            return ast.Name(id=name, ctx=ast.Load())

        if isinstance(descriptor, data.Scalar):
            self._append_node(tn.AssignNode(name=name, value=CodeBlock(_unparse(value))))
        else:
            self._append_node(tn.StatementNode(code=CodeBlock(f'{name} = {_unparse(value)}')))
        return ast.Name(id=name, ctx=ast.Load())

    def _register_binding(self, name: str, descriptor: data.Data, kind: str) -> None:
        self._store_binding(name, descriptor, kind=kind)

    def _store_binding(self,
                       name: str,
                       descriptor: Optional[data.Data],
                       *,
                       kind: str,
                       structure: Optional[Any] = None) -> None:
        cloned = _clone_descriptor(descriptor) if descriptor is not None else None
        stored_structure = copy.deepcopy(structure)
        self.bindings[name] = _Binding(descriptor=cloned, kind=kind, structure=stored_structure)
        if cloned is None:
            return
        self.root.containers[name] = _clone_descriptor(cloned)
        if self.scope_stack[-1] is not self.root:
            self.scope_stack[-1].containers[name] = _clone_descriptor(cloned)
        self.globals[name] = cloned

    def _clone_constants(self, constants: Optional[Dict[str, Tuple[data.Data,
                                                                   Any]]]) -> Dict[str, Tuple[data.Data, Any]]:
        if not constants:
            return {}
        return {name: (_clone_descriptor(descriptor), value) for name, (descriptor, value) in constants.items()}

    def _infer_internal_iterator_binding(self, name: str, value: ast.AST,
                                         annotated_descriptor: Optional[data.Data]) -> None:
        inferred_binding = self.inferred_bindings.get(name)
        if inferred_binding is not None:
            self._store_binding(name,
                                inferred_binding.descriptor,
                                kind=inferred_binding.kind,
                                structure=inferred_binding.structure)
            return

        scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
        if scalar_descriptor is not None:
            self._store_binding(name, scalar_descriptor, kind='iterator-index', structure=scalar_descriptor)

    def _descriptor_from_structure(self, structure: Any, runtime_value: Optional[Any] = None) -> Optional[data.Data]:
        if isinstance(structure, data.Data):
            return _clone_descriptor(structure)

        if not isinstance(structure, (list, tuple)):
            return None

        dtype = dtypes.pyobject()
        if structure:
            first = structure[0]
            if all(
                    isinstance(element, data.Scalar) and isinstance(first, data.Scalar) and element.dtype == first.dtype
                    for element in structure):
                dtype = first.dtype

        descriptor_type = PythonList if isinstance(structure, list) else PythonTuple
        descriptor = descriptor_type(dtype=dtype, shape=(len(structure), ), transient=True)

        if runtime_value is not None and not isinstance(runtime_value, (list, tuple)):
            return descriptor
        return descriptor

    def _structure_from_ast(self, node: ast.AST) -> Tuple[Optional[Any], bool]:
        if isinstance(node, ast.Name):
            binding = self.bindings.get(node.id)
            if binding is None:
                return (None, False)
            structure = binding.structure if binding.structure is not None else binding.descriptor
            uses_internal = binding.kind.startswith('iterator') or self._is_internal_iterator_binding_name(node.id)
            return (copy.deepcopy(structure), uses_internal)

        if isinstance(node, (ast.Tuple, ast.List)):
            elements: List[Any] = []
            uses_internal = False
            for element in node.elts:
                substructure, sub_internal = self._structure_from_ast(element)
                if substructure is None:
                    return (None, False)
                elements.append(substructure)
                uses_internal = uses_internal or sub_internal
            if isinstance(node, ast.List):
                return (elements, uses_internal)
            return (tuple(elements), uses_internal)

        return (None, False)

    def _materialize_structured_assignment_value(self, value: ast.AST) -> ast.AST:
        if not isinstance(value, (ast.Tuple, ast.List)):
            return value

        structure, _ = self._structure_from_ast(value)
        if structure is None:
            return value

        descriptor = self._descriptor_from_structure(structure)
        if descriptor is None:
            return value

        name = self._fresh_transient_name('__stree_tuple_tmp')
        kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
        self._store_binding(name, descriptor, kind=kind, structure=structure)
        self._append_node(tn.StatementNode(code=CodeBlock(f'{name} = {_unparse(value)}')))
        return ast.Name(id=name, ctx=ast.Load())

    def _bind_structured_assignment(self, target: ast.AST, value: ast.AST) -> bool:
        structure, _ = self._structure_from_ast(value)
        if structure is None:
            return False
        return self._bind_target_structure(target, structure)

    def _bind_target_structure(self, target: ast.AST, structure: Any) -> bool:
        if isinstance(target, ast.Name):
            descriptor = self._descriptor_from_structure(structure)
            if descriptor is None:
                return False
            kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
            self._store_binding(target.id, descriptor, kind=kind, structure=structure)
            return True

        if isinstance(target, ast.Starred):
            if not isinstance(structure, list):
                structure = list(structure) if isinstance(structure, tuple) else [structure]
            return self._bind_target_structure(target.value, structure)

        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(structure, (list, tuple)):
            starred_indices = [index for index, element in enumerate(target.elts) if isinstance(element, ast.Starred)]
            if len(starred_indices) > 1:
                return False
            if not starred_indices:
                if len(target.elts) != len(structure):
                    return False
                return all(
                    self._bind_target_structure(subtarget, substructure)
                    for subtarget, substructure in zip(target.elts, structure))

            starred_index = starred_indices[0]
            if len(structure) < len(target.elts) - 1:
                return False

            prefix_targets = target.elts[:starred_index]
            suffix_targets = target.elts[starred_index + 1:]
            prefix_structures = structure[:starred_index]
            suffix_structures = structure[len(structure) - len(suffix_targets):]
            middle_structure = list(structure[starred_index:len(structure) - len(suffix_targets)])

            return all(
                self._bind_target_structure(subtarget, substructure)
                for subtarget, substructure in zip(prefix_targets, prefix_structures)) and self._bind_target_structure(
                    target.elts[starred_index], middle_structure) and all(
                        self._bind_target_structure(subtarget, substructure)
                        for subtarget, substructure in zip(suffix_targets, suffix_structures))

        return False

    def _ensure_reference_binding(self, name: str, descriptor: data.Data) -> data.Data:
        existing = self.bindings.get(name)
        if existing is not None and existing.descriptor is not None and isinstance(existing.descriptor, data.Reference):
            return _clone_descriptor(existing.descriptor)
        ref_desc = data.Reference.view(descriptor)
        self._register_binding(name, ref_desc, kind='reference')
        return _clone_descriptor(ref_desc)

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

    def _is_aliasable_descriptor(self, descriptor: data.Data) -> bool:
        return not isinstance(descriptor, data.Scalar)

    def _should_bind_expression_as_reference(self, value: ast.AST, descriptor: data.Data) -> bool:
        if not self._is_aliasable_descriptor(descriptor):
            return False
        return isinstance(value, ast.Attribute)

    # ------------------------------------------------------------------ #
    #  Function-call detection for nested @dace.program calls              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_dace_program_call(node: ast.AST) -> bool:
        """Return True when *node* is a call to a ``@dace.program``."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Constant):
            return False
        value = node.func.value
        from dace import SDFG
        return hasattr(value, '__sdfg__') and not isinstance(value, SDFG)

    def _extract_argument_mapping(self, call_node: ast.Call) -> Dict[str, str]:
        """Build ``{callee_param: caller_expression}`` from a call AST node."""
        callee = call_node.func.value
        sig = inspect.signature(callee.f)
        params = [p for p in sig.parameters.values() if p.name != 'self']

        mapping: Dict[str, str] = {}
        for i, arg in enumerate(call_node.args):
            if i < len(params):
                mapping[params[i].name] = self._format_runtime_expression(arg)
        for kw in call_node.keywords:
            mapping[kw.arg] = self._format_runtime_expression(kw.value)
        return mapping

    def _emit_function_call(self, call_node: ast.Call, return_targets: Optional[List[str]] = None) -> None:
        """Create a :class:`FunctionCallScope` placeholder and append it."""
        callee = call_node.func.value
        callee_name = getattr(callee.f, '__name__', None) or getattr(callee, 'name', '<anonymous>')
        arguments = self._extract_argument_mapping(call_node)

        scope = tn.FunctionCallScope(
            children=[],
            call=tn.FrontendFunctionCall(callee_name=callee_name, arguments=arguments),
        )
        # Transient metadata consumed by the inlining pass.
        scope._callee_program = callee
        scope._call_node = call_node
        scope._return_targets = return_targets
        self._append_node(scope)

    def _materialize_call_args(self, call_node: ast.Call) -> None:
        """Materialize array-valued call arguments into temporaries in-place."""
        ctx = self._expression_planning_context()
        for i, arg in enumerate(call_node.args):
            call_node.args[i] = self.expression_support.plan_expression(ctx, arg, materialize_root=True)
        for kw in call_node.keywords:
            kw.value = self.expression_support.plan_expression(ctx, kw.value, materialize_root=True)

    def _should_lower_as_library_call(self, node: ast.Call) -> bool:
        call_name = astutils.rname(node.func)
        if call_name in _INTERNAL_ITERATOR_HELPERS:
            return False
        if call_name in {'range', 'prange', 'parrange'}:
            return False
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'reshape':
            return False
        return isinstance(node.func, ast.Attribute)

    def _is_internal_iterator_helper_call(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and astutils.rname(node.func) in _INTERNAL_ITERATOR_HELPERS

    def _is_internal_iterator_binding_name(self, name: str) -> bool:
        return name.startswith('__dace_iter_')

    def _should_bind_as_reference(self, value: ast.AST, source: data.Data) -> bool:
        if isinstance(source, data.Scalar):
            return False
        return isinstance(value, ast.Name)

    def _can_promote_to_reference(self, existing: data.Data, source: data.Data) -> bool:
        if isinstance(existing, data.Scalar) or isinstance(source, data.Scalar):
            return False
        if hasattr(existing, 'is_equivalent'):
            return existing.is_equivalent(source)
        return type(existing) is type(source)

    def _evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(self.globals)
        context.update({
            name: binding.descriptor
            for name, binding in self.bindings.items() if binding.descriptor is not None
        })
        context.update(self.root.symbols)
        for scope in self.scope_stack:
            context.update(scope.symbols)
        return context

    def _numpy_lowering_context(self) -> NumpyLoweringContext:
        return NumpyLoweringContext(bindings=self.bindings,
                                    evaluation_context=self._evaluation_context,
                                    resolve_output_target=self._resolve_output_target,
                                    tasklet_name=self._tasklet_name)

    def _expression_planning_context(self) -> ExpressionPlanningContext:
        return ExpressionPlanningContext(infer_descriptor=self._infer_plannable_expression_descriptor,
                                         materialize_expression=self._materialize_temporary_expression,
                                         resolve_data_access=self._resolve_data_access,
                                         collect_input_memlets=self._collect_input_memlets,
                                         resolve_output_target=self._resolve_output_target)

    def _infer_plannable_expression_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        generic_descriptor = self.expression_support.infer_expression_descriptor(self._expression_planning_context(),
                                                                                 node)
        if generic_descriptor is not None:
            return generic_descriptor

        numpy_descriptor = self.numpy_support.infer_expression_descriptor(self._numpy_lowering_context(), node)
        if numpy_descriptor is not None:
            return numpy_descriptor

        scalar_descriptor = self._infer_scalar_descriptor(node, None)
        if scalar_descriptor is not None:
            return scalar_descriptor

        access = self._resolve_data_access(node)
        if access is not None:
            _, _, descriptor, view_descriptor = access
            result = _clone_descriptor(view_descriptor or descriptor)
            result.transient = True
            return result

        return None

    def _emit_if_chain(self, node: ast.If) -> None:
        parent = self.scope_stack[-1]
        current = node
        if_scope = tn.IfScope(condition=CodeBlock(_unparse(current.test)), children=[])
        if_scope.parent = parent
        parent.children.append(if_scope)
        self._visit_body(if_scope, current.body)

        orelse = current.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If):
            current = orelse[0]
            elif_scope = tn.ElifScope(condition=CodeBlock(_unparse(current.test)), children=[])
            elif_scope.parent = parent
            parent.children.append(elif_scope)
            self._visit_body(elif_scope, current.body)
            orelse = current.orelse

        if orelse:
            else_scope = tn.ElseScope(children=[])
            else_scope.parent = parent
            parent.children.append(else_scope)
            self._visit_body(else_scope, orelse)

    def _parse_for_indices(self, node: ast.AST) -> List[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            names = []
            for elt in node.elts:
                if not isinstance(elt, ast.Name):
                    raise TypeError('Only identifier loop targets are supported in the schedule-tree frontend')
                names.append(elt.id)
            return names
        raise TypeError('Only identifier loop targets are supported in the schedule-tree frontend')

    def _parse_for_iterator(self, node: ast.AST) -> Tuple[str, List[Tuple[str, str, str]]]:
        schedule_target = node
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            schedule_target = node.left

        if isinstance(schedule_target, ast.Call):
            iterator = astutils.rname(schedule_target.func)
            if iterator not in {'range', 'prange', 'parrange'}:
                raise TypeError(f'Unsupported for-loop iterator {iterator!r}')

            args = schedule_target.args
            if len(args) == 1:
                return 'range', [('0', _unparse(args[0]), '1')]
            if len(args) == 2:
                return 'range', [(_unparse(args[0]), _unparse(args[1]), '1')]
            if len(args) == 3:
                return 'range', [(_unparse(args[0]), _unparse(args[1]), _unparse(args[2]))]
            raise TypeError(f'Invalid number of arguments for {iterator!r}')

        if isinstance(schedule_target, ast.Subscript):
            iterator = astutils.rname(schedule_target.value)
            if iterator != 'dace.map':
                raise TypeError(f'Unsupported for-loop iterator {iterator!r}')
            return 'dace.map', self._parse_map_ranges(schedule_target)

        raise TypeError('Unsupported for-loop iterator expression in schedule-tree frontend')

    def _parse_map_ranges(self, node: ast.Subscript) -> List[Tuple[str, str, str]]:
        slice_node = node.slice
        if isinstance(slice_node, ast.Tuple):
            dims = list(slice_node.elts)
        else:
            dims = [slice_node]

        ranges: List[Tuple[str, str, str]] = []
        for dim in dims:
            if isinstance(dim, ast.Slice):
                start = '0' if dim.lower is None else _unparse(dim.lower)
                stop = _unparse(dim.upper) if dim.upper is not None else ''
                step = '1' if dim.step is None else _unparse(dim.step)
                ranges.append((start, stop, step))
            else:
                expr = _unparse(dim)
                ranges.append((expr, expr, '1'))
        return ranges
