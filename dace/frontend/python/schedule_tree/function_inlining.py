# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Bottom-up parallel inlining of ``@dace.program`` calls in schedule trees.

After the schedule-tree builder emits :class:`FunctionCallScope` placeholders
for every nested ``@dace.program`` call, :func:`resolve_function_calls`
collects them, generates the callee schedule trees (in parallel when there are
multiple independent callees), and inlines the results.
"""

import ast
import copy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from dace.cli.progress import optional_progressbar
from dace import data
from dace.data.pydata import PythonClass
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.utils import find_new_name

# -------------------------------------------------------------------- #
#  Public entry point                                                    #
# -------------------------------------------------------------------- #


def resolve_function_calls(root: tn.ScheduleTreeRoot) -> None:
    """
    Collect all :class:`FunctionCallScope` nodes in *root*, parse each
    callee's schedule tree (in parallel where possible), and inline the
    results.

    Callee schedule trees may themselves contain nested calls; since
    ``to_schedule_tree`` calls ``resolve_function_calls`` recursively,
    bottom-up ordering is automatic — leaf-level callees are fully
    resolved before their callers.
    """
    scopes = _collect_function_call_scopes(root)
    if not scopes:
        return

    # Build callee schedule trees — keyed by (callee_id, arg_types).
    callee_trees = _build_callee_trees(scopes)

    # Inline each call site.
    for scope in scopes:
        key = _callee_key(scope)
        callee_tree = callee_trees[key]
        _inline_callee(scope, callee_tree, root)


# -------------------------------------------------------------------- #
#  Collecting FunctionCallScope nodes                                    #
# -------------------------------------------------------------------- #


def _collect_function_call_scopes(root: tn.ScheduleTreeRoot) -> List[tn.FunctionCallScope]:
    result: List[tn.FunctionCallScope] = []
    for node in root.preorder_traversal():
        if isinstance(node, tn.FunctionCallScope):
            result.append(node)
    return result


# -------------------------------------------------------------------- #
#  Building callee schedule trees (with parallelism)                     #
# -------------------------------------------------------------------- #


def _callee_key(scope: tn.FunctionCallScope) -> int:
    """
    Return a hashable key for a call-scope's callee.

    For now this is just the callee identity.  When we need to support
    multiple specialisations of the same function for different argument
    types, this should be extended to include the type signature.
    """
    return hash((id(scope._callee_program), _specialization_key(scope)))


def _specialization_key(scope: tn.FunctionCallScope) -> Tuple:
    return (_specialization_values_key(getattr(scope, '_call_args',
                                               [])), _specialization_kwargs_key(getattr(scope, '_call_kwargs', {})),
            tuple(
                sorted((name, ast.dump(lambda_node))
                       for name, lambda_node in getattr(scope, '_lambda_bindings', {}).items())),
            tuple(sorted((name, id(value)) for name, value in getattr(scope, '_callable_bindings', {}).items())))


def _specialization_values_key(values: List[object]) -> Tuple:
    return tuple(_specialization_value_key(value) for value in values)


def _specialization_kwargs_key(values: Dict[str, object]) -> Tuple:
    return tuple(sorted((name, _specialization_value_key(value)) for name, value in values.items()))


def _specialization_value_key(value: object) -> Tuple[str, str]:
    if isinstance(value, data.Data):
        return ('descriptor', repr(value))
    return (type(value).__name__, repr(value))


def _build_callee_trees(scopes: List[tn.FunctionCallScope]) -> Dict[int, tn.ScheduleTreeRoot]:
    """
    For every unique callee referenced by *scopes*, build its schedule
    tree.  When there are multiple independent callees, parse them in
    parallel.  Returns a mapping from :func:`_callee_key` to the parsed
    :class:`ScheduleTreeRoot`.
    """
    # De-duplicate by callee identity.
    unique: Dict[int, tn.FunctionCallScope] = {}
    for scope in scopes:
        key = _callee_key(scope)
        if key not in unique:
            unique[key] = scope

    if len(unique) == 1:
        # Only one callee — no need for thread overhead.
        scope = next(iter(unique.values()))
        tree = _parse_callee(scope)
        return {_callee_key(scope): tree}

    results: Dict[int, tn.ScheduleTreeRoot] = {}
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_parse_callee, scope): key for key, scope in unique.items()}
        completed = optional_progressbar(as_completed(futures), title='Parsing nested DaCe functions', n=len(futures))
        for future in completed:
            results[futures[future]] = future.result()
    return results


def _parse_callee(scope: tn.FunctionCallScope) -> tn.ScheduleTreeRoot:
    """
    Parse a callee ``DaceProgram`` into its schedule tree.

    The callee's ``to_schedule_tree`` method triggers preprocessing +
    schedule-tree building + recursive ``resolve_function_calls``, so
    leaf-level callees are fully inlined before we return.
    """
    callee = scope._callee_program
    call_args = tuple(getattr(scope, '_call_args', []))
    call_kwargs = dict(getattr(scope, '_call_kwargs', {}))
    lambda_bindings = dict(getattr(scope, '_lambda_bindings', {}))
    callable_bindings = dict(getattr(scope, '_callable_bindings', {}))

    if hasattr(callee, '__schedule_tree__'):
        return callee.__schedule_tree__(*call_args,
                                        lambda_bindings=lambda_bindings,
                                        callable_bindings=callable_bindings,
                                        **call_kwargs)

    return callee._generate_schedule_tree(call_args,
                                          call_kwargs,
                                          lambda_bindings=lambda_bindings,
                                          callable_bindings=callable_bindings)


# -------------------------------------------------------------------- #
#  Inlining a callee tree into a FunctionCallScope                       #
# -------------------------------------------------------------------- #


def _inline_callee(scope: tn.FunctionCallScope, callee_tree: tn.ScheduleTreeRoot,
                   caller_root: tn.ScheduleTreeRoot) -> None:
    """
    Inline *callee_tree* into *scope*, renaming containers to match the
    caller's namespace, merging descriptors, and rewriting return nodes.
    """
    arguments = scope.call.arguments  # callee_param -> caller_expr
    callee_arg_names = set(callee_tree.arg_names)
    captured_names = set(getattr(scope, '_captured_names', set()))

    # 1. Build rename map: callee name -> caller name.
    rename_map = _build_rename_map(arguments, callee_tree, caller_root, callee_arg_names, captured_names)

    # 2. Deep-copy callee body.
    body = copy.deepcopy(callee_tree.children)

    # 3. Rename all data references in the cloned body.
    renamer = _ContainerRenamer(rename_map)
    body = [renamer.visit(child) for child in body]
    body = [child for child in body if child is not None]

    # 4. Propagate callee argument descriptor upgrades back to the caller.
    _propagate_argument_descriptor_updates(arguments, callee_tree, caller_root)

    # 5. Merge callee transient containers and symbols into the caller.
    for cname, desc in callee_tree.containers.items():
        new_name = rename_map.get(cname, cname)
        if new_name not in caller_root.containers:
            caller_root.containers[new_name] = copy.deepcopy(desc)
    for sname, stype in callee_tree.symbols.items():
        caller_root.symbols.setdefault(sname, stype)

    # 6. Merge callee constants and callbacks.
    for cname, cval in callee_tree.constants.items():
        caller_root.constants.setdefault(cname, cval)
    for cbname, cbval in callee_tree.callback_mapping.items():
        caller_root.callback_mapping.setdefault(cbname, cbval)

    # 7. Handle return values.
    return_targets = getattr(scope, '_return_targets', None)
    body = _rewrite_returns(body, return_targets, rename_map)

    # 8. Populate the scope.
    scope.children = body
    for child in body:
        child.parent = scope


def _propagate_argument_descriptor_updates(arguments: Dict[str, str], callee_tree: tn.ScheduleTreeRoot,
                                           caller_root: tn.ScheduleTreeRoot) -> None:
    for callee_param, caller_expr in arguments.items():
        callee_descriptor = callee_tree.containers.get(callee_param)
        if not isinstance(callee_descriptor, PythonClass):
            continue
        caller_descriptor = caller_root.containers.get(caller_expr)
        if caller_descriptor is None or isinstance(caller_descriptor, PythonClass):
            continue
        caller_root.containers[caller_expr] = copy.deepcopy(callee_descriptor)


# -------------------------------------------------------------------- #
#  Rename-map construction                                               #
# -------------------------------------------------------------------- #


def _build_rename_map(arguments: Dict[str, str], callee_tree: tn.ScheduleTreeRoot, caller_root: tn.ScheduleTreeRoot,
                      callee_arg_names: Set[str], captured_names: Set[str]) -> Dict[str, str]:
    """
    Build ``{callee_name: caller_name}`` for every container in the
    callee's schedule tree.

    * Arguments are mapped via *arguments* (callee_param -> caller_expr).
    * Transients that collide with caller names get fresh names.
    """
    rename: Dict[str, str] = {}
    occupied = set(caller_root.containers.keys())

    # Map callee parameters to caller arguments.
    for callee_param, caller_expr in arguments.items():
        rename[callee_param] = caller_expr

    # Explicit global/nonlocal captures must keep the caller-visible name.
    for captured_name in captured_names:
        rename[captured_name] = captured_name
        occupied.add(captured_name)

    # Handle callee-internal transients (everything not in arg_names).
    for cname in callee_tree.containers:
        if cname in rename:
            continue
        if cname in callee_arg_names:
            # Argument not in the mapping (e.g. default-valued) — keep as-is.
            continue
        new_name = find_new_name(cname, list(occupied))
        rename[cname] = new_name
        occupied.add(new_name)

    return rename


# -------------------------------------------------------------------- #
#  Container renaming transformer                                        #
# -------------------------------------------------------------------- #


class _ContainerRenamer(tn.ScheduleNodeTransformer):
    """Rename data-container references throughout a schedule sub-tree."""

    def __init__(self, rename_map: Dict[str, str]) -> None:
        self._map = {k: v for k, v in rename_map.items() if k != v}

    # -- helpers --------------------------------------------------------

    def _rename(self, name: str) -> str:
        return self._map.get(name, name)

    def _rename_memlet(self, memlet: Optional[Memlet]) -> Optional[Memlet]:
        if memlet is None:
            return None
        if memlet.data in self._map:
            memlet.data = self._map[memlet.data]
        return memlet

    def _rename_memlet_dict(self, d):
        if isinstance(d, dict):
            return {k: self._rename_memlet(copy.deepcopy(m)) for k, m in d.items()}
        if isinstance(d, set):
            return {self._rename_memlet(copy.deepcopy(m)) for m in d}
        return d

    def _rename_code_block(self, cb: Optional[CodeBlock]) -> Optional[CodeBlock]:
        if cb is None:
            return None
        text = cb.as_string
        for old, new in self._map.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
        return CodeBlock(text)

    # -- leaf node visitors ---------------------------------------------

    def visit_CopyNode(self, node: tn.CopyNode):
        node.target = self._rename(node.target)
        self._rename_memlet(node.memlet)
        return node

    def visit_DynScopeCopyNode(self, node: tn.DynScopeCopyNode):
        node.target = self._rename(node.target)
        self._rename_memlet(node.memlet)
        return node

    def visit_ViewNode(self, node: tn.ViewNode):
        node.target = self._rename(node.target)
        node.source = self._rename(node.source)
        self._rename_memlet(node.memlet)
        return node

    def visit_RefSetNode(self, node: tn.RefSetNode):
        node.target = self._rename(node.target)
        self._rename_memlet(node.memlet)
        if node.source_expr is not None:
            for old, new in self._map.items():
                node.source_expr = re.sub(r'\b' + re.escape(old) + r'\b', new, node.source_expr)
        return node

    def visit_TaskletNode(self, node: tn.TaskletNode):
        node.in_memlets = self._rename_memlet_dict(node.in_memlets)
        node.out_memlets = self._rename_memlet_dict(node.out_memlets)
        if isinstance(node.node, tn.FrontendTasklet):
            node.node = tn.FrontendTasklet(name=node.node.name, code=self._rename_code_block(node.node.code))
        return node

    def visit_LibraryCall(self, node: tn.LibraryCall):
        node.in_memlets = self._rename_memlet_dict(node.in_memlets)
        node.out_memlets = self._rename_memlet_dict(node.out_memlets)
        return node

    def visit_AssignNode(self, node: tn.AssignNode):
        node.name = self._rename(node.name)
        node.value = self._rename_code_block(node.value)
        return node

    def visit_ReassignExternalNode(self, node: tn.ReassignExternalNode):
        node.value = self._rename_code_block(node.value)
        return node

    def visit_StatementNode(self, node: tn.StatementNode):
        node.code = self._rename_code_block(node.code)
        return node

    def visit_PythonCallbackNode(self, node: tn.PythonCallbackNode):
        node.code = self._rename_code_block(node.code)
        if node.outlined_function_code is not None:
            node.outlined_function_code = self._rename_code_block(node.outlined_function_code)
        if node.outlined_call_code is not None:
            node.outlined_call_code = self._rename_code_block(node.outlined_call_code)
        return node

    def visit_RaiseNode(self, node: tn.RaiseNode):
        if node.exception_type is not None:
            node.exception_type = self._rename_code_block(node.exception_type)
        node.args = [self._rename_code_block(argument) for argument in node.args]
        node.kwargs = {name: self._rename_code_block(value) for name, value in node.kwargs.items()}
        return node

    def visit_ReturnNode(self, node: tn.ReturnNode):
        node.values = [self._rename(v) for v in node.values]
        return node


# -------------------------------------------------------------------- #
#  Return-value rewriting                                                #
# -------------------------------------------------------------------- #


def _rewrite_returns(body: List[tn.ScheduleTreeNode], return_targets: Optional[List[str]],
                     rename_map: Dict[str, str]) -> List[tn.ScheduleTreeNode]:
    """
    Replace :class:`ReturnNode` instances in *body* with assignments to
    *return_targets*.  If *return_targets* is ``None`` (the call was used
    as a bare statement), remove ``ReturnNode`` instances entirely.
    """
    result: List[tn.ScheduleTreeNode] = []
    for node in body:
        if isinstance(node, tn.ReturnNode):
            if return_targets and node.values:
                for target, value_name in zip(return_targets, node.values):
                    if value_name != target:
                        result.append(tn.AssignNode(name=target, value=CodeBlock(value_name)))
            # else: bare call — drop the return
        elif isinstance(node, tn.ScheduleTreeScope):
            node.children = _rewrite_returns(node.children, return_targets, rename_map)
            result.append(node)
        else:
            result.append(node)
    return result
