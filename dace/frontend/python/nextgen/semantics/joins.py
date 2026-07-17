# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Join merging for branch-scoped name bindings.

After lowering an ``if``/``elif``/``else`` chain, each branch may have rebound
names independently. This module reconciles the branch-end binding states into
a single post-join state:

1. **Identical bindings** across all paths are kept as-is. This is the
   dominant case: scalar names are containers updated in place, so plain
   conditional assignments never diverge.
2. **Shape- and dtype-compatible container divergence** (each path bound the
   name to a different repository container) merges — the φ node of the
   SSA-lite binding design, materialized only when needed. Array bindings
   merge through a :class:`~dace.data.Reference` container that each branch
   re-points with a :class:`RefSetNode` (a pointer set, preserving Python
   aliasing semantics for writes after the join); scalar bindings, which are
   immutable in Python, merge through a plain container with a full-range
   copy at each branch tail.
3. **Conditional definitions** (a name bound on some paths only) keep the
   binding of the paths that define it; reading the name after the join when
   an undefined path was taken is a user error, matching the stable frontend.
4. **Diverging compile-time sequences** (differing static values, or a static
   value on one path and a container on another) materialize the static
   paths as constant containers and join the container merge of rule 2.
5. Anything else (kind divergence, shape/dtype divergence, non-constant
   sequence elements) raises :class:`UnsupportedFeatureError`; the
   control-flow rule rolls the whole chain back and re-lowers it as a single
   Python callback, preserving totality.
"""
import copy
from typing import List, Optional

from dace import data
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.subsets import Range
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.semantics.context import Binding, BindingSnapshot


def merge_branches(before: BindingSnapshot, branch_ends: List[BindingSnapshot],
                   branch_scopes: List[Optional[tn.ScheduleTreeScope]], statement, state) -> None:
    """
    Reconcile the binding states at the ends of a branch chain and update the
    context to the merged post-join state.

    :param before: Binding state before the chain (the caller must have
                   restored the context to it already).
    :param branch_ends: Binding state at the end of each path. A path that
                        does not execute any branch body (the implicit
                        fall-through of a chain without ``else``) is
                        represented by ``before`` itself.
    :param branch_scopes: The emitted scope of each path, aligned with
                          ``branch_ends``; None for the implicit fall-through
                          (an ``ElseScope`` is created lazily if that path
                          needs a merge copy).
    :param statement: The source statement, for error locations.
    :param state: The active lowering state.
    :raises UnsupportedFeatureError: If the bindings cannot be merged soundly.
    """
    names = set(before.bindings)
    for end in branch_ends:
        names.update(end.bindings)

    for name in sorted(names):
        before_binding = before.bindings.get(name)
        # Per-path effective (binding, static value): paths that never rebound
        # the name keep its pre-chain binding.
        effective = []
        for end in branch_ends:
            binding = end.bindings.get(name, before_binding)
            static = None
            if binding is not None and binding.kind == 'static':
                static = end.static_values.get(name, before.static_values.get(name))
            effective.append((binding, static))
        defined = [(binding, static) for binding, static in effective if binding is not None]
        if not defined:
            continue

        reference_binding, reference_static = defined[0]
        if all(_same_binding(binding, reference_binding, static, reference_static) for binding, static in defined):
            _reinstate(name, reference_binding, reference_static, state)
            continue

        kinds = {binding.kind for binding, _ in defined}
        if not kinds <= {'container', 'static'}:
            raise UnsupportedFeatureError(
                f'Cannot merge conditional rebinding of "{name}" across branches '
                f'(diverging binding kinds: {sorted(kinds)})', state.context.filename, statement)

        # Per-path container names; compile-time sequences materialize as
        # constant containers so they can join the container merge. Sequences
        # with non-constant elements raise, which the control-flow rule turns
        # into a whole-chain callback.
        path_containers: List[Optional[str]] = []
        for binding, static in effective:
            if binding is None:
                path_containers.append(None)
            elif binding.kind == 'container':
                path_containers.append(binding.container)
            else:
                # Deferred import: the mechanism layer imports semantics
                from dace.frontend.python.nextgen.lowering.mechanisms import static_values
                access = static_values.materialize(static, state, name_hint=f'__const_{name}')
                path_containers.append(access.container)

        descriptors = [state.context.containers[container] for container in path_containers if container is not None]
        merged_descriptor = _merged_descriptor(name, descriptors, statement, state)
        # Arrays are mutable: the merged name must alias the branch's actual
        # container so post-join writes reach it — a runtime reference set,
        # not a copy. Scalars are immutable, so a value copy is exact.
        reference_merge = isinstance(merged_descriptor, data.Array)
        if reference_merge:
            merged_descriptor = data.Reference.view(merged_descriptor)
        merged = state.context.add_container(name, merged_descriptor)

        for index, container in enumerate(path_containers):
            if container is None or container == merged:
                continue  # Undefined on this path: nothing to merge
            scope = branch_scopes[index]
            if scope is None:
                scope = _implicit_else(branch_scopes, index, state)
            source_descriptor = state.context.containers[container]
            if reference_merge:
                merge_node = tn.RefSetNode(target=merged,
                                           memlet=Memlet(data=container, subset=Range.from_array(source_descriptor)),
                                           src_desc=source_descriptor,
                                           ref_desc=copy.deepcopy(merged_descriptor))
            else:
                merge_node = tn.CopyNode(target=merged,
                                         memlet=Memlet(data=container,
                                                       subset=Range.from_array(source_descriptor),
                                                       other_subset=Range.from_array(merged_descriptor)))
            scope.add_child(merge_node)
        state.context.bind(name, merged)


def _same_binding(a: Binding, b: Binding, a_static, b_static) -> bool:
    """Whether two bindings denote the same meaning (φ-free join)."""
    if a.kind != b.kind:
        return False
    if a.kind == 'container':
        return a.container == b.container
    if a.kind == 'static':
        return a_static is b_static
    return True


def _reinstate(name: str, binding: Binding, static, state) -> None:
    """Adopt an agreed-upon binding (e.g., a conditional definition) into the
    restored post-chain context."""
    state.context.bindings[name] = binding
    if binding.kind == 'static' and static is not None:
        state.context.static_values[name] = static


def _merged_descriptor(name: str, descriptors: List[data.Data], statement, state) -> data.Data:
    """A fresh descriptor for the merged container, requiring dtype and shape
    agreement across all paths."""
    first = descriptors[0]
    for descriptor in descriptors[1:]:
        if descriptor.dtype != first.dtype:
            raise UnsupportedFeatureError(f'Cannot merge conditional rebinding of "{name}": dtype mismatch',
                                          state.context.filename, statement)
        if tuple(descriptor.shape) != tuple(first.shape):
            raise UnsupportedFeatureError(f'Cannot merge conditional rebinding of "{name}": shape mismatch',
                                          state.context.filename, statement)
        if isinstance(descriptor, data.View) or isinstance(first, data.View):
            raise UnsupportedFeatureError(f'Cannot merge conditional rebinding of "{name}" through views',
                                          state.context.filename, statement)
    if isinstance(first, data.Scalar):
        return data.Scalar(first.dtype)
    return data.Array(first.dtype, list(first.shape))


def _implicit_else(branch_scopes: List[Optional[tn.ScheduleTreeScope]], index: int, state) -> tn.ScheduleTreeScope:
    """Lazily create the ElseScope of the implicit fall-through path, so merge
    copies for the not-taken case have a home."""
    scope = tn.ElseScope(children=[])
    state.emitter.emit(scope)
    branch_scopes[index] = scope
    return scope
