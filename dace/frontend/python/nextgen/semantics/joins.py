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
   name to a different repository container) allocates one merged container
   and appends a full-range copy at the tail of every diverging branch — the
   φ node of the SSA-lite binding design, materialized only when needed.
3. **Conditional definitions** (a name bound on some paths only) keep the
   binding of the paths that define it; reading the name after the join when
   an undefined path was taken is a user error, matching the stable frontend.
4. Anything else (kind divergence, shape/dtype divergence, differing
   compile-time static values) raises :class:`UnsupportedFeatureError`; the
   control-flow rule rolls the whole chain back and re-lowers it as a single
   Python callback, preserving totality.
"""
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
        if kinds != {'container'}:
            raise UnsupportedFeatureError(
                f'Cannot merge conditional rebinding of "{name}" across branches '
                f'(diverging binding kinds: {sorted(kinds)})', state.context.filename, statement)

        descriptors = [state.context.containers[binding.container] for binding, _ in defined]
        merged_descriptor = _merged_descriptor(name, descriptors, statement, state)
        merged = state.context.add_container(name, merged_descriptor)

        for index, (binding, _) in enumerate(effective):
            if binding is None or binding.container == merged:
                continue  # Undefined on this path: nothing to copy
            scope = branch_scopes[index]
            if scope is None:
                scope = _implicit_else(branch_scopes, index, state)
            source_descriptor = state.context.containers[binding.container]
            copy_node = tn.CopyNode(target=merged,
                                    memlet=Memlet(data=binding.container,
                                                  subset=Range.from_array(source_descriptor),
                                                  other_subset=Range.from_array(merged_descriptor)))
            scope.add_child(copy_node)
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
