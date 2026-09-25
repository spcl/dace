# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Flattening loop nests that traverse contiguous memory into single loops."""
import ast
from typing import Dict, Optional, Tuple

import sympy

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (iteration_spaces, names_in_subtrees, names_read,
                                                            names_written, range_analysis, repository_of, trip_count)
from dace.sdfg.state import LoopRegion


def _loop(scope: tn.ScheduleTreeNode, repository) -> Optional[Tuple[str, object]]:
    """``(variable, iteration space)`` of a ``for`` loop with unit step, or ``None``."""
    if not isinstance(scope, tn.ForScope):
        return None
    spaces = iteration_spaces(scope, repository)
    if len(spaces) != 1 or spaces[0][2] is None or spaces[0][2].stride != 1:
        return None
    variable, space = spaces[0][1:]
    bounds = {str(s) for s in sympy.sympify(space.start).free_symbols | sympy.sympify(space.end).free_symbols}
    if variable in space.body_defined or bounds & space.body_defined:
        return None  # The body changes the loop variable or the bounds
    return variable, space


def _is_zero(expr) -> bool:
    return sympy.simplify(sympy.sympify(expr)) == 0


def _flat_access(memlet: Memlet, desc: data.Data, outer: sympy.Symbol, inner: sympy.Symbol, start: int,
                 count: int) -> Optional[tuple]:
    """How a single-element access indexes the flattened loop: ``()`` if it depends on neither loop variable,
    ``(inner dimension, outer dimension, offset)`` if the inner variable indexes a whole contiguous row of ``desc``
    (``index = inner + c`` over the row ``[0, count)``) and the outer variable the next row (``outer + d``, the
    dimension's stride being ``count`` times the inner one), so that the element is ``flat + c + count * d`` in a view
    that merges the two dimensions. ``None`` otherwise."""
    if memlet.subset.num_elements() != 1 or memlet.wcr is not None:
        return None
    indices = [sympy.expand(sympy.sympify(i)) for i in memlet.subset.min_element()]
    uses = [(k, e) for k, e in enumerate(indices) if e.has(outer) or e.has(inner)]
    if not uses:
        return ()
    if (not isinstance(desc, data.Array) or isinstance(desc, data.View) or len(uses) != 2
            or any(not _is_zero(o) for o in desc.offset)):
        return None
    inner_dims = [(k, e - inner) for k, e in uses if e.coeff(inner) == 1 and not (e - inner).has(inner, outer)]
    outer_dims = [(k, e - outer) for k, e in uses if e.coeff(outer) == 1 and not (e - outer).has(inner, outer)]
    if len(inner_dims) != 1 or len(outer_dims) != 1:
        return None
    (row_dim, row_offset), (next_dim, next_offset) = inner_dims[0], outer_dims[0]
    # The inner loop covers the whole row, halo and padding included, and rows are adjacent in memory
    if not _is_zero(desc.shape[row_dim] - count) or not _is_zero(start + row_offset):
        return None
    if not _is_zero(desc.strides[next_dim] - count * desc.strides[row_dim]):
        return None
    return row_dim, next_dim, row_offset + count * next_offset


def flatten_contiguous_nests(stree: tn.ScheduleTreeScope, min_outer_trip_count: int = 2) -> int:
    """
    Merge a loop and the loop it immediately contains into one loop when together they traverse contiguous memory.

    ``for j in range(J): for i in range(0, S): B[i, j + 1] = f(A[i, j])`` becomes
    ``for t in range(0, S * J): B_flat[t + S] = f(A_flat[t])``, where ``A_flat`` and ``B_flat`` are views that merge
    the two dimensions. This removes the short inner loop and its remainder handling from vectorized code. Applies to
    a perfect nest of unit-step ``for`` loops whose inner loop has a constant trip count ``S`` and a body of tasklets
    that do not refer to the loop variables other than in single-element accesses, where every access that depends on
    the loop variables indexes one dimension of extent exactly ``S`` with ``i + c`` over the whole dimension (so rows
    with halos or padding that the loop does not cover, or strides that pad rows, keep the nest) and one dimension
    with ``j + d`` whose stride is ``S`` times the first's. The iterations run in the same order as before.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param min_outer_trip_count: Only flatten nests whose outer loop runs at least this many times, when known.
    :return: The number of loop nests flattened.
    """
    root = stree.get_root()
    repository = repository_of(root)
    containers = root.containers
    taken = set(root.symbols) | set(containers)
    taken |= names_in_subtrees([root], names_read) | names_in_subtrees([root], names_written)
    views: Dict[tuple, str] = {}
    counter = [0]

    def fresh(base: str) -> str:
        while f'{base}{counter[0]}' in taken:
            counter[0] += 1
        name = f'{base}{counter[0]}'
        taken.add(name)
        return name

    def view_of(name: str, row_dim: int, next_dim: int) -> str:
        key = (name, row_dim, next_dim)
        if key not in views:
            desc = containers[name]
            shape = list(desc.shape)
            strides = list(desc.strides)
            shape[row_dim] = shape[row_dim] * shape[next_dim]
            del shape[next_dim], strides[next_dim]
            view = data.ArrayView(desc.dtype,
                                  shape,
                                  transient=True,
                                  storage=desc.storage,
                                  strides=strides,
                                  offset=[0] * len(shape),
                                  total_size=desc.total_size,
                                  lifetime=dtypes.AllocationLifetime.Scope)
            views[key] = data.find_new_name(f'{name}_flat', containers)
            containers[views[key]] = view
            taken.add(views[key])
        return views[key]

    flat_names: Dict[Tuple[str, str], str] = {}

    def flatten(outer_scope: tn.ForScope, defined: set) -> Optional[list]:
        outer = _loop(outer_scope, repository)
        if outer is None or len(outer_scope.children) != 1:
            return None
        inner_scope = outer_scope.children[0]
        inner = _loop(inner_scope, repository)
        if inner is None:
            return None
        (outer_var, outer_space), (inner_var, inner_space) = outer, inner
        count = trip_count(inner_space)
        start = range_analysis()._num(inner_space.start)
        if count is None or count < 1 or start is None:
            return None
        outer_count = trip_count(outer_space)
        if outer_count is not None and outer_count < min_outer_trip_count:
            return None
        body = inner_scope.children
        if not body or not all(isinstance(s, tn.TaskletNode) for s in body):
            return None
        o, i = symbolic.symbol(outer_var), symbolic.symbol(inner_var)
        plan = []
        for statement in body:
            code_names = {n.id for s in statement.node.code.code for n in ast.walk(s) if isinstance(n, ast.Name)}
            if {outer_var, inner_var} & code_names:
                return None
            accesses = {}
            for kind, memlets in (('in', statement.in_memlets), ('out', statement.out_memlets)):
                for connector, memlet in memlets.items():
                    if memlet.data not in containers:
                        return None
                    access = _flat_access(memlet, containers[memlet.data], o, i, start, count)
                    if access is None:
                        return None
                    accesses[kind, connector] = access
            plan.append(accesses)
        # Rewrite
        # Loops flattened from the same variables share a name, so that contiguous ones can be merged
        if (outer_var, inner_var) not in flat_names:
            flat_names[outer_var, inner_var] = fresh('__flat')
        flat_var = flat_names[outer_var, inner_var]
        t = symbolic.symbol(flat_var)
        for statement, accesses in zip(body, plan):
            for (kind, connector), access in accesses.items():
                if access == ():
                    continue
                memlets = statement.in_memlets if kind == 'in' else statement.out_memlets
                memlet = memlets[connector]
                row_dim, next_dim, offset = access
                indices = list(memlet.subset.min_element())
                indices[row_dim] = t + offset
                del indices[next_dim]
                name = view_of(memlet.data, row_dim, next_dim)
                memlets[connector] = Memlet(data=name,
                                            subset=subsets.Range([(x, x, 1) for x in indices]),
                                            dynamic=memlet.dynamic)
        first = start + count * outer_space.start
        end = start + count * (outer_space.end + 1)
        header = LoopRegion(f'{outer_scope.loop.label}_flat', f'{flat_var} < {symbolic.symstr(end)}', flat_var,
                            f'{flat_var} = {symbolic.symstr(first)}', f'{flat_var} = {flat_var} + 1')
        flat = tn.ForScope(loop=header, children=[])
        flat.add_children(body)
        definitions = []
        for (name, row_dim, next_dim), view in views.items():
            if view in defined:
                continue  # Defined before an earlier loop of the same scope
            if any(m.data == view for s in body for m in list(s.in_memlets.values()) + list(s.out_memlets.values())):
                defined.add(view)
                definitions.append(
                    tn.ViewNode(target=view,
                                source=name,
                                memlet=Memlet.from_array(name, containers[name]),
                                src_desc=containers[name],
                                view_desc=containers[view]))
        return definitions + [flat]

    flattened = 0

    def visit(scope: tn.ScheduleTreeScope):
        nonlocal flattened
        result, changed, defined = [], False, set()
        for child in scope.children:
            replacement = flatten(child, defined) if isinstance(child, tn.ForScope) else None
            if replacement is None:
                if isinstance(child, tn.ScheduleTreeScope):
                    visit(child)
                result.append(child)
                continue
            result += replacement
            flattened += 1
            changed = True
        if changed:
            scope.children = []
            scope.add_children(result)

    visit(stree)
    return flattened
