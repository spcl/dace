# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shrinking transients to the part of them that one loop iteration uses (e.g., fields to planes, planes to
scalars)."""
import ast
import copy
from typing import Dict, List, Set, Tuple, cast

import sympy

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (iteration_spaces, memlets_of, names_read, names_written,
                                                            repository_of)
from dace.sdfg.analysis.schedule_tree.passes.transient_reuse import Liveness, Use, usable_accesses


def _sequential_loops(node: tn.ScheduleTreeNode) -> List[tn.ScheduleTreeScope]:
    """The loops enclosing ``node`` whose iterations run one after the other, innermost first."""
    result, parent = [], node.parent
    while parent is not None:
        if isinstance(parent, tn.ForScope):
            result.append(parent)
        elif isinstance(parent, tn.MapScope) and parent.node.map.schedule == dtypes.ScheduleType.Sequential:
            result.append(parent)
        parent = parent.parent
    return result


def _packed_strides(shape: list, strides: list) -> list:
    """Contiguous strides for ``shape`` that keep the order of the dimensions in memory given by ``strides``."""
    order = sorted(range(len(shape)),
                   key=lambda d: (symbolic.overapproximate(strides[d])
                                  if symbolic.issymbolic(strides[d]) else strides[d], d))
    result, stride = [0] * len(shape), 1
    for d in order:
        result[d] = stride
        stride = stride * shape[d]
    return result


def _condition_accesses(code: CodeBlock, containers: dict) -> Tuple[List[Tuple[str, ast.Subscript]], Set[str]]:
    """Find direct, scalar subscripts of containers in condition code; report unsupported uses as opaque."""
    if code.language != dtypes.Language.Python:
        return [], set()
    if not isinstance(code.code, list):
        return [], set(containers)
    body = cast(List[ast.stmt], code.code)
    module = ast.Module(body=body, type_ignores=[])

    parents = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    accesses, opaque = [], set()
    for name_node in (n for n in ast.walk(module)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in containers):
        parent = parents.get(name_node)
        if not isinstance(parent, ast.Subscript) or parent.value is not name_node:
            opaque.add(name_node.id)
            continue
        grandparent = parents.get(parent)
        if isinstance(grandparent, ast.Subscript) and grandparent.value is parent:
            opaque.add(name_node.id)
            continue
        indices = parent.slice.elts if isinstance(parent.slice, ast.Tuple) else [parent.slice]
        if (len(indices) != len(containers[name_node.id].shape) or any(isinstance(i, ast.Slice) for i in indices)):
            opaque.add(name_node.id)
            continue
        accesses.append((name_node.id, parent))
    return accesses, opaque


def _rewrite_condition(code: CodeBlock, accesses: List[ast.Subscript], name: str, contracted: Set[int], scalar: bool):
    if not isinstance(code.code, list):
        return
    replacements = {id(access) for access in accesses}

    class Rewriter(ast.NodeTransformer):

        def visit_Subscript(self, node):
            if id(node) not in replacements:
                return self.generic_visit(node)
            if scalar:
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            indices = [
                ast.copy_location(ast.Constant(value=0), index) if d in contracted else index
                for d, index in enumerate(indices)
            ]
            if isinstance(node.slice, ast.Tuple):
                node.slice.elts = indices
            else:
                node.slice = indices[0]
            return self.generic_visit(node)

    tree = ast.Module(body=cast(List[ast.stmt], code.code), type_ignores=[])
    tree = Rewriter().visit(tree)
    ast.fix_missing_locations(tree)
    code.code = tree.body


def refine_loop_local_transients(stree: tn.ScheduleTreeScope,
                                 trust_reads: bool = False,
                                 refine_in_conditions: bool = True) -> int:
    """
    Shrink each transient array to the part that one iteration of a loop uses, when no value it holds is used by
    another iteration: a dimension that every access indexes with exactly the variable ``v`` of an enclosing
    sequential loop (that encloses all accesses) is reduced to one element, and an array reduced to one element
    becomes a scalar. For example, a field ``T[i, j, k]`` computed and consumed within each iteration of a vertical
    loop over ``k`` becomes a plane ``T[i, j, 0]`` reused by every level, and a value that one loop body computes and
    consumes at the same point becomes a scalar. This keeps loop-local data in cache (and registers) instead of
    streaming whole fields through memory.

    Iteration ``v`` only accesses the elements at index ``v`` in that dimension, which no other iteration accesses, so
    sharing one element among all iterations is valid as long as the loop is sequential and every iteration writes
    what it reads before reading it. A transient an iteration may read before writing (e.g., halo points that a
    stencil reads but only some iterations compute) is left alone: such a read sees an element no iteration wrote
    (e.g., zero-initialized), where the shared element would hold what an earlier iteration wrote. Transients that are viewed, accessed
    other than through single-element-per-dimension memlets, or accessed outside the loop are left alone. Run this
    while producers and consumers are in one loop (e.g., after fusing loops, before index-set splitting separates
    iterations into several loops).

    :param stree: The schedule tree to transform in place.
    :param trust_reads: Assume that no iteration reads an element of a transient before writing it, i.e., that such
                        reads would see undefined values anyway (as with temporaries on the stack of each call), and
                        shrink such transients too. Results may then change where those values reach outputs (e.g.,
                        halo points that the program never computes).
    :param refine_in_conditions: Analyze and rewrite direct transient subscripts in Python conditions. A condition
        access is unsupported if it refers to a container without subscripting it, uses a slice, supplies a number of
        indices different from the container rank, or uses chained/indirect subscripting (for example,
        ``A[i][j]``). Such accesses in Python conditions remain opaque, preventing that container from being refined.
        Non-Python conditions code are not analyzed.
    :return: The number of dimensions removed.
    """
    root = stree.get_root()
    containers = root.containers
    repository = repository_of(root)
    viewed = {n.source for n in root.preorder_traversal() if isinstance(n, tn.ViewNode)}
    viewed |= {n.target for n in root.preorder_traversal() if isinstance(n, tn.ViewNode)}

    # Every memlet of each container, and containers accessed in ways not analyzed here
    accesses: Dict[str, List[Tuple[tn.ScheduleTreeNode, dict | None, str | None, ast.Subscript | None]]] = {}
    condition_accesses: Dict[str, List[Tuple[tn.ScheduleTreeNode, ast.Subscript]]] = {}
    opaque: Set[str] = set()
    for node in root.preorder_traversal():
        covered = set()
        for attr in ('in_memlets', 'out_memlets'):
            memlets = getattr(node, attr, None)
            if isinstance(memlets, dict):
                for connector, memlet in memlets.items():
                    accesses.setdefault(memlet.data, []).append((node, memlets, connector, None))
                    covered.add(memlet.data)
                    if memlet.wcr is not None or memlet.other_subset is not None:
                        opaque.add(memlet.data)
            elif memlets is not None:
                opaque |= {m.data for m in memlets_of(node, attr)}
        opaque |= {m.data for m in memlets_of(node, 'memlet')}
        condition = getattr(node, 'condition', None)
        if refine_in_conditions and isinstance(condition, CodeBlock):
            found, unsupported = _condition_accesses(condition, containers)
            opaque |= unsupported
            for name, access in found:
                accesses.setdefault(name, []).append((node, None, None, access))
                condition_accesses.setdefault(name, []).append((node, access))
                covered.add(name)
        opaque |= (names_read(node) | names_written(node)) & containers.keys() - covered

    live_uses, _ = usable_accesses(root)
    for name, scope_accesses in condition_accesses.items():  # Conditions are read before their scope runs
        for node, access in scope_accesses:
            indices = access.slice.elts if isinstance(access.slice, ast.Tuple) else [access.slice]
            memlets = {'__condition': Memlet(f'{name}[{", ".join(ast.unparse(i) for i in indices)}]')}
            live_uses.setdefault(name, []).append(Use(node, memlets, '__condition', False))
    liveness = Liveness(root, trust_reads=False)

    removed = 0
    for name, uses in accesses.items():
        desc = containers.get(name)
        if (desc is None or not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, data.View)
                or name in viewed or name in opaque or getattr(desc, 'may_alias', False)):
            continue
        if any(memlets is not None and memlets[connector].subset.num_elements() != 1
               for _, memlets, connector, _ in uses):
            continue
        # Loops enclosing every access
        loops = None
        for node, _, _, _ in uses:
            enclosing = _sequential_loops(node)
            loops = enclosing if loops is None else [l for l in loops if any(l is e for e in enclosing)]
        contracted: Set[int] = set()
        for loop in loops or []:
            spaces = iteration_spaces(loop, repository)
            for dim_index, var, space in spaces:
                if space is None:
                    continue
                v = sympy.Symbol(var)  # Indices are parsed from strings, without assumptions
                dims = set()
                for _, memlets, connector, access in uses:
                    if access is None:
                        assert memlets is not None and connector is not None
                        indices = [sympy.sympify(str(x)) for x in memlets[connector].subset.min_element()]
                    else:
                        index_nodes = access.slice.elts if isinstance(access.slice, ast.Tuple) else [access.slice]
                        indices = [sympy.sympify(ast.unparse(index)) for index in index_nodes]
                    exact = [d for d, x in enumerate(indices) if sympy.expand(x - v) == 0]
                    elsewhere = [d for d, x in enumerate(indices) if x.has(v) and d not in exact]
                    if len(exact) != 1 or elsewhere:
                        dims = None
                        break
                    dims.add(exact[0])
                if dims is None or len(dims) != 1:
                    continue
                dim = dims.pop()
                if dim in contracted or desc.shape[dim] == 1:
                    continue
                if not trust_reads and (name not in live_uses or liveness.exposed_in(loop, live_uses[name])):
                    continue  # An iteration may read an element before writing it
                contracted.add(dim)
        if not contracted:
            continue
        # Rewrite: the contracted dimensions become single-element (index 0), or the array a scalar
        shape = [1 if d in contracted else s for d, s in enumerate(desc.shape)]
        removed += len(contracted)
        if all(s == 1 for s in shape):
            containers[name] = data.Scalar(desc.dtype, transient=True, storage=desc.storage)
            for _, memlets, connector, _ in uses:
                if memlets is None:
                    continue
                old = memlets[connector]
                memlets[connector] = Memlet(data=name, subset=subsets.Range([(0, 0, 1)]), dynamic=old.dynamic)
            grouped = {}
            for node, access in condition_accesses.get(name, []):
                grouped.setdefault(id(node), (node.condition, []))[1].append(access)
            for code, code_accesses in grouped.values():
                _rewrite_condition(code, code_accesses, name, contracted, scalar=True)
            continue
        new = copy.deepcopy(desc)
        new.set_shape(shape, strides=_packed_strides(shape, list(desc.strides)))
        containers[name] = new
        for _, memlets, connector, _ in uses:
            if memlets is None:
                continue
            old = memlets[connector]
            indices = [0 if d in contracted else x for d, x in enumerate(old.subset.min_element())]
            memlets[connector] = Memlet(data=name,
                                        subset=subsets.Range([(x, x, 1) for x in indices]),
                                        dynamic=old.dynamic)
        grouped = {}
        for node, access in condition_accesses.get(name, []):
            grouped.setdefault(id(node), (node.condition, []))[1].append(access)
        for code, code_accesses in grouped.values():
            _rewrite_condition(code, code_accesses, name, contracted, scalar=False)
    return removed
