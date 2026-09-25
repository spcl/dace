# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shrinking transients to the part of them that one loop iteration uses (e.g., fields to planes, planes to
scalars)."""
import copy
from typing import Dict, List, Set, Tuple

import sympy

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (iteration_spaces, memlets_of, names_read, names_written,
                                                            repository_of)


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


def refine_loop_local_transients(stree: tn.ScheduleTreeScope) -> int:
    """
    Shrink each transient array to the part that one iteration of a loop uses, when no value it holds is used by
    another iteration: a dimension that every access indexes with exactly the variable ``v`` of an enclosing
    sequential loop (that encloses all accesses) is reduced to one element, and an array reduced to one element
    becomes a scalar. For example, a field ``T[i, j, k]`` computed and consumed within each iteration of a vertical
    loop over ``k`` becomes a plane ``T[i, j, 0]`` reused by every level, and a value that one loop body computes and
    consumes at the same point becomes a scalar. This keeps loop-local data in cache (and registers) instead of
    streaming whole fields through memory.

    Iteration ``v`` only accesses the elements at index ``v`` in that dimension, which no other iteration accesses, so
    sharing one element among all iterations is valid as long as the loop is sequential; reads of elements that the
    iteration has not written would read undefined values before and after. Transients that are viewed, accessed
    other than through single-element-per-dimension memlets, or accessed outside the loop are left alone. Run this
    while producers and consumers are in one loop (e.g., after fusing loops, before index-set splitting separates
    iterations into several loops).

    :param stree: The schedule tree to transform in place.
    :return: The number of dimensions removed.
    """
    root = stree.get_root()
    containers = root.containers
    repository = repository_of(root)
    viewed = {n.source for n in root.preorder_traversal() if isinstance(n, tn.ViewNode)}
    viewed |= {n.target for n in root.preorder_traversal() if isinstance(n, tn.ViewNode)}

    # Every memlet of each container, and containers accessed in ways not analyzed here
    accesses: Dict[str, List[Tuple[tn.ScheduleTreeNode, dict, str]]] = {}
    opaque: Set[str] = set()
    for node in root.preorder_traversal():
        covered = set()
        for attr in ('in_memlets', 'out_memlets'):
            memlets = getattr(node, attr, None)
            if isinstance(memlets, dict):
                for connector, memlet in memlets.items():
                    accesses.setdefault(memlet.data, []).append((node, memlets, connector))
                    covered.add(memlet.data)
                    if memlet.wcr is not None or memlet.other_subset is not None:
                        opaque.add(memlet.data)
            elif memlets is not None:
                opaque |= {m.data for m in memlets_of(node, attr)}
        opaque |= {m.data for m in memlets_of(node, 'memlet')}
        opaque |= (names_read(node) | names_written(node)) & containers.keys() - covered

    removed = 0
    for name, uses in accesses.items():
        desc = containers.get(name)
        if (desc is None or not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, data.View)
                or name in viewed or name in opaque or getattr(desc, 'may_alias', False)):
            continue
        if any(u[1][u[2]].subset.num_elements() != 1 for u in uses):
            continue
        # Loops enclosing every access
        loops = None
        for node, _, _ in uses:
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
                for node, memlets, connector in uses:
                    indices = [sympy.sympify(str(x)) for x in memlets[connector].subset.min_element()]
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
                contracted.add(dim)
        if not contracted:
            continue
        # Rewrite: the contracted dimensions become single-element (index 0), or the array a scalar
        shape = [1 if d in contracted else s for d, s in enumerate(desc.shape)]
        removed += len(contracted)
        if all(s == 1 for s in shape):
            containers[name] = data.Scalar(desc.dtype, transient=True, storage=desc.storage)
            for node, memlets, connector in uses:
                old = memlets[connector]
                memlets[connector] = Memlet(data=name, subset=subsets.Range([(0, 0, 1)]), dynamic=old.dynamic)
            continue
        new = copy.deepcopy(desc)
        new.set_shape(shape, strides=_packed_strides(shape, list(desc.strides)))
        containers[name] = new
        for node, memlets, connector in uses:
            old = memlets[connector]
            indices = [0 if d in contracted else x for d, x in enumerate(old.subset.min_element())]
            memlets[connector] = Memlet(data=name,
                                        subset=subsets.Range([(x, x, 1) for x in indices]),
                                        dynamic=old.dynamic)
    return removed
