# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from dace import data as dt, Memlet, SDFG
from dace.sdfg.analysis.schedule_tree import treenodes as tnodes
from dace.sdfg.analysis.schedule_tree import utils as tutils
from dace.sdfg.graph import NodeNotFoundError
from typing import Dict
from warnings import warn


_dataflow_nodes = (tnodes.ViewNode, tnodes.RefSetNode, tnodes.CopyNode, tnodes.DynScopeCopyNode, tnodes.TaskletNode, tnodes.LibraryCall)


def _update_memlets(data: Dict[str, dt.Data], memlets: Dict[str, Memlet], index: str, replace: Dict[str, bool]):
    for conn, memlet in memlets.items():
        if memlet.data in data:
            subset = index if replace[memlet.data] else f"{index}, {memlet.subset}"
            memlets[conn] = Memlet(data=memlet.data, subset=subset)


def _augment_data(data: Dict[str, dt.Data], map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode, sdfg: SDFG):
    
    # Generate map-related indices, sizes and, strides
    map = map_scope.node.map
    index = ", ".join(map.params)
    size = map.range.size()
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i+1] * size[i+1]

    # Augment data descriptors
    replace = dict()
    for name, desc in data.items():
        if isinstance(desc, dt.Scalar):
            sdfg.arrays[name] = dt.Array(desc.dtype, size, True, storage=desc.storage)
            replace[name] = True
        else:
            mult = desc.shape[0]
            desc.shape = (*size, *desc.shape)
            new_strides = [s * mult for s in strides]
            desc.strides = (*new_strides, *desc.strides)
            replace[name] = False
    
    # Update memlets
    frontier = list(tree.children)
    while frontier:
        node = frontier.pop()
        if isinstance(node, _dataflow_nodes):
            try:
                _update_memlets(data, node.in_memlets, index, replace)
                _update_memlets(data, node.out_memlets, index, replace)
            except AttributeError:
                subset = index if replace[node.target] else f"{index}, {node.memlet.subset}"
                node.memlet = Memlet(data=node.target, subset=subset)
        if hasattr(node, 'children'):
            frontier.extend(node.children)


def map_fission(map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode, sdfg: SDFG) -> bool:
    """
    Applies the MapFission transformation to the input MapScope.

    :param map_scope: The MapScope.
    :param tree: The ScheduleTree.
    :param sdfg: The (top-level) SDFG.
    :return: True if the transformation applies successfully, otherwise False.
    """

    ####################################
    # Check if MapFission can be applied

    # Basic check: cannot fission an empty MapScope or one that has a single dataflow child.
    num_children = len(map_scope.children)
    if num_children == 0 or (num_children == 1 and isinstance(map_scope.children[0], _dataflow_nodes)):
        return False
    
    # State-scope check: if the body consists of a single state-scope, certain conditions apply.
    partition = tutils.partition_scope_body(map_scope)
    if len(partition) == 1:
        child = partition[0]
        conditions = []
        if isinstance(child, list):
            # If-Elif-Else-Scope
            for c in child:
                if isinstance(c, (tnodes.IfScope, tnodes.ElifScope)):
                    conditions.append(c.condition)
        elif isinstance(child, tnodes.ForScope):
            conditions.append(child.header.condition)
        elif isinstance(child, tnodes.WhileScope):
            conditions.append(child.header.test)
        for cond in conditions:
            map = map_scope.node.map
            if any(p in cond.get_free_symbols() for p in map.params):
                return False
            # TODO: How to run the check below in the ScheduleTree?
            # for s in cond.get_free_symbols():
            #     for e in graph.edges_by_connector(self.nested_sdfg, s):
            #         if any(p in e.data.free_symbols for p in map.params):
            #             return False
    
    data_to_augment = dict() 
    for scope in partition:
        if isinstance(scope, _dataflow_nodes):
            try:
                _, _, sd = tutils.find_tnode_in_sdfg(scope, sdfg)
                for _, memlet in scope.out_memlets.items():
                    data_to_augment[memlet.data] = sd.arrays[memlet.data]
            except NodeNotFoundError:
                warn(f"Tree node {scope} not found in SDFG {sdfg}. Switching to unsafe data lookup.")
                for _, memlet in scope.out_memlets.items():
                    for sd in sdfg.all_sdfgs_recursive():
                        if memlet.data in sd.arrays:
                            data_to_augment[memlet.data] = sd.arrays[memlet.data]
                            break
            except NotImplementedError:
                warn(f"Tree node {scope} is unsupported. Switching to unsafe data lookup.")
                for sd in sdfg.all_sdfgs_recursive():
                    if scope.target in sd.arrays:
                        data_to_augment[scope.target] = sd.arrays[scope.target]
                        break
    _augment_data(data_to_augment, map_scope, tree, sdfg)
    
    parent_scope = tutils.find_parent(map_scope, tree)
    idx = parent_scope.children.index(map_scope)
    parent_scope.children.pop(idx)
    while len(partition) > 0:
        child_scope = partition.pop()
        if not isinstance(child_scope, list):
            child_scope = [child_scope]
        scope = tnodes.MapScope(child_scope, deepcopy(map_scope.node))
        parent_scope.children.insert(idx, scope)
