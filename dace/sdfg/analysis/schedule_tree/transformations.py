# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from dace import data as dt, Memlet, SDFG
from dace.sdfg.analysis.schedule_tree import treenodes as tnodes
from dace.sdfg.analysis.schedule_tree import utils as tutils
from typing import Dict, Set


_dataflow_nodes = (tnodes.ViewNode, tnodes.RefSetNode, tnodes.CopyNode, tnodes.DynScopeCopyNode, tnodes.TaskletNode, tnodes.LibraryCall)


def _update_memlets(data: Dict[str, dt.Data], memlets: Dict[str, Memlet], index: str, replace: Dict[str, bool]):
    for conn, memlet in memlets.items():
        if memlet.data in data:
            subset = index if replace[memlet.data] else f"{index}, {memlet.subset}"
            memlets[conn] = Memlet(data=memlet.data, subset=subset)


# def _augment_data(data: Dict[str, dt.Data], map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode, sdfg: SDFG):
def _augment_data(data: Set[str], map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode, sdfg: SDFG):
    
    # Generate map-related indices, sizes and, strides
    map = map_scope.node.map
    index = ", ".join(f"{p} - {r[0]}" for p, r in zip(map.params, map.range))
    size = map.range.size()
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i+1] * size[i+1]

    # Augment data descriptors
    replace = dict()
    # for name, nsdfg in data.items():
    for name in data:
        # desc = nsdfg.arrays[name]
        desc = map_scope.containers[name]
        if isinstance(desc, dt.Scalar):
            # nsdfg.arrays[name] = dt.Array(desc.dtype, size, True, storage=desc.storage)
            desc = dt.Array(desc.dtype, size, True, storage=desc.storage)
            replace[name] = True
        else:
            mult = desc.shape[0]
            desc.shape = (*size, *desc.shape)
            new_strides = [s * mult for s in strides]
            desc.strides = (*new_strides, *desc.strides)
            replace[name] = False
        del map_scope.containers[name]
        map_scope.parent.containers[name] = desc
        # if sdfg.parent:
        #     nsdfg_node = nsdfg.parent_nsdfg_node
        #     nsdfg_state = nsdfg.parent
        #     nsdfg_node.out_connectors = {**nsdfg_node.out_connectors, name: None}
        #     sdfg.arrays[name] = deepcopy(nsdfg.arrays[name])
        #     access = nsdfg_state.add_access(name)
        #     nsdfg_state.add_edge(nsdfg_node, name, access, None, Memlet.from_array(name, sdfg.arrays[name]))
    
    # Update memlets
    frontier = list(tree.children)
    while frontier:
        node = frontier.pop()
        if isinstance(node, _dataflow_nodes):
            try:
                _update_memlets(data, node.in_memlets, index, replace)
                _update_memlets(data, node.out_memlets, index, replace)
            except AttributeError:
                if node.memlet.data in data:
                    subset = index if replace[node.memlet.data] else f"{index}, {node.memlet.subset}"
                    node.memlet = Memlet(data=node.memlet.data, subset=subset)
        if hasattr(node, 'children'):
            frontier.extend(node.children)


def map_fission(map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode) -> bool:
    """
    Applies the MapFission transformation to the input MapScope.

    :param map_scope: The MapScope.
    :param tree: The ScheduleTree.
    :return: True if the transformation applies successfully, otherwise False.
    """

    sdfg = map_scope.sdfg

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
    
    # data_to_augment = dict()
    data_to_augment = set()
    frontier = list(partition)
    while len(frontier) > 0:
        scope = frontier.pop()
        if isinstance(scope, _dataflow_nodes):
            try:
                for _, memlet in scope.out_memlets.items():
                    if memlet.data in map_scope.containers:
                        data_to_augment.add(memlet.data)
                    # if scope.sdfg.arrays[memlet.data].transient:
                    #     data_to_augment[memlet.data] = scope.sdfg
            except AttributeError:
                if scope.target in map_scope.containers:
                    data_to_augment.add(scope.target)
                # if scope.target in scope.sdfg.arrays and scope.sdfg.arrays[scope.target].transient:
                #     data_to_augment[scope.target] = scope.sdfg
        if hasattr(scope, 'children'):
            frontier.extend(scope.children)
    _augment_data(data_to_augment, map_scope, tree, sdfg)
    
    parent_scope = map_scope.parent
    idx = parent_scope.children.index(map_scope)
    parent_scope.children.pop(idx)
    while len(partition) > 0:
        child_scope = partition.pop()
        if not isinstance(child_scope, list):
            child_scope = [child_scope]
        scope = tnodes.MapScope(sdfg, False, child_scope, deepcopy(map_scope.node))
        scope.parent = parent_scope
        parent_scope.children.insert(idx, scope)
    
    return True


def if_fission(if_scope: tnodes.IfScope, distribute: bool = False) -> bool:

    from dace.sdfg.nodes import CodeBlock

    parent_scope = if_scope.parent
    idx = parent_scope.children.index(if_scope)

    # Check transformation conditions
    # Scope must not have subsequent elif or else scopes
    if len(parent_scope.children) > idx + 1 and isinstance(parent_scope.children[idx+1],
                                                           (tnodes.ElifScope, tnodes.ElseScope)):
        return False
    
    # Apply transformations
    partition = tutils.partition_scope_body(if_scope)
    parent_scope.children.pop(idx)
    while len(partition) > 0:
        child_scope = partition.pop()
        if isinstance(child_scope, list) and len(child_scope) == 1 and isinstance(child_scope[0], tnodes.IfScope) and distribute:
            scope = tnodes.IfScope(if_scope.sdfg, False, child_scope[0].children, CodeBlock(f"{if_scope.condition.as_string} and {child_scope[0].condition.as_string}"))
        else:
            if not isinstance(child_scope, list):
                child_scope = [child_scope]
            scope = tnodes.IfScope(if_scope.sdfg, False, child_scope, deepcopy(if_scope.condition))
        scope.parent = parent_scope
        parent_scope.children.insert(idx, scope)

    return True


def wcr_to_reduce(map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode) -> bool:

    pass