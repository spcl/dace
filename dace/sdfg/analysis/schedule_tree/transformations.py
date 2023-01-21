# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace import data as dt, dtypes, Memlet, SDFG, subsets
from dace.sdfg import nodes as dnodes
from dace.sdfg.analysis.schedule_tree import treenodes as tnodes
from dace.sdfg.analysis.schedule_tree import utils as tutils
import re
from typing import Dict, List, Set, Tuple, Union


_dataflow_nodes = (tnodes.ViewNode, tnodes.RefSetNode, tnodes.CopyNode, tnodes.DynScopeCopyNode, tnodes.TaskletNode, tnodes.LibraryCall)


def _get_loop_size(loop: Union[tnodes.MapScope, tnodes.ForScope]) -> Tuple[str, list, list]:
    # Generate loop-related indices, sizes and, strides
    if isinstance(loop, tnodes.MapScope):
        map = loop.node.map
        index = ", ".join(f"{p}/{r[2]}-{r[0]}" if r[2] != 1 else f"{p}-{r[0]}" for p, r in zip(map.params, map.range))
        size = map.range.size()
    else:
        itervar = loop.header.itervar
        start = loop.header.init
        # NOTE: Condition expression may be inside parentheses
        par = re.search(f"^\s*(\(?)\s*{itervar}", loop.header.condition.as_string).group(1) == '('
        if par:
            stop_match = re.search(f"\(\s*{itervar}\s*([<>=]+)\s*(.+)\s*\)", loop.header.condition.as_string)
        else:
            stop_match = re.search(f"{itervar}\s*([<>=]+)\s*(.+)", loop.header.condition.as_string)
        stop_op = stop_match.group(1)
        assert stop_op in ("<", "<=", ">", ">=")
        stop = stop_match.group(2)
        # NOTE: Update expression may be inside parentheses
        par = re.search(f"^\s*(\(?)\s*{itervar}", loop.header.update).group(1) == '('
        if par:
            step_match = re.search(f"\(\s*{itervar}\s*([(*+-/%)]+)\s*([a-zA-Z0-9_]+)\s*\)", loop.header.update)
        else:
            step_match = re.search(f"{itervar}\s*([(*+-/%)]+)\s*([a-zA-Z0-9_]+)", loop.header.update)
        try:
            step_op = step_match.group(1)
            step = step_match.group(2)
            if step_op == '+':
                step = int(step)
                index = f"{itervar}/{step}-{start}" if step != 1 else f"{itervar}-{start}"
            else:
                raise ValueError
        except (AttributeError, ValueError):
            step = 1 if '<' in stop_op  else -1
            index = itervar
        if "=" in stop_op:
            stop = f"{stop} + ({step})"
        size = subsets.Range.from_string(f"{start}:{stop}:{step}").size()

    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i+1] * size[i+1]
    
    return index, size, strides


def _update_memlets(data: Dict[str, dt.Data], memlets: Dict[str, Memlet], index: str, replace: Dict[str, bool]):
    for conn, memlet in memlets.items():
        if memlet.data in data:
            subset = index if replace[memlet.data] else f"{index}, {memlet.subset}"
            memlets[conn] = Memlet(data=memlet.data, subset=subset)


def _augment_data(data: Set[str], loop: Union[tnodes.MapScope, tnodes.ForScope], tree: tnodes.ScheduleTreeNode):
    
    # # Generate loop-related indices, sizes and, strides
    # if isinstance(loop, tnodes.MapScope):
    #     map = loop.node.map
    #     index = ", ".join(f"{p}/{r[2]}-{r[0]}" if r[2] != 1 else f"{p}-{r[0]}" for p, r in zip(map.params, map.range))
    #     size = map.range.size()
    # else:
    #     itervar = loop.header.itervar
    #     start = loop.header.init
    #     # NOTE: Condition expression may be inside parentheses
    #     par = re.search(f"^\s*(\(?)\s*{itervar}", loop.header.condition.as_string).group(1) == '('
    #     if par:
    #         stop_match = re.search(f"\(\s*{itervar}\s*([<>=]+)\s*(.+)\s*\)", loop.header.condition.as_string)
    #     else:
    #         stop_match = re.search(f"{itervar}\s*([<>=]+)\s*(.+)", loop.header.condition.as_string)
    #     stop_op = stop_match.group(1)
    #     assert stop_op in ("<", "<=", ">", ">=")
    #     stop = stop_match.group(2)
    #     # NOTE: Update expression may be inside parentheses
    #     par = re.search(f"^\s*(\(?)\s*{itervar}", loop.header.update).group(1) == '('
    #     if par:
    #         step_match = re.search(f"\(\s*{itervar}\s*([(*+-/%)]+)\s*([a-zA-Z0-9_]+)\s*\)", loop.header.update)
    #     else:
    #         step_match = re.search(f"{itervar}\s*([(*+-/%)]+)\s*([a-zA-Z0-9_]+)", loop.header.update)
    #     try:
    #         step_op = step_match.group(1)
    #         step = step_match.group(2)
    #         if step_op == '+':
    #             step = int(step)
    #             index = f"{itervar}/{step}-{start}" if step != 1 else f"{itervar}-{start}"
    #         else:
    #             raise ValueError
    #     except (AttributeError, ValueError):
    #         step = 1 if '<' in stop_op  else -1
    #         index = itervar
    #     if "=" in stop_op:
    #         stop = f"{stop} + ({step})"
    #     size = subsets.Range.from_string(f"{start}:{stop}:{step}").size()

    # strides = [1] * len(size)
    # for i in range(len(size) - 2, -1, -1):
    #     strides[i] = strides[i+1] * size[i+1]

    index, size, strides = _get_loop_size(loop)

    # Augment data descriptors
    replace = dict()
    for name in data:
        desc = loop.containers[name]
        if isinstance(desc, dt.Scalar):

            desc = dt.Array(desc.dtype, size, True, storage=desc.storage)
            replace[name] = True
        else:
            mult = desc.shape[0]
            desc.shape = (*size, *desc.shape)
            new_strides = [s * mult for s in strides]
            desc.strides = (*new_strides, *desc.strides)
            replace[name] = False
        del loop.containers[name]
        loop.parent.containers[name] = desc
    
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



def loop_fission(loop: Union[tnodes.MapScope, tnodes.ForScope], tree: tnodes.ScheduleTreeNode) -> List[Union[tnodes.MapScope, tnodes.ForScope]]:
    """
    Applies the LoopFission transformation to the input MapScope or ForScope.

    :param loop: The MapScope or ForScope.
    :param tree: The ScheduleTree.
    :return: True if the transformation applies successfully, otherwise False.
    """

    sdfg = loop.sdfg

    ####################################
    # Check if LoopFission can be applied

    # Basic check: cannot fission an empty MapScope/ForScope or one that has a single child.
    partition = tutils.partition_scope_body(loop)
    if len(partition) < 2:
        return [loop]

    index, _, _ = _get_loop_size(loop)
    
    data_to_augment = set()
    assignments = dict()
    frontier = list(partition)
    while len(frontier) > 0:
        scope = frontier.pop()
        if isinstance(scope, _dataflow_nodes):
            try:
                for _, memlet in scope.out_memlets.items():
                    if memlet.data in loop.containers:
                        data_to_augment.add(memlet.data)
            except AttributeError:
                if scope.target in loop.containers:
                    data_to_augment.add(scope.target)
        elif isinstance(scope, tnodes.AssignNode):
            symbol = tree.symbols[scope.name]
            loop.containers[f"{scope.name}_arr"] = dt.Scalar(symbol.dtype, transient=True)
            data_to_augment.add(f"{scope.name}_arr")
            repl_dict = {scope.name: '__out'}
            out_memlets = {'__out': Memlet(data=f"{scope.name}_arr", subset='0')}
            in_memlets = dict()
            for i, memlet in enumerate(scope.edge.get_read_memlets(scope.parent.sdfg.arrays)):
                repl_dict[str(memlet)] = f'__in{i}'
                in_memlets[f'__in{i}'] = memlet
            scope.edge.replace_dict(repl_dict)
            tasklet = dnodes.Tasklet('some_label', in_memlets.keys(), {'__out'},
                                     f"__out = {scope.edge.assignments['__out']}")
            tnode = tnodes.TaskletNode(tasklet, in_memlets, out_memlets)
            tnode.parent = loop
            idx = loop.children.index(scope)
            loop.children[idx] = tnode
            idx = partition.index(scope)
            partition[idx] = tnode
            edge = copy.deepcopy(scope.edge)
            edge.assignments['__out'] = f"{scope.name}_arr[{index}]"
            assignments[scope.name] = (dnodes.CodeBlock(f"{scope.name}[{index}]"), edge, idx)
        elif hasattr(scope, 'children'):
            frontier.extend(scope.children)
    _augment_data(data_to_augment, loop, tree)
    print(data_to_augment)


    new_scopes = []
    # while partition:
    #     child = partition.pop(0)
    for i, child in enumerate(partition):
        if not isinstance(child, list):
            child = [child]
        
        for c in list(child):
            idx = child.index(c)
            # Reverse access?
            for name, (value, edge, index) in assignments.items():
                if index == i:
                    continue
                if c.is_data_used(name, True):
                    child.insert(idx, tnodes.AssignNode(f"{name}", copy.deepcopy(value), copy.deepcopy(edge)))

        if isinstance(loop, tnodes.MapScope):
            scope = tnodes.MapScope(sdfg, False, child, copy.deepcopy(loop.node))
        else:
            scope = tnodes.ForScope(sdfg, False, child, copy.copy(loop.header))
        for child in scope.children:
            child.parent = scope
            if isinstance(child, tnodes.ScheduleTreeScope):
                scope.containers.update(child.containers)
        scope.parent = loop.parent
        new_scopes.append(scope)

    return new_scopes


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
    _augment_data(data_to_augment, map_scope, tree)
    
    parent_scope = map_scope.parent
    idx = parent_scope.children.index(map_scope)
    parent_scope.children.pop(idx)
    while len(partition) > 0:
        child_scope = partition.pop()
        if not isinstance(child_scope, list):
            child_scope = [child_scope]
        scope = tnodes.MapScope(sdfg, False, child_scope, copy.deepcopy(map_scope.node))
        scope.parent = parent_scope
        parent_scope.children.insert(idx, scope)
    
    return True


def if_fission(if_scope: tnodes.IfScope, assume_canonical: bool = False, distribute: bool = False) -> List[tnodes.IfScope]:

    from dace.sdfg.nodes import CodeBlock

    # Check transformation conditions
    # Scope must not have subsequent elif or else scopes
    if not assume_canonical:
        idx = if_scope.parent.children.index(if_scope)
        if len(if_scope.parent.children) > idx + 1 and isinstance(if_scope.parent.children[idx+1],
                                                                  (tnodes.ElifScope, tnodes.ElseScope)):
            return [if_scope]
    if len(if_scope.children) < 2 and not (isinstance(if_scope.children[0], tnodes.IfScope) and distribute):
        return [if_scope]
    
    new_scopes = []
    partition = tutils.partition_scope_body(if_scope)
    while partition:
        child = partition.pop(0)
        if isinstance(child, list) and len(child) == 1 and isinstance(child[0], tnodes.IfScope) and distribute:
            scope = tnodes.IfScope(if_scope.sdfg, False, child[0].children, CodeBlock(f"{if_scope.condition.as_string} and {child[0].condition.as_string}"))
            scope.containers.update(child[0].containers)           
        else:
            if not isinstance(child, list):
                child = [child]
            scope = tnodes.IfScope(if_scope.sdfg, False, child, copy.deepcopy(if_scope.condition))
        for child in scope.children:
            child.parent = scope
            if isinstance(child, tnodes.ScheduleTreeScope):
                scope.containers.update(child.containers)
        scope.parent = if_scope.parent
        new_scopes.append(scope)

    return new_scopes


def wcr_to_reduce(map_scope: tnodes.MapScope, tree: tnodes.ScheduleTreeNode) -> bool:

    pass