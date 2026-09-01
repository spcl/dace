from dace import dtypes, data
from dace.sdfg import nodes, SDFG, SDFGState
from dace.transformation.passes.offloading.OffloadingIRNode import OffloadingIRNode
from dace.sdfg.utils import get_last_view_node



##################################################
###                Scope Dict                  ###
### is expensive to generate, should be cached ###
##################################################

def get_sdfg_scope_dict(sdfg):
    scopes = {}
    for state in sdfg.states():
        scopes[state] = state.scope_dict() 
    return scopes


###################################
###  Checking Common Conditions ###
###################################

def has_GPU_schedule(node):
    schedule = None
    if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
        schedule = node.map.schedule
    elif isinstance(node, nodes.LibraryNode):
        schedule = node.schedule
    else:
        assert False
    return schedule in dtypes.GPU_SCHEDULES

def is_array_stored_on_GPU(sdfg, array_name):
    storage = sdfg.arrays[array_name].storage
    return storage == dtypes.StorageType.GPU_Global or storage in dtypes.GPU_STORAGES

def is_scalar(data_name:str, sdfg:SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Scalar)

def is_array(data_name:str, sdfg:SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Array)

def is_view(data_name:str, sdfg:SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.View)

def is_length1_array(data_name:str, sdfg:SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Array) and len(desc.shape) == 1 and desc.shape[0] == 1


#######################
###  SDFG Traversal ###
#######################

def get_children(state, node):
    return {e.dst for e in state.out_edges(node)}

def get_predecessors(state, node):
    return {e.src for e in state.in_edges(node)}


def traverse_IR(IR:OffloadingIRNode, method):
    def recursion(node, visited_set):
        if node in visited_set:
            return
        visited_set.add(node)

        method(node)
        
        for next in node.next:
            recursion(next, visited_set)

    return recursion(IR, set())

def traverse_same_level(IR:OffloadingIRNode, method): #DFS
    queue = IR.next.copy()
    while queue:
        curr = queue.pop()
        if curr.type == OffloadingIRNode.STATE or curr.type == OffloadingIRNode.EDGE: # data node
            method(curr)
            queue += curr.next

        elif curr.is_open_node():
            method(curr)
            queue += curr.close.next

        elif curr.type == OffloadingIRNode.CLOSE:
            break

        else:
            assert False


########################################
###  Get Arrays Used by Access Nodes ###
########################################

def get_data_used_by_incoming_access_nodes(sdfg:SDFG, state:SDFGState, node:nodes.Node, include_scalars:bool=False) -> set[str]:

    def recursion(node:nodes.Node, visited_set:set[nodes.Node]):
        if node in visited_set: # the visited set is necessary for edge cases, e.g. an access node A whose predecessor B is a view node refering back to A
            return set()
        visited_set.add(node)

        # find accessed arrays
        arrays : set[str] = set()
        if isinstance(node, nodes.AccessNode): 
            data_name = node.data
            if is_array(data_name, sdfg):
                arrays.add(data_name)

            elif is_view(data_name, sdfg): # trace it if it is a view
                original = get_last_view_node(state, node) # once the view access node is known, its original access node can be found and it's data added
                arrays |= recursion(original, visited_set)
                
            elif include_scalars and is_scalar(data_name, sdfg):
                arrays.add(data_name)

        # check if more access nodes UPstream
        for n in get_predecessors(state, node):
            if isinstance(n, nodes.AccessNode):
                arrays |= recursion(n, visited_set)

        return arrays
    
    return recursion(node, set())

def get_data_used_by_outgoing_access_nodes(sdfg:SDFG, state:SDFGState, node:nodes.Node, include_scalars:bool=False) -> set[str]:
    
    def recursion(node:nodes.Node, visited_set:set[nodes.Node]):
        if node in visited_set: # the visited set is necessary for edge cases, e.g. an access node A whose successor B is a view node refering back to A
            return set()
        visited_set.add(node)

        # find accessed arrays
        arrays : set[str] = set()
        if isinstance(node, nodes.AccessNode): 
            data_name = node.data

            if is_array(data_name, sdfg):
                arrays.add(data_name)

            elif is_view(data_name, sdfg): # trace it if it is a view
                original = get_last_view_node(state, node) # once the view access node is known, its original access node can be found and it's data added
                arrays |= recursion(original, visited_set)

            elif include_scalars and is_scalar(data_name, sdfg):
                arrays.add(data_name)
                
        # check if more access nodes DOWNstream
        for n in get_children(state, node):
            if isinstance(n, nodes.AccessNode):
                arrays |= recursion(n, visited_set)
                
        return arrays
    
    return recursion(node, set())


############################
###  Map Creation Helper ###
############################

def get_new_map_identifiers(state: SDFGState, map_label: str, map_param: str):
    existing_labels = {getattr(node, "label", None) for node in state.nodes()}
    existing_params = set()
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            existing_params |= set(node.map.params)
    
    suffix = 0
    new_label = map_label
    while new_label in existing_labels:
        suffix += 1
        new_label = f"{map_label}_{suffix}"

    suffix = 0
    new_param = map_param
    while new_param in existing_params:
        suffix += 1
        new_param = f"{map_param}_{suffix}"

    return new_label, new_param

