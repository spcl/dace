# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This file collects all helper functions and the top level function
required to transform sdfg's containing hbm-multibank arrays into 
their corresponding representation with only single hbm-bank arrays.

It might at some point be beneficial to add this to the codegen, or to the utils,
it doesn't necessarily have to stay here. 
"""

from copy import deepcopy
from re import A
from numpy import isin

from sympy.codegen.ast import Variable
from dace.codegen import exceptions as cgx
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes as nd
from dace import data, memlet, subsets, symbolic, sdfg as sd
from dace.sdfg import state, graph

from typing import Union
from typing import Any

def selectNodesConditional(sdfg, condition):
    #Get all nodes that fullfill a condition
    targets = list(filter(lambda x : condition(x[0], x[1]),sdfg.all_nodes_recursive()))
    return targets

def parseHBMBank(arrayname, array): 
    """
    Reads the hbm bank-specification of an array if present.
    :param arrayname: The name of the array
    :param array: The array
    :return: None on not present, (low, high) otherwise, 
    where low == high is possible.
    """
    if(not "hbmbank" in array.location):
        return None
    errormsg = ("location['hbmbank'] must be a string"
        f" with format 'int' or 'i1:i2' with i1, i2"
        f" type int and i1<i2 for {arrayname}")
    banks = array.location["hbmbank"]
    if(not isinstance(banks, str)):
        cgx.CodegenError(errormsg)
    split = banks.split(":")
    if(len(split) == 1):
        try:
            val = int(banks)
            return (val, val)
        except ValueError:
            cgx.CodegenError(errormsg)
    elif(len(split) == 2):
        try:
            low = int(split[0])
            high = int(split[1])
            if(low > high):
                raise ValueError()
            return (low, high)
        except ValueError:
            cgx.CodegenError(errormsg)
    raise RuntimeError("Reached unreachable state")

def parseHBMAlignment(arrayname, array):
    """
    Tries to read and parse alignment, if present,
    otherwise assumes default alignment. Returns 
    the axes along which the array is split.
    :param arrayname: The name of the array
    :param array: The array
    :return: A list of axes along which array is split
    """
    ndim = len(array.shape)
    splitaxes = []
    for i in range(ndim):
        splitaxes.append(i)
    alignment = []
    for i in range(ndim-1):
        alignment.append(i)

    if "hbmalignment" in array.location:
            alignment = array.location["hbmalignment"]
    if(isinstance(alignment, str) and alignment == 'even'):
        alignment = []

    if(not isinstance(alignment, list)):
        cgx.CodegenError("hbmalignment must be 'even' "
        f"or a list of axes in {arrayname}")
    alignment.sort()
    lastval = None
    for val in alignment:
        if(val >= ndim or (lastval != None and lastval == val)):
            cgx.CodegenError("alignment list contains duplicates "
                f"or non existing axes in {arrayname}")
        lastval = val
        splitaxes.remove(val)
    return splitaxes

def collectAndParseHBMArrays(sdfg : sd.SDFG) -> "dict[(str, sd.SDFG), dict[str, Any]]":
    """
    Finds all arrays that are spread across multiple HBM
    banks
    """
    arrays = sdfg.arrays_recursive()

    handledArrays = {}  #(oldarrayname, sdfg) -> 
        #{ndim -> int, splitcount->int, splitaxes->[int]}
    for currentsdfg, arrayname, array in arrays:
        parsed = parseHBMBank(arrayname, array)
        if(parsed == None):
            continue
        low, high = parsed
        if(high - low == 0):
            continue

        count = high - low + 1
        shape = array.shape
        ndim = len(shape)
        splitaxes = parseHBMAlignment(arrayname, array)
        splitdim = len(splitaxes)
        if(splitdim == 0):
            cgx.CodegenError("for an array divided across multiple hbm-banks "
                "there must be at least 1 allowed split dimension" 
                f"in {arrayname}")
        splitcount = round(count ** (1 / splitdim))
        if(splitcount ** splitdim != count):
            cgx.CodegenError("for an array divided across mutiple hbm-banks "
                "the equation 'hbmbanks == x ** splitdims' must hold where "
                "hbmbanks is the number of used banks, splitdims is the total "
                "count of axes along which splitting is allowed and x is an "
                "arbitrary integer. This is necessary so the number of splits "
                "in each direction is the same). This does not hold for "
                f"{arrayname}")
        handledArrays[(arrayname, currentsdfg)] = {"ndim" : ndim,
            "splitcount" : splitcount, "splitaxes" : splitaxes, "lowbank" : low}
    return handledArrays
            
def getNonExistingName(suggestedname : str, checkagainstcollection) -> str:
    counter = 0
    while(suggestedname in checkagainstcollection):
        suggestedname = suggestedname + f"_{counter}"
        counter += 1
    return suggestedname

def getAttachedHBMMemlets(state, node, hbmarraylist):
    hbminput = []
    hbmoutput = []
    for input in state.in_edges(node):
        if((input.data.data, state.parent) in hbmarraylist):
            hbminput.append(input)
    for output in state.out_edges(node):
        if((output.data.data, state.parent) in hbmarraylist):
            hbmoutput.append(output)
    return (hbminput, hbmoutput)

def downward_topological(sdfg : state.SDFGState, source : nd.Node):
    """
    Starts from a source node and moves in a depth first
    approach in topological order along memlets in the state. 
    Moves only in the direction of dataflow (towards out edges)
    """
    visited = set()
    getDownChildren = lambda s : [x.dst for x in sdfg.out_edges(s)]
    stack = getDownChildren(source)
    yield source
    visited.add(source)
    while stack:
        child = stack[-1]
        stack.pop()
        if(child in visited):
            continue
        proc = True
        for pred in sdfg.in_edges(child):
            if(pred.src not in visited):
                proc = False
                break
        if(proc):
            visited.add(child)
            stack.extend(getDownChildren(child))
            yield child


def unroll_map(state : state.SDFGState, entry : nd.MapEntry, exit : nd.MapExit):
    """
    1 D only, assume valid
    """
    begin, end, stride = entry.map.range[0]
    begin = int(str(begin))
    end = int(str(end))
    stride = int(str(stride))
    varname = entry.map.params[0]
    
    def intern_recreateMemletWithReplace(edge, varname : str, replace : int):
        #Copy a memlet from an edge and replace varname with replace in 
        #subset/volume
        oldmemlet = edge.data
        symbolicvar = symbolic.symbol(varname)
        repldict = [(symbolicvar, replace)]

        mem = deepcopy(oldmemlet)

        if(symbolic.issymbolic(oldmemlet.volume)):
            mem.volume = mem.volume.subs(repldict)
        if(isinstance(oldmemlet.other_subset, subsets.Range)):
            mem.other_subset.replace(repldict)
        if(isinstance(oldmemlet.subset, subsets.Range)):
            mem.subset.replace(repldict)
        return mem

    for k in range(begin, end+1, stride):
        #Copy/reconnect the scope in the map #unroll times
        oldToNewMap = {}
        for v in downward_topological(state, entry):
            if(v == entry):
                continue
            if(v == exit):
                break
            newnode = deepcopy(v)
            state.add_node(newnode)
            oldToNewMap[v] = newnode

            for edge in state.in_edges(v):
                mem = intern_recreateMemletWithReplace(edge, varname, k)
                if(edge.src == entry):
                    path = state.memlet_path(edge)
                    prev = path[path.index(edge)-1]
                    state.add_edge(prev.src, prev.src_conn,
                    newnode, edge.dst_conn, mem)
                else:
                    state.add_edge(oldToNewMap[edge.src], edge.src_conn,
                    newnode, edge.dst_conn, mem)
        for edge in state.in_edges(exit):
            mem = intern_recreateMemletWithReplace(edge, varname, k)
            path = state.memlet_path(edge)
            next = path[path.index(edge) + 1]
            state.add_edge(oldToNewMap[edge.src], edge.src_conn,
            next.dst, next.dst_conn, mem)
    for v in downward_topological(state, entry):
        #Erase the old map
        state.remove_node(v)
        if(v == exit):
            break

def expand_hbm_multiarrays(sdfg : sd.SDFG) -> sd.SDFG:
    """
    This function removes arrays split across k > 1 banks into k new arrays, 
    each with it's own hbm index. (and a new name). 
    Memlets/Accessnodes accessing the array get redefined according 
    to the subsets they access. This includes fully unrolling
    maps that access the memlet (if their index is used as bank index).
    Copymemlets from/to host are created based on hbmalignment.
    """

    info = collectAndParseHBMArrays(sdfg)
    if(len(info) == 0):
        return
    
    arraylist : dict[(str, sd.SDFG), list[(str, data.Array)]] = {}
    statelist : set[sd.SDFGState] = set()

    for arrayname, arraysdfg in info.keys():
        #Create mapping from old to new arrays
        curinfo = info[(arrayname, arraysdfg)]
        oldarray : data.Array = arraysdfg.arrays[arrayname]
        shape = oldarray.shape
        splitcount = curinfo["splitcount"]
        axes = curinfo["splitaxes"]
        arraylist[(arrayname, arraysdfg)] = []
        for i in range(splitcount*len(axes)):
            newname = f"{arrayname}_hbm{i}"
            newname = getNonExistingName(newname, arraysdfg.arrays)
            newshapelist = []
            for d in range(curinfo["ndim"]):
                if d in axes:
                    strip = shape[d] // splitcount
                    if(i == 0):
                        strip += shape[d] % splitcount
                    newshapelist.append(strip)
                else:
                    newshapelist.append(shape[d])
            newshape = tuple(newshapelist)

            newname, newarray = arraysdfg.add_array(
                newname, 
                newshape, 
                oldarray.dtype,
                transient=False,
                strides=oldarray.strides,
                offset=oldarray.offset,
                lifetime=oldarray.lifetime,
                debuginfo=oldarray.debuginfo,
                allow_conflicts=oldarray.allow_conflicts
            )
            newarray.location["hbmbank"] = f"{curinfo['lowbank'] + i}"
            arraylist[(arrayname, arraysdfg)].append((newname, newarray))

    for lookatsdfg in sdfg.all_sdfgs_recursive():
        #Find all states where HBM-multibanks are used
        for state in lookatsdfg.states():
            for node in state.nodes():
                if(isinstance(node, nd.AccessNode)):
                    if((node.data, lookatsdfg) in arraylist):
                        statelist.add(state)

    for state in statelist:
        accessnodestoexpand = []
        mapstounroll = {}
        start_nodes = sdutil.find_source_nodes(state)

        for node in sdutil.dfs_topological_sort(state, start_nodes):
            #Find all accessnodes and maps which should be expanded and store
            #them in accessnodestoexpand and mapstounroll
            if(isinstance(node, nd.AccessNode)):
                #multiply the node number-of-bank times and store in accesnodelist
                currentindex = (node.data, state.parent)
                if(currentindex in arraylist):
                    accessnodestoexpand.append(node)
            elif(isinstance(node, nd.MapEntry)):
                #decide wheter to unroll the map
                hbmin : list[graph.MultiConnectorEdge] = None
                hbmout : list[graph.MultiConnectorEdge] = None
                hbmin , hbmout  = getAttachedHBMMemlets(state, node, arraylist)
                if(node.map.get_param_num() != 1):  #TODO: In case this has bound hbmmemlets need to throw an error here
                    continue
                variable = node.map.params[0]
                for current in hbmout:
                    low, high, stride = current.data.subset[0]
                    lowstr = symbolic.free_symbols_and_functions(low)
                    highstr = symbolic.free_symbols_and_functions(high)
                    if(variable in lowstr or variable in highstr):
                        #TODO: Check that low == high, and this is
                        #a constant (i.e. just the variable)
                        mapstounroll[node.map] = node
                        break
            elif(isinstance(node, nd.MapExit)):
                #if this map will be unrolled (known because we've already seen the MapEntry), store this MapExit
                if(node.map in mapstounroll):
                    mapstounroll[node.map] = (mapstounroll[node.map], node)
            
        for entry, exit in mapstounroll.values():
            unroll_map(state, entry, exit)
        for node in accessnodestoexpand:
            currentindex = (node.data, state.parent)
            refArrays = arraylist[currentindex]
            refInfo = info[currentindex]
            generated_nodes : dict[str, nd.AccessNode] = {}

            nodeInputEdges = state.in_edges(node)
            nodeConnections = nodeInputEdges + state.out_edges(node)
            for index in range(len(nodeConnections)):
                edge = nodeConnections[index]
                oldmem = edge.data
                firstrange = oldmem.subset[0]
                #TODO: Check if range is really constant and in bounds, stride == 1
                rangelow = int(str(firstrange[0]))
                rangehigh = int(str(firstrange[1])) + 1
                
                for index in range(rangelow, rangehigh):
                    #For each bank referenced by the memlet create a accessnode for
                    #that bank if it does not already exist, and reconnect with single bank memlet
                    newrefArrayname, newrefArray = refArrays[index]
                    if(newrefArrayname not in generated_nodes):
                        newnode = deepcopy(node)
                        newnode.data = newrefArrayname
                        state.add_node(newnode)
                        generated_nodes[newrefArrayname] = newnode
                    newnode = generated_nodes[newrefArrayname]

                    if(index < len(nodeInputEdges)):
                        #This is an input edge. Connect accordingly
                        state.add_edge(edge.src, edge.src_conn, newnode, edge.dst_conn, deepcopy(oldmem))
                    else:
                        #This is an output edge
                        state.add_edge(newnode, edge.src_conn, edge.dst, edge.dst_conn, deepcopy(oldmem))
            state.remove_node(node)

        #TODO: Replace all hbm memlets with their real representation. For multimemlets add the right connectors to scopes.
            

    return sdfg