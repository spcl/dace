# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This file collects all helper functions and the top level function
required to transform sdfg's containing hbm-multibank arrays into 
their corresponding representation with only single hbm-bank arrays.

This is only kept as 1 file, after that some of those function should go to utils,
the rest into a file in the codegen.
"""

from copy import deepcopy

from dace.codegen import exceptions as cgx
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes as nd
from dace.sdfg import graph 
from dace.sdfg import state as statenamespace
from dace import data, memlet, subsets, symbolic, sdfg as sd

from typing import Iterable, Union, Any
            
def getNonExistingName(suggestedname : str, 
    checkagainstcollection : Iterable) -> str:
    """
    Small helper that returns a new name through appending f'_{number}'
    """
    counter = 0
    while(suggestedname in checkagainstcollection):
        suggestedname = suggestedname + f"_{counter}"
        counter += 1
    return suggestedname

def downward_topological(state : statenamespace.SDFGState, source : nd.Node):
    """
    Starts from a source node and moves in a depth first
    approach in topological order along memlets in the state. 
    Moves only in the direction of dataflow (towards out edges)
    """
    visited = set()
    getDownChildren = lambda s : [x.dst for x in state.out_edges(s)]
    stack = getDownChildren(source)
    yield source
    visited.add(source)
    while stack:
        child = stack[-1]
        stack.pop()
        if(child in visited):
            continue
        proc = True
        for pred in state.in_edges(child):
            if(pred.src not in visited):
                proc = False
                break
        if(proc):
            visited.add(child)
            stack.extend(getDownChildren(child))
            yield child

def unroll_map(state : statenamespace.SDFGState, entry : nd.MapEntry, exit : nd.MapExit):
    """
    Completly unrolls a map. The map has to be 1 dimensional and has to 
    only depend on constants (in terms of begin, end and stride).
    unroll_map assumes that the map passed is valid, if it is not
    then the result is undefined.
    This is an inplace operation on the passed state.
    """
    begin, end, stride = entry.map.range[0]
    begin = int(str(begin))
    end = int(str(end))
    stride = int(str(stride))
    varname = entry.map.params[0]

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
                mem = deepcopy(edge.data)
                mem.replace({varname : str(k)})
                if(edge.src == entry):
                    path = state.memlet_path(edge)
                    prev = path[path.index(edge) - 1]
                    state.add_edge(prev.src, prev.src_conn,
                    newnode, edge.dst_conn, mem)
                else:
                    state.add_edge(oldToNewMap[edge.src], edge.src_conn,
                    newnode, edge.dst_conn, mem)
        for edge in state.in_edges(exit):
            mem = deepcopy(edge.data)
            mem.replace({varname : str(k)})
            path = state.memlet_path(edge)
            next = path[path.index(edge) + 1]
            state.add_edge(oldToNewMap[edge.src], edge.src_conn,
            next.dst, next.dst_conn, mem)
    for v in downward_topological(state, entry):
        #Erase the old map
        state.remove_node(v)
        if(v == exit):
            break

def parseHBMBank(arrayname : str, array : data.Array) -> "tuple[int, int]": 
    """
    Reads the hbm bank-specification of an array if present.

    :param arrayname: The name of the array
    :param array: The array
    :return: None if not present, (low, high) otherwise, where low == high is possible.
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

def parseHBMAlignment(arrayname : str, array : data.Array) -> "list[int]":
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
    banks, and parses their properties using parseHMBAlignment
    and parseHBMBank.

    :return: A mapping from (arrayname, sdfg of the array) to a mapping
    from string that contains collected information.
    'ndim': contains the dimension of the array
    'splitcount': contains how many times this array is split
        on each of the axes along which it is split 
        (ie splitcount==2 on an 2d-array which is split along axis 0
        and axis 1 => There are 4 parts)
    'splitaxes': List that contains axes along which the array is split
    'lowbank': The lowest bank index this array is placed on
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

def _recursive_splice_hbmmemlettree(state : statenamespace.SDFGState,
        tree : memlet.MemletTree, flowsTowardsRoot : bool) -> "dict[int, list[graph.MultiConnectorEdge]]":
    """
    applying unroll results in an sdfg that still contains multibank memlets
    on the paths to the unrolled maps. This function "splices" those memlets,
    such that all memlets from/to the unrolled maps are single bank only. Note that
    the resulting state may still contain multibankmemlets between accessnodes.

    :param flowsTowardsRoot: Does the tree carry data towards the accessnode or away from it?
    """
    #TODO: Handle othersubset as well
    outgoingmemlets : dict[int, list[graph.MultiConnectorEdge]] = {}
    outgoingmemletcount = 0
    returnval = {}
    for child in tree.children:
        result = _recursive_splice_hbmmemlettree(state, child, flowsTowardsRoot)
        for banknum in result.keys():
            if(banknum in outgoingmemlets):
                outgoingmemlets[banknum].extend(result[banknum])
            else:
                outgoingmemlets[banknum] = result[banknum]
    for mlist in outgoingmemlets.values():
        outgoingmemletcount += len(mlist)
    edge : graph.MultiConnectorEdge = tree.edge
    mem : memlet.Memlet = edge.data
    if(outgoingmemletcount > 1):
        subsetlow = int(str(mem.subset[0][0]))
        subsethigh = int(str(mem.subset[0][1]))
        stride = int(str(mem.subset[0][2]))
        if(stride != 1):
            raise NotImplementedError()
        oldinputconnectorname = None
        oldoutputconnectorname = None
        referedNode : nd.Node = None
        if(flowsTowardsRoot):
            referedNode = edge.src
            oldinputconnectorname = next(iter(outgoingmemlets.items()))[1][0].dst_conn
            oldoutputconnectorname = edge.src_conn
        else:
            referedNode = edge.dst
            oldoutputconnectorname = next(iter(outgoingmemlets.items()))[1][0].src_conn
            oldinputconnectorname = edge.dst_conn

        for banknum in outgoingmemlets.keys():
            assert(banknum >= subsetlow and banknum <= subsethigh) #TODO: Handle stride !=1
            newmem = deepcopy(mem)
            newmem.subset[0] = (symbolic.pystr_to_symbolic(banknum),
                        symbolic.pystr_to_symbolic(banknum),
                        symbolic.pystr_to_symbolic(1))
            newmem.volume = symbolic.pystr_to_symbolic("floor(" + 
                str(newmem.volume) + f" / {outgoingmemletcount})")
            connectsToList = outgoingmemlets[banknum]
            connectstoconnectorname = None
            newedge = None

            if(flowsTowardsRoot):
                edgeconnectorname = f"{edge.src_conn}_{banknum}"
                connectstoconnectorname = f"{connectsToList[0].dst_conn}_{banknum}"
                referedNode.add_out_connector(edgeconnectorname)
                referedNode.add_in_connector(connectstoconnectorname)
                newedge = state.add_edge(edge.src, edgeconnectorname, 
                    edge.dst, edge.dst_conn, newmem)
            else:
                edgeconnectorname = f"{edge.dst_conn}_{banknum}"
                connectstoconnectorname = f"{connectsToList[0].src_conn}_{banknum}"
                referedNode.add_in_connector(edgeconnectorname)
                referedNode.add_out_connector(connectstoconnectorname)
                newedge = state.add_edge(edge.src, edge.src_conn,
                    edge.dst, edgeconnectorname, newmem)
            returnval[banknum] = [newedge]
            
            for connectsTo in connectsToList:
                if(flowsTowardsRoot):
                    connectsTo.dst_conn = connectstoconnectorname
                else:
                    connectsTo.src_conn = connectstoconnectorname

        state.remove_edge(edge)
        referedNode.remove_in_connector(oldinputconnectorname)
        referedNode.remove_out_connector(oldoutputconnectorname)
    else:
        banknum = int(str(edge.data.subset[0][0]))
        returnval[banknum] = [edge]
    return returnval

def expand_hbm_multiarrays(sdfg : sd.SDFG) -> sd.SDFG:
    """
    This function removes arrays split across k > 1 banks into k new arrays, 
    each with it's own hbm index. (and a new name). 
    Memlets/Accessnodes accessing the array get redefined according 
    to the subsets they access. This includes fully unrolling
    maps that access the memlet (if their index is used as bank index).
    Copymemlets are created based on hbmalignment.
    """
                
    info = collectAndParseHBMArrays(sdfg)
    if(len(info) == 0):
        return

    #mapping from old arrays to a list of the new arrays
    arraylist : dict[(str, sd.SDFG),  list[(str, data.Array)]] = {}
    #all states that contain hbm multibanks
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
                strides=deepcopy(oldarray.strides),
                offset=deepcopy(oldarray.offset),
                lifetime=deepcopy(oldarray.lifetime),
                debuginfo=deepcopy(oldarray.debuginfo),
                allow_conflicts=deepcopy(oldarray.allow_conflicts)
            )
            newarray.location["hbmbank"] = f"{curinfo['lowbank'] + i}"
            arraylist[(arrayname, arraysdfg)].append((newname, newarray))
        sdfg.remove_data(arrayname, validate=False)

    for lookatsdfg in sdfg.all_sdfgs_recursive():
        #Find all states where HBM-multibanks are used
        for state in lookatsdfg.states():
            for node in state.nodes():
                if(isinstance(node, nd.AccessNode)):
                    if((node.data, lookatsdfg) in arraylist):
                        statelist.add(state)

    for state in statelist:
        def intern_iterate_attached_memlets(node, hbmarraylist):
            #small helper to iterate over all memlets of a node
            #Why this is nested: Very simple pattern and used a lot 
            #in the following part, but not generally useful
            for inp in state.in_edges(node):
                if((inp.data.data, state.parent) in hbmarraylist):
                    yield 'in', inp
            for outp in state.out_edges(node):
                if((outp.data.data, state.parent) in hbmarraylist):
                    yield 'out', outp

        accessnodestoexpand = []
        mapstounroll = {}
        start_nodes = sdutil.find_source_nodes(state)
        for node in sdutil.dfs_topological_sort(state, start_nodes):
            #Find all accessnodes and maps which should be expanded and store
            #them in accessnodestoexpand and mapstounroll
            if(isinstance(node, nd.AccessNode)):
                currentindex = (node.data, state.parent)
                if(currentindex in arraylist):
                    accessnodestoexpand.append(node)
            elif(isinstance(node, nd.MapEntry)):
                if(node.map.get_param_num() != 1):  
                    #TODO: In case this has bound hbmmemlets need to throw an error here
                    continue
                variable = node.map.params[0]
                for inorout, current in intern_iterate_attached_memlets(node, arraylist):
                    if(inorout == 'in'):
                        continue
                    low, high, stride = current.data.subset[0]
                    lowstr = symbolic.free_symbols_and_functions(low)
                    highstr = symbolic.free_symbols_and_functions(high)
                    if(variable in lowstr or variable in highstr):
                        #TODO: Check that low == high, and this is
                        #a constant (i.e. just the variable)
                        mapstounroll[node.map] = node
                        break
            elif(isinstance(node, nd.MapExit)):
                if(node.map in mapstounroll):
                    mapstounroll[node.map] = (mapstounroll[node.map], node)

        #unroll all maps
        for entry, exit in mapstounroll.values():
            unroll_map(state, entry, exit)

        #expand all memlets such that only single bank memlets stay
        for accessnode in accessnodestoexpand:
            for inorout, watchededge in intern_iterate_attached_memlets(accessnode, arraylist):
                root = state.memlet_tree(watchededge)
                _recursive_splice_hbmmemlettree(state, root, inorout == 'in')
        
        #expand all accessnodes, so that each accessnode only accesses a single bank
        for node in accessnodestoexpand:
            currentindex = (node.data, state.parent)
            refArrays = arraylist[currentindex]
            generated_nodes : dict[str, nd.AccessNode] = {}

            for inorout, edge in intern_iterate_attached_memlets(node, arraylist):
                oldmem = edge.data
                firstrange = oldmem.subset[0]
                banklow = int(str(firstrange[0]))
                bankhigh = int(str(firstrange[1]))
                stride = int(str(firstrange[2]))
                #TODO: support stride != 1
                if(stride != 1):
                    raise NotImplementedError()

                #For each bank referenced by the memlet create an accessnode for
                #that bank if it does not already exist, and reconnect with single bank memlet
                for currentbank in range(banklow, bankhigh +1):
                    newrefArrayname, newrefArray = refArrays[currentbank]
                    if(newrefArrayname not in generated_nodes):
                        newnode = deepcopy(node)
                        newnode.data = newrefArrayname
                        state.add_node(newnode)
                        generated_nodes[newrefArrayname] = newnode
                    newnode = generated_nodes[newrefArrayname]

                    if(inorout == 'in'):
                        state.add_edge(edge.src, edge.src_conn, newnode, 
                            edge.dst_conn, deepcopy(oldmem))
                    else:
                        state.add_edge(newnode, edge.src_conn, edge.dst, 
                            edge.dst_conn, deepcopy(oldmem))
            state.remove_node(node)
        
        #TODO: Remove the bank index, if necessary modify the subsets, change the array of all memlets

    return sdfg