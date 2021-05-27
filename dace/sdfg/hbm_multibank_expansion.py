# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This file collects all helper functions and the top level function
required to transform sdfg's containing hbm-multibank arrays into 
their corresponding representation with only single hbm-bank arrays.

This is only kept as 1 file for development, after that some of those functions should go to utils,
the rest into a file in the codegen.
"""

from copy import deepcopy

from dace.sdfg import utils as sdutil
from dace.sdfg import nodes as nd
from dace.sdfg import graph 
from dace.sdfg import state as statenamespace
from dace import data, memlet, subsets, symbolic, dtypes, sdfg as sd

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

def get_unroll_map_properties(state : statenamespace.SDFGState, entry : nd.MapEntry) -> "Union[str, list[int]]":
    if(len(entry.map.params) != 1):
        return "only supported for 1 dimensional maps at the moment"
    if(not entry.map.unroll):
        return "map.unroll must be set to allow unrolling"
    low, high, stride = entry.map.range[0]
    try:    
        low = int(str(low))
        high = int(str(high))
        stride = int(str(stride))
    except:
        return "all the arguments have to be integers to support unrolling"
    paramvals = []
    for i in range(low, high+1, stride):
        paramvals.append(i)
    return paramvals

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

"""
def parseHBMBank(arrayname : str, array : data.Array) -> "tuple[int, int]": 
    ""
    Reads the hbm bank-specification of an array if present.

    :param arrayname: The name of the array
    :param array: The array
    :return: None if not present, (low, high) otherwise, where low == high is possible.
    ""
    if(not "hbmbank" in array.location):
        return None
    errormsg = ("location['hbmbank'] must be a string"
        f" with format 'int' or 'i1:i2' with i1, i2"
        f" type int and i1<i2 for {arrayname}")
    banks = array.location["hbmbank"]
    if(not isinstance(banks, str)):
        raise TypeError(errormsg)
    split = banks.split(":")
    if(len(split) == 1):
        try:
            val = int(banks)
            return (val, val)
        except ValueError:
            raise ValueError(errormsg)
    elif(len(split) == 2):
        try:
            low = int(split[0])
            high = int(split[1])
            if(low > high):
                raise ValueError()
            return (low, high)
        except ValueError:
            raise ValueError(errormsg)
    raise ValueError(errormsg)
"""

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
        raise ValueError("hbmalignment must be 'even' "
        f"or a list of axes in {arrayname}")
    alignment.sort()
    lastval = None
    for val in alignment:
        if(val >= ndim or (lastval != None and lastval == val)):
            raise ValueError("alignment list contains duplicates "
                f"or non existing axes in {arrayname}")
        lastval = val
        splitaxes.remove(val)
    return splitaxes

def parseHBMArray(arrayname : str, array : data.Array) -> "dict[str, Any]":
    """
    parses HBM properties of an array (hbmbank and hbmalignment). 
    Returns none if hbmbank is not present as property

    :return: A mapping from (arrayname, sdfg of the array) to a mapping
    from string that contains collected information.
    'ndim': contains the dimension of the array
    'splitcount': contains how many times this array is split
        on each of the axes along which it is split 
        (ie splitcount==2 on an 2d-array which is split along axis 0
        and axis 1 => There are 4 parts)
    'splitaxes': List that contains axes along which the array is split
    'lowbank': The lowest bank index this array is placed on
    'shape': The shape of the whole array
    'numbanks': The number of banks across which this array spans
    """
    if("hbmbank" not in array.location):
        return None
    hbmbankrange : subsets.Range = array.location["hbmbank"]
    if(not isinstance(hbmbankrange, subsets.Range)):
        raise TypeError(f"Locationproperty 'hbmbank' must be of type subsets.Range for {arrayname}")
    low, high, stride = hbmbankrange[0]
    if(stride != 1):
        raise NotImplementedError(f"Locationproperty 'hbmbank' does not support stride != 1 for {arrayname}")
    count = high - low + 1
    shape = array.shape
    ndim = len(shape)
    if(low == high):
        return {"ndim" : ndim, "shape" : array.shape,
            "splitcount" : 1, "splitaxes" : [], "lowbank" : low, "numbank": 1}
    splitaxes = parseHBMAlignment(arrayname, array)
    splitdim = len(splitaxes)
    if(splitdim == 0):
        raise ValueError("for an array divided across multiple hbm-banks "
            "there must be at least 1 allowed split dimension" 
            f"in {arrayname}")
    splitcount = round(count ** (1 / splitdim))
    if(splitcount ** splitdim != count):
        raise ValueError("for an array divided across mutiple hbm-banks "
            "the equation 'hbmbanks == x ** splitdims' must hold where "
            "hbmbanks is the number of used banks, splitdims is the total "
            "count of axes along which splitting is allowed and x is an "
            "arbitrary integer. This is necessary so the number of splits "
            "in each direction is the same). This does not hold for "
            f"{arrayname}")
    return {"ndim" : ndim, "shape" : array.shape,
            "splitcount" : splitcount, "splitaxes" : splitaxes, "lowbank" : low,
            "numbank": splitcount ** len(splitaxes)}

def findAndParseHBMMultibank(sdfg : sd.SDFG) -> "dict[(str, sd.SDFG), dict[str, Any]]":
    """
    Finds and parses HBM arrays that are spread across multiple banks in the sdfg
    """
    arrays = sdfg.arrays_recursive()
    handledArrays = {}  #(oldarrayname, sdfg) -> 
        #{ndim -> int, splitcount->int, splitaxes->[int]}
    for currentsdfg, arrayname, array in arrays:
        collected = parseHBMArray(arrayname, array)
        if(collected == None or collected['numbank'] == 1):
            continue
        handledArrays[(arrayname, currentsdfg)] = collected
    return handledArrays

def spliceHBMMemlet(state : statenamespace.SDFGState, edge : graph.MultiConnectorEdge, spliceSource : bool,
    aimsAtBank : int, divideVolumeBy : int, followupConnector : str = None, edgeconnectorname = None):
    """
    This is a helper for hbm memlets.
    'Splices', i.e. duplicates and reconnects an edge to a newly created connector. Volume and subset 
    of the new memlet are modified to target a single bank. Optionally adds a new corresponding connector.

    :param spliceSource: Wheter to splice at the source or at the destination of the edge
    :param aimsAtBank: The bank index of the spliced memlet
    :param divideVolumeBy: An integer by which the volume of the spliced memlet is divided
    :param followupConnector: The old name of a followup connector
    :param edgeconnectorname: If None will generate a unique new connector name, otherwise it will
        take the provided names as given for the edge connector
    """
    newmem = deepcopy(edge.data)
    newmem.subset[0] = (symbolic.pystr_to_symbolic(aimsAtBank),
                symbolic.pystr_to_symbolic(aimsAtBank),
                symbolic.pystr_to_symbolic(1))
    newmem.volume = symbolic.pystr_to_symbolic("floor(" + 
        str(newmem.volume) + f" / {divideVolumeBy})")

    if(spliceSource):
        referedNode = edge.src
        if not edgeconnectorname:
            edgeconnectorname = getNonExistingName(f"{edge.src_conn}_{aimsAtBank}", edge.src.out_connectors)
        if followupConnector:
            connectstoconnectorname = getNonExistingName(f"{followupConnector}_{aimsAtBank}", 
                edge.src.in_connectors)
            referedNode.add_in_connector(connectstoconnectorname)
        referedNode.add_out_connector(edgeconnectorname)
        newedge = state.add_edge(edge.src, edgeconnectorname, 
            edge.dst, edge.dst_conn, newmem)
    else:
        referedNode = edge.dst
        if not edgeconnectorname:
            edgeconnectorname =  getNonExistingName(f"{edge.dst_conn}_{aimsAtBank}", edge.src.in_connectors)
        if followupConnector:
            connectstoconnectorname = getNonExistingName(f"{followupConnector}_{aimsAtBank}",
                edge.src.out_connectors)
            referedNode.add_out_connector(connectstoconnectorname)
        referedNode.add_in_connector(edgeconnectorname)        
        newedge = state.add_edge(edge.src, edge.src_conn, 
             edge.dst, edgeconnectorname, newmem)
    if(followupConnector):
        return (newedge, connectstoconnectorname)
    else:
        return newedge
    

def recursive_splice_hbmmemlettree(state : statenamespace.SDFGState,
        tree : memlet.MemletTree, flowsTowardsRoot : bool) -> "dict[int, list[graph.MultiConnectorEdge]]":
    """
    applying unroll results in an sdfg that still contains multibank memlets
    on the paths to the unrolled maps. This function "splices" those memlets,
    such that all memlets from/to the unrolled maps are single bank only. Note that
    the resulting state may still contain multibankmemlets between accessnodes, or on 
    nested sdfgs

    :param flowsTowardsRoot: Does the tree carry data towards the accessnode or away from it?
    """
    outgoingmemlets : dict[int, graph.MultiConnectorEdge] = {}
    returnval = {}
    for child in tree.children:
        result = recursive_splice_hbmmemlettree(state, child, flowsTowardsRoot)
        for banknum in result.keys():
            if(banknum in outgoingmemlets):
                #Would make the computation of the volume terrible
                raise ValueError("Accessing the same hbmbank multiple times at the same connector is disallowed")
            else:
                outgoingmemlets[banknum] = result[banknum]
    edge : graph.MultiConnectorEdge = tree.edge
    mem : memlet.Memlet = edge.data

    if(len(outgoingmemlets) > 1):
        subsetlow = int(str(mem.subset[0][0]))
        subsethigh = int(str(mem.subset[0][1]))
        stride = int(str(mem.subset[0][2]))
        if(stride != 1):
            raise NotImplementedError()
        
        if(flowsTowardsRoot):
            oldoutconnector = edge.src_conn
            oldinconnector = next(iter(outgoingmemlets.values())).dst_conn
            referedNode = edge.src
        else:
            oldinconnector = edge.dst_conn
            oldoutconnector = next(iter(outgoingmemlets.values())).src_conn
            referedNode = edge.dst
        for banknum in outgoingmemlets.keys():
            #Create the new edges and connectors and reconnect 
            assert(banknum >= subsetlow and banknum <= subsethigh) #TODO: Handle stride !=1
            connectsTo = outgoingmemlets[banknum]
            if(flowsTowardsRoot):
                newedge, connectstoconnectorname = spliceHBMMemlet(state, edge, True, 
                    banknum, len(outgoingmemlets), connectsTo.dst_conn)
            else:
                newedge, connectstoconnectorname = spliceHBMMemlet(state, edge, False,
                    banknum, len(outgoingmemlets), connectsTo.src_conn)
            returnval[banknum] = [newedge]
            if(flowsTowardsRoot):
                connectsTo.dst_conn = connectstoconnectorname
            else:
                connectsTo.src_conn = connectstoconnectorname
        state.remove_edge(edge)
        referedNode.remove_in_connector(oldinconnector)
        referedNode.remove_out_connector(oldoutconnector)
    else:
        banknum = int(str(edge.data.subset[0][0]))
        returnval[banknum] = edge
    return returnval

def getHBMBankOffset(refInfo : "dict[str, Any]", bank : int) -> "list[str]":
    """
    Returns the offset of a bank in a hbm multibankarray
    :param refInfo: A dict containing info about the array as returned by parseHBMArray
    :param bank: The bank for which to compute the offset
    """
    dim = refInfo['ndim']
    splitaxes = refInfo['splitaxes']
    splitcount = refInfo['splitcount']
    oldshape = refInfo['shape']
    bankoffset = []
    for d in range(dim):
        if(d not in splitaxes):
            bankoffset.append(0)
        else:
            currentoffset = bank % splitcount
            bankoffset.append(currentoffset)
            bank = bank // splitcount
    trueoffset = []
    for d in range(dim):
        if(bankoffset[d] == 0):
            trueoffset.append("0")
        else:
            trueoffset.append(
                f"({oldshape[d]}//{splitcount})*{bankoffset[d]}"
            )
    return trueoffset

def getHBMHostRange(bank : int, refInfo : "dict[str, Any]",
        hbmrange : subsets.Range) -> "subsets.Range":
    """
    Returns the range corresponding to the 'virtual' index of a
    hbm-array given a bank and a range off the array on that bank
    
    :param bank: The bank
    :param refInfo: A dict containing info about the array as returned by parseHBMArray
    :param hbmrange: The subset on the hbm bank. Note that this is assumed to have a
    bank index (as it's first index) which is equal to :param bank:
    """
    hostsubset = ""
    dim = refInfo['ndim']
    trueoffset = getHBMBankOffset(refInfo, bank)
    for d in range(dim):
        hostsubset += (f"{trueoffset[d]}:{str(hbmrange[d+1][1])} + "
            f"1 + {trueoffset[d]}")
        if(d < dim-1):
            hostsubset += ","
    return subsets.Range.from_string(hostsubset)

def getHBMTrueShape(virtualshape : "Any", splitaxes : "list[int]", splitcount : int) -> "list[int]":
    """
    :returns: the shape of a part-array on one HBMbank for an HBM-multibank-array
    """
    newshapelist = []
    for d in range(len(virtualshape)):
        if d in splitaxes:
            newshapelist.append(virtualshape[d] // splitcount)
        else:
            newshapelist.append(virtualshape[d])
    return newshapelist

def expand_hbm_multiarrays(sdfg : sd.SDFG) -> sd.SDFG:
    """
    This function removes arrays split across k > 1 banks into k new arrays, 
    each with it's own hbm index. (and a new name). 
    Memlets/Accessnodes accessing the array get redefined according 
    to the subsets they access. This includes fully unrolling
    maps that access the memlet (if their index is used as bank index).
    Copymemlets are created based on hbmalignment.
    """
                
    info = findAndParseHBMMultibank(sdfg)
    if(len(info) == 0):
        return

    #mapping from old arrays to a list of the new arrays
    arraylist : dict[(str, sd.SDFG),  list[(str, data.Array)]] = {}
    #set of all newly created arrays
    newcreatedarrays : set[(str, sd.SDFG)] = set()
    #all states that contain hbm multibanks
    statelist : set[sd.SDFGState] = set()
    CPU_STORAGE_TYPES = {
        dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal,
        dtypes.StorageType.CPU_Pinned, dtypes.StorageType.Default
    }

    #Create the new arrays and store them in a mapping from old to new
    for arrayname, arraysdfg in info.keys():
        curinfo = info[(arrayname, arraysdfg)]
        oldarray : data.Array = arraysdfg.arrays[arrayname]
        shape = oldarray.shape
        splitcount = curinfo["splitcount"]
        axes = curinfo["splitaxes"]
        numbanks = curinfo["numbank"]
        arraylist[(arrayname, arraysdfg)] = []
        newshape = tuple(getHBMTrueShape(shape, axes, splitcount))

        for i in range(numbanks):
            newname = f"{arrayname}_hbm{i}"
            newname = getNonExistingName(newname, arraysdfg.arrays)
            newname, newarray = arraysdfg.add_array(    #TODO: See if alignment=0 is actually ok, add support for offset != 0 
                newname, newshape, oldarray.dtype,
                oldarray.storage, oldarray.transient,
                offset=deepcopy(oldarray.offset),
                lifetime=deepcopy(oldarray.lifetime), 
                debuginfo=deepcopy(oldarray.debuginfo),
                allow_conflicts=oldarray.allow_conflicts, 
                alignment=0,
                may_alias=oldarray.may_alias,
            )
            newarray.location["hbmbank"] = subsets.Range.from_string(str(curinfo['lowbank'] + i))
            arraylist[(arrayname, arraysdfg)].append((newname, newarray))
            newcreatedarrays.add((newname, arraysdfg))
        arraysdfg.remove_data(arrayname, validate=False)

    #Find all states where HBM-multibanks are used
    for lookatsdfg in sdfg.all_sdfgs_recursive():
        for state in lookatsdfg.states():
            for node in state.nodes():
                if(isinstance(node, nd.AccessNode)):
                    if((node.data, lookatsdfg) in arraylist):
                        statelist.add(state)

    for state in statelist:
        def intern_iterate_attached_memlets(node, hbmarraylist=[], allmemlets=False):
            #small helper to iterate over all memlets of a node
            for inp in state.in_edges(node):
                if((inp.data.data, state.parent) in hbmarraylist or allmemlets):
                    yield 'in', inp
            for outp in state.out_edges(node):
                if((outp.data.data, state.parent) in hbmarraylist or allmemlets):
                    yield 'out', outp

        accessnodestoexpand = set()
        mapstounroll = {}
        nestedsdfgs : list[nd.NestedSDFG]= []

        #Find all accessnodes, maps and nestedSDFGs which should be expanded
        start_nodes = sdutil.find_source_nodes(state)
        for node in sdutil.dfs_topological_sort(state, start_nodes):
            if(isinstance(node, nd.AccessNode)):
                currentindex = (node.data, state.parent)
                if(currentindex in arraylist):
                    accessnodestoexpand.add(node)
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
            elif(isinstance(node, nd.NestedSDFG)):
                nestedsdfgs.append(node)

        #unroll all maps
        for entry, exit in mapstounroll.values():
            unroll_map(state, entry, exit)

        #Splice hbm inputs to nested sdfg's
        for nsdfg in nestedsdfgs:
            for inorout, edge in intern_iterate_attached_memlets(nsdfg, arraylist):
                if(inorout == 'in'):
                    currentname = edge.dst_conn
                else:
                    currentname = edge.src_conn
                targets = arraylist[(currentname, nsdfg.sdfg)]
                for i in range(len(targets)):
                    spliceHBMMemlet(state, edge, inorout == 'out', i,
                        len(targets), edgeconnectorname=targets[i][0])
                state.remove_edge_and_connectors(edge)

        #expand all memlets leading from/to maps such that only single bank memlets stay
        for accessnode in accessnodestoexpand:
            for inorout, watchededge in intern_iterate_attached_memlets(accessnode, arraylist):
                root = state.memlet_tree(watchededge)
                recursive_splice_hbmmemlettree(state, root, inorout == 'in')

        #expand all accessnodes, so that each accessnode only accesses a single bank
        for node in accessnodestoexpand:
            currentindex = (node.data, state.parent)
            refArrays = arraylist[currentindex]
            refInfo = info[currentindex]
            generated_nodes : dict[str, nd.AccessNode] = {}

            for inorout, edge in intern_iterate_attached_memlets(node,allmemlets=True):
                hbmtohost = False
                hosttohbm = False
                if(isinstance(edge.src, nd.AccessNode) and isinstance(edge.dst, nd.AccessNode)):
                    #This is a copy memlet
                    hbmtohost = edge.src == node and edge.dst.desc(state).storage in CPU_STORAGE_TYPES
                    hosttohbm = edge.dst == node and edge.src.desc(state).storage in CPU_STORAGE_TYPES
                    if(not hosttohbm and not hbmtohost):
                        raise NotImplementedError("At the moment copy memlets between data nodes on "
                            "the fpga are not supported")
                oldmem = edge.data
                #TODO: Throw errors if the correct subset does not exist, or if parsing fails
                if(hosttohbm):
                    refsubset = oldmem.dst_subset
                elif(hbmtohost):
                    refsubset = oldmem.src_subset
                else:
                    refsubset = oldmem.subset
                banklow = int(str(refsubset[0][0]))
                bankhigh = int(str(refsubset[0][1]))
                stride = int(str(refsubset[0][2]))
                if(banklow != bankhigh and not hosttohbm and not hbmtohost):
                    raise ValueError("Found not expanded multibank memlets")

                #TODO: support stride != 1
                if(stride != 1):
                    raise NotImplementedError()

                def updateSubsets(mem : memlet.Memlet, bank : int):
                    """
                    Updates the accessed subsets. At the moment there is no notion of what it means to
                    have a defined src_subset for "host to hbm" respectively dst_subset for 
                    "hbm to host", hence those values are simply overwritten if they exist. Also
                    note that the corresponding opposite subset must be set.
                    """
                    mem = deepcopy(mem)
                    if(hosttohbm):
                        mem.src_subset = getHBMHostRange(bank, refInfo, mem.dst_subset)
                        mem.dst_subset[0] = subsets.Range.from_string(str(bank))[0]
                        mem.volume = mem.src_subset.num_elements()
                    elif(hbmtohost):
                        mem.dst_subset = getHBMHostRange(bank, refInfo, mem.src_subset)
                        mem.src_subset[0] = subsets.Range.from_string(str(bank))[0]
                        mem.volume = mem.dst_subset.num_elements()
                    return mem

                #For each bank referenced by the memlet create an accessnode for
                #that bank if it does not already exist, and reconnect with single bank memlet
                for k in range(bankhigh + 1 - banklow):
                    currentbank = banklow + k
                    newrefArrayname, newrefArray = refArrays[currentbank]
                    if(newrefArrayname not in generated_nodes):
                        newnode = deepcopy(node)
                        newnode.data = newrefArrayname
                        state.add_node(newnode)
                        generated_nodes[newrefArrayname] = newnode
                    newnode = generated_nodes[newrefArrayname]

                    if(inorout == 'in'):
                        state.add_edge(edge.src, edge.src_conn, newnode, 
                            edge.dst_conn, updateSubsets(oldmem, currentbank))
                    else:
                        state.add_edge(newnode, edge.src_conn, edge.dst, 
                            edge.dst_conn, updateSubsets(oldmem, currentbank))
            state.remove_node(node)
        
        #Remove the bank index, change the data source
        for edge in state.edges():
            mem : memlet.Memlet = edge.data
            if((mem.data, state.parent) in arraylist):
                banklow = int(str(mem.subset[0][0]))
                bankhigh = int(str(mem.subset[0][1]))
                #TODO: Raise a nice error
                assert(banklow == bankhigh)
                mem.data = arraylist[(mem.data, state.parent)][banklow][0]
                mem.subset.pop({0})
            elif(isinstance(edge.dst, nd.AccessNode) and (edge.dst.data, state.parent) in newcreatedarrays):
                banklow = int(str(mem.other_subset[0][0]))
                bankhigh = int(str(mem.other_subset[0][1]))
                #TODO: Raise a nice error
                assert(banklow == bankhigh)
                mem.other_subset.pop({0})
    return sdfg