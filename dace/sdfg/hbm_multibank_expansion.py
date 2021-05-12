# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This file collects all helper functions and the top level function
required to transform sdfg's containing hbm-multibank arrays into 
their corresponding representation with only single hbm-bank arrays.

It might at some point be beneficial to add this to the codegen, or to the utils,
it doesn't necessarily have to stay here. 
"""

from dace.codegen import exceptions as cgx
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes as nd
from dace import data, memlet, dtypes, sdfg as sd

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

def collectAndParseHBMArrays(sdfg : sd.SDFG) -> "dict[(str, sd.SDFG), dict[str, any]]":
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

def addHBMAccessNodeEntry(accessnodelist, arraylist, accessnode, state):
    refArrays = arraylist[accessnode.data]
    locationcount = refArrays["splitcount"] * len(refArrays["splitaxes"])
    listindex = (accessnode, state)
    accessnodelist[listindex] = []
    for i in range(locationcount):
        newrefArray = refArrays["arraynames"][i]
        newnode = nd.AccessNode(newrefArray, accessnode.access,
            accessnode.debuginfo)
        state.add_node(newnode)
        accessnodelist[listindex].append(newnode)

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
    statelist : list[sd.SDFGState] = []

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

    for node, parent in sdfg.all_nodes_recursive():
        #Find all states where HBM-multibanks are used
        if(isinstance(node, nd.AccessNode)):
            if(node.data in arraylist):
                statelist.append(parent)

    for oldstate in statelist:
        newstate = sd.SDFGState(oldstate.label, oldstate.parent, oldstate._debuginfo, oldstate.location)
        #rebuild the entire state with expanded HBM-Accesses
        start_nodes = list(v for v in oldstate.nodes() if len(list(oldstate.predecessors(v))) == 0)
        for node in sdutil.dfs_topological_sort(oldstate, start_nodes): #sure this has the right type? (oldstate)
            if(isinstance(node, nd.AccessNode)):
                i =0
                #multiply the node
            elif(isinstance(node, nd.MapEntry)):
                i= 0
                #do something really clever to move into the scope (and decide for unrolling)
            else:
                i =0
                #Probably don't do anything fancy. Look at the attached memlets and copy or replace a simplified version.
                #will need to do this based on looking backwards so nodes are already handled. Maybe This can be done just like that 
                #for acceessNodes and Mapentries, so maybe doing this in else is not even required.
            


    """
    __OLD

    
    
    for i in range(count):
                newshapelist = []
                for d in range(ndim):
                    if d in splitaxes:
                        strip = shape[d] // splitcount
                        if(i == 0):
                            strip += shape[d] % splitcount
                        newshapelist.append(strip)
                    else:
                        newshapelist.append(shape[d])
                newshape = tuple(newshapelist)

                newarrayname, newarray = sdfg.add_array(f"{accessnode.data}_hbm{i}", 
                    newshape, oldarray.dtype)
                newarray.location["hbmbank"] = f"{low + i}"
                handledArrays[accessnode.data]["arraynames"].append(newarrayname)
            
    #Remove the old arrays
    #for old in handledArrays.keys():
    #    sdfg.remove_data(old[0])
    """

    return sdfg