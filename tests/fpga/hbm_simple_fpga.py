import dace
import dace.sdfg.nodes
from dace.sdfg import utils as sdutils
from IPython.display import Code
from dace.codegen import exceptions as cgx

import dace.cli.sdfv as sdfv

#Helper for getting a node and potentially apply a function
def mapNode(sdfg, condition = lambda node, state : True, call = lambda node, state : False):
    targets = list(filter(lambda x : condition(x[0], x[1]),sdfg.all_nodes_recursive()))
    for n in targets:
        call(n[0], n[1])
    zipped = list(zip(*targets))
    return list(zipped[0]), list(zipped[1])

def expand_hbm_multiarrays(sdfg):
    """
    This function removes arrays split across k > 1 banks into k new arrays, 
    each with it's own hbm index. (and a new name). 
    Memlets/Accessnodes accessing the array get redefined according 
    to the subsets they access. This includes fully unrolling
    maps that access the memlet. Copymemlets from/to host are created based on hbmalignment.
    """
    def selectNodesConditional(sdfg, condition):
        #Get all nodes that fullfill a condition
        targets = list(filter(lambda x : condition(x[0], x[1]),sdfg.all_nodes_recursive()))
        zipped = list(zip(*targets))
        return list(zipped[0]), list(zipped[1])

    def tryParseHbm(sdfg, node): 
        #Reads the hbm bank-specification of an accessnode if present.
        #Returns none on fail/not present, (low, high) otherwise, 
        #where low == high is possible
        desc = node.desc(sdfg)
        if(not "hbmbank" in desc.location):
            return None
        banks = desc.location["hbmbank"]
        split = banks.split(":")
        if(len(split) == 1):
            try:
                val = int(banks)
                return (val, val)
            except ValueError:
                pass
        elif(len(split) == 2):
            try:
                low = int(split[0])
                high = int(split[1])
                if(high > low):
                    raise ValueError()
            except ValueError:
                pass
        return None

    def isNodeHBMMultiAccess(node, state):
        #Returns true if node is an accessnode
        #spread accross multiple HBM-banks
        if(not isinstance(node, dace.sdfg.nodes.AccessNode)):
            return False
        c = tryParseHbm(state.sdfg, node)
        if(c == None):
            return False
        return c[1] - c[0] > 0

    multiaccesstuple = selectNodesConditional(sdfg, isNodeHBMMultiAccess)
    handledArrays = {} #dace.array -> [dace.array]

    for currentnode in multiaccesstuple:    #Create a map of the old arrays to the new ones
        accessnode, state = currentnode
        oldarray = accessnode.desc(sdfg)
        linkedTo = None
        if oldarray in handledArrays:
            linkedTo = handledArrays[oldarray]
        else:
            low, high = tryParseHbm(sdfg, accessnode) 
            count = high - low
            shape = oldarray.shape
            ndim = len(shape)
            splitaxes = []
            for i in range(ndim):
                splitaxes.append(i)
            alignment = 'even'
            if "hbmalignment" in oldarray.location:
                alignment = oldarray.location["hbmalignment"]
            if(alignment != 'even'): #Check/parse alignment
                if(not isinstance(alignment, list)):
                    cgx.CodegenError("hbmalignment must be 'even' "
                    f"or a list axes in {accessnode.data}")
                duplicatecheck = {}
                for val in alignment:
                    if(val in duplicatecheck or val >= ndim):
                        cgx.CodegenError("alignment list contains duplicates "
                            f"or non existing axes in {accessnode.data}")
                    duplicatecheck[val] = True
                    splitaxes.remove(val)
            splitdim = len(splitaxes)
            if(splitdim == 0):
                cgx.CodegenError("for an array divided across multiple hbm-banks "
                    "there must be at least 1 allowed split dimension" 
                    f"in {accessnode.data}")
            splitcount = round(count ** (1 / splitdim))
            if(splitcount ** splitdim != count):
                cgx.CodegenError("for an array divided across mutiple hbm-banks "
                    "the equation 'hbmbanks == x ** splitdims' must hold where "
                    "hbmbanks is the number of used banks, splitdims is the total "
                    "count of axes along which splitting is allowed and x is an "
                    "arbitrary integer. (So the number of splits in each direction "
                    f"is the same) This does not hold for {accessnode.data}")
            for i in range(count):
                newshape = None
                newarray = sdfg.add_array(f"{accessnode.data}_hbm{i}", 
                    newshape, oldarray.dtype)
                #TODO

    return sdfg

def create_sdfg():
    N = dace.symbol("N")
    sdfg = dace.SDFG("vadd_hbm")

    in1 = sdfg.add_array("in1", [N], dace.float32)
    in2 = sdfg.add_array("in2", [N], dace.float32)
    out = sdfg.add_array("out", [N], dace.float32)

    state = sdfg.add_state()

    n_in1 = state.add_read("in1")
    n_in2 = state.add_read("in2")
    n_out = state.add_write("out")

    tasklet, mapentry, mapexit = state.add_mapped_tasklet(
        "Addition", 
        dict(i="0:N"), 
        dict(inp1=dace.Memlet.simple(n_in1.data, "0:N"), inp2=dace.Memlet.simple(n_in2.data, "0:N")), 
        '''out[i] = in1[i] + in2[i]''', 
        dict(out1=dace.Memlet.simple(n_out.data, "0:N"))
    )

    state.add_edge(n_in1, None, mapentry, None, dace.Memlet.simple(n_in1.data, "0:N"))
    state.add_edge(n_in2, None, mapentry, None, dace.Memlet.simple(n_in2.data, "0:N"))
    state.add_edge(mapexit, None, n_out, None, dace.Memlet.simple(n_out.data, "0:N"))

    sdfg.fill_scope_connectors()
    sdfg.apply_fpga_transformations()

    sdfg.arrays["fpga_in1"].location["hbmbank"] = "0"
    sdfg.arrays["fpga_in2"].location["hbmbank"] = "1"
    sdfg.arrays["fpga_out"].location["hbmbank"] = "2"
    """
    sdfg.arrays["fpga_in1"].location["bank"] = "0"
    sdfg.arrays["fpga_in2"].location["bank"] = "1"
    sdfg.arrays["fpga_out"].location["bank"] = "2"
    """

    return sdfg
    
if __name__ == '__main__':
    sdfg = create_sdfg()
    sdfg = expand_hbm_multiarrays(sdfg)
    sdfv.view(sdfg)
    #code = Code(sdfg.generate_code()[0].code, language='cpp')
    #print(code)