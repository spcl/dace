import dace
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace import dtypes
from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA



"""


sdfg = example.to_sdfg()
ota = OtA()
ota.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
ota.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)
ota.insert_copies_cfregion(sdfg, sdfg)

"""


"""
thesis
introduction: rephrase thesis globals
list contributions
related work: how do others lower things to GPU
implementation: document code and ask claude to summarize
evaluation: test against nbench, test sdfgs which failed previously
"""


# copy insertion tests (copy inserted)

# A -> add -> tmp(A) -> mul -> C
def simple_sdfg():
    sdfg = dace.SDFG("minimal_chain")
    state = sdfg.add_state()

    sdfg.add_array("X", [1], dace.float64)      # input
    sdfg.add_array("Y", [1], dace.float64)      # output
    sdfg.add_transient("A", [1], dace.float64)  # intermediate access node

    X = state.add_access("X")
    A = state.add_access("A")
    Y = state.add_access("Y")

    t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
    t2 = state.add_tasklet("comp2", {"a"}, {"y"}, "y = a * 2")

    state.add_edge(X, None, t1, "x", dace.Memlet("X[0]"))
    state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))
    state.add_edge(A, None, t2, "a", dace.Memlet("A[0]"))
    state.add_edge(t2, "y", Y, None, dace.Memlet("Y[0]"))

    return sdfg, state, A

def simple_sdfg_with_copy():
    # Build SDFG
    sdfg = dace.SDFG("minimal_chain_with_copy")
    state = sdfg.add_state()

    sdfg.add_array("X", [1], dace.float64)            # input to computation 1
    sdfg.add_transient("A", [1], dace.float64)        # output of computation 1
    sdfg.add_transient("A_copy", [1], dace.float64)   # explicit copy of A
    sdfg.add_array("Y", [1], dace.float64)            # output of computation 2

    X = state.add_access("X")
    A = state.add_access("A")
    A_copy = state.add_access("A_copy")
    Y = state.add_access("Y")

    t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
    t2 = state.add_tasklet("comp2", {"a"}, {"y"}, "y = a * 2")
    
    # computation 1: X -> A
    state.add_edge(X, None, t1, "x", dace.Memlet("X[0]"))
    state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))

    # explicit copy: A -> A_copy
    state.add_edge(A, None, A_copy, None, dace.Memlet("A[0] -> A_copy[0]"))

    # computation 2: A_copy -> Y
    state.add_edge(A_copy, None, t2, "a", dace.Memlet("A_copy[0]"))
    state.add_edge(t2, "y", Y, None, dace.Memlet("Y[0]"))
    return sdfg

def medium_sdfg():
    sdfg = dace.SDFG("minimal_chain_multiuse_A")
    state = sdfg.add_state()

    # I/O
    sdfg.add_array("X", [1], dace.float64)
    sdfg.add_array("Y", [1], dace.float64)

    # intermediates
    sdfg.add_transient("A", [1], dace.float64)
    sdfg.add_transient("B", [1], dace.float64)
    sdfg.add_transient("C", [1], dace.float64)
    sdfg.add_transient("D", [1], dace.float64)

    X = state.add_access("X")
    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")
    D = state.add_access("D")
    Y = state.add_access("Y")

    # original two computations
    t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
    t2 = state.add_tasklet("comp2", {"a"}, {"b"}, "b = a * 2")

    # new computations, all reading A
    t3 = state.add_tasklet("comp3", {"a"}, {"c"}, "c = a + 3")
    t4 = state.add_tasklet("comp4", {"a"}, {"d"}, "d = a - 1")

    # final merge into Y
    t5 = state.add_tasklet("merge", {"b", "c", "d"}, {"y"}, "y = (b + c + d) / 3.0")

    # X -> t1 -> A
    state.add_edge(X, None, t1, "x", dace.Memlet("X[0]"))
    state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))

    # A -> (t2, t3, t4)
    state.add_edge(A, None, t2, "a", dace.Memlet("A[0]"))
    state.add_edge(A, None, t3, "a", dace.Memlet("A[0]"))
    state.add_edge(A, None, t4, "a", dace.Memlet("A[0]"))

    # (t2, t3, t4) -> (B, C, D)
    state.add_edge(t2, "b", B, None, dace.Memlet("B[0]"))
    state.add_edge(t3, "c", C, None, dace.Memlet("C[0]"))
    state.add_edge(t4, "d", D, None, dace.Memlet("D[0]"))

    # (B, C, D) -> t5 -> Y
    state.add_edge(B, None, t5, "b", dace.Memlet("B[0]"))
    state.add_edge(C, None, t5, "c", dace.Memlet("C[0]"))
    state.add_edge(D, None, t5, "d", dace.Memlet("D[0]"))
    state.add_edge(t5, "y", Y, None, dace.Memlet("Y[0]"))

    return sdfg, state, A

def complex_sdfg():
    TS = dace.symbol("TS")
    @dace.program
    def example(A: dace.float64[100, 100], B: dace.float64[100, 100], C: dace.float64[100, 100], D: dace.float64[100, 100], E: dace.float64[100]) -> dace.float64[100, 100]:
        for t1 in range(TS):
            for i, j in dace.map[0:100, 0:100]:
                C[i, j] = A[i, j] + B[i, j]
        for t2 in range(2):
            for j in range(100):
                for i in dace.map[0:100]:
                    E[i] = E[i] + C[i, j]
            for i in range(1, 100):
                E[i] = (E[i-1] + E[i]) / 100.0
        for t3 in range(2):
            for i, j in dace.map[0:100, 0:100]:
                D[i, j] = E[i] * 2.0 + C[i, j]

    return example.to_sdfg()


def two_state_sdfg():
    sdfg = dace.SDFG("two_states")

    # Arrays
    sdfg.add_array("in", [1], dace.float64)
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)

    # States + transition
    s1 = sdfg.add_state("s1", is_start_block=True)
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    # s1: in -> A
    in_node = s1.add_access("in")
    a_s1 = s1.add_access("A")
    t1 = s1.add_tasklet("t1", {"x"}, {"y"}, "y = x + 1")
    s1.add_edge(in_node, None, t1, "x", dace.Memlet("in[0]"))
    s1.add_edge(t1, "y", a_s1, None, dace.Memlet("A[0]"))

    # s2: use A inside a map, write to out
    a_s2 = s2.add_access("A")
    out_s2 = s2.add_access("out")

    me, mx = s2.add_map("m", dict(i="0:1"))
    t2 = s2.add_tasklet("t2", {"a"}, {"y"}, "y = a * 2")

    s2.add_memlet_path(a_s2, me, t2, memlet=dace.Memlet("A[i]"), dst_conn="a")
    s2.add_memlet_path(t2, mx, out_s2, memlet=dace.Memlet("out[i]"), src_conn="y")

    return sdfg

### implementation


def pretty_print(gpu_set, cpu_set):
    print("both GPU and CPU:")
    for name in gpu_set & cpu_set:
        print("\t", name)

    print("\nGPU only:")
    for name in gpu_set - cpu_set:
        print("\t", name)
    
    print("\nCPU only:")
    for name in cpu_set - gpu_set:
        print("\t", name)

#def _insert_node_on_edge(state, edge, new_node):
#    """Insert new_node on a dataflow edge (MultiConnectorEdge) inside a state."""
#    state.add_edge(edge.src, edge.src_conn, new_node, None,          edge.data) # add A -> new
#    state.add_edge(new_node, None,          edge.dst, edge.dst_conn, edge.data)      # add new -> B
#    state.remove_edge(edge)                                                     # del A -> B


def insert_copy(sdfg, state, access_node, suffix) -> nodes.AccessNode:
    # determine names
    access_name = access_node.data
    copy_name = access_name + "_" + suffix

    # create copy node
    sdfg.add_transient(copy_name, [1], dace.float64)
    access_copy = state.add_access(copy_name)

    # disconnect access_node from all children and connect to copy instead
    for edge in state.out_edges(access_node):
        memlet = edge.data # rename arrays in memlet
        memlet.replace({access_name: copy_name})

        state.add_edge(access_copy, None, edge.dst, edge.dst_conn, memlet) # new edge
        state.remove_edge(edge) # disconnect / delete old
    
    # connect access_node to access_copy (must be done second, else connection gets deleted in loop above)
    state.add_edge(access_node, None, access_copy, None, dace.Memlet(f"{access_node} -> {copy_name}"))

    return access_copy

def insert_gpu_copy(sdfg, state, access_node):
    copy : nodes.AccessNode = insert_copy(sdfg, state, access_node, "gpu")
    sdfg.arrays[access_node.data].storage   = dtypes.StorageType.Default
    sdfg.arrays[copy.data].storage          = dtypes.StorageType.GPU_Global

def insert_cpu_copy(sdfg, state, access_node):
    copy : nodes.AccessNode = insert_copy(sdfg, state, access_node, "cpu")
    sdfg.arrays[access_node.data].storage   = dtypes.StorageType.GPU_Global
    sdfg.arrays[copy.data].storage          = dtypes.StorageType.Default

### tests ###

class OffloadingIRNode:

    def __init__(self, block, cpu_set=set(), gpu_set=set(), next=set()):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.block : ControlFlowBlock = block
        self.cpu_set : set[str] = cpu_set
        self.gpu_set : set[str] = gpu_set
        self.next : set[OffloadingIRNode] = next

    def __repr__(self):
        return f"{self.block}: cpu = {self.cpu_set}, gpu = {self.gpu_set}\n\t{self.block} next: {"; ".join([next.__repr__() for next in self.next])}\n"
    def __str__(self): 
        return self.__repr__()
    
    # utility functions
    def is_empty(self):
        return self.block is None
    
    def append_node(self, node):
        self.next.add(node)

    def find_all_leaves(self, result:set):
        if self.is_leaf():
            result.add(self)
        else:
            for next in self.next:
                next.find_all_leaves(result)

    def is_leaf(self):
        return not self.next

    # static makers
    def make_empty():
        return OffloadingIRNode(None, set(), set(), set())
    

def sdfg_to_IR(sdfg:SDFG):
    all_arrays = {name for name in sdfg.arrays}
    # TODO: check if anything is already GPU or not default and create error
    
    # create inital node where all arrays must be on CPU
    IR = OffloadingIRNode.make_empty()
    IR.cpu_set = all_arrays # all arrays are initially assumed to be on CPU

    # create IR
    _parse_to_IR(sdfg, sdfg, IR)
    print(IR)

    # all arrays which aren't used by this state retain their previous status
    # ASSUMPTION: arrays are either gpu or cpu within a state
    def propagate_arrays(node):
        for next in node.next:
            next_arrays = next.cpu_set | next.gpu_set
            #print(next_arrays)
            for array in node.cpu_set:
                if not array in next_arrays:
                    next.cpu_set.add(array)
            for array in node.gpu_set:
                if not array in next_arrays:
                    next.gpu_set.add(array)
            propagate_arrays(next)

    propagate_arrays(IR)
    print(IR)

    # clean up
    remove_empty_nodes(IR)
    print(IR)

    # add final node where all arrays must be on CPU again
    final = OffloadingIRNode.make_empty()
    final.cpu_set = all_arrays

    leaves = set()
    IR.find_all_leaves(leaves)
    
    for leaf in leaves:
        leaf.append_node(final)
    print(IR)

    return IR

def _parse_to_IR(sdfg:SDFG, cfr:ControlFlowRegion, curr_node:OffloadingIRNode) -> OffloadingIRNode:
    # edges
    # if there are edge accesses, use a new node to represent this controlflow region
    # if copies are later necessary, they will be added before the entire region
    if cfr.parent_graph is not None:
        arrays = set()
        edge : dace.sdfg.InterstateEdge
        for edge in cfr.parent_graph.edges(cfr):
            arrays |= set(edge.data.used_arrays(sdfg.arrays))

        if arrays:
            new_node = OffloadingIRNode(cfr, cpu_set=arrays, gpu_set=set(), next=set())
            curr_node.append_node(new_node)
            curr_node = new_node
        

    # nodes
    block : ControlFlowBlock
    for block in cfr.bfs_nodes():

        # non-nested state
        if isinstance(block, SDFGState):
            state : SDFGState = block
            gpu_set,cpu_set = OtA().get_data_locations_of_state(sdfg, state) # beating heart of this function
            new_node = OffloadingIRNode(state, cpu_set, gpu_set, set())
            curr_node.append_node(new_node)
            curr_node = new_node

        # if else
        elif isinstance(block, ConditionalBlock):
            # connect current node to new node reprenting the branching condition
            # find all array accesses in condition and add to cpu set
            # if condition necessitates copies, they will be added before the block
            branch_condition = OffloadingIRNode(block)
            for memlet in block.get_meta_read_memlets():
                if memlet.data in sdfg.arrays:
                    branch_condition.cpu_set.add(memlet.data)
            curr_node.append_node(branch_condition)

            # parse branches and connect each branch head to branch condition
            tails = set()
            for _, branch in block.branches:
                branch_head : OffloadingIRNode = _parse_to_IR(sdfg, branch, branch_condition)
                branch_head.find_all_leaves(tails)

            # connect all tails to empty connector node (= new current node)
            curr_node = OffloadingIRNode.make_empty()
            for tail in tails:
                tail.append_nodes(curr_node)

        # loop
        elif isinstance(block, LoopRegion):
            # parse loop region and connect to current node
            loop : LoopRegion = block
            head : OffloadingIRNode = _parse_to_IR(sdfg, loop, curr_node) # linked list representing all internal nodes of loop

            # get array accesses of init_statement, update_statement, and loop_condition add them to head's cpu_set
            for memlet in loop.get_meta_read_memlets():
                if memlet.data in sdfg.arrays:
                    head.cpu_set.add(memlet.data)
            
            # connect all tails to empty connector node
            # connect all tails to head again (-> loop)
            tails = head.find_all_leaves()
            curr_node = OffloadingIRNode.make_empty()
            for tail in tails:
                tail.append_nodes(curr_node)
                tail.append_nodes(head)

        # nested region -> flatten   
        elif isinstance(block, (ControlFlowRegion, nodes.NestedSDFG) ):
            _parse_to_IR(block, curr_node)

        # do nothing
        elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
            pass 

        else:
            raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

    return curr_node


def remove_empty_nodes(node:OffloadingIRNode):
    # NOTE: if given node is empty, it won't be removed - only children are checked
    empties = {next for next in node.next if next.is_empty()}
    
    for empty in empties:
        node.next.remove(empty)
        for nextnext in empty.next:
            node.append_node(nextnext)

    for next in node.next:
        remove_empty_nodes(next)

def eval_IR(sdfg, IR:OffloadingIRNode):
    # modifies SDFG in place & inserts all necessary copies
    # TODO: collect all gpu & all cpu copies intp one state
    for next in IR.next:

        if IR.cpu_set & IR.gpu_set:
            updated_c, updated_g = print(f"insert intrastate copy for {IR}")
            IR.cpu_set = updated_c
            IR.gpu_set = updated_g

        gpu_copies = IR.cpu_set & next.gpu_set
        if gpu_copies:
            OtA().create_interstate_copy(sdfg, IR.block, next.block, gpu_copies, to_gpu=True)
            #next.cpu_set = updated_c
            #next.gpu_set = updated_g
            print(f"insert gpu copy for {gpu_copies} between {IR.block} and {next.block}")# -> {updated_c}, {updated_g}")

        cpu_copies = IR.gpu_set & next.cpu_set
        if cpu_copies:
            OtA().create_interstate_copy(sdfg, IR.block, next.block, cpu_copies, to_gpu=False)
            #next.cpu_set = updated_c
            #next.gpu_set = updated_g
            print(f"insert cpu copy for {cpu_copies} between {IR.block} and {next.block}")# -> {updated_c}, {updated_g}")

        """# ensure correct schedules
        for array_name in IR.cpu_set & next.cpu_set:
            sdfg.arrays[array_name].storage = dtypes.StorageType.Default
        
        for array_name in IR.gpu_set & next.gpu_set:
            sdfg.arrays[array_name].storage = dtypes.StorageType.GPU_Global
        """
    
    for next in IR.next:
        eval_IR(sdfg, next)

import numpy as np

def test_two_state_sdfg(input:float):
    sdfg = two_state_sdfg()
    sdfg.validate()

    orig_output = np.array([0.0])
    sdfg(**{"in": np.array([input]), "A":np.array([0.0]), "out": orig_output})
    
    ota = OtA()
    ota.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
    ota.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)

    IR_two_state_sdfg = sdfg_to_IR(sdfg)
    print("\nIR:", IR_two_state_sdfg, "\n")
    """ expected:
        IR_two_state_sdfg = OffloadingIRNode( s1, {"in", "A"}, set(), 
            {
                OffloadingIRNode( s2, set(), {"out", "A"}, set())
            }
    )"""

    eval_IR(sdfg, IR_two_state_sdfg)
    #sdfg.view()
    sdfg.validate()   


    sdfg._recompile = True
    new_output = np.array([0.0])
    sdfg(**{"in": np.array([input]), "A":np.array([0.0]), "out": new_output})

    assert np.allclose(orig_output, new_output)

test_two_state_sdfg(10.0)


# TODO: create testcases with numerical correctness

# TODO: scalars have pass by copy right now -> if map writes to single scalar, detect and raise error or convert to array of length one and properly offload
# curently not offloaded -> incorrect -> run "replace_all_length1_arrays_with_scalars" at start, then replace back iff written to by GPU / map

# TODO: dace rep -> tests -> npbench -> polybench -> copy2d, copy1d, heat3d, ... -> lib nodes, wcr edges etc. -> use as test cases
# views, subset and wcr edges might create problems -> if they do, discuss with Yakup, might not need to handle this

# TODO: clean up -> PR branch -> write thesis!

# create showcases for all four base cases
# test and get to work
#   -> todo: make interstate edge copy method also work with controlflowblocks (?)
# create mixed test cases
# run on big sdfg
# tidy up code, move to offloading pass


# next meeting
# use npbench -> polybench and s-cases

