import dace
from dace import dtypes, properties
from dace.sdfg import nodes, SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

from typing import Any, Dict, Tuple, List, Optional, Set, Type, Union
import numpy as np

@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """
    Docstring for OffloadToAccelerator
    """
    
    CATEGORY: str = 'Offload To Accelerator'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False
    
    #def depends_on(self) -> Set[Union[Type['Pass'], 'Pass']]:
    #    return set()
    
    #def report(self, pass_retval: Any) -> Optional[str]:
    #    """
    #    Returns a user-readable string report based on the results of this pass.
    #
    #    :param pass_retval: The return value from applying this pass.
    #    :return: A string with the user-readable report, or None if nothing to report.
    #    """
    #    return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """

        self.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
        self.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)

        self.find_and_insert_copies(sdfg)
        print("program is ready")


    def set_toplevel_to_GPU(self, sdfg: SDFG, type:Type):
        assert type in (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)

        for state in sdfg.states():
            scope_dict = state.scope_dict() 

            for node in state.nodes():
                if not isinstance(node, type): # filter
                    continue
                    
                if scope_dict[node] is None: # toplevel node -> offload
                    self.set_schedule(node)

                else: # within nested scope -> must not offload (defensive check)
                    if self.has_GPU_schedule(node):
                        raise RuntimeError("Invalid SDFG for OffloadToAccelerator pass." \
                        "All maps must have default or CPU schedule before pass." \
                        f"Node {node} has schedule type {self.get_schedule(node)}" )

    
    def find_and_insert_copies(self, sdfg):
        gpu_access_nodes = set()

        for state in sdfg.states():
            for node in state.nodes():
                if self.is_schedule_node(node) and self.has_GPU_schedule(node): # map entry, exit or library node

                    if isinstance(node, nodes.MapEntry):
                        #print(f"checking map entry node {node}")
                        for input in self.get_all_input_nodes(state, node):
                            self._insert_copies(sdfg, state, input, node, gpu_access_nodes)

                    elif isinstance(node, nodes.MapExit):
                        #print(f"checking map exit node {node}")
                        for output in self.get_all_output_nodes(state, node):
                            self._insert_copies(sdfg, state, output, node, gpu_access_nodes)
                        
                    elif isinstance(node, nodes.LibraryNode):
                        #print(f"checking library node {node}")
                        neighbours = self.get_all_input_nodes(state, node)
                        neighbours |= self.get_all_output_nodes(state, node)
                        for neighbour in neighbours:
                            self._insert_copies(sdfg, state, neighbour, node, gpu_access_nodes)

                    else:
                        raise RuntimeError(f"unknown schedule node {node} in find_and_insert_copies")


    def _insert_copies(self, sdfg:dace.SDFG, state:dace.SDFGState, node:nodes.Node, prev:nodes.Node, gpu_access_nodes:set):
        # if wrong node or already visited, return
        if not isinstance(node, nodes.AccessNode): return
        if node in gpu_access_nodes: return
        
        # otherwise, mark as data needed on GPU
        gpu_access_nodes.add(node)
        
        # get neighbours = input + output - prev
        inputs = self.get_all_input_nodes(state, node)
        neighbours = inputs | self.get_all_output_nodes(state, node) 
        if prev in neighbours: neighbours.remove(prev) # prevent cylical recursion

        # check all neighbours and insert copies where needed
        #print(f"checking node {node} with neighbours {neighbours}\n")
        for neighbour in neighbours:

            if self.is_schedule_node(neighbour):
                # if neighbour is schedule node it must have GPU schedule, no action needed
                if not self.has_GPU_schedule(neighbour):
                    raise RuntimeError(f"Node {node} is expected to be scheduled for GPU but isn't.")
            
            elif isinstance(neighbour, nodes.AccessNode):
                # if the neighbouring access node is already marked as GPU, no action needed
                # otherwise, assume current node's data lives on GPU and recurse on the neighbour
                if not neighbour in gpu_access_nodes:
                    self._insert_copies(sdfg, state, neighbour, node, gpu_access_nodes)

            else:
                # different node, i.e. control flow -> data needed on CPU: insert copy
                raise NotImplemented(f"Copy needed between {node} and {neighbour}")

        # if its a non-transient node, a copy from/to userspace is also needed
        if not node.desc(sdfg)._transient:
      
            if not inputs and node.has_reads(state):
                print(f"copy input node {node} to GPU at beginning of the program")
                # raise NotImplemented
            elif node.has_writes(state): # can also be a result if it isn't returned (pass by pointer), so checking for no outputs doesn't work
                print(f"copy result node {node} back from GPU at the end of the program")
                # raise NotImplemented  
        
            # Q: Heat3d: array B is partially overwritten but never read from 
            #    do we still need an initial copy of B to the GPU or can be do a partial copy back?
            # A?: all arrays need to be copied back because of sideeffects?


    def is_schedule_node(self, node):
        return isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit) or isinstance(node, nodes.LibraryNode)

    def get_schedule(self, node):
        if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
            return node.map.schedule
        elif isinstance(node, nodes.LibraryNode):
            return node.schedule
        else:
            assert False
        
    def set_schedule(self, node):
        if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
            node.map.schedule = dtypes.ScheduleType.GPU_Device
        elif isinstance(node, nodes.LibraryNode):
            node.schedule = dtypes.ScheduleType.GPU_Device
        else:
            assert False

    def has_GPU_schedule(self, node):
        GPU_SCHEDULES = (
            dtypes.ScheduleType.GPU_Device,
            dtypes.ScheduleType.GPU_Default
        )
        return self.get_schedule(node) in GPU_SCHEDULES

    def get_all_input_nodes(self, state, node, type=None):
        input_nodes = set()
        for e in state.in_edges(node):
            root_src = state.memlet_tree(e).root().edge.src
            if not type or isinstance(root_src, type):
                input_nodes.add(root_src)
        return input_nodes

    def get_all_output_nodes(self, state, node, type=None):
        output_nodes = set()
        for e in state.out_edges(node):
            root_dst = state.memlet_tree(e).root().edge.dst
            if not type or isinstance(root_dst, type):
                output_nodes.add(root_dst)
        return output_nodes
        


"""
############ TESTING ##############
from dace.transformation.interstate import LoopToMap
import dace.memlet as mm


def handmade_map():
    sdfg = dace.SDFG("simple")
    sdfg.add_array("A", [16], dace.float32)
    sdfg.add_array("B", [16], dace.float32)
    state = sdfg.add_state()

    A = state.add_read("A")       # AccessNode
    B = state.add_write("B")      # AccessNode
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, "b = a * 2")  # double nodes

    entry, exit = state.add_map("m", dict(i="0:16")) # create entry & exit with shared map object

    # add entry & exit connectors
    entry.add_in_connector("IN_A")
    entry.add_out_connector("OUT_A")
    exit.add_in_connector("IN_B")
    exit.add_out_connector("OUT_B")

    # add entry & exit edges
    state.add_edge(A, None, entry, "IN_A", dace.memlet.Memlet("A[i]")) # src, src_conn, dst, dst_conn, memlet
    state.add_edge(entry, "OUT_A", tasklet, "a", dace.memlet.Memlet("A[i]"))
    state.add_edge(tasklet, "b", exit, "IN_B", dace.memlet.Memlet("B[i]"))
    state.add_edge(exit, "OUT_B", B, None, dace.memlet.Memlet("B[i]"))

    return sdfg

@dace.program
def matadd(A, B, C):
    C[:] = A + B

@dace.program
def nested_loop(A, B):
    C = np.empty((A.shape[0], B.shape[0]), dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            C[i, j] = A[i] * B[j] + B[j] - A[i+1]
    return C

@dace.program
def nested_loop2(A, B):
    C = np.empty((A.shape[0]), dtype=A.dtype)
    for i in range(A.shape[0]):
        C[i] = A[i] * B[i]
    return C

@dace.program
def gemm(A, B, C):
    C[:] = A @ B  # BLAS GEMM library node


def dual_input():
    sdfg = dace.SDFG("two_inputs")
    sdfg.add_array("A", [10], dace.float32)
    sdfg.add_array("B", [10], dace.float32)
    sdfg.add_array("C", [10], dace.float32)
    state = sdfg.add_state()

    A = state.add_read("A")
    B = state.add_read("B")
    C = state.add_write("C")  # AccessNode with two inputs

    t1 = state.add_tasklet("t1", {"a"}, {"c"}, "c = a")
    t2 = state.add_tasklet("t2", {"b"}, {"c"}, "c = b")

    state.add_edge(A, None, t1, "a", mm.Memlet("A[0:10]"))
    state.add_edge(B, None, t2, "b", mm.Memlet("B[0:10]"))

    # Two incoming edges to C (using WCR to combine)
    state.add_edge(t1, "c", C, None, mm.Memlet("C[0:10]", wcr="lambda x, y: x + y"))
    state.add_edge(t2, "c", C, None, mm.Memlet("C[0:10]", wcr="lambda x, y: x + y"))

    return sdfg


@dace.program
def two_writers(A, B, C):
    # Two independent writes into C (will require WCR when lowered)
    for i in range(len(C)):
        C[i] = A[i]
        C[i] = C[i] + B[i]



A = np.random.rand(8, 4).astype(np.float32)
B = np.random.rand(4, 6).astype(np.float32)
C = np.zeros((8, 6), dtype=np.float32)
sdfg = gemm.to_sdfg(A, B, C)

A = np.arange(4, dtype=np.float32)
B = np.arange(4, dtype=np.float32)
sdfg = nested_loop2.to_sdfg(A, B)
sdfg.apply_transformations_repeated(LoopToMap)

A = np.arange(4, dtype=np.float32)
B = np.arange(2, dtype=np.float32)
sdfg = nested_loop.to_sdfg(A, B)
sdfg.apply_transformations_repeated(LoopToMap)

A = np.array([1.0], dtype=np.float32)
B = np.array([2.0], dtype=np.float32)
C = np.zeros_like(A)
sdfg : SDFG = two_writers.to_sdfg(A, B, C)
sdfg.apply_transformations_repeated(LoopToMap)

A = np.random.rand(64, 64).astype(np.float32)
B = np.random.rand(64, 64).astype(np.float32)
C = np.zeros_like(A)
sdfg = matadd.to_sdfg(A, B, C)



sdfg = handmade_map()
p = OffloadToAccelerator()
p.apply_pass(sdfg, {})

sdfg.view()

"""


