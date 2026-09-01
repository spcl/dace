from dace.sdfg import nodes, SDFG
from dace import dtypes
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offloading_helpers import get_sdfg_scope_dict
from typing import Type

class SchedulePhase():
    """
    Heuristic: offload toplevel nodes
    example: 

    library node -> GPU
    map -> GPU
        library node -> Seq
        map -> Seq
            library node -> Seq
            map -> Seq
                library node -> Seq
    """

    def apply(self, sdfg:SDFG, sdfg_scope_dict:dict=None, verbose=False):
        self.verbose = verbose

        if sdfg_scope_dict:
            self.sdfg_scope_dict = sdfg_scope_dict
        else:
            self.sdfg_scope_dict = get_sdfg_scope_dict(sdfg)

        self.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
        self.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)
                

    def set_toplevel_to_GPU(self, sdfg: SDFG, type:Type):
        assert type in (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)

        for state in sdfg.states():
            scope_dict = self.sdfg_scope_dict[state] if state in self.sdfg_scope_dict else None # None -> within nested SDFGs

            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    self.set_toplevel_to_GPU(node.sdfg, type)
                    continue

                if not isinstance(node, type): # filter
                    continue
                    
                if scope_dict and node in scope_dict and scope_dict[node] is None: # toplevel node -> change schedule
                    if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
                        node.map.schedule = dtypes.ScheduleType.GPU_Device
                        if self.verbose: print(f"Phase1: set map {node} to GPU schedule")

                    elif isinstance(node, nodes.LibraryNode):
                        node.schedule = dtypes.ScheduleType.GPU_Device
                        if self.verbose: print(f"Phase1: set libnode {node} to GPU schedule")
                    else:
                        assert False

                else: # within nested scope -> must not have GPU schedule
                    node.schedule = dtypes.ScheduleType.Sequential # set specifically as Default can be lowered to Cuda in the wrong places
                    if self.verbose: print(f"Phase1: set {"libnode" if isinstance(node, nodes.LibraryNode) else "map"} {node} to sequential schedule")
    