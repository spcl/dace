
from dace import dtypes
from dace.sdfg import nodes, SDFG
from dace.transformation.passes.vectorization.length_one_array_scalar_conversion import (
    ConvertLengthOneArraysToScalars,
    ConvertScalarsToLengthOneArrays,
)
import dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offloading_helpers as helpers

class SingleElementValuePhase():

    def apply(self, sdfg:SDFG, exceptions:set=None, verbose=False):
        self.verbose = verbose
        self.exceptions = exceptions if exceptions is not None else set() # results passed back by values as long as track_hybrid_states is not None
        return self.change_single_element_data_containers(sdfg)


    def change_single_element_data_containers(self, sdfg:SDFG):

        all_scalars : set[str]= {data_name for data_name in sdfg.arrays if helpers.is_scalar(data_name, sdfg)}
        all_len1arrays : set[str]= {data_name for data_name in sdfg.arrays if helpers.is_length1_array(data_name, sdfg)} 

        gpu_written = self.get_gpu_written_data(sdfg)
        
        to_len1_arrays = all_scalars & gpu_written - self.exceptions # don't apply double - shouldn't be an issue for arrays
        to_scalars = all_len1arrays - gpu_written - self.exceptions # but is an issue for non-transient scalars, because they add a copy state that still has the original
        to_scalars = {name for name in to_scalars if not name.startswith("__return")} # __return must always be by reference in case CPU/GPU reads results back
     
        if to_len1_arrays:
            if self.verbose: print(f"Phase3: to_len1_arrays:\n{to_len1_arrays}\n")
            ConvertScalarsToLengthOneArrays(
                recursive=True,
                preserve_abi=True,
                filter=to_len1_arrays,
            ).apply_pass(sdfg, {})

            for data_name in to_len1_arrays: # prevents frequent copies within busy loops
                sdfg.arrays[data_name].lifetime = dtypes.AllocationLifetime.SDFG
        
        if to_scalars:
            if self.verbose: print(f"Phase3: to_scalars:\n{to_scalars}\n\n")
            ConvertLengthOneArraysToScalars(
                recursive=True,
                preserve_abi=True,
                filter=to_scalars,
            ).apply_pass(sdfg, {})

        return to_scalars | to_len1_arrays

    
    def get_gpu_written_data(self, sdfg:SDFG):
        gpu_written = set()
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and helpers.has_GPU_schedule(node):
                    gpu_written |= helpers.get_data_used_by_outgoing_access_nodes(sdfg, state, node, include_scalars=True)
        
        return gpu_written