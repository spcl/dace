# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from dace import SDFG, SDFGState, data, dtypes, subsets
from dace import memlet as mm
from dace import symbolic
from dace.codegen import common
from dace.codegen.targets import cpp
from dace.codegen.targets.cpp import unparse_cr
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import symbolic_to_cpp
from dace.config import Config
from dace.dtypes import StorageType
from dace.frontend import operations
from dace.sdfg import nodes, scope_contains_scope
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import helpers

class CopyContext:

    def __init__(self, sdfg: SDFG, state: SDFGState, src_node: nodes.Node, dst_node: nodes.Node, 
                 edge: MultiConnectorEdge[mm.Memlet], gpustream_assignments: Dict[nodes.Node, Union[int, str]]):
        
        # Store the basic context as attributes
        self.sdfg = sdfg
        self.state = state
        self.src_node = src_node
        self.dst_node = dst_node
        self.edge = edge
        self.gpustream_assignments = gpustream_assignments

        memlet = edge.data

        self.copy_shape = memlet.subset.size_exact()
        if isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode):
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = self.get_accessnode_to_accessnode_copy_info()
        else:
            copy_shape = memlet.subset.size_exact()
            src_strides = dst_strides = src_expr = dst_expr = None
        
        self.copy_shape = copy_shape
        self.src_strides = src_strides
        self.dst_strides = dst_strides
        self.src_expr = src_expr
        self.dst_expr = dst_expr

    def get_storage_type(self, node: nodes.Node):
        if isinstance(node, nodes.Tasklet):
            storage_type = StorageType.Register

        elif isinstance(node, nodes.AccessNode):
            storage_type = node.desc(self.sdfg).storage

        else:
            raise NotImplementedError(
                f"Unsupported node type {type(node)} for storage type retrieval; "
                "expected AccessNode or Tasklet. Please extend this method accordingly."
            )
        
        return storage_type

    def get_assigned_gpustream(self) -> str:
        src_stream = self.gpustream_assignments.get(self.src_node)
        dst_stream = self.gpustream_assignments.get(self.dst_node)

        # 1. Catch unsupported cases
        if src_stream is None or dst_stream is None:
            raise ValueError("GPU stream assignment missing for source or destination node.")

        if src_stream != dst_stream:
            raise ValueError(
                f"Mismatch in assigned GPU streams: src_node has '{src_stream}', "
                f"dst_node has '{dst_stream}'. They must be the same."
            )
        
        # 2. Generate GPU stream expression
        
        gpustream = src_stream
        if gpustream == 'nullptr':
            raise NotImplementedError("nullptr GPU stream not supported yet.")

        gpustream_var_name_prefix = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[1]
        gpustream_expr = f"{gpustream_var_name_prefix}{gpustream}" 

        return gpustream_expr

    def get_memory_location(self) -> Tuple[str, str]:
        src_storage = self.get_storage_type(self.src_node)
        dst_storage = self.get_storage_type(self.dst_node)
        src_location = 'Device' if src_storage == dtypes.StorageType.GPU_Global else 'Host'
        dst_location = 'Device' if dst_storage == dtypes.StorageType.GPU_Global else 'Host'

        return src_location, dst_location

    def get_ctype(self) -> Any:
        sdfg = self.sdfg
        src_node, dst_node = self.src_node, self.dst_node

        if isinstance(src_node, nodes.AccessNode):
            return src_node.desc(sdfg).ctype
        
        if isinstance(dst_node, nodes.AccessNode):
            return dst_node.desc(sdfg).ctype

        raise NotImplementedError(
            f"Cannot determine ctype: neither src nor dst node is an AccessNode. "
            f"Got src_node type: {type(src_node).__name__}, dst_node type: {type(dst_node).__name__}. "
            "Please extend this case or fix the issue."
        )

    def get_accessnode_to_accessnode_copy_info(self):
        src_node, dst_node = self.src_node, self.dst_node
        sdfg = self.sdfg
        edge = self.edge
        memlet = self.edge.data
        state = self.state
        copy_shape = self.copy_shape

        if not (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)):
            raise TypeError(
                f"get_accessnode_to_accessnode_copy_info requires both source and destination "
                f"to be AccessNode instances, but got {type(src_node).__name__} and {type(dst_node).__name__}."
            )
        
        src_nodedesc = src_node.desc(sdfg)
        dst_nodedesc = dst_node.desc(sdfg)

        src_subset = memlet.get_src_subset(edge, state)
        dst_subset = memlet.get_dst_subset(edge, state)
        
        if src_subset is None:
            src_subset = subsets.Range.from_array(src_nodedesc)

        if dst_subset is None:
            dst_subset = subsets.Range.from_array(dst_nodedesc)

        src_strides = src_subset.absolute_strides(src_nodedesc.strides)
        dst_strides = dst_subset.absolute_strides(dst_nodedesc.strides)

        # Try to turn into degenerate/strided ND copies
        result = cpp.ndcopy_to_strided_copy(
            copy_shape,
            src_nodedesc.shape,
            src_strides,
            dst_nodedesc.shape,
            dst_strides,
            memlet.subset,
            src_subset,
            dst_subset,
        )
        if result is not None:
            copy_shape, src_strides, dst_strides = result
        else:
            # If other_subset is defined, reduce its dimensionality by
            # removing the "empty" dimensions (size = 1) and filter the
            # corresponding strides out
            src_strides = ([stride
                            for stride, s in zip(src_strides, src_subset.size()) if s != 1] + src_strides[len(src_subset):]
                        )  # Include tiles
            if not src_strides:
                src_strides = [1]
            dst_strides = ([stride
                            for stride, s in zip(dst_strides, dst_subset.size()) if s != 1] + dst_strides[len(dst_subset):]
                        )  # Include tiles
            if not dst_strides:
                dst_strides = [1]
            copy_shape = [s for s in copy_shape if s != 1]
            if not copy_shape:
                copy_shape = [1]

        # Extend copy shape to the largest among the data dimensions,
        # and extend other array with the appropriate strides
        if len(dst_strides) != len(copy_shape) or len(src_strides) != len(copy_shape):
            if memlet.data == src_node.data:
                copy_shape, dst_strides = cpp.reshape_strides(src_subset, src_strides, dst_strides, copy_shape)
            elif memlet.data == dst_node.data:
                copy_shape, src_strides = cpp.reshape_strides(dst_subset, dst_strides, src_strides, copy_shape)


        src_name = src_node.data
        if (src_nodedesc.transient and src_nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External)):
            ptr_name = f'__state->__{sdfg.cfg_id}_{src_name}'
        else:
            ptr_name = src_name

        if isinstance(src_nodedesc, data.Scalar) and src_nodedesc.storage in dtypes.GPU_MEMORY_STORAGES_EXPERIMENTAL_CUDACODEGEN:
            parent_nsdfg_node = state.sdfg.parent_nsdfg_node
            if parent_nsdfg_node is not None and src_name in parent_nsdfg_node.in_connectors:
                src_expr = f"&{ptr_name}"
            else:
                src_expr = ptr_name

        elif isinstance(src_nodedesc, data.Scalar):
            src_expr = f"&{ptr_name}"

        elif isinstance(src_nodedesc, data.Array):
            src_offset = cpp.cpp_offset_expr(src_nodedesc, src_subset)
            src_expr = f"{ptr_name} + {src_offset}" if src_offset != "0" else ptr_name

        else:
            raise NotImplementedError(
                f"Expected {src_name} to be either data.Scalar or data.Array, "
                f"but got {type(src_nodedesc).__name__}."
            )
            
        dst_name = dst_node.data
        if (dst_nodedesc.transient and dst_nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External)):
            ptr_name = f'__state->__{sdfg.cfg_id}_{dst_name}'
        else:
            ptr_name = dst_name

        if isinstance(dst_nodedesc, data.Scalar) and dst_nodedesc.storage in dtypes.GPU_MEMORY_STORAGES_EXPERIMENTAL_CUDACODEGEN:
            parent_nsdfg_node = state.sdfg.parent_nsdfg_node
            if parent_nsdfg_node is not None and dst_name in parent_nsdfg_node.in_connectors:
                dst_expr = f"&{ptr_name}"
            else:
                dst_expr = ptr_name

        elif isinstance(dst_nodedesc, data.Scalar):
            dst_expr = f"&{ptr_name}"

        elif isinstance(dst_nodedesc, data.Array):
            dst_offset = cpp.cpp_offset_expr(dst_nodedesc, dst_subset)
            dst_expr = f"{ptr_name} + {dst_offset}" if dst_offset != "0" else ptr_name

        else:
            raise NotImplementedError(
                f"Expected {dst_name} to be either data.Scalar or data.Array, "
                f"but got {type(dst_nodedesc).__name__}."
            )

        return copy_shape, src_strides, dst_strides, src_expr, dst_expr


class CopyStrategy(ABC):
    """Abstract base class for memory copy strategies."""

    @abstractmethod
    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Return True if this strategy can handle the given memory copy.
        """
        raise NotImplementedError('Abstract class')

    @abstractmethod
    def generate_copy(self, copy_context: CopyContext) -> str:
        """
        Generates and returns the copy code for the supported pattern.
        """
        raise NotImplementedError('Abstract class')
    

class OutOfKernelCopyStrategy(CopyStrategy):
    """
    Copy strategy for memory transfers that occur outside of kernel execution.

    This pattern often occurs when generating host-to-device copies for kernel inputs
    (since kernels cannot access host memory directly), and device-to-host copies
    to retrieve results for further processing.
    """

    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Determines whether the data movement is a host<->device memory copy.

        This function returns True if:
        - We are not currently generating kernel code
        - The copy occurs between two AccessNodes
        - The data descriptors of source and destination are not views.
        - The storage types of either src or dst is CPU_Pinned or GPU_Device
        - We do not have a CPU-to-CPU copy
        """
        # Retrieve needed information
        state = copy_context.state
        src_node, dst_node = copy_context.src_node, copy_context.dst_node

        # 1. Ensure copy is not occuring within a kernel
        scope_dict = state.scope_dict()
        deeper_node = dst_node if scope_contains_scope(scope_dict, src_node, dst_node) else src_node

        parent_map_tuple = helpers.get_parent_map(state, deeper_node)
        while parent_map_tuple is not None:
            parent_map, parent_state = parent_map_tuple
            if parent_map.map.schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
                return False
            else:
                parent_map_tuple = helpers.get_parent_map(parent_state, parent_map)

        # 2. Check whether copy is between two AccessNodes
        if not (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)):
            return False
        
        # 3. The data descriptors of source and destination are not views
        if isinstance(src_node.desc(state), data.View) or isinstance(dst_node.desc(state), data.View):
            return False

        # 4. Check that one StorageType of either src or dst is CPU_Pinned or GPU_Device
        src_storage = copy_context.get_storage_type(src_node)
        dst_storage = copy_context.get_storage_type(dst_node)
        if not (src_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned) or 
                dst_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned)):
            return False

        # 5. Check that this is not a CPU to CPU copy
        cpu_storage_types = [StorageType.CPU_Heap, StorageType.CPU_ThreadLocal, StorageType.CPU_Pinned]
        if src_storage in cpu_storage_types and dst_storage in cpu_storage_types:
            return False

        return True
    
    def generate_copy(self, copy_context: CopyContext) -> str:
        """Execute host-device copy with CUDA memory operations"""

        # Guard 
        memlet = copy_context.edge.data
        if memlet.wcr is not None:
            src_location, dst_location = copy_context.get_memory_location()
            raise NotImplementedError(f'Accumulate {src_location} to {dst_location} not implemented')

        # Based on the copy dimension, call appropiate helper function
        num_dims = len(copy_context.copy_shape)
        if num_dims == 1:
            copy_call = self._generate_1d_copy(copy_context)

        elif num_dims == 2:
            copy_call = self._generate_2d_copy(copy_context)

        else:
            # sanity check
            assert num_dims > 2, f"Expected copy shape with more than 2 dimensions, but got {num_dims}."
            copy_call = self._generate_nd_copy(copy_context)

        return copy_call
    
    def _generate_1d_copy(self, copy_context: CopyContext) -> str:
        """
        Generates a 1D memory copy between host and device using the GPU backend.

        Uses {backend}MemcpyAsync for contiguous memory. For strided memory, 
        {backend}Memcpy2DAsync is leveraged to efficiently handle the stride along one dimension.
        """
        # ----------- Retrieve relevant copy parameters --------------
        backend: str = common.get_gpu_backend()

        # Due to applicable(), src and dst node must be AccessNodes
        copy_shape, src_strides, dst_strides, src_expr, dst_expr = copy_context.get_accessnode_to_accessnode_copy_info()

        src_location, dst_location = copy_context.get_memory_location()
        is_contiguous_copy = (src_strides[-1] == 1) and (dst_strides[-1] == 1)
        ctype = copy_context.get_ctype()
        gpustream = copy_context.get_assigned_gpustream()

        # ----------------- Generate backend call --------------------

        if is_contiguous_copy:
            # Memory is linear: can use {backend}MemcpyAsync
            copysize = ' * '.join(symbolic_to_cpp(copy_shape))
            copysize += f' * sizeof({ctype})'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'
            call = f'DACE_GPU_CHECK({backend}MemcpyAsync({dst_expr}, {src_expr}, {copysize}, {kind}, {gpustream}));\n'

        else:
            # Memory is strided: use {backend}Memcpy2DAsync with dpitch/spitch
            # This allows copying a strided 1D region
            dpitch = f'{symbolic_to_cpp(dst_strides[0])} * sizeof({ctype})'
            spitch = f'{symbolic_to_cpp(src_strides[0])} * sizeof({ctype})'
            width = f'sizeof({ctype})'
            height = symbolic_to_cpp(copy_shape[0])
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        return call
    
    def _generate_2d_copy(self, copy_context: CopyContext) -> None:
        """
        Generates a 2D memory copy using {backend}Memcpy2DAsync.

        Three main cases are handled:
        - Copy between row-major stored arrays with contiguous rows.
        - Copy between column-major stored arrays with contiguous columns.
        - A special case where a 2D copy can still be represented.

        Raises:
            NotImplementedError: Raised if the source and destination strides do not match any of the handled patterns.
            Such cases indicate an unsupported 2D copy and should be examined separately.
            They can be implemented if valid, or a more descriptive error should be raised if the path should not occur.

        Note:
            {backend}Memcpy2DAsync supports strided copies along only one dimension (row or column), 
            but not both simultaneously.
        """

        # ----------- Extract relevant copy parameters --------------
        backend: str = common.get_gpu_backend()

        # Due to applicable(), src and dst node must be AccessNodes
        copy_shape, src_strides, dst_strides, src_expr, dst_expr = copy_context.get_accessnode_to_accessnode_copy_info()
        src_location, dst_location = copy_context.get_memory_location()
        ctype = copy_context.get_ctype()
        gpustream = copy_context.get_assigned_gpustream()

        # ----------------- Generate backend call if supported --------------------

        # Case: Row-major layout, rows are not strided.
        if (src_strides[1] == 1) and (dst_strides[1] == 1):
            dpitch = f'{symbolic_to_cpp(dst_strides[0])} * sizeof({ctype})'
            spitch = f'{symbolic_to_cpp(src_strides[0])} * sizeof({ctype})'
            width = f'{symbolic_to_cpp(copy_shape[1])} * sizeof({ctype})'
            height = f'{symbolic_to_cpp(copy_shape[0])}'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Case: Column-major layout, no columns are strided.
        elif (src_strides[0] == 1) and (dst_strides[0] == 1):
            dpitch = f'{symbolic_to_cpp(dst_strides[1])} * sizeof({ctype})'
            spitch = f'{symbolic_to_cpp(src_strides[1])} * sizeof({ctype})'
            width = f'{symbolic_to_cpp(copy_shape[0])} * sizeof({ctype})'
            height = f'{symbolic_to_cpp(copy_shape[1])}'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Special case
        elif (src_strides[0] / src_strides[1] == copy_shape[1] and dst_strides[0] / dst_strides[1] == copy_shape[1]):
            # Consider as an example this copy: A[0:I, 0:J, K] -> B[0:I, 0:J] with
            # copy shape [I, J], src_strides[J*K, K], dst_strides[J, 1]. This can be represented with a
            # {backend}Memcpy2DAsync call!

            dpitch = f'{symbolic_to_cpp(dst_strides[1])} * sizeof({ctype})'
            spitch = f'{symbolic_to_cpp(src_strides[1])} * sizeof({ctype})'
            width = f'sizeof({ctype})'
            height = symbolic_to_cpp(copy_shape[0] * copy_shape[1])
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        else:
            raise NotImplementedError(
                f"Unsupported 2D memory copy: shape={copy_shape}, src_strides={src_strides}, dst_strides={dst_strides}."
                "Please implement this case if it is valid, or raise a more descriptive error if this path should not be taken."
            )

        return call

    def _generate_nd_copy(self, copy_context: CopyContext) -> None:
        """
        Generates GPU code for copying N-dimensional arrays using 2D memory copies.

        Uses {backend}Memcpy2DAsync for the last two dimensions, with nested loops
        for any outer dimensions. Expects the copy to be contiguous and between
        row-major storage locations.
        """
        # ----------- Extract relevant copy parameters --------------
        backend: str = common.get_gpu_backend()

        # Due to applicable(), src and dst node must be AccessNodes
        copy_shape, src_strides, dst_strides, src_expr, dst_expr = copy_context.get_accessnode_to_accessnode_copy_info()

        src_location, dst_location = copy_context.get_memory_location()
        ctype = copy_context.get_ctype()
        gpustream = copy_context.get_assigned_gpustream()
        num_dims = len(copy_shape)

        # ----------- Guard for unsupported Pattern --------------
        if not (src_strides[-1] == 1) and (dst_strides[-1] == 1):
            src_node, dst_node = copy_context.src_node, copy_context.dst_node
            src_storage = copy_context.get_storage_type(src_node)
            dst_storage = copy_context.get_storage_type(dst_node)
            raise NotImplementedError(
                "N-dimensional GPU memory copies, that are strided or contain column-major arrays, are currently not supported.\n"
                f"  Source node: {src_node} (storage: {src_storage})\n"
                f"  Destination node: {copy_context.dst_node} (storage: {dst_storage})\n"
                f"  Source strides: {src_strides}\n"
                f"  Destination strides: {dst_strides}\n"
                f"  copy shape: {copy_shape}\n"
                )
        
        # ----------------- Generate and write backend call(s) --------------------

        call = ""
        # Write for-loop headers
        for dim in range(num_dims - 2):
            call += f"for (int __copyidx{dim} = 0; __copyidx{dim} < {copy_shape[dim]}; ++__copyidx{dim}) {{\n"

        # Write Memcopy2DAsync
        offset_src = ' + '.join(f'(__copyidx{d} * ({symbolic_to_cpp(s)}))' for d, s in enumerate(src_strides[:-2]))
        offset_dst = ' + '.join(f'(__copyidx{d} * ({symbolic_to_cpp(s)}))' for d, s in enumerate(dst_strides[:-2]))

        src = f'{src_expr} + {offset_src}'
        dst = f'{dst_expr} + {offset_dst}'

        dpitch = f'{symbolic_to_cpp(dst_strides[-2])} * sizeof({ctype})'
        spitch = f'{symbolic_to_cpp(src_strides[-2])} * sizeof({ctype})'
        width = f'{symbolic_to_cpp(copy_shape[-1])} * sizeof({ctype})'
        height = symbolic_to_cpp(copy_shape[-2])
        kind = f'{backend}Memcpy{src_location}To{dst_location}'

        # Generate call and write it
        call += f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst}, {dpitch}, {src}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Write for-loop footers
        for dim in range(num_dims - 2):
            call += "\n}"

        # Return the code
        return call
    

class SyncCollaboritveGPUCopyStrategy(CopyStrategy):
    """
    Implements (synchronous) collaborative GPU copy operations. 

    This strategy generates the appropriate code for copies performed 
    inside GPU kernels, where multiple threads cooperate to move data 
    between gpu memory spaces (e.g., global to shared memory). 
    """

    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Checks if the copy is eligible for a collaborative GPU-to-GPU copy.

        Conditions:
        1. The copy is between two AccessNodes
        2. The copy is between GPU memory StorageTypes (shared or global).
        3. The innermost non-sequential map is a GPU_Device-scheduled map i.e. 
           the copy occurs within a kernel but is not within a GPU_ThreadBlock map.
        """
        # --- Condition 1: src and dst are AccessNodes ---
        src_node, dst_node = copy_context.src_node, copy_context.dst_node
        if not (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)):
            return False
        
        # --- Condition 2: GPU to GPU memory transfer ---
        src_storage, dst_storage = copy_context.get_storage_type(src_node), copy_context.get_storage_type(dst_node)
        gpu_storages = {dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared}

        if not (src_storage in gpu_storages and dst_storage in gpu_storages):
            return False
        
        # --- Condition 3: Next non-sequential Map is a GPU_Device Map ---
        next_nonseq_parent_map = self._next_non_seq_parent_map(copy_context)
        if next_nonseq_parent_map is None:
            return False
        else:
            return next_nonseq_parent_map.map.schedule == dtypes.ScheduleType.GPU_Device
    
    def generate_copy(self, copy_context: CopyContext, kernel_dimensions_maps: Dict[nodes.MapEntry, Tuple[List, List]]) -> str:
        """
        Generates a GPU copy call as a string using DaCe's runtime CUDA copy functions.

        The function determines the appropriate templated copy function from
        `dace/libraries/runtime/include/dace/cuda/copy.cuh` and constructs
        the call string with the necessary arguments, including kernel block
        dimensions and optional accumulation/reduction information.

        Parameters
        ----------
        copy_context : CopyContext
            Helper object containing information about the copy.

        kernel_dimensions_maps : Dict[nodes.MapEntry, Tuple[List, List]]
            Kernel map (GPU_Devie scheduled map) entry nodes to (grid_dims, block_dims); 
            block_dims needed in templating.

        Returns
        -------
        str
            The GPU copy call in C++ as a string.

        Notes
        -----
        - The kernel block size could be derived, but since this function is typically called
          from `ExperimentalCUDACodeGen`, it is provided as input to avoid recomputation.
        - The template functions use a parameter called 'is_async', which is set to True here
          because `ExperimentalCUDACodeGen` inserts "__syncthreads()" explicitly in tasklets.
        """
        # ----------- Retrieve relevant copy information --------------

        # Due to applicable(), src and dst node must be AccessNodes
        copy_shape, src_strides, dst_strides, src_expr, dst_expr = copy_context.get_accessnode_to_accessnode_copy_info()
        sdfg = copy_context.sdfg
        dtype = copy_context.src_node.desc(sdfg).dtype
        ctype = dtype.ctype

        # Get copy function name (defined in runtime library)
        num_dims = len(copy_shape)
        src_node, dst_node = copy_context.src_node, copy_context.dst_node
        src_storage, dst_storage = copy_context.get_storage_type(src_node), copy_context.get_storage_type(dst_node)
        src_storage_name = self._get_storagename(src_storage)
        dst_storage_name = self._get_storagename(dst_storage)
        function_name = f"dace::{src_storage_name}To{dst_storage_name}{num_dims}D"

        # Extract WCR info (accumulation template + optional custom reduction)
        accum, custom_reduction = self._get_accumulation_info(copy_context)
        custom_reduction = [custom_reduction] if custom_reduction else []

        # Get parent kernel block dimensions (guaranteed GPU_Device) and sync flag
        parent_kernel = self._next_non_seq_parent_map(copy_context)
        block_dims = ", ".join(symbolic_to_cpp(kernel_dimensions_maps[parent_kernel][1]))
        synchronized = "true" # Legacy 'is_async'; sync barriers handled by passes (see docstring)

        # ------------------------- Generate copy call ----------------------------

        if any(symbolic.issymbolic(s, copy_context.sdfg.constants) for s in copy_shape):
            args_list = ([src_expr] + src_strides + [dst_expr] + custom_reduction + dst_strides + copy_shape)
            args = ", ".join(symbolic_to_cpp(args_list))
            call = f"{function_name}Dynamic<{ctype}, {block_dims}, {synchronized}>{accum}({args});"

        elif function_name == "dace::SharedToGlobal1D":
            copy_size = ', '.join(symbolic_to_cpp(copy_shape))
            accum = accum or '::Copy'
            args_list = ([src_expr] + src_strides + [dst_expr] + dst_strides + custom_reduction)
            args = ", ".join(symbolic_to_cpp(args_list))
            call = f"{function_name}<{ctype}, {block_dims}, {copy_size}, {synchronized}>{accum}({args});"

        else:
            copy_size = ', '.join(symbolic_to_cpp(copy_shape))
            args_list = ([src_expr] + src_strides + [dst_expr] + custom_reduction)
            args = ", ".join(symbolic_to_cpp(args_list))
            dst_strides_unpacked = ", ".join(symbolic_to_cpp(dst_strides))
            call = f"{function_name}<{ctype}, {block_dims}, {copy_size}, {dst_strides_unpacked}, {synchronized}>{accum}({args});"
        
        return call

    def _get_accumulation_info(self, copy_context: CopyContext) -> Tuple[str, str]:
        """
        Extracts write-conflict resolution (WCR) information from the copy context
        and returns the accumulation/reduction template components needed for the 
        final templated function call in `generate_copy()`.

        This method processes WCR information from the memlet and generates the 
        appropriate C++ template strings for both predefined and custom reductions.

        Parameters
        ----------
        copy_context : CopyContext
            Copy context containing the copy operation details, including
            the memlet with WCR information.

        Returns
        -------
        Tuple[str, str]
            A tuple containing:
            - accum : str  
            Template accumulation string for the function call. Empty string if no WCR,  
            `"::template Accum<ReductionType>"` for predefined reductions, or `"::template Accum"` for custom reductions.  
            - custom_reduction : str  
            C++ formatted custom reduction code string. Empty string for no WCR or predefined reductions,  
            unparsed custom reduction code for custom reductions.
        """
        sdfg = copy_context.sdfg
        dtype = copy_context.src_node.desc(sdfg).dtype
        memlet = copy_context.edge.data
        wcr = memlet.wcr
        reduction_type = operations.detect_reduction_type(wcr)

        if wcr is None:
            accum, custom_reduction = "", ""

        elif reduction_type != dtypes.ReductionType.Custom:
            # Use predefined reduction
            reduction_type_str = str(reduction_type).split(".")[-1]  # e.g., "Sum"
            accum = f"::template Accum<dace::ReductionType::{reduction_type_str}>"
            custom_reduction = ""

        else:
            accum = "::template Accum" 
            custom_reduction = unparse_cr(sdfg, wcr, dtype)

        return accum, custom_reduction

    def _get_storagename(self, storage: dtypes.StorageType):
        """
        Returns a string containing the name of the storage location.

        Example: dtypes.StorageType.GPU_Shared will return "Shared".
        """
        storage_name = str(storage)
        return storage_name[storage_name.rindex('_') + 1:]
    
    def _next_non_seq_parent_map(self, copy_context: CopyContext) -> Optional[nodes.MapEntry]:
        """
        Traverse up the parent map chain from the deeper of src_node or dst_node 
        in `copy_context` and return the first parent MapEntry whose schedule 
        is not sequential.

        Parameters
        ----------
        copy_context : CopyContext
            Context information about the memory copy.

        Returns
        -------
        Optional[nodes.MapEntry]
            The first non-sequential parent MapEntry encountered, or None if no 
            such parent exists.
        """
        src_node, dst_node = copy_context.src_node, copy_context.dst_node
        state = copy_context.state
        scope_dict = state.scope_dict()

        # Determine which node (src or dst) is in the deeper scope
        deeper_node = dst_node if scope_contains_scope(scope_dict, src_node, dst_node) else src_node
        current_node = deeper_node
        while (current_node is None or not isinstance(current_node, nodes.MapEntry)
               or current_node.map.schedule == dtypes.ScheduleType.Sequential):
            parent = helpers.get_parent_map(state, current_node)
            if parent is None:
                current_node = None
                break
            current_node, state = parent

        return current_node