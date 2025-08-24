# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from dace import SDFG, SDFGState, data, dtypes, subsets
from dace import memlet as mm
from dace.codegen import common
from dace.codegen.targets import cpp
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import symbolic_to_cpp
from dace.config import Config
from dace.dtypes import StorageType
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

    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Determines whether the data movement is a host<->device memory copy.

        This function returns True if:
        - We are not currently generating kernel code
        - The copy occurs between two AccessNodes
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

        # 2. Check whether copy is between to AccessNodes
        if not (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)):
            return False

        # 3. Check that one StorageType of either src or dst is CPU_Pinned or GPU_Device
        src_storage = copy_context.get_storage_type(src_node)
        dst_storage = copy_context.get_storage_type(dst_node)
        if not (src_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned) or 
                dst_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned)):
            return False

        # 4. Check that this is not a CPU to CPU copy
        cpu_storage_types = [StorageType.CPU_Heap, StorageType.CPU_ThreadLocal, StorageType.CPU_Pinned]
        if src_storage in cpu_storage_types and dst_storage in cpu_storage_types:
            return False
        

        if isinstance(src_node.desc(state), data.View) or isinstance(dst_node.desc(state), data.View):
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
        Emits code for a 1D memory copy between host and device using GPU backend.
        Uses {backend}MemcpyAsync for contiguous memory and uses {backend}Memcpy2DAsync
        for strided memory copies.
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
        """Generates code for a 2D copy, falling back to 1D flattening if applicable."""

        # ----------- Extract relevant copy parameters --------------
        backend: str = common.get_gpu_backend()

        # Due to applicable(), src and dst node must be AccessNodes
        copy_shape, src_strides, dst_strides, src_expr, dst_expr = copy_context.get_accessnode_to_accessnode_copy_info()

        src_location, dst_location = copy_context.get_memory_location()
        is_contiguous_copy = (src_strides[-1] == 1) and (dst_strides[-1] == 1)
        ctype = copy_context.get_ctype()
        gpustream = copy_context.get_assigned_gpustream()

        # ----------------- Generate backend call if supported --------------------

        if is_contiguous_copy:
            dpitch = f'{symbolic_to_cpp(dst_strides[0])} * sizeof({ctype})'
            spitch = f'{symbolic_to_cpp(src_strides[0])} * sizeof({ctype})'
            width = f'{symbolic_to_cpp(copy_shape[1])} * sizeof({ctype})'
            height = f'{symbolic_to_cpp(copy_shape[0])}'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        elif src_strides[-1] != 1 or dst_strides[-1] != 1:
            # TODO: Checks this, I am not sure but the old code and its description
            # seems to be more complicated here than necessary..
            # But worth to mention: we essentially perform flattening

            # NOTE: Special case of continuous copy
            # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
            # with copy shape [I, J] and strides [J*K, K], [J, 1]

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
        # TODO: comment
        # ----------- Extract relevant copy parameters --------------
        backend: str = common.get_gpu_backend()

        # Due to applicable(), src and dst node must be AccessNodes
        copy_shape, src_strides, dst_strides, src_expr, dst_expr = copy_context.get_accessnode_to_accessnode_copy_info()

        src_location, dst_location = copy_context.get_memory_location()
        ctype = copy_context.get_ctype()
        gpustream = copy_context.get_assigned_gpustream()
        num_dims = len(copy_shape)

        # ----------- Guard for unsupported Pattern --------------
        is_contiguous_copy = (src_strides[-1] == 1) and (dst_strides[-1] == 1)
        if not is_contiguous_copy:
            src_node, dst_node = copy_context.src_node, copy_context.dst_node
            src_storage = copy_context.get_storage_type(src_node)
            dst_storage = copy_context.get_storage_type(dst_node)
            raise NotImplementedError(
                "Strided GPU memory copies for N-dimensional arrays are not currently supported.\n"
                f"  Source node: {src_node} (storage: {src_storage})\n"
                f"  Destination node: {copy_context.dst_node} (storage: {dst_storage})\n"
                f"  Source strides: {src_strides}\n"
                f"  Destination strides: {dst_strides}\n")
        
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