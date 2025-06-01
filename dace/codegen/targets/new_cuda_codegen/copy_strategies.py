from abc import ABC, abstractmethod
from typing import Tuple

from dace import symbolic
from dace import Memlet, dtypes
from dace.dtypes import StorageType
from dace.codegen.targets.new_cuda_codegen.experimental_cuda import ExperimentalCUDACodeGen, CUDAStreamManager, product



from dace.codegen.prettycode import CodeIOStream
from dace.sdfg import SDFG, nodes
from dace.sdfg.nodes import Node
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

from dace.codegen.targets.cpp import memlet_copy_to_absolute_strides


# TODO: Adapt documentation if src strides is None!
class CopyContext:
    """
    Stores and derives all information required for memory copy operations on GPUs.

    This class exists because memory copy logic often requires a large amount of context,
    including node references, expressions, layout, and backend details. Handling all this
    ad hoc makes the code harder to follow and maintain.

    CopyContext centralizes this information and provides helper functions to clarify
    what values are needed for code generation and why. This improves readability,
    simplifies copy emission logic, and makes future extensions easier.
    """
    def __init__(self, codegen: ExperimentalCUDACodeGen, cuda_stream_manager: CUDAStreamManager, state_id: int,
                 src_node: Node, dst_node: Node, edge: Tuple[Node, str, Node, str, Memlet], sdfg: SDFG, 
                 cfg: ControlFlowRegion, dfg: StateSubgraphView, callsite_stream: CodeIOStream):
        
        # Store general context information for the copy operation, such as:
        # - which code generator is responsible,
        # - which edge and SDFG/state context related to the copy,
        # - and where the generated code is written (callsite stream).
        self.codegen = codegen
        self.state_id = state_id
        self.src_node = src_node
        self.dst_node = dst_node
        self.edge = edge
        self.sdfg = sdfg
        self.cfg = cfg
        self.dfg = dfg
        self.callsite_stream = callsite_stream
        
        # Additional information frequently needed
        self.backend = codegen.backend
        self.state_dfg = cfg.state(state_id)
        self.cudastream = cuda_stream_manager.get_stream_edge(src_node, dst_node)
        self.src_storage = self.get_storage_type(src_node)
        self.dst_storage = self.get_storage_type(dst_node)


        if isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode):
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = memlet_copy_to_absolute_strides(
                                    codegen._dispatcher, sdfg, self.state_dfg, edge, src_node, dst_node, codegen._cpu_codegen._packed_types)
        else:
            _, _, _, _, memlet = edge
            copy_shape = [symbolic.overapproximate(s) for s in memlet.subset.bounding_box_size()]
            
            # if src and dst node are not AccessNodes, these are undefined 
            src_strides = dst_strides = src_expr = dst_expr = None
        

        self.copy_shape = copy_shape
        self.src_strides = src_strides
        self.dst_strides = dst_strides
        self.src_expr = src_expr
        self.dst_expr = dst_expr

        self.num_dims = len(copy_shape)

    def get_storage_type(self, node: Node):
            
        if isinstance(node, nodes.Tasklet):
            storage_type = StorageType.Register
        else:
            storage_type = node.desc(self.sdfg).storage
        
        return storage_type
    
    def get_copy_call_parameters(self) -> Tuple[str, str, str, str, str, str, any]:
        """
        Returns all essential parameters required to emit a backend memory copy call.

        This method determines both structural and backend-specific information 
        needed to perform a memory copy, including memory locations, pointer 
        expressions, and data types. In cases where either the source or 
        destination is not a data access node, pointer expressions may be unavailable.

        Returns
        -------
        Tuple[str, Optional[str], Optional[str], str, str, str, any]
            A tuple containing:
            - backend (str): Name of the backend used (e.g., 'cuda', 'hip').
            - src_expr (Optional[str]): Source pointer expression, or None if unavailable.
            - dst_expr (Optional[str]): Destination pointer expression, or None if unavailable.
            - src_location (str): Memory location of the source ('Host' or 'Device').
            - dst_location (str): Memory location of the destination ('Host' or 'Device').
            - cudastream (str): Backend-specific stream identifier.
            - ctype (any): The C type of the data being copied.
        """
        src_location = 'Device' if self.src_storage == dtypes.StorageType.GPU_Global else 'Host'
        dst_location = 'Device' if self.dst_storage == dtypes.StorageType.GPU_Global else 'Host'

        # Use the destination data type 
        ctype = self.dst_node.desc(self.sdfg).ctype

        # NOTE: I implicitly assume it is the same dtype as of the src.
        assert ctype == self.src_node.desc(self.sdfg).dtype.ctype, \
            "Source and destination data types must match for the memory copy."

        return self.backend, self.src_expr, self.dst_expr, src_location, dst_location, self.cudastream, ctype
    
    def get_transfer_layout(self) -> Tuple[list, list, list]:
        """
        Returns layout information required for emitting a memory copy.

        Returns
        -------
        copy_shape : List
            The size (extent) of each dimension to be copied.
            Singleton dimensions (i.e., dimensions of size 1) are omitted.
            Example: [J, K, 1] becomes [J, K]
        src_strides : List or None
            Stride values of the source expression, per dimension if
            source and destination are of type AccessNode, else None.
        dst_strides : List or None
            Stride values of the destination expression, per dimension if
            source and destination are of type AccessNode, else None.
        """
        return self.copy_shape, self.src_strides, self.dst_strides

    def get_write_context(self) -> Tuple[CodeIOStream, ControlFlowRegion, int, Node, Node]:
        """
        Returns all context required to emit code into the callsite stream with proper SDFG annotations.

        Returns
        -------
        callsite_stream : CodeIOStream
            The output stream where backend code is written.
        cfg : ControlFlowRegion
            The control flow region containing the current state.
        state_id : int
            The ID of the SDFG state being generated.
        src_node : Node
            The source node involved in the copy.
        dst_node : Node
            The destination node involved in the copy.
        """
        return self.callsite_stream, self.cfg, self.state_id, self.src_node, self.dst_node

    def is_contiguous_copy(self) -> bool:
        """
        Returns True if the memory copy is contiguous in the last dimension
        for both source and destination.
        """
        return (self.src_strides[-1] == 1) and (self.dst_strides[-1] == 1)


    

class CopyStrategy(ABC):

    @abstractmethod
    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Return True if this strategy can handle the given memory copy.
        """
        raise NotImplementedError('Abstract class')

    @abstractmethod
    def generate_copy(self, copy_context: CopyContext) -> None:
        """
        Generates the copy code for the supported pattern.
        """
        raise NotImplementedError('Abstract class')
    

class OutOfKernelCopyStrategy(CopyStrategy):
          
    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Determines whether the data movement is a host<->device memory copy.

        This function returns True if:
        - We are not currently generating kernel code
        - The copy occurs between two AccessNodes
        - The storage types involve a CPU and a GPU (but not CPU-to-CPU or GPU-to-GPU)

        This check is used to detect and handle transfers between host and device memory spaces.
        """
        
        # TODO: I don't understand why all of these conditions are needed, look into it

        cpu_storage_types = [StorageType.CPU_Heap, StorageType.CPU_ThreadLocal, StorageType.CPU_Pinned]
        not_in_kernel_code = not ExperimentalCUDACodeGen._in_kernel_code

        is_between_access_nodes = (
            isinstance(copy_context.src_node, nodes.AccessNode) and
            isinstance(copy_context.dst_node, nodes.AccessNode)
        )
        

        involves_gpu_or_pinned = (
            copy_context.src_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned) or
            copy_context.dst_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned)
        )

        is_not_cpu_to_cpu = not (
            copy_context.src_storage in cpu_storage_types and
            copy_context.dst_storage in cpu_storage_types
        )

        is_gpu_host_copy = (
            not_in_kernel_code and
            is_between_access_nodes and
            involves_gpu_or_pinned and
            is_not_cpu_to_cpu
        )

        return is_gpu_host_copy
    
    def generate_copy(self, copy_context: CopyContext) -> None:
        """Execute host-device copy with CUDA memory operations"""

        
        num_dims = copy_context.num_dims

        if num_dims == 1:
            self._generate_1d_copy(copy_context)
        elif num_dims == 2:
            self._generate_2d_copy(copy_context)
        elif num_dims > 2:
            self._generate_nd_copy(copy_context)
        else: # num_dims = 0
            raise NotImplementedError(
                f"ExternalCudaCopyStrategy does not support memory copies with {num_dims} dimensions "
                f"(copy shape: {copy_context.copy_shape}). "
            )
            
    def _generate_1d_copy(self, copy_context: CopyContext) -> None:
        """
        Emits code for a 1D memory copy between host and device using GPU backend.
        Uses {backend}MemcpyAsync for contiguous memory and uses {backend}Memcpy2DAsync
        for strided memory copies.
        """
        
        # ----------- Extract relevant copy parameters --------------
        copy_shape, src_strides, dst_strides= copy_context.get_transfer_layout()

        backend, src_expr, dst_expr, src_location, dst_location, cudastream, ctype = \
                copy_context.get_copy_call_parameters()

        # ----------------- Generate backend call --------------------
        if copy_context.is_contiguous_copy():
            # Memory is linear: can use {backend}MemcpyAsync
            num_bytes = f'{product(copy_shape)} * sizeof({ctype})'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}MemcpyAsync({dst_expr}, {src_expr}, {num_bytes}, {kind}, {cudastream}));\n'
            
        else:
            # Memory is strided: use {backend}Memcpy2DAsync with dpitch/spitch
            # This allows copying a strided 1D region 
            dpitch = f'{dst_strides[0]} * sizeof({ctype})'
            spitch = f'{src_strides[0]} * sizeof({ctype})'
            width  = f'sizeof({ctype})'
            height = copy_shape[0]
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'

        # ----------------- Write copy call to code stream --------------------
        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()
        callsite_stream.write(call, cfg, state_id, [src_node, dst_node])

    def _generate_2d_copy(self, copy_context: CopyContext) -> None:
        """Generates code for a 2D copy, falling back to 1D flattening if applicable."""

        # ----------- Extract relevant copy parameters --------------
        copy_shape, src_strides, dst_strides= copy_context.get_transfer_layout()

        backend, src_expr, dst_expr, src_location, dst_location, cudastream, ctype = \
                copy_context.get_copy_call_parameters()


        # ----------------- Generate backend call if supported --------------------

        if copy_context.is_contiguous_copy():
            dpitch = f'{dst_strides[0]} * sizeof({ctype})'
            spitch = f'{src_strides[0]} * sizeof({ctype})'
            width = f'{copy_shape[1]} * sizeof({ctype})'
            height = f'{copy_shape[0]}'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'
            
        elif src_strides[-1] == 1 or dst_strides[-1] == 1: 
            # TODO: Checks this, I am not sure but the old code and its description
            # seems to be more complicated here than necessary.. 
            # But worth to mention: we essentiall flatten 

            # NOTE: Special case of continuous copy
            # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
            # with copy shape [I, J] and strides [J*K, K], [J, 1]

            dpitch = f'{dst_strides[1]} * sizeof({ctype})'
            spitch = f'{src_strides[1]} * sizeof({ctype})'
            width  = f'sizeof({ctype})'
            height = copy_shape[0] * copy_shape[1]
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'
        
        else:
            raise NotImplementedError('2D copy only supported with one stride')
        

        # ----------------- Write copy call to code stream --------------------
        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()
        callsite_stream.write(call, cfg, state_id, [src_node, dst_node])

    def _generate_nd_copy(self, copy_context: CopyContext) -> None:

        # ----------- Guard for unsupported Pattern --------------
        if not copy_context.is_contiguous_copy():
            raise NotImplementedError(
                "Strided GPU memory copies for N-dimensional arrays are not currently supported.\n"
                f"  Source node: {copy_context.src_node} (storage: {copy_context.src_storage})\n"
                f"  Destination node: {copy_context.dst_node} (storage: {copy_context.dst_storage})\n"
                f"  Source strides: {copy_context.src_strides}\n"
                f"  Destination strides: {copy_context.dst_strides}\n"
            )
        
        # ----------- Extract relevant copy parameters --------------
        copy_shape, src_strides, dst_strides= copy_context.get_transfer_layout()

        backend, src_expr, dst_expr, src_location, dst_location, cudastream, ctype = \
                copy_context.get_copy_call_parameters()

        num_dims = copy_context.num_dims
        # ----------------- Generate and write backend call(s) --------------------

        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()

        # Write for-loop headers
        for dim in range(num_dims - 2):
            callsite_stream.write(
                f"for (int __copyidx{dim} = 0; __copyidx{dim} < {copy_shape[dim]}; ++__copyidx{dim}) {{")
            
        # Write Memcopy2DAsync
        offset_src = ' + '.join(f'(__copyidx{d} * ({s}))' for d, s in enumerate(src_strides[:-2]))
        offset_dst = ' + '.join(f'(__copyidx{d} * ({s}))' for d, s in enumerate(dst_strides[:-2]))

        src = f'{src_expr} + {offset_src}' 
        dst = f'{dst_expr} + {offset_dst}' 

        dpitch = f'{dst_strides[-2]} + sizeof({ctype})'
        spitch = f'{src_strides[-2]} + sizeof({ctype})'
        width  = f'{copy_shape[-1]} + sizeof({ctype})'
        height = copy_shape[-2]
        kind = f'{backend}Memcpy{src_location}To{dst_location}'

        # Generate call and write it
        call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst}, {dpitch}, {src}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'
        callsite_stream.write(call, cfg, state_id, [src_node, dst_node])

        # Write for-loop footers
        for d in range(num_dims - 2):
                callsite_stream.write("}")

