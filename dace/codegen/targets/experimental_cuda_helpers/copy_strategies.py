from abc import ABC, abstractmethod
from typing import Tuple

from dace import symbolic
from dace import Memlet, dtypes
from dace.dtypes import StorageType
from dace.codegen.targets.experimental_cuda import ExperimentalCUDACodeGen, GPUStreamManager, KernelSpec
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import product, symbolic_to_cpp, emit_sync_debug_checks

from dace.codegen.prettycode import CodeIOStream
from dace.sdfg import SDFG, nodes
from dace.sdfg.nodes import Node
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

from dace.codegen.targets.cpp import memlet_copy_to_absolute_strides, unparse_cr


# TODO: Review Documentation once done here. And also, take care of the other
# two strategies below.
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

    def __init__(self, codegen: ExperimentalCUDACodeGen, gpu_stream_manager: GPUStreamManager, state_id: int,
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
        self.cudastream = gpu_stream_manager.get_stream_edge(src_node, dst_node)
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

        # Should be symmetric
        ctype_src = self.src_node.desc(self.sdfg).dtype.ctype
        ctype_dst = self.dst_node.desc(self.sdfg).dtype.ctype
        ctype = ctype_dst
        assert ctype_src == ctype_dst, (f"Source and destination data types must match for the memory copy: "
                                        f"{ctype_src} != {ctype_dst}")

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

    def get_memory_location(self) -> Tuple[str, str]:
        src_location = 'Device' if self.src_storage == dtypes.StorageType.GPU_Global else 'Host'
        dst_location = 'Device' if self.dst_storage == dtypes.StorageType.GPU_Global else 'Host'

        return src_location, dst_location


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

        is_between_access_nodes = (isinstance(copy_context.src_node, nodes.AccessNode)
                                   and isinstance(copy_context.dst_node, nodes.AccessNode))

        involves_gpu_or_pinned = (copy_context.src_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned)
                                  or copy_context.dst_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned))

        is_not_cpu_to_cpu = not (copy_context.src_storage in cpu_storage_types
                                 and copy_context.dst_storage in cpu_storage_types)

        is_gpu_host_copy = (not_in_kernel_code and is_between_access_nodes and involves_gpu_or_pinned
                            and is_not_cpu_to_cpu)

        return is_gpu_host_copy

    def generate_copy(self, copy_context: CopyContext) -> None:
        """Execute host-device copy with CUDA memory operations"""

        # guard
        _, _, _, _, memlet = copy_context.edge
        if memlet.wcr is not None:
            src_location, dst_location = copy_context.get_memory_location()
            raise NotImplementedError(f'Accumulate {src_location} to {dst_location} not implemented')

        # call corresponding helper function
        num_dims = copy_context.num_dims
        if num_dims == 1:
            self._generate_1d_copy(copy_context)
        elif num_dims == 2:
            self._generate_2d_copy(copy_context)
        else:
            # sanity check
            assert num_dims > 2, f"Expected copy shape with more than 2 dimensions, but got {num_dims}."
            self._generate_nd_copy(copy_context)

        # We use library calls thus for debugging we provide sync option
        emit_sync_debug_checks(copy_context.backend, copy_context.callsite_stream)

    def _generate_1d_copy(self, copy_context: CopyContext) -> None:
        """
        Emits code for a 1D memory copy between host and device using GPU backend.
        Uses {backend}MemcpyAsync for contiguous memory and uses {backend}Memcpy2DAsync
        for strided memory copies.
        """

        # ----------- Extract relevant copy parameters --------------
        copy_shape, src_strides, dst_strides = copy_context.get_transfer_layout()

        backend, src_expr, dst_expr, src_location, dst_location, cudastream, ctype = \
                copy_context.get_copy_call_parameters()

        # ----------------- Generate backend call --------------------
        if copy_context.is_contiguous_copy():
            # Memory is linear: can use {backend}MemcpyAsync
            copysize = ' * '.join(symbolic_to_cpp(copy_shape))
            copysize += f' * sizeof({ctype})'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'
            call = f'DACE_GPU_CHECK({backend}MemcpyAsync({dst_expr}, {src_expr}, {copysize}, {kind}, {cudastream}));\n'

        else:
            # Memory is strided: use {backend}Memcpy2DAsync with dpitch/spitch
            # This allows copying a strided 1D region
            dpitch = f'{dst_strides[0]} * sizeof({ctype})'
            spitch = f'{src_strides[0]} * sizeof({ctype})'
            width = f'sizeof({ctype})'
            height = copy_shape[0]
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'

        # ----------------- Write copy call to code stream --------------------
        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()
        callsite_stream.write(call, cfg, state_id, [src_node, dst_node])

    def _generate_2d_copy(self, copy_context: CopyContext) -> None:
        """Generates code for a 2D copy, falling back to 1D flattening if applicable."""

        # ----------- Extract relevant copy parameters --------------
        copy_shape, src_strides, dst_strides = copy_context.get_transfer_layout()

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

        elif src_strides[-1] != 1 or dst_strides[-1] != 1:
            # TODO: Checks this, I am not sure but the old code and its description
            # seems to be more complicated here than necessary..
            # But worth to mention: we essentiall flatten

            # NOTE: Special case of continuous copy
            # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
            # with copy shape [I, J] and strides [J*K, K], [J, 1]

            dpitch = f'{dst_strides[1]} * sizeof({ctype})'
            spitch = f'{src_strides[1]} * sizeof({ctype})'
            width = f'sizeof({ctype})'
            height = copy_shape[0] * copy_shape[1]
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst_expr}, {dpitch}, {src_expr}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'

        else:
            raise NotImplementedError(
                f"Unsupported 2D memory copy: shape={copy_shape}, src_strides={src_strides}, dst_strides={dst_strides}."
                " Please implement this case if it is valid, or raise a more descriptive error if this path should not be taken."
            )

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
                f"  Destination strides: {copy_context.dst_strides}\n")

        # ----------- Extract relevant copy parameters --------------
        copy_shape, src_strides, dst_strides = copy_context.get_transfer_layout()

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
        width = f'{copy_shape[-1]} + sizeof({ctype})'
        height = copy_shape[-2]
        kind = f'{backend}Memcpy{src_location}To{dst_location}'

        # Generate call and write it
        call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync({dst}, {dpitch}, {src}, {spitch}, {width}, {height}, {kind}, {cudastream}));\n'
        callsite_stream.write(call, cfg, state_id, [src_node, dst_node])

        # Write for-loop footers
        for dim in range(num_dims - 2):
            callsite_stream.write("}")


################ TODO, Might need to modified further #############


# Below: Does collaborative copy
class SyncCollaboritveGPUCopyStrategy(CopyStrategy):

    def applicable(self, copy_context: CopyContext) -> bool:
        """
        Checks if the copy is eligible for a collaborative GPU-to-GPU copy.

        Conditions:
        1. The copy is between GPU memory types (shared or global).
        2. The innermost non-sequential map is scheduled on GPU_Device.
        """
        from dace.sdfg import scope_contains_scope
        from dace.transformation import helpers

        # --- Condition 1: GPU to GPU memory transfer ---
        gpu_storages = {dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared}
        if not (copy_context.src_storage in gpu_storages and copy_context.dst_storage in gpu_storages):
            return False

        dst_node = copy_context.dst_node
        if isinstance(dst_node, nodes.AccessNode) and dst_node.async_copy:
            return False

        # --- Condition 2: Inside a GPU_Device map scope ---
        state = copy_context.state_dfg
        scope_dict = state.scope_dict()

        # Determine which node (src or dst) is in the deeper scope
        src, dst = copy_context.src_node, copy_context.dst_node
        deeper_scope_node = dst if scope_contains_scope(scope_dict, src, dst) else src

        # Determine the schedule type of the innermost non-sequential map.
        # If no such map exists, use the default schedule.
        current_node = deeper_scope_node
        while (current_node is None or not isinstance(current_node, nodes.MapEntry)
               or current_node.map.schedule == dtypes.ScheduleType.Sequential):

            parent = helpers.get_parent_map(state, current_node)
            if parent is None:
                current_node = None
                break
            current_node, state = parent

        if current_node is None:
            schedule_type = dtypes.SCOPEDEFAULT_SCHEDULE[None]
        else:
            schedule_type = current_node.map.schedule

        return schedule_type == dtypes.ScheduleType.GPU_Device

    def generate_copy(self, copy_context: CopyContext) -> None:

        from dace.frontend import operations

        # Get required copy information
        copy_shape, src_strides, dst_strides = copy_context.get_transfer_layout()
        src_expr, dst_expr = copy_context.src_expr, copy_context.dst_expr

        sdfg = copy_context.sdfg
        dtype = copy_context.src_node.desc(sdfg).dtype
        ctype = dtype.ctype

        # Get copy function name (defined in runtime library)
        num_dims = copy_context.num_dims
        src_storage_name = self._get_storagename(copy_context.src_storage)
        dst_storage_name = self._get_storagename(copy_context.dst_storage)

        function_name = f"dace::{src_storage_name}To{dst_storage_name}{num_dims}D"

        # Check for write-conflict resolution (WCR), it affects function call
        accum = ''
        custom_reduction = []
        _, _, _, _, memlet = copy_context.edge
        wcr = memlet.wcr

        if wcr is not None:
            reduction_type = operations.detect_reduction_type(wcr)

            if reduction_type != dtypes.ReductionType.Custom:
                # Use predefined reduction
                reduction_type_str = str(reduction_type).split('.')[-1]  # e.g., "Sum"
                reduction_template = f"<{reduction_type_str}>"
            else:
                custom_reduction = [unparse_cr(sdfg, wcr, dtype)]
                reduction_template = ""

            accum = f"::template Accum{reduction_template}"

        # Dispatch to the correct backend copy template based on copy characteristics

        # get always used stuff
        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()

        # Retrieve kernel specs from the ExperimentalCUDACodegen instance (held in a dedicated class)
        # Only there block_dims is stored, which is needed in this case
        kernel_specifications: KernelSpec = copy_context.codegen._current_kernel_spec
        block_dims = ', '.join(symbolic_to_cpp(kernel_specifications.block_dims))

        # was called "is_async" previously. It determines whether a "__syncthreads()" is called at the
        # end of the copy. In ExperimentalCUDACodegen, a pass is responsible to insert such sync barriers,
        # so it is synchronized and we do not need "implicit" synchronization
        synchronized = "true"

        if any(symbolic.issymbolic(s, copy_context.sdfg.constants) for s in copy_shape):
            args_list = ([src_expr] + src_strides + [dst_expr] + custom_reduction + dst_strides + copy_shape)
            args = ", ".join(symbolic_to_cpp(args_list))
            callsite_stream.write(f"{function_name}Dynamic<{ctype}, {block_dims}, {synchronized}>{accum}({args});", cfg,
                                  state_id, [src_node, dst_node])

        elif function_name == "dace::SharedToGlobal1D":
            # special case: use a new template struct that provides functions for copy and reduction
            copy_size = ', '.join(symbolic_to_cpp(copy_shape))
            accum = accum or '::Copy'
            args_list = ([src_expr] + src_strides + [dst_expr] + dst_strides + custom_reduction)
            args = ", ".join(symbolic_to_cpp(args_list))
            callsite_stream.write(
                f"{function_name}<{ctype}, {block_dims}, {copy_size}, {synchronized}>{accum}({args});", cfg, state_id,
                [src_node, dst_node])

        else:
            copy_size = ', '.join(symbolic_to_cpp(copy_shape))
            args_list = ([src_expr] + src_strides + [dst_expr] + custom_reduction)
            args = ", ".join(symbolic_to_cpp(args_list))
            dst_strides_unpacked = ", ".join(symbolic_to_cpp(dst_strides))
            callsite_stream.write(
                f"{function_name}<{ctype}, {block_dims}, {copy_size}, {dst_strides_unpacked}, {synchronized}>{accum}({args});",
                cfg, state_id, [src_node, dst_node])

    def _get_storagename(self, storage: dtypes.StorageType):
        """
        Returns a string containing the name of the storage location.

        Example: dtypes.StorageType.GPU_Shared will return "Shared".
        """
        storage_name = str(storage)
        return storage_name[storage_name.rindex('_') + 1:]


class AsyncCollaboritveGPUCopyStrategy(CopyStrategy):

    def applicable(self, copy_context: CopyContext) -> bool:

        from dace.sdfg import scope_contains_scope
        from dace.transformation import helpers

        # --- Condition 1: GPU to GPU memory transfer ---
        gpu_storages = {dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared}
        if not (copy_context.src_storage in gpu_storages and copy_context.dst_storage in gpu_storages):
            return False

        dst_node = copy_context.dst_node
        if not (isinstance(dst_node, nodes.AccessNode) and dst_node.async_copy):
            return False

        # --- Condition 2: Inside a GPU_Device map scope ---
        state = copy_context.state_dfg
        scope_dict = state.scope_dict()

        # Determine which node (src or dst) is in the deeper scope
        src, dst = copy_context.src_node, copy_context.dst_node
        deeper_scope_node = dst if scope_contains_scope(scope_dict, src, dst) else src

        # Determine the schedule type of the innermost non-sequential map.
        # If no such map exists, use the default schedule.
        current_node = deeper_scope_node
        while (current_node is None or not isinstance(current_node, nodes.MapEntry)
               or current_node.map.schedule == dtypes.ScheduleType.Sequential):

            parent = helpers.get_parent_map(state, current_node)
            if parent is None:
                current_node = None
                break
            current_node, state = parent

        if current_node is None:
            schedule_type = dtypes.SCOPEDEFAULT_SCHEDULE[None]
        else:
            schedule_type = current_node.map.schedule

        return schedule_type == dtypes.ScheduleType.GPU_Device

    def generate_copy(self, copy_context: CopyContext):

        # Show Yakup:
        # Asynchronous memory copies are only allowed if they are contiguous
        if not copy_context.is_contiguous_copy():
            raise NotImplementedError("Asynchronous memory copies are not supported for not contigous memory copies")

        # Get required copy information
        copy_shape, src_strides, dst_strides = copy_context.get_transfer_layout()
        src_expr, dst_expr = copy_context.src_expr, copy_context.dst_expr

        sdfg = copy_context.sdfg
        dtype = copy_context.src_node.desc(sdfg).dtype
        ctype = dtype.ctype

        # Get write context:
        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()
        # copy dimension
        num_dims = len(copy_shape)

        if num_dims == 1:
            pipeline = dst_node.async_pipeline
            size = f'{product(copy_shape)} *sizeof({ctype})'
            callsite_stream.write(f"cuda::memcpy_async(block, {dst_expr}, {src_expr}, {size}, {pipeline});\n", cfg,
                                  state_id, [src_node, dst_node])

        elif num_dims > 1:

            # No built-in functionality for higher dimension copies-
            # But solvable looping and doing 1D copies

            # write for-loop header:
            for dim in range(num_dims - 1):
                callsite_stream.write(
                    f"for (int __copyidx{dim} = 0; __copyidx{dim} < {copy_shape[dim]}; ++__copyidx{dim}) {{")

            offset_src = ' + '.join(f'(__copyidx{d} * ({s}))' for d, s in enumerate(src_strides[:-1]))
            offset_dst = ' + '.join(f'(__copyidx{d} * ({s}))' for d, s in enumerate(dst_strides[:-1]))

            size = f'{copy_shape[-1]} *sizeof({ctype})'
            src = f'{src_expr} + {offset_src}'
            dst = f'{dst_expr} + {offset_dst}'

            callsite_stream.write(f"cuda::memcpy_async(block, {dst}, {src}, {size}, {pipeline});\n", cfg, state_id,
                                  [src_node, dst_node])

            # Write for-loop footers
            for dim in range(num_dims - 2):
                callsite_stream.write("}")

        else:
            # Should not be possible- otherwise, doing nothing is also okay
            # because a empty copy shape means we don't copy anything
            pass

        emit_sync_debug_checks(copy_context.backend, copy_context.callsite_stream)


class FallBackGPUCopyStrategy(CopyStrategy):

    def applicable(self, copy_context: CopyContext) -> bool:
        return True

    def generate_copy(self, copy_context: CopyContext):
        callsite_stream, cfg, state_id, src_node, dst_node = copy_context.get_write_context()
        sdfg = copy_context.sdfg
        dfg = copy_context.dfg
        edge = copy_context.edge
        cpu_codegen = copy_context.codegen._cpu_codegen
        cpu_codegen.copy_memory(sdfg, cfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)
