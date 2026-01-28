# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
from dace import SDFG, SDFGState, data, dtypes, subsets
from dace import memlet as mm
from dace.codegen import common
from dace.codegen.targets import cpp
from dace.codegen.targets.cpp import sym2cpp
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import generate_sync_debug_call
from dace.dtypes import StorageType
from dace.sdfg import nodes, scope_contains_scope
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import helpers


class CopyContext:
    """
    Encapsulates inputs required for copy operations and exposes helper
    methods to derive additional information. This keeps copy strategies
    lightweight by letting them focus only on the relevant logic.
    """

    def __init__(self, sdfg: SDFG, state: SDFGState, src_node: nodes.Node, dst_node: nodes.Node,
                 edge: MultiConnectorEdge[mm.Memlet]):

        # Store the basic context as attributes
        self.sdfg = sdfg
        self.state = state
        self.src_node = src_node
        self.dst_node = dst_node
        self.edge = edge

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
        """
        Return the storage type associated with a given SDFG node.

        Tasklets are assumed to use register storage, while AccessNodes
        return the storage type from their data descriptor. Raises
        NotImplementedError for unsupported node types.
        """
        if isinstance(node, nodes.Tasklet):
            storage_type = StorageType.Register

        elif isinstance(node, nodes.AccessNode):
            storage_type = node.desc(self.sdfg).storage

        else:
            raise NotImplementedError(f"Unsupported node type {type(node)} for storage type retrieval; "
                                      "expected AccessNode or Tasklet. Please extend this method accordingly.")

        return storage_type

    def get_assigned_gpustream(self) -> str:
        """
        Return the GPU stream expression assigned to both source and destination nodes.
        Defaults to `__dace_current_stream` placeholder, which can be changed by the scheduling pass
        """
        # 2. Generate GPU stream expression
        gpustream = "__dace_current_stream"
        gpustream_expr = gpustream

        return gpustream_expr

    def get_memory_location(self) -> Tuple[str, str]:
        """
        Determine whether the source and destination nodes reside in device or host memory.

        Uses the storage type of each node to classify it as either 'Device'
        (GPU global memory) or 'Host' (all other storage types).
        Used for GPU related copies outside the kernel (e.g. to construct
        cudaMemcpyHostToDevice for example).

        Returns
        -------
        Tuple[str, str]
            (src_location, dst_location) where each is either 'Device' or 'Host'.
        """
        src_storage = self.get_storage_type(self.src_node)
        dst_storage = self.get_storage_type(self.dst_node)
        src_location = 'Device' if src_storage == dtypes.StorageType.GPU_Global else 'Host'
        dst_location = 'Device' if dst_storage == dtypes.StorageType.GPU_Global else 'Host'

        return src_location, dst_location

    def get_ctype(self) -> Any:
        """
        Determine the C data type (ctype) of the source or destination node.

        The ctype is resolved from the data descriptor of the first node
        (source or destination) that is an AccessNode (assumed to be the same
        if both are AccessNodes).

        Returns
        -------
        Any
            The C type string (e.g., "float*", "int32") associated with the node.

        Raises
        ------
        NotImplementedError
            If neither the source nor the destination node is an AccessNode.
        """
        sdfg = self.sdfg
        src_node, dst_node = self.src_node, self.dst_node

        if isinstance(src_node, nodes.AccessNode):
            return src_node.desc(sdfg).ctype

        if isinstance(dst_node, nodes.AccessNode):
            return dst_node.desc(sdfg).ctype

        raise NotImplementedError(
            f"Cannot determine ctype: neither src nor dst node is an AccessNode. "
            f"Got src_node type: {type(src_node).__name__}, dst_node type: {type(dst_node).__name__}. "
            "Please extend this case or fix the issue.")

    def get_accessnode_to_accessnode_copy_info(self):
        """
        Compute copy shape, absolute strides, and pointer expressions for a copy
        between two AccessNodes. Tries to mimic
        cpp.memlet_copy_to_absolute_strides without requiring a dispatcher.

        Returns
        -------
        (copy_shape, src_strides, dst_strides, src_expr, dst_expr)

        Raises
        ------
        TypeError
            If either endpoint is not an AccessNode.
        NotImplementedError
            If a descriptor is not Scalar or Array.
        """

        # ---------------------------- helpers ----------------------------
        def _collapse_strides(strides, subset):
            """Remove size-1 dims; keep tile strides; default to [1] if none remain."""
            n = len(subset)
            collapsed = [st for st, sz in zip(strides, subset.size()) if sz != 1]
            collapsed.extend(strides[n:])  # include tiles
            if len(collapsed) == 0:
                return [1]
            return collapsed

        def _ptr_name(desc, name):
            if desc.transient and desc.lifetime in (dtypes.AllocationLifetime.Persistent,
                                                    dtypes.AllocationLifetime.External):
                return f'__state->__{sdfg.cfg_id}_{name}'
            return name

        def _expr_for(desc, name, subset):
            ptr = _ptr_name(desc, name)

            if isinstance(desc, data.Scalar):
                # GPU scalar special-case
                if desc.storage in dtypes.GPU_STORAGES:
                    parent = state.sdfg.parent_nsdfg_node
                    if parent is not None and name in parent.in_connectors:
                        return f"&{ptr}"
                    return ptr
                # CPU (or other) scalars
                return f"&{ptr}"

            if isinstance(desc, data.Array):
                offset = cpp.cpp_offset_expr(desc, subset)
                return f"{ptr} + {offset}" if offset != "0" else ptr

            raise NotImplementedError(
                f"Expected {name} to be either data.Scalar or data.Array, but got {type(desc).__name__}.")

        # ---------------------------- Get copy info ----------------------------
        # Get needed information
        src_node, dst_node = self.src_node, self.dst_node
        sdfg, edge, state = self.sdfg, self.edge, self.state
        memlet, copy_shape = self.edge.data, self.copy_shape

        # Guard - only applicable if src and dst are AccessNodes
        if not (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)):
            raise TypeError(
                f"get_accessnode_to_accessnode_copy_info requires both source and destination "
                f"to be AccessNode instances, but got {type(src_node).__name__} and {type(dst_node).__name__}.")

        # Get node descriptors
        src_nodedesc = src_node.desc(sdfg)
        dst_nodedesc = dst_node.desc(sdfg)

        # Resolve subsets (fallback to full range)
        src_subset = memlet.get_src_subset(edge, state)
        dst_subset = memlet.get_dst_subset(edge, state)

        if src_subset is None:
            src_subset = subsets.Range.from_array(src_nodedesc)

        if dst_subset is None:
            dst_subset = src_subset
            # dst_subset = subsets.Range.from_array(dst_nodedesc)

        # Get strides
        src_strides = src_subset.absolute_strides(src_nodedesc.strides)
        dst_strides = dst_subset.absolute_strides(dst_nodedesc.strides)

        # Try to convert to a degenerate/strided ND copy first
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
            src_strides = _collapse_strides(src_strides, src_subset)
            dst_strides = _collapse_strides(dst_strides, dst_subset)
            copy_shape = [s for s in copy_shape if s != 1] or [1]

        # Extend copy shape to the largest among the data dimensions,
        # and extend other array with the appropriate strides
        if len(dst_strides) != len(copy_shape) or len(src_strides) != len(copy_shape):
            if memlet.data == src_node.data:
                copy_shape, dst_strides = cpp.reshape_strides(src_subset, src_strides, dst_strides, copy_shape)
            elif memlet.data == dst_node.data:
                copy_shape, src_strides = cpp.reshape_strides(dst_subset, dst_strides, src_strides, copy_shape)

        return copy_shape, src_strides, dst_strides, src_node.data, dst_node.data


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
            if parent_map.map.schedule in dtypes.GPU_SCHEDULES + dtypes.EXPERIMENTAL_GPU_SCHEDULES:
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
        if not (src_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned)
                or dst_storage in (StorageType.GPU_Global, StorageType.CPU_Pinned)):
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
            copysize = ' * '.join(sym2cpp(copy_shape))
            copysize += f' * sizeof({ctype})'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'
            call = f'DACE_GPU_CHECK({backend}MemcpyAsync(_out_{dst_expr}, _in_{src_expr}, {copysize}, {kind}, {gpustream}));\n'

        else:
            # Memory is strided: use {backend}Memcpy2DAsync with dpitch/spitch
            # This allows copying a strided 1D region
            dpitch = f'{sym2cpp(dst_strides[0])} * sizeof({ctype})'
            spitch = f'{sym2cpp(src_strides[0])} * sizeof({ctype})'
            width = f'sizeof({ctype})'
            height = sym2cpp(copy_shape[0])
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync(_out_{dst_expr}, {dpitch}, _in_{src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Potentially snychronization required if syncdebug is set to true in configurations
        call = call + generate_sync_debug_call()
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
            dpitch = f'{sym2cpp(dst_strides[0])} * sizeof({ctype})'
            spitch = f'{sym2cpp(src_strides[0])} * sizeof({ctype})'
            width = f'{sym2cpp(copy_shape[1])} * sizeof({ctype})'
            height = f'{sym2cpp(copy_shape[0])}'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync(_out_{dst_expr}, {dpitch}, _in_{src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Case: Column-major layout, no columns are strided.
        elif (src_strides[0] == 1) and (dst_strides[0] == 1):
            dpitch = f'{sym2cpp(dst_strides[1])} * sizeof({ctype})'
            spitch = f'{sym2cpp(src_strides[1])} * sizeof({ctype})'
            width = f'{sym2cpp(copy_shape[0])} * sizeof({ctype})'
            height = f'{sym2cpp(copy_shape[1])}'
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync(_out_{dst_expr}, {dpitch}, _in_{src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Special case
        elif (src_strides[0] / src_strides[1] == copy_shape[1] and dst_strides[0] / dst_strides[1] == copy_shape[1]):
            # Consider as an example this copy: A[0:I, 0:J, K] -> B[0:I, 0:J] with
            # copy shape [I, J], src_strides[J*K, K], dst_strides[J, 1]. This can be represented with a
            # {backend}Memcpy2DAsync call!

            dpitch = f'{sym2cpp(dst_strides[1])} * sizeof({ctype})'
            spitch = f'{sym2cpp(src_strides[1])} * sizeof({ctype})'
            width = f'sizeof({ctype})'
            height = sym2cpp(copy_shape[0] * copy_shape[1])
            kind = f'{backend}Memcpy{src_location}To{dst_location}'

            call = f'DACE_GPU_CHECK({backend}Memcpy2DAsync(_out_{dst_expr}, {dpitch}, _in_{src_expr}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        else:
            raise NotImplementedError(
                f"Unsupported 2D memory copy: shape={copy_shape}, src_strides={src_strides}, dst_strides={dst_strides}."
                "Please implement this case if it is valid, or raise a more descriptive error if this path should not be taken."
            )

        # Potentially snychronization required if syncdebug is set to true in configurations
        call = call + generate_sync_debug_call()
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
                f"  copy shape: {copy_shape}\n")

        # ----------------- Generate and write backend call(s) --------------------

        call = ""
        # Write for-loop headers
        for dim in range(num_dims - 2):
            call += f"for (int __copyidx{dim} = 0; __copyidx{dim} < {copy_shape[dim]}; ++__copyidx{dim}) {{\n"

        # Write Memcopy2DAsync
        offset_src = ' + '.join(f'(__copyidx{d} * ({sym2cpp(s)}))' for d, s in enumerate(src_strides[:-2]))
        offset_dst = ' + '.join(f'(__copyidx{d} * ({sym2cpp(s)}))' for d, s in enumerate(dst_strides[:-2]))

        src = f'{src_expr} + {offset_src}'
        dst = f'{dst_expr} + {offset_dst}'

        dpitch = f'{sym2cpp(dst_strides[-2])} * sizeof({ctype})'
        spitch = f'{sym2cpp(src_strides[-2])} * sizeof({ctype})'
        width = f'{sym2cpp(copy_shape[-1])} * sizeof({ctype})'
        height = sym2cpp(copy_shape[-2])
        kind = f'{backend}Memcpy{src_location}To{dst_location}'

        # Generate call and write it
        call += f'DACE_GPU_CHECK({backend}Memcpy2DAsync(_out_{dst}, {dpitch}, _in_{src}, {spitch}, {width}, {height}, {kind}, {gpustream}));\n'

        # Potentially synchronization required if syncdebug is set to true in configurations
        call += generate_sync_debug_call()

        # Write for-loop footers
        for dim in range(num_dims - 2):
            call += "\n}"

        # Return the code
        return call
