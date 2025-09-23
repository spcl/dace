# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os
from typing import Union

from dace import dtypes, registry
from dace.codegen import common
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.scope import is_devicelevel_gpu_kernel
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_TX_MARKERS)
class GPUTXMarkersProvider(InstrumentationProvider):
    """ Timing instrumentation that adds NVTX/rocTX ranges to SDFGs and states. """
    NVTX_HEADER_INCLUDE = '#include <nvtx3/nvToolsExt.h>'
    ROCTX_HEADER_INCLUDE = '#include <roctx.h>'

    def __init__(self):
        self.backend = common.get_gpu_backend()
        # Check if ROCm TX libraries and headers are available
        rocm_path = os.getenv('ROCM_PATH', '/opt/rocm')
        roctx_header_paths = [
            os.path.join(rocm_path, 'roctracer/include/roctx.h'),
            os.path.join(rocm_path, 'include/roctracer/roctx.h')
        ]
        roctx_library_path = os.path.join(rocm_path, 'lib', 'libroctx64.so')
        self.enable_rocTX = any(os.path.isfile(path)
                                for path in roctx_header_paths) and os.path.isfile(roctx_library_path)
        self.include_generated = False
        super().__init__()

    def _print_include(self, sdfg: SDFG) -> None:
        """ Prints the include statement for the NVTX/rocTX library for a given SDFG. """
        if self.include_generated:
            return
        if self.backend == 'cuda':
            sdfg.append_global_code(self.NVTX_HEADER_INCLUDE, 'frame')
        elif self.backend == 'hip':
            if self.enable_rocTX:
                sdfg.append_global_code(self.ROCTX_HEADER_INCLUDE, 'frame')
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)
        self.include_generated = True

    def print_include(self, stream: CodeIOStream) -> None:
        """ Prints the include statement for the NVTX/rocTX library in stream. """
        if stream is None:
            return
        if self.backend == 'cuda':
            stream.write(self.NVTX_HEADER_INCLUDE)
        elif self.backend == 'hip':
            if self.enable_rocTX:
                stream.write(self.ROCTX_HEADER_INCLUDE)
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)

    def print_range_push(self, name: str, sdfg: SDFG, stream: CodeIOStream) -> None:
        if stream is None:
            return
        self._print_include(sdfg)
        if name is None:
            name = 'None'
        if self.backend == 'cuda':
            stream.write(f'nvtxRangePush("{name}");')
        elif self.backend == 'hip':
            if self.enable_rocTX:
                stream.write(f'roctxRangePush("{name}");')
        else:
            raise NameError(f'GPU backend "{self.backend}" not recognized')

    def print_range_pop(self, stream: CodeIOStream) -> None:
        if stream is None:
            return
        if self.backend == 'cuda':
            stream.write('nvtxRangePop();')
        elif self.backend == 'hip':
            if self.enable_rocTX:
                stream.write('roctxRangePop();')
        else:
            raise NameError(f'GPU backend "{self.backend}" not recognized')

    def _is_sdfg_in_device_code(self, sdfg: SDFG) -> bool:
        """ Check if the SDFG is in device code and not top level SDFG. """
        sdfg_parent_state = sdfg.parent
        while sdfg_parent_state is not None:
            sdfg_parent_node = sdfg.parent_nsdfg_node
            if is_devicelevel_gpu_kernel(sdfg, sdfg_parent_state, sdfg_parent_node):
                return True
            sdfg_parent_state = sdfg_parent_state.sdfg.parent
        return False

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'sdfg_{sdfg.name}', sdfg, local_stream)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(local_stream)

    def on_state_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                       global_stream: CodeIOStream) -> None:
        if state.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'state_{state.label}', sdfg, local_stream)

    def on_state_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                     global_stream: CodeIOStream) -> None:
        if state.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(local_stream)

    def on_copy_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, src_node: nodes.Node,
                      dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet], local_stream: CodeIOStream,
                      global_stream: CodeIOStream, copy_shape, src_strides, dst_strides) -> None:
        if state.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, src_node):
            # Don't instrument device code
            return
        self.print_range_push(f'copy_{src_node.label}_to_{dst_node.label}', sdfg, local_stream)

    def on_copy_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet], local_stream: CodeIOStream,
                    global_stream: CodeIOStream) -> None:
        if state.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, src_node):
            # Don't instrument device code
            return
        self.print_range_pop(local_stream)

    def on_scope_entry(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.EntryNode,
                       outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if node.map.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, node):
            # Don't instrument device code
            return
        self.print_range_push(f'scope_{node.label}', sdfg, outer_stream)

    def on_scope_exit(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.ExitNode,
                      outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        entry_node = state.entry_node(node)
        if entry_node.map.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, entry_node):
            # Don't instrument device code
            return
        self.print_range_pop(outer_stream)

    def on_sdfg_init_begin(self, sdfg: SDFG, callsite_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        # cannot push rocTX markers before initializing HIP
        if self.enable_rocTX:
            return
        self.print_range_push(f'init_{sdfg.name}', sdfg, callsite_stream)

    def on_sdfg_init_end(self, sdfg: SDFG, callsite_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        # cannot push rocTX markers before initializing HIP so there's no marker to pop
        if self.enable_rocTX:
            return
        self.print_range_pop(callsite_stream)

    def on_sdfg_exit_begin(self, sdfg: SDFG, callsite_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'exit_{sdfg.name}', sdfg, callsite_stream)

    def on_sdfg_exit_end(self, sdfg: SDFG, callsite_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(callsite_stream)

    def on_allocation_begin(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG], stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'alloc_{sdfg.name}', sdfg, stream)

    def on_allocation_end(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG], stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(stream)

    def on_deallocation_begin(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG], stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'dealloc_{sdfg.name}', sdfg, stream)

    def on_deallocation_end(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG], stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(stream)
