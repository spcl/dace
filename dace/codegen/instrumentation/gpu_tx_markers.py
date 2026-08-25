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
from dace.sdfg.scope import is_devicelevel_gpu_kernel
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState


def roctx_available() -> bool:
    """Whether this machine has both halves of roctracer's marker library."""
    rocm_path = os.getenv('ROCM_PATH', '/opt/rocm')
    headers = [
        os.path.join(rocm_path, 'roctracer/include/roctx.h'),
        os.path.join(rocm_path, 'include/roctracer/roctx.h')
    ]
    return (any(os.path.isfile(path) for path in headers)
            and os.path.isfile(os.path.join(rocm_path, 'lib', 'libroctx64.so')))


def nvtx_available() -> bool:
    """Whether the CUDA toolkit's marker headers are reachable. NVTX v3 is header-only."""
    cuda_path = os.getenv('CUDA_HOME') or os.getenv('CUDA_PATH') or '/usr/local/cuda'
    return os.path.isfile(os.path.join(cuda_path, 'include', 'nvtx3', 'nvToolsExt.h'))


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_TX_MARKERS)
class GPUTXMarkersProvider(InstrumentationProvider):
    """ Timing instrumentation that adds NVTX/rocTX ranges to SDFGs and states. """

    #: Keyed by marker library. The library's name is also the prefix of its range calls.
    HEADER_INCLUDE = {'nvtx': '#include <nvtx3/nvToolsExt.h>', 'roctx': '#include <roctx.h>'}

    def __init__(self):
        self.backend = common.get_gpu_backend()
        if self.backend not in ('cuda', 'hip'):
            raise NameError('GPU backend "%s" not recognized' % self.backend)
        # Which marker library to use follows the platform, not the backend. HIP's NVIDIA platform
        # is the CUDA toolkit underneath: it has NVTX and no roctracer at all.
        if self.backend == 'cuda':
            self.library = 'nvtx'
        elif roctx_available():
            self.library = 'roctx'
        elif nvtx_available():
            self.library = 'nvtx'
        else:
            self.library = None
        self.include_generated = False
        super().__init__()

    def _print_include(self, sdfg: SDFG) -> None:
        """ Prints the include statement for the NVTX/rocTX library for a given SDFG. """
        if self.include_generated or self.library is None:
            return
        sdfg.append_global_code(self.HEADER_INCLUDE[self.library], 'frame')
        self.include_generated = True

    def print_include(self, stream: CodeIOStream) -> None:
        """ Prints the include statement for the NVTX/rocTX library in stream. """
        if stream is None or self.include_generated or self.library is None:
            return
        stream.write(self.HEADER_INCLUDE[self.library])
        self.include_generated = True

    def print_range_push(self, name: str, sdfg: SDFG, stream: CodeIOStream) -> None:
        if stream is None or self.library is None:
            return
        self._print_include(sdfg)
        if name is None:
            name = 'None'
        stream.write(f'{self.library}RangePush("{name}");')

    def print_range_pop(self, stream: CodeIOStream) -> None:
        if stream is None or self.library is None:
            return
        stream.write(f'{self.library}RangePop();')

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
        self.print_include(global_stream)
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
        if is_devicelevel_gpu_kernel(sdfg, state, src_node) or is_devicelevel_gpu_kernel(sdfg, state, dst_node):
            # Don't instrument device code
            return
        self.print_range_push(f'copy_{src_node.label}_to_{dst_node.label}', sdfg, local_stream)

    def on_copy_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet], local_stream: CodeIOStream,
                    global_stream: CodeIOStream) -> None:
        if state.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, src_node) or is_devicelevel_gpu_kernel(sdfg, state, dst_node):
            # Don't instrument device code
            return
        self.print_range_pop(local_stream)

    def on_node_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node,
                      outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if not isinstance(node, nodes.CodeNode) or node.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, node):
            # Don't instrument device code
            return
        self.print_range_push(node.label, sdfg, outer_stream)

    def on_node_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node,
                    outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if not isinstance(node, nodes.CodeNode) or node.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if is_devicelevel_gpu_kernel(sdfg, state, node):
            # Don't instrument device code
            return
        self.print_range_pop(outer_stream)

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
        if self.library == 'roctx':
            return
        self.print_range_push(f'init_{sdfg.name}', sdfg, callsite_stream)

    def on_sdfg_init_end(self, sdfg: SDFG, callsite_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        # cannot push rocTX markers before initializing HIP so there's no marker to pop
        if self.library == 'roctx':
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

    def on_allocation_begin(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                            stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'alloc_{sdfg.name}', sdfg, stream)

    def on_allocation_end(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                          stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(stream)

    def on_deallocation_begin(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                              stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_push(f'dealloc_{sdfg.name}', sdfg, stream)

    def on_deallocation_end(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                            stream: CodeIOStream) -> None:
        if sdfg.instrument != dtypes.InstrumentationType.GPU_TX_MARKERS:
            return
        # We only want to instrument allocations at the SDFG or state level
        if not isinstance(scope, (SDFGState, SDFG)):
            return
        if self._is_sdfg_in_device_code(sdfg):
            # Don't instrument device code
            return
        self.print_range_pop(stream)
