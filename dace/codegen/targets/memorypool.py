import dace
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.sdfg.analysis.live_sets import live_sets
from dace.codegen.targets.target import DefinedType
from dace import registry
import itertools

_MP_STORAGE_TYPES = ['CPU_Pool', 'GPU_Pool']

#@registry.autoregister_params(name='memorypool')
class MemoryPoolCodegen(TargetCodeGenerator):
    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        self._block_size = 512

        # Mark all transients as CPU_Pool
        for s in sdfg.arrays:
            if sdfg.arrays[s].transient:
                sdfg.arrays[s].storage = dace.StorageType.CPU_Pool

        # Get graph analysis
        self.alloc_dealloc_states, self.maximum_live_set, \
            self.maximum_live_set_states, self.shared_transients = live_sets(sdfg)

        print('alloc_dealloc_states', self.alloc_dealloc_states)
        print('maximum_live_set', self.maximum_live_set)
        print('maximum_live_set_states', self.maximum_live_set_states)
        print('shared_transients', self.shared_transients)

        # Simple initialization of MemoryPool:
        self.initialization = True

        cpp_code = '''#include <dace/memory_pool.h>'''
        if 'frame' not in sdfg.global_code:
            sdfg.global_code['frame'] = dace.properties.CodeBlock('', dace.dtypes.Language.CPP)
        sdfg.global_code['frame'].code += cpp_code

        # Register array allocation/deallocation
        for dtype in _MP_STORAGE_TYPES:
            enum_type = dace.StorageType[dtype]
            self._dispatcher.register_array_dispatcher(enum_type, self)

        cpu_storages = [
            dace.StorageType.CPU_Heap, dace.StorageType.CPU_Pinned,
            dace.StorageType.CPU_ThreadLocal, dace.StorageType.Register
        ]
        gpu_storages = [
            dace.StorageType.GPU_Global,
            dace.StorageType.GPU_Shared,
            dace.StorageType.Register
        ]

        for src_storage, dst_storage in itertools.product(
                [_MP_STORAGE_TYPES[0]], cpu_storages):
            src_storage = dace.StorageType[src_storage]
            self._dispatcher.register_copy_dispatcher(src_storage, dst_storage,
                                                      None, self)
            self._dispatcher.register_copy_dispatcher(dst_storage, src_storage,
                                                      None, self)

        for src_storage, dst_storage in itertools.product(
                [_MP_STORAGE_TYPES[1]], gpu_storages):
            src_storage = dace.StorageType[src_storage]
            self._dispatcher.register_copy_dispatcher(src_storage, dst_storage,
                                                      None, self)
            self._dispatcher.register_copy_dispatcher(dst_storage, src_storage,
                                                      None, self)

        # Register node dispatcher
        self._dispatcher.register_node_dispatcher(self)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                        function_stream, callsite_stream)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        if self.initialization:
            m_size = 0
            for t in sdfg.transients():
                m_size += int((sdfg.arrays[t].total_size // self._block_size + 1) * self._block_size)

            callsite_stream.write(
                '''MemoryPool<false> MPool;
                   MPool.ReserveMemory({m_size},{block_size});'''.format(m_size=m_size, block_size=self._block_size)
            )
            for t in self.shared_transients:
                callsite_stream.write(
                    '''double *{array} = (double*)MPool.Alloc({size});'''.format(
                    array=t, size=sdfg.arrays[node.data].total_size,
                     m_size=self.maximum_live_set[1])
            )
            self.initialization = False
        self._dispatcher.defined_vars.add(node.label, DefinedType.Pointer)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        pass

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge, function_stream,
                    callsite_stream):
        self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge, function_stream,
                    callsite_stream)

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream, callsite_stream):
        pass


def extend_dace():

    TargetCodeGenerator.register(MemoryPoolCodegen, name='memorypool')

    for dtype in _MP_STORAGE_TYPES:
        dace.StorageType.register(dtype)
