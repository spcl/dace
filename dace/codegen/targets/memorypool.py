import dace
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.sdfg.analysis.live_sets import live_sets
import itertools

_MP_STORAGE_TYPES = ['CPU_Pool', 'GPU_Pool']


class MemoryPoolCodegen(TargetCodeGenerator):
    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()


        # Register array allocation/deallocation
        for dtype in _MP_STORAGE_TYPES:
            enum_type = dace.StorageType[dtype]
            self._dispatcher.register_array_dispatcher(enum_type, self)

        self.alloc_dealloc_states, self.maximum_live_set, self.maximum_live_set_states, self.static_transients = live_sets(sdfg)

        cpu_storages = [
            dace.StorageType.CPU_Heap, dace.StorageType.CPU_Pinned,
            dace.StorageType.CPU_ThreadLocal
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

        self._dispatcher.register_node_dispatcher(self)

        mempools = set()
        for sd, aname, array in sdfg.arrays_recursive():
            if array.lifetime == dace.AllocationLifetime.MemoryPool:
                mempools.add(array.storage)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        callsite_stream.write(
            'printf(\"test\");'
        )
        self._cpu_codegen.generate_node(sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        function_stream.write(
            ''' #include <dace/memory_pool.h>
            '''
        )

        callsite_stream.write(
            ''' MemoryPool MPool;
            MPool.ReserveMemory({m_size});
            double *{array} = (double*)MPool.Alloc({size});
            '''.format(array=node.label, size=sdfg.arrays[node.data].total_size, m_size=self.maximum_live_set[1])
        )

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        pass

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge, function_stream,
                    callsite_stream):
        callsite_stream.write("""printf(\"copyy\");
            std::memcpy({dst_node},{src_node}, sizeof({src_node}) * 64);
            """.format(src_node=src_node, dst_node=dst_node, size=sdfg.arrays[src_node.data].total_size)
        )

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream, callsite_stream):
        pass


def extend_dace():

    TargetCodeGenerator.register(MemoryPoolCodegen, name='memorypool')

    for dtype in _MP_STORAGE_TYPES:
        dace.StorageType.register(dtype)
