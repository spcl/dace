import dace
from dace import dtypes
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.sdfg.analysis.live_sets import live_sets
from dace.codegen.targets.target import DefinedType
from dace import registry
import itertools


# @registry.autoregister_params(name='memorypool')
class MemoryPoolCodegen(TargetCodeGenerator):
    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        self._block_size = 512

        # Get graph analysis
        #self.alloc_dealloc_states, self.maximum_live_set, \
        #    self.maximum_live_set_states, self.shared_transients = live_sets(sdfg)

        # Simple initialization of MemoryPool:
        self.initialization = True
        cpp_code = '''#include <dace/memory_pool.h>'''
        if 'frame' not in sdfg.global_code:
            sdfg.global_code['frame'] = dace.properties.CodeBlock('', dace.dtypes.Language.CPP)
        sdfg.global_code['frame'].code += cpp_code

        self.cpu_storages = [dace.StorageType.CPU_Heap, dace.StorageType.Default]
        self.gpu_storages = [dace.StorageType.GPU_Global]

        # Mark all transients as CPU_Pool/GPU_Pool
        self.cpu_size = 0
        self.gpu_size = 0
        for s in sdfg.arrays:
            array = sdfg.arrays[s]
            if array.transient:
                original = array.storage
                if array.storage in self.cpu_storages:
                    array.lifetime = dace.AllocationLifetime.Pool
                    self.cpu_size += int((array.total_size * array.dtype.bytes
                                          // self._block_size + 1) * self._block_size)
                elif array.storage in self.gpu_storages:
                    array.lifetime = dace.AllocationLifetime.Pool
                    self.gpu_size += int((array.total_size * array.dtype.bytes
                                          // self._block_size + 1) * self._block_size)

        # Register array allocation/deallocation
        for dtype in [dace.StorageType.CPU_Heap, dace.StorageType.GPU_Global]:
            self._dispatcher.register_array_dispatcher(
                dtype, self,
                lambda sdfg, node: node.desc(sdfg).lifetime == dtypes.AllocationLifetime.Pool
            )

        # Register node dispatcher
        #self._dispatcher.register_node_dispatcher(self)



    def generate_node(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        #callsite_stream.write("//{node}".format(node=node))

        # state = sdfg.states()[state_id]
        # if state in self.alloc_dealloc_states.keys():
        #     alloc_dealloc = self.alloc_dealloc_states[state]
        #     if node in alloc_dealloc:
        #         alloc, dealloc = alloc_dealloc[node]
        #         for t in dealloc:
        #             array = sdfg.arrays[t]
        #             if array.storage in self.cpu_storages:
        #                 callsite_stream.write(
        #                     '''CPU_Pool.Dealloc({name});'''.format(name=t)
        #                 )
        #             elif array.storage in self.gpu_storages:
        #                 callsite_stream.write(
        #                     '''GPU_Pool.Dealloc({name});'''.format(name=t)
        #                 )
        #
        #         for t in alloc:
        #             array = sdfg.arrays[t]
        #             if array.storage in self.cpu_storages:
        #                 callsite_stream.write(
        #                     '''{type} *{name} = ({type}*)CPU_Pool.Alloc({size});'''.format(
        #                         name=t, size=array.total_size*array.dtype.bytes,
        #                         type=array.dtype.ctype
        #                     )
        #                 )
        #             elif array.storage in self.gpu_storages:
        #                 callsite_stream.write(
        #                     '''{type} *{name} = ({type}*)GPU_Pool.Alloc({size});'''.format(
        #                         name=t, size=array.total_size*array.dtype.bytes,
        #                         type=array.dtype.ctype
        #                     )
        #                 )
        '''self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                        function_stream, callsite_stream)'''

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        if self.initialization:
            if self.cpu_size:
                callsite_stream.write(
                    '''MemoryPool<false> CPU_Pool({m_size},{block_size});'''.format(
                        m_size=self.cpu_size, block_size=self._block_size)
                )
            if self.gpu_size:
                callsite_stream.write(
                    '''MemoryPool<true> GPU_Pool({m_size},{block_size});'''.format(
                        m_size=self.gpu_size, block_size=self._block_size)
                )

            for t in sdfg.transients():
                array = sdfg.arrays[t]
                if array.storage in self.cpu_storages:
                    callsite_stream.write(
                        '''{type} *{name} = ({type}*)CPU_Pool.Alloc({size});'''.format(
                            name=t, size=array.total_size*array.dtype.bytes, type=array.dtype.ctype
                        )
                    )
                elif array.storage in self.gpu_storages:
                    callsite_stream.write(
                        '''{type} *{name} = ({type}*)GPU_Pool.Alloc({size});'''.format(
                            name=t, size=array.total_size * array.dtype.bytes,
                            type=array.dtype.ctype
                        )
                    )

            self.initialization = False
        self._dispatcher.defined_vars.add(node.label, DefinedType.Pointer)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        pass

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge, function_stream,
                    callsite_stream):
        self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge,
                                      function_stream, callsite_stream)

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream, callsite_stream):
        pass


def extend_dace():

    TargetCodeGenerator.register(MemoryPoolCodegen, name='memorypool')

