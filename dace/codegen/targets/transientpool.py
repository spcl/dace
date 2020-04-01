import dace
from dace.codegen.targets.target import TargetCodeGenerator
from dace import registry
from dace.codegen.targets.framecode import DaCeCodeGenerator
import itertools
from dace import nodes

_MP_STORAGE_TYPES = ['CPU_Pool', 'GPU_Pool']


def extend_dace():
    # Register code generator
    TargetCodeGenerator.register(MPCodeGen, name='transient')

    # Register storage types
    for dtype in _MP_STORAGE_TYPES:
        dace.StorageType.register(dtype)

@registry.autoregister_params(name='transient')
class MPCodeGen(TargetCodeGenerator):


    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher

        # Register array allocation/deallocation
        for dtype in _MP_STORAGE_TYPES:
            enum_type = dace.StorageType[dtype]
            self._dispatcher.register_array_dispatcher(enum_type, self)

        # Register copies to/from tensor cores
        gpu_storages = [
            dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned,
            dace.StorageType.GPU_Shared, dace.StorageType.GPU_Stack,
            dace.StorageType.Register
        ]
        for src_storage, dst_storage in itertools.product(
                _MP_STORAGE_TYPES, gpu_storages):
            src_storage = dace.StorageType[src_storage]
            self._dispatcher.register_copy_dispatcher(src_storage, dst_storage,
                                                      None, self)
            self._dispatcher.register_copy_dispatcher(dst_storage, src_storage,
                                                      None, self)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)
        print('allocating: ', name)
        callsite_stream.write(
            'double *tmp = new double DACE_ALIGN(64)[{}];'.format(nodedesc.total_size)
        )

    def initialize_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        pass

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        pass

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream):
        src_desc = (src_node.desc(sdfg)
                    if isinstance(src_node, nodes.AccessNode) else None)
        if not src_desc:
            local_name = dfg.memlet_path(edge)[0].src_conn
            callsite_stream.write(
                'auto %s = %s;' % (local_name, dst_node.data), sdfg, state_id,
                [src_node, dst_node])
            return