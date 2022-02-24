# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import config, data as dt, dtypes, registry, SDFG
from dace.sdfg import nodes, is_devicelevel_gpu
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.scope import is_devicelevel_fpga
from dace.sdfg.state import SDFGState
from dace.codegen.targets import cpp
import os
from typing import Tuple


class DataInstrumentationProviderMixin:
    def _setup_gpu_runtime(self, sdfg: SDFG, global_stream: CodeIOStream):
        if self.gpu_runtime_init:
            return
        self.gpu_runtime_init = True
        self.backend = config.Config.get('compiler', 'cuda', 'backend')
        if self.backend == 'cuda':
            header_name = 'cuda_runtime.h'
        elif self.backend == 'hip':
            header_name = 'hip/hip_runtime.h'
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)

        global_stream.write('#include <%s>' % header_name)

        # For other file headers
        sdfg.append_global_code('\n#include <%s>' % header_name, None)

    def _generate_copy_to_host(self, node: nodes.AccessNode, desc: dt.Array, ptr: str) -> Tuple[str, str, str]:
        """ Copies to host and returns (preamble, postamble, name of new host pointer). """
        new_ptr = f'__dinstr_{node.data}'
        new_desc = dt.Array(desc.dtype, [desc.total_size - desc.start_offset])
        csize = cpp.sym2cpp(desc.total_size - desc.start_offset)

        # Emit synchronous memcpy
        preamble = f'''
        {{
        {new_desc.as_arg(name=new_ptr)} = new {desc.dtype.ctype}[{csize}];
        {self.backend}Memcpy({new_ptr}, {ptr}, sizeof({desc.dtype.ctype}) * ({csize}), {self.backend}MemcpyDeviceToHost);
        '''

        postamble = f'''
        delete[] {new_ptr};
        }}
        '''

        return preamble, postamble, new_ptr


@registry.autoregister_params(type=dtypes.DataInstrumentationType.Save)
class SaveProvider(InstrumentationProvider, DataInstrumentationProviderMixin):
    """ Data instrumentation code generator that stores arrays to a file. """
    def __init__(self):
        super().__init__()
        self.gpu_runtime_init = False
        from dace.codegen.targets.framecode import DaCeCodeGenerator  # Avoid import loop
        self.codegen: DaCeCodeGenerator = None

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream,
                      codegen: 'DaCeCodeGenerator'):
        # Initialize serializer versioning object
        if sdfg.parent is None:
            self.codegen = codegen
            path = os.path.abspath(os.path.join(sdfg.build_folder, 'data'))
            codegen.statestruct.append('dace::DataSerializer *serializer;')
            sdfg.append_init_code(f'__state->serializer = new dace::DataSerializer("{path}");\n')

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream):
        # Teardown serializer versioning object
        if sdfg.parent is None:
            sdfg.append_exit_code('delete __state->serializer;\n')

    def on_node_end(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode, outer_stream: CodeIOStream,
                    inner_stream: CodeIOStream, global_stream: CodeIOStream):
        from dace.codegen.dispatcher import DefinedType  # Avoid import loop

        if is_devicelevel_gpu(sdfg, state, node) or is_devicelevel_fpga(sdfg, state, node):
            # Only run on host code
            return

        desc = node.desc(sdfg)

        # Obtain a pointer for arrays and scalars
        ptrname = cpp.ptr(node.data, desc, sdfg)
        defined_type, _ = self.codegen.dispatcher.defined_vars.get(node.data)
        if defined_type == DefinedType.Scalar:
            ptrname = '&' + ptrname

        # Create UUID
        state_id = sdfg.node_id(state)
        node_id = state.node_id(node)
        uuid = f'{sdfg.sdfg_id}_{state_id}_{node_id}'

        # Get optional pre/postamble for instrumenting device data
        preamble, postamble = '', ''
        if desc.storage == dtypes.StorageType.GPU_Global:
            self._setup_gpu_runtime(sdfg, global_stream)
            preamble, postamble, ptrname = self._generate_copy_to_host(node, desc, ptrname)

        # Write code
        inner_stream.write(preamble, sdfg, state_id, node_id)
        inner_stream.write(
            f'__state->serializer->save({ptrname}, {cpp.sym2cpp(desc.total_size - desc.start_offset)}, '
            f'"{node.data}", "{uuid}");\n', sdfg, state_id, node_id)
        inner_stream.write(postamble, sdfg, state_id, node_id)


@registry.autoregister_params(type=dtypes.DataInstrumentationType.Restore)
class RestoreProvider(InstrumentationProvider, DataInstrumentationProviderMixin):
    """ Data instrumentation that restores arrays from a file, generated by the ``Save`` data instrumentation type. """
    def __init__(self):
        super().__init__()
        self.gpu_runtime_init = False
        from dace.codegen.targets.framecode import DaCeCodeGenerator  # Avoid import loop
        self.codegen: DaCeCodeGenerator = None

