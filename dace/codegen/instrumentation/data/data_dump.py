# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data as dt, dtypes, registry, SDFG
from dace.sdfg import nodes, is_devicelevel_gpu
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.scope import is_devicelevel_fpga
from dace.sdfg.state import SDFGState
from dace.codegen import common
from dace.codegen import cppunparse
from dace.codegen.targets import cpp
from dace.properties import CodeBlock
import os
import warnings
from typing import Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator


class DataInstrumentationProviderMixin:

    def _setup_gpu_runtime(self, sdfg: SDFG, global_stream: CodeIOStream):
        if self.gpu_runtime_init:
            return
        self.gpu_runtime_init = True
        self.backend = common.get_gpu_backend()
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

    def _generate_copy_to_device(self, node: nodes.AccessNode, desc: dt.Array, ptr: str) -> Tuple[str, str, str]:
        """ Copies restored data to device and returns (preamble, postamble, name of new host pointer). """
        new_ptr = f'__dinstr_{node.data}'
        new_desc = dt.Array(desc.dtype, [desc.total_size - desc.start_offset])
        csize = cpp.sym2cpp(desc.total_size - desc.start_offset)

        # Emit synchronous memcpy
        preamble = f'''
        {{
        {new_desc.as_arg(name=new_ptr)} = new {desc.dtype.ctype}[{csize}];
        '''

        postamble = f'''
        {self.backend}Memcpy({ptr}, {new_ptr}, sizeof({desc.dtype.ctype}) * ({csize}), {self.backend}MemcpyHostToDevice);
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

    def _save_var(self, varname: str, vardesc: dt.Data, sdfg: SDFG, local_stream: CodeIOStream,
                  global_stream: CodeIOStream, state: Optional[SDFGState] = None,
                  node: Optional[SDFGState] = None, filename: Optional[str] = None) -> None:
        from dace.codegen.dispatcher import DefinedType  # Avoid import loop

        if isinstance(vardesc, dt.Structure):
            for member_name, member_desc in vardesc.members.items():
                self._save_var(varname + '->' + member_name, member_desc, sdfg, local_stream, global_stream,
                               state, node, varname + '.' + member_name)
        else:
            ptrname = cpp.ptr(varname, vardesc, sdfg, self.codegen)
            try:
                defined_type, _ = self.codegen.dispatcher.defined_vars.get(ptrname)
            except KeyError:
                print('skipping saving of' + ptrname)
                return

            if defined_type == DefinedType.Scalar:
                ptrname = '&' + ptrname

            # Create UUID
            if node is not None:
                state_id = state.parent_graph.node_id(state)
                node_id = state.node_id(node)
                uuid = f'{sdfg.cfg_id}_{state_id}_{node_id}'
            elif state is not None:
                state_id = state.parent_graph.node_id(state)
                node_id = None
                uuid = f'{sdfg.cfg_id}_{state_id}'
            else:
                state_id = None
                node_id = None
                uuid = f'{sdfg.cfg_id}'

            # Get optional pre/postamble for instrumenting device data
            preamble, postamble = '', ''
            if vardesc.storage == dtypes.StorageType.GPU_Global and node is not None:
                self._setup_gpu_runtime(sdfg, global_stream)
                preamble, postamble, ptrname = self._generate_copy_to_host(node, vardesc, ptrname)

            # Encode runtime shape and strides
            shape = ', '.join(cpp.sym2cpp(s) for s in vardesc.shape)
            strides = ', '.join(cpp.sym2cpp(s) for s in vardesc.strides)

            if filename is None:
                filename = varname
            filename = filename.replace('->', '.')

            local_stream.write(preamble, sdfg, state_id, node_id)
            local_stream.write(
                f'__state->serializer->save({ptrname}, {cpp.sym2cpp(vardesc.total_size - vardesc.start_offset)}, '
                f'"{filename}", "{uuid}", {shape}, {strides});\n', sdfg, state_id, node_id)
            local_stream.write(postamble, sdfg, state_id, node_id)

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream,
                      codegen: 'DaCeCodeGenerator'):
        # Initialize serializer versioning object
        if sdfg.parent is None:
            self.codegen = codegen
            path = os.path.abspath(os.path.join(sdfg.build_folder, 'data')).replace('\\', '/')
            codegen.statestruct.append('dace::DataSerializer *serializer;')
            sdfg.append_init_code(f'__state->serializer = new dace::DataSerializer("{path}");\n')

            # If the SDFG is supposed to save or restore its initial state, do that here.
            if sdfg.save_restore_initial_state == dtypes.DataInstrumentationType.Save:
                for argname, desc in sdfg.arglist().items():
                    self._save_var(argname, desc, sdfg, local_stream, global_stream)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream):
        # Teardown serializer versioning object
        if sdfg.parent is None:
            sdfg.append_exit_code('delete __state->serializer;\n')

    def on_state_begin(self, sdfg: SDFG, state: SDFGState, local_stream: CodeIOStream, global_stream: CodeIOStream):
        if state.symbol_instrument == dtypes.DataInstrumentationType.No_Instrumentation:
            return

        condition_preamble, condition_postamble = '', ''
        condition: Optional[CodeBlock] = state.symbol_instrument_condition
        if condition is not None and not condition.as_string == '1':
            cond_string = None
            if condition.language == dtypes.Language.CPP:
                cond_string = condition.as_string
            elif condition.language == dtypes.Language.Python:
                cond_string = cppunparse.py2cpp(condition.code[0], expr_semicolon=False)
            else:
                warnings.warn('Unrecognized language %s in codeblock' % condition.language)
                cond_string = condition.as_string
            condition_preamble = f'if ({cond_string})' + ' {'
            condition_postamble = '}'

        state_id = sdfg.node_id(state)
        local_stream.write(condition_preamble, sdfg, state_id)
        defined_symbols = state.defined_symbols()
        for sym, _ in defined_symbols.items():
            local_stream.write(
                f'__state->serializer->save_symbol("{sym}", "{state_id}", {cpp.sym2cpp(sym)});\n', sdfg, state_id
            )
        local_stream.write(condition_postamble, sdfg, state_id)

    def on_node_end(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode, outer_stream: CodeIOStream,
                    inner_stream: CodeIOStream, global_stream: CodeIOStream):
        from dace.codegen.dispatcher import DefinedType  # Avoid import loop

        if is_devicelevel_gpu(sdfg, state, node) or is_devicelevel_fpga(sdfg, state, node):
            # Only run on host code
            return

        condition_preamble, condition_postamble = '', ''
        condition: Optional[CodeBlock] = node.instrument_condition
        if condition is not None and not condition.as_string == '1':
            cond_string = None
            if condition.language == dtypes.Language.CPP:
                cond_string = condition.as_string
            elif condition.language == dtypes.Language.Python:
                cond_string = cppunparse.py2cpp(condition.code[0], expr_semicolon=False)
            else:
                warnings.warn('Unrecognized language %s in codeblock' % condition.language)
                cond_string = condition.as_string
            condition_preamble = f'if ({cond_string})' + ' {'
            condition_postamble = '}'

        desc = node.desc(sdfg)

        # Create UUID
        state_id = state.parent_graph.node_id(state)
        node_id = state.node_id(node)

        # Write code
        inner_stream.write(condition_preamble, sdfg, state_id, node_id)
        self._save_var(node.data, desc, sdfg, inner_stream, global_stream, state, node)
        inner_stream.write(condition_postamble, sdfg, state_id, node_id)


@registry.autoregister_params(type=dtypes.DataInstrumentationType.Restore)
class RestoreProvider(InstrumentationProvider, DataInstrumentationProviderMixin):
    """ Data instrumentation that restores arrays from a file, generated by the ``Save`` data instrumentation type. """

    def __init__(self):
        super().__init__()
        self.gpu_runtime_init = False
        from dace.codegen.targets.framecode import DaCeCodeGenerator  # Avoid import loop
        self.codegen: DaCeCodeGenerator = None

    def _restore_var(self, varname: str, vardesc: dt.Data, sdfg: SDFG, local_stream: CodeIOStream,
                     global_stream: CodeIOStream, state: Optional[SDFGState] = None,
                     node: Optional[SDFGState] = None, filename: Optional[str] = None) -> None:
        from dace.codegen.dispatcher import DefinedType  # Avoid import loop

        if isinstance(vardesc, dt.Structure):
            for member_name, member_desc in vardesc.members.items():
                self._restore_var(varname + '->' + member_name, member_desc, sdfg, local_stream, global_stream,
                                  state, node, varname + '.' + member_name)
        else:
            ptrname = cpp.ptr(varname, vardesc, sdfg, self.codegen)
            try:
                defined_type, _ = self.codegen.dispatcher.defined_vars.get(ptrname)
            except KeyError:
                print('skipping restoring of' + ptrname)
                return

            if defined_type == DefinedType.Scalar:
                ptrname = '&' + ptrname

            # Create UUID
            if node is not None:
                state_id = state.parent_graph.node_id(state)
                node_id = state.node_id(node)
                uuid = f'{sdfg.cfg_id}_{state_id}_{node_id}'
            elif state is not None:
                state_id = state.parent_graph.node_id(state)
                node_id = None
                uuid = f'{sdfg.cfg_id}_{state_id}'
            else:
                state_id = None
                node_id = None
                uuid = f'{sdfg.cfg_id}'

            # Get optional pre/postamble for instrumenting device data
            preamble, postamble = '', ''
            if vardesc.storage == dtypes.StorageType.GPU_Global and node is not None:
                self._setup_gpu_runtime(sdfg, global_stream)
                preamble, postamble, ptrname = self._generate_copy_to_device(node, vardesc, ptrname)

            if filename is None:
                filename = varname
            filename = filename.replace('->', '.')

            # Write code
            local_stream.write(preamble, sdfg, state_id, node_id)
            local_stream.write(
                f'__state->serializer->restore({ptrname}, {cpp.sym2cpp(vardesc.total_size - vardesc.start_offset)}, '
                f'"{filename}", "{uuid}");\n', sdfg, state_id, node_id)
            local_stream.write(postamble, sdfg, state_id, node_id)

    def _generate_report_setter(self, sdfg: SDFG) -> str:
        return f'''
        DACE_EXPORTED void __dace_set_instrumented_data_report({cpp.mangle_dace_state_struct_name(sdfg)} *__state, const char *dirpath) {{
            __state->serializer->set_folder(dirpath);
        }}
        '''

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream,
                      codegen: 'DaCeCodeGenerator'):
        # Initialize serializer versioning object
        if sdfg.parent is None:
            self.codegen = codegen
            codegen.statestruct.append('dace::DataSerializer *serializer;')
            sdfg.append_init_code(f'__state->serializer = new dace::DataSerializer("");\n')

            # Add method that controls serializer input
            global_stream.write(self._generate_report_setter(sdfg))

            if sdfg.save_restore_initial_state == dtypes.DataInstrumentationType.Restore:
                for argname, desc in sdfg.arglist().items():
                    self._restore_var(argname, desc, sdfg, local_stream, global_stream)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream):
        # Teardown serializer versioning object
        if sdfg.parent is None:
            sdfg.append_exit_code('delete __state->serializer;\n')

    def on_state_begin(self, sdfg: SDFG, state: SDFGState, local_stream: CodeIOStream, global_stream: CodeIOStream):
        if state.symbol_instrument == dtypes.DataInstrumentationType.No_Instrumentation:
            return

        condition_preamble, condition_postamble = '', ''
        condition: Optional[CodeBlock] = state.symbol_instrument_condition
        if condition is not None and not condition.as_string == '1':
            cond_string = None
            if condition.language == dtypes.Language.CPP:
                cond_string = condition.as_string
            elif condition.language == dtypes.Language.Python:
                cond_string = cppunparse.py2cpp(condition.code[0], expr_semicolon=False)
            else:
                warnings.warn('Unrecognized language %s in codeblock' % condition.language)
                cond_string = condition.as_string
            condition_preamble = f'if ({cond_string})' + ' {'
            condition_postamble = '}'

        state_id = sdfg.node_id(state)
        local_stream.write(condition_preamble, sdfg, state_id)
        defined_symbols = state.defined_symbols()
        for sym, sym_type in defined_symbols.items():
            local_stream.write(
                f'{cpp.sym2cpp(sym)} = __state->serializer->restore_symbol<{sym_type.ctype}>("{sym}", "{state_id}");\n',
                sdfg, state_id
            )
        local_stream.write(condition_postamble, sdfg, state_id)

    def on_node_begin(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode, outer_stream: CodeIOStream,
                      inner_stream: CodeIOStream, global_stream: CodeIOStream):
        from dace.codegen.dispatcher import DefinedType  # Avoid import loop

        if is_devicelevel_gpu(sdfg, state, node) or is_devicelevel_fpga(sdfg, state, node):
            # Only run on host code
            return

        condition_preamble, condition_postamble = '', ''
        condition: Optional[CodeBlock] = node.instrument_condition
        if condition is not None and not condition.as_string == '1':
            cond_string = None
            if condition.language == dtypes.Language.CPP:
                cond_string = condition.as_string
            elif condition.language == dtypes.Language.Python:
                cond_string = cppunparse.py2cpp(condition.code[0], expr_semicolon=False)
            else:
                warnings.warn('Unrecognized language %s in codeblock' % condition.language)
                cond_string = condition.as_string
            condition_preamble = f'if ({cond_string})' + ' {'
            condition_postamble = '}'

        desc = node.desc(sdfg)

        state_id = sdfg.node_id(state)
        node_id = state.node_id(node)

        # Write code
        inner_stream.write(condition_preamble, sdfg, state_id, node_id)
        self._restore_var(node.data, desc, sdfg, inner_stream, global_stream, state, node)
        inner_stream.write(condition_postamble, sdfg, state_id, node_id)
