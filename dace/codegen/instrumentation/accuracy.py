# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace import dtypes, registry
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream


@registry.autoregister_params(type=dtypes.InstrumentationType.Accuracy)
class AccuracyProvider(InstrumentationProvider):
    """ """
    def write_array(self, array, name, stream):
        sizes = array.sizes()
        stream.write('''myfile << "[";''')
        for i, size in enumerate(sizes):
            stream.write(
                '''for(int dace__count_{i} = 0; dace__count_{i} < {size}; dace__count_{i}++){{'''
                .format(i=i, size=size))

        arrayBrackets = '['
        for i, _ in enumerate(sizes):
            if i != 0:
                arrayBrackets = arrayBrackets + ' + '
            arrayBrackets = arrayBrackets + '''(dace__count_{i}'''.format(i=i)
            for j, _ in list(enumerate(sizes))[::-1]:
                if i == j:
                    break
                arrayBrackets = arrayBrackets + ''' * dace__count_{j}'''.format(
                    j=j)
            arrayBrackets = arrayBrackets + ')'
        arrayBrackets = arrayBrackets + ']'

        stream.write('''myfile << {name}{arrayBrackets} << ", ";'''.format(
            name=name, arrayBrackets=arrayBrackets))

        for size in sizes:
            stream.write('}')
        stream.write('myfile << "]\\n";')

    def on_save_value(self,
                      state_name: str,
                      stream: CodeIOStream,
                      sdfg=None,
                      state=None,
                      node=None):
        state_id = -1
        if state is not None:
            state_id = sdfg.node_id(state)

        # get the parent sdfg so the reports gets saved in the same folder
        parent_sdfg = sdfg
        while parent_sdfg.parent_sdfg is not None:
            parent_sdfg = parent_sdfg.parent_sdfg

        folderpath = os.path.join(parent_sdfg.build_folder, 'accuracy',
                                  parent_sdfg.hash_sdfg())
        if not os.path.exists(folderpath):
            os.makedirs(folderpath, exist_ok=True)

        for name, array in sdfg.arrays.items():
            filepath = os.path.join(folderpath, name + '_' + str(state_id) +
                                    '.bin').replace('\\', '/')

            # Calculation for the array length, example: W * H
            array_length = str(array.sizes()[0])
            for size in array.sizes()[1::]:
                array_length = array_length + ' * ' + str(size)

            if not array.transient:
                array_access = name
            else:
                array_access = '''__state->__{sdfg_id}_{arrayname}'''.format(
                    sdfg_id=sdfg.sdfg_id, arrayname=name)

            stream.write('''
                const void *_buffer_{arrayname};
                _buffer_{arrayname} = {array_access};
                FILE *{file_var} = fopen("{filepath}", "wb");
                fwrite(_buffer_{arrayname}, sizeof({array_access}[0]), {array_length}, {file_var});
                fclose({file_var});
            '''.format(filepath=filepath,
                       arrayname=name,
                       array_access=array_access,
                       array_length=array_length,
                       file_var='file_' + name + '_' + str(state_id)))

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Accuracy:
            self.on_save_value('State %s' % state.label, local_stream, sdfg,
                               state)
