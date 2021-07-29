# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace import dtypes, registry
from dace.sdfg.nodes import AccessNode
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

        for _, name, array in sdfg.arrays_recursive():
            stream.write(
                '''myfile << "{sdfgName} StateId: {state_id} ArrayName: {name}\\n";'''
                .format(state_id=state_id, name=name, sdfgName=sdfg.name))
            self.write_array(array, name, stream)

    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        global_stream.write('''#include <iostream>
            #include <fstream>
            using namespace std;''')

        filename = os.path.join(sdfg.build_folder,
                                'accuracy_report.txt').replace('\\', '\\\\')
        local_stream.write('''ofstream myfile;
            myfile.open ("{file}");'''.format(file=filename))

        # For other file headers
        sdfg.append_global_code(
            '''#include <iostream>
            #include <fstream>''', None)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        local_stream.write('''myfile.close();''')

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Accuracy:
            self.on_save_value('State %s' % state.label, local_stream, sdfg,
                               state)
