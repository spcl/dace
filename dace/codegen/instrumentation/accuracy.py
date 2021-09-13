# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace import dtypes, registry, data
from dace.sdfg import state as st
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream


@registry.autoregister_params(type=dtypes.InstrumentationType.Accuracy)
class AccuracyProvider(InstrumentationProvider):
    """ Creates a Binary file at the end of every state 
        for each array in the relevant scope """
    def on_save_value(self, state_name: str, stream: CodeIOStream, sdfg, state):
        state_id = sdfg.node_id(state)

        # get the parent sdfg so the reports gets saved in the same folder
        parent_sdfg = sdfg
        while parent_sdfg.parent_sdfg is not None:
            parent_sdfg = parent_sdfg.parent_sdfg

        folderpath = os.path.join(parent_sdfg.build_folder, 'accuracy',
                                  parent_sdfg.hash_sdfg())
        if not os.path.exists(folderpath):
            os.makedirs(folderpath, exist_ok=True)

        for an in state.data_nodes():
            name = an.data
            array = sdfg.data(name)

            if (isinstance(array, data.Array)
                    and array.storage != dtypes.StorageType.Register):

                # The file path the data will be saved to
                filepath = os.path.join(
                    folderpath, name + '_' + str(sdfg.sdfg_id) + '_' +
                    str(state_id) + '.bin').replace('\\', '/')

                # Array length calculation as string, example: W * H
                array_length = str(array.sizes()[0])
                for size in array.sizes()[1::]:
                    array_length = array_length + ' * ' + str(size)

                if not dtypes.can_access(dtypes.ScheduleType.Default,
                                         array.storage):
                    continue

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
                           file_var='file_' + name + '_' + str(sdfg.sdfg_id) +
                           '_' + str(state_id)))

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Accuracy:
            self.on_save_value('State %s' % state.label, local_stream, sdfg,
                               state)
