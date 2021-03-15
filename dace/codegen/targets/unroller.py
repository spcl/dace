# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from dace.codegen import cppunparse
from itertools import product


@registry.autoregister_params(name='unroll')
class UnrollCodeGen(TargetCodeGenerator):
    """ A constant-range map unroller code generator. """
    target_name = 'unroll'
    title = 'Unrolled'
    language = 'cpp'

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        # Register dispatchers
        self._dispatcher.register_map_dispatcher(dace.ScheduleType.Unrolled, self)

    def get_generated_codeobjects(self):
        return []

    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        
        entry_node = scope.source_nodes()[0]

        index_list = []
        for begin, end, stride in entry_node.map.range:
            l = []
            while begin <= end:
                l.append(begin)
                begin += stride
            index_list.append(l)

        for indices in product(*index_list):
            callsite_stream.write('{')
            for param, index in zip(entry_node.map.params, indices):
                callsite_stream.write(f'auto {param} = {sym2cpp(index)};')
            self._dispatcher.dispatch_subgraph(sdfg, scope, state_id,
                                            function_stream, callsite_stream,
                                            skip_entry_node=True,
                                            skip_exit_node=True)
            callsite_stream.write('}')
