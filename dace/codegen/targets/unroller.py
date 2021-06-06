# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from typing import Any, Tuple
import dace
from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from dace.codegen import cppunparse
from itertools import product

from dace.sdfg import replace, state as state
import dace.subsets

def backup_replacement_fields(subgraph: 'dace.sdfg.state.StateGraphView') -> Tuple:
    """
    Creates mappings from all nodes/edges to all the fields that are affected 
    by replace, before overwritting all those fields with deepcopies of themselves.
    The old fields can be restored using use_replacement_fields_backup
    (Maybe this is usefull to someone else and should go to replace?)
    """
    oldproperties = {}
    oldedgedata = {}
    oldedgesubsets = {}
    oldedgevolumes = {}

    for node in subgraph.nodes():
        for propclass, propval in node.properties():
            pname = propclass.attr_name
            oldproperties[node] = (pname, propval)
            setattr(node, pname, deepcopy(propval))
    for edge in subgraph.edges():
        oldedgedata[edge] = edge.data.data
        edge.data.data = deepcopy(edge.data.data)
        oldedgesubsets[edge] = (edge.data.subset, edge.data.other_subset)
        edge.data.subset = deepcopy(edge.data.subset)
        edge.data.other_subset = deepcopy(edge.data.other_subset)
        oldedgevolumes[edge] = edge.data.volume
        edge.data.volume = deepcopy(edge.data.volume)
    return (oldproperties, oldedgedata, oldedgesubsets, oldedgevolumes)

def use_replacement_fields_backup(subgraph : "dace.sdfg.state.StateGraphView",
                            backup : "Tuple"):
    """
    Restores values saved using backup_replacement_fields.
    """
    oldproperties, oldedgedata, oldedgesubsets, oldedgevolumes = backup
    for node in oldproperties:
        pname, v = oldproperties[node]
        setattr(node, pname, v)
    for edge, data in oldedgedata.items():
        edge.data.data = data
    for edge, subsets in oldedgesubsets.items():
        sub, osub = subsets
        edge.data.subset = sub
        edge.data.other_subset = osub
    for edge, volume in oldedgevolumes.items():
        edge.data.volume = volume

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
            backup = backup_replacement_fields(scope)
            callsite_stream.write('{')
            for param, index in zip(entry_node.map.params, indices):
                #callsite_stream.write(f'auto {param} = {sym2cpp(index)};')
                scope.replace(str(param), str(index))
            self._dispatcher.dispatch_subgraph(sdfg, scope, state_id,
                                            function_stream, callsite_stream,
                                            skip_entry_node=True,
                                            skip_exit_node=True)
            callsite_stream.write('}')
            use_replacement_fields_backup(scope, backup)
