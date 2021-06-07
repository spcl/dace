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
from dace.sdfg import state
import dace.subsets
import dace.sdfg
from dace.sdfg.replace import deepreplace
from dace.sdfg import utils
from dace.sdfg import nodes as nd
from dace import Config

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

def use_replacement_fields_backup(backup : "Tuple"):
    """
    Restores values saved using backup_replacement_fields. Will apply
    the changes to the nodes/edges objects it saw during backup, in other
    words those must still exist.
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

def deep_replacement_field_backup(subgraph: 'dace.sdfg.state.StateGraphView') -> "list[Tuple]":
    """
    Same as replacement_field_backup, but saves also inside of NestedSDFG's
    """
    backups = []
    backups.append(backup_replacement_fields(subgraph))
    for node in subgraph.nodes():
        if(isinstance(node, nd.NestedSDFG)):
            for ns in node.sdfg.nodes():
                backups.extend(deep_replacement_field_backup(ns))
    return backups

def use_deep_replacement_fields_backup(backups: "list[Tuple]"):
    """
    Counterpart to deep_replacement_field_backup,restores all values saved by it.
    """
    for backup in backups:
        use_replacement_fields_backup(backup)

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

        #Generate new names for nsdfgs, as they will be generated multiple times
        def nsdfg_replace_name(scope, paramname, paramval):
            unique_functions_conf = Config.get('compiler', 'unique_functions')
            if unique_functions_conf is True:
                unique_functions_conf = 'hash'
            elif unique_functions_conf is False:
                unique_functions_conf = 'none'
            if unique_functions_conf == 'hash':
                unique_functions = True
                unique_functions_hash = True
            elif unique_functions_conf == 'unique_name':
                unique_functions = True
                unique_functions_hash = False
            elif unique_functions_conf == 'none':
                unique_functions = False
            else:
                raise ValueError(
                    f'Unknown unique_functions configuration: {unique_functions_conf}'
                )
            backup = []
            for node in scope.nodes():
                if(isinstance(node, nd.NestedSDFG)):
                    if unique_functions and not unique_functions_hash and node.unique_name != "":
                        backup.append((node, True, node.unique_name))
                        node.unique_name = f"{node.unique_name}_{param}{paramval}"
                    else:
                        backup.append((node, False, node.sdfg.name))
                        node.sdfg.name = f"{node.sdfg.name}_{param}{paramval}"
                    for nstate in node.sdfg.nodes():
                        backup.extend(nsdfg_replace_name(nstate, paramname, paramval))
            return backup

        def nsdfg_restore_name(backup):
            for node, isUniqueName, value in backup:
                if isUniqueName:
                    node.unique_name = value
                else:
                    node.sdfg.name = value

        for begin, end, stride in entry_node.map.range:
            l = []
            while begin <= end:
                l.append(begin)
                begin += stride
            index_list.append(l)

        for indices in product(*index_list):
            backups = deep_replacement_field_backup(scope)
            callsite_stream.write('{')
            nsdfg_names = None
            for param, index in zip(entry_node.map.params, indices):
                #callsite_stream.write(f'auto {param} = {sym2cpp(index)};')
                deepreplace(scope, str(param), str(index))
                if nsdfg_names is None:
                    nsdfg_names = nsdfg_replace_name(scope, str(param), str(index))
                else:
                    nsdfg_replace_name(scope, str(param), str(index))
            #sdfg.view()
            self._dispatcher.dispatch_subgraph(sdfg, scope, state_id,
                                            function_stream, callsite_stream,
                                            skip_entry_node=True,
                                            skip_exit_node=True)
            callsite_stream.write('}')
            nsdfg_restore_name(nsdfg_names)
            use_deep_replacement_fields_backup(backups)
