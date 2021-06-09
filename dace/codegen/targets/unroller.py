# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict, Tuple

from numpy import longlong
import dace
from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from itertools import product
from dace.sdfg import state
import dace.subsets
import dace.sdfg
from dace.sdfg import nodes as nd
from dace import Config
import dace.codegen.targets.common
from dace import dtypes, data as dt

def backup_statescope_fields(
        subgraph: 'dace.sdfg.state.StateGraphView') -> Tuple:
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
            setattr(node, pname, copy.deepcopy(propval))
    for edge in subgraph.edges():
        oldedgedata[edge] = edge.data.data
        edge.data.data = copy.deepcopy(edge.data.data)
        oldedgesubsets[edge] = (edge.data.subset, edge.data.other_subset)
        edge.data.subset = copy.deepcopy(edge.data.subset)
        edge.data.other_subset = copy.deepcopy(edge.data.other_subset)
        oldedgevolumes[edge] = edge.data.volume
        edge.data.volume = copy.deepcopy(edge.data.volume)
    return (oldproperties, oldedgedata, oldedgesubsets, oldedgevolumes)

def use_statescope_fields_backup(backup: "Tuple"):
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
        self._dispatcher.register_map_dispatcher(dace.ScheduleType.Unrolled,
                                                 self)

    def get_generated_codeobjects(self):
        return []

    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):

        entry_node = scope.source_nodes()[0]
        index_list = []

        #Generate new names for nsdfgs, and adds defined variables to constants
        def nsdfg_prepare_unroll(scope, paramname, paramval):
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
                if (isinstance(node, nd.NestedSDFG)):
                    if unique_functions and not unique_functions_hash and node.unique_name != "":
                        backup.append(
                            (node, True, copy.deepcopy(node.unique_name),
                             copy.deepcopy(node.symbol_mapping),
                             copy.deepcopy(node.sdfg.constants_prop)))
                        node.unique_name = f"{node.unique_name}_{param}{paramval}"
                    else:
                        backup.append(
                            (node, False, copy.deepcopy(node.sdfg.name),
                             copy.deepcopy(node.symbol_mapping),
                             copy.deepcopy(node.sdfg.constants_prop)))
                        node.sdfg.name = f"{node.sdfg.name}_{param}{paramval}"
                    for nstate in node.sdfg.nodes():
                        backup.extend(
                            nsdfg_prepare_unroll(nstate, paramname, paramval))
                    if param in node.symbol_mapping:
                        node.symbol_mapping.pop(param)
                    node.sdfg.add_constant(param, longlong(paramval))
            return backup

        def nsdfg_after_unroll(backup):
            for node, isUniqueName, name, symbols, constants in backup:
                if isUniqueName:
                    node.unique_name = name
                else:
                    node.sdfg.name = name
                node.symbol_mapping = symbols
                node.sdfg.constants_prop = constants

        for begin, end, stride in entry_node.map.range:
            l = []
            while begin <= end:
                l.append(begin)
                begin += stride
            index_list.append(l)

        sdfgconsts = sdfg.constants_prop
        sdfg.constants_prop = copy.deepcopy(sdfgconsts)

        for indices in product(*index_list):
            #backups = backup_statescope_fields(scope)
            callsite_stream.write('{')
            nsdfg_unroll_info = None
            for param, index in zip(entry_node.map.params, indices):
                #dace.sdfg.replace(scope, str(param), str(index))
                if nsdfg_unroll_info is None:
                    nsdfg_unroll_info = nsdfg_prepare_unroll(
                        scope, str(param), str(index))
                else:
                    nsdfg_prepare_unroll(scope, str(param), str(index))
                callsite_stream.write(
                    "constexpr %s %s = %s;\n" %
                    ('long long', param, dace.codegen.targets.common.sym2cpp(index)), sdfg)
                sdfg.add_constant(param, longlong(index))
            
            callsite_stream.write('{')
            self._dispatcher.dispatch_subgraph(
                sdfg,
                scope,
                state_id,
                function_stream,
                callsite_stream,
                skip_entry_node=True,
                skip_exit_node=True,
            )
            callsite_stream.write('}')
            callsite_stream.write('}')
            nsdfg_after_unroll(nsdfg_unroll_info)
            #use_statescope_fields_backup(backups)
            
        sdfg.constants_prop = sdfgconsts