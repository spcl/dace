# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict, Tuple

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
import dace.codegen.targets.common
from dace import dtypes, data as dt


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

    #Generate new names for nsdfgs, and adds defined variables to constants
    def nsdfg_prepare_unroll(self, scope: ScopeSubgraphView, paramname: str, paramval: str):
        backup = []
        for node in scope.nodes():
            if (isinstance(node, nd.NestedSDFG)):
                backup.append((node, node.unique_name, node.sdfg.name, node.symbol_mapping, node.sdfg.constants_prop))
                node.unique_name = copy.deepcopy(node.unique_name)
                node.sdfg.name = copy.deepcopy(node.sdfg.name)
                node.symbol_mapping = copy.deepcopy(node.symbol_mapping)
                node.sdfg.constants_prop = copy.deepcopy(node.sdfg.constants_prop)
                node.unique_name = f"{node.unique_name}_{paramname}{paramval}"
                node.sdfg.name = f"{node.sdfg.name}_{paramname}{paramval}"
                for nstate in node.sdfg.nodes():
                    backup.extend(self.nsdfg_prepare_unroll(nstate, paramname, paramval))
                if paramname in node.symbol_mapping:
                    node.symbol_mapping.pop(paramname)
                node.sdfg.add_constant(paramname, int(paramval))
        return backup

    def nsdfg_after_unroll(self, backup: "list[tuple[str, str, dict, dict]]"):
        for node, unique_name, name, symbols, constants in backup:
            node.unique_name = unique_name
            node.sdfg.name = name
            node.symbol_mapping = symbols
            node.sdfg.constants_prop = constants

    #TODO: Expand the unroller so it can also generate openCL code
    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView, state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):

        entry_node: nd.MapEntry = scope.source_nodes()[0]
        index_list = []

        for begin, end, stride in entry_node.map.range:
            l = []
            while begin <= end:
                l.append(begin)
                begin += stride
            index_list.append(l)

        sdfgconsts = sdfg.constants_prop
        sdfg.constants_prop = copy.deepcopy(sdfg.constants_prop)

        mapsymboltypes = entry_node.new_symbols(sdfg, scope, [entry_node.map.params])
        for indices in product(*index_list):
            callsite_stream.write('{')
            nsdfg_unroll_info = None
            for param, index in zip(entry_node.map.params, indices):
                if nsdfg_unroll_info is None:
                    nsdfg_unroll_info = self.nsdfg_prepare_unroll(scope, str(param), str(index))
                else:
                    self.nsdfg_prepare_unroll(scope, str(param), str(index))
                callsite_stream.write(
                    f"constexpr {mapsymboltypes[param]} {param} = "
                    f"{dace.codegen.targets.common.sym2cpp(index)};\n", sdfg)
                sdfg.add_constant(param, int(index))

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
            self.nsdfg_after_unroll(nsdfg_unroll_info)

        sdfg.constants_prop = sdfgconsts
