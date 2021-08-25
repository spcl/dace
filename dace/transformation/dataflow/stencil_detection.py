# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes for detecting stencils """

from copy import deepcopy
from numbers import Number
from typing import Dict, List
import dace
import sympy
from dace import data, dtypes, registry, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as pm


@registry.autoregister_params(singlestate=True)
class StencilDetection(pm.Transformation):
    """ Detects Maps that perform stencil operations and substitutes them with
        a StencilNode.
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(StencilDetection.map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):

        map_entry = graph.node(candidate[StencilDetection.map_entry])
        map_exit = graph.exit_node(map_entry)
        params = [dace.symbol(p) for p in map_entry.map.params]

        inputs = dict()
        for _, _, _, _, m in graph.out_edges(map_entry):
            if not m.data:
                continue
            desc = sdfg.arrays[m.data]
            if desc not in inputs.keys():
                inputs[desc] = []
            inputs[desc].append(m.subset)

        stencil_found = False
        for desc, accesses in inputs.items():
            if isinstance(desc, dace.data.Scalar):
                continue
            elif isinstance(desc, (dace.data.Array, dace.data.View)):
                if list(desc.shape) == [1]:
                    continue
                first_access = None
                for a in accesses:
                    if a.num_elements() != 1:
                        return False
                    if first_access:
                        new_access = deepcopy(a)
                        new_access.offset(first_access, True)
                        for idx in new_access.min_element():
                            if not isinstance(idx, Number):
                                return False
                            if idx != 0:
                                stencil_found = True
                    else:
                        first_access = a
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if isinstance(idx, sympy.Symbol):
                            bidx = idx
                        elif isinstance(idx, sympy.Add):
                            if len(idx.free_symbols) != 1:
                                return False
                            bidx = list(idx.free_symbols)[0]
                        else:
                            return False
                        if bidx in unmatched_indices:
                            unmatched_indices.remove(bidx)
                    if len(unmatched_indices) > 0:
                        return False
            else:
                return False

        outputs = dict()
        for _, _, _, _, m in graph.in_edges(map_exit):
            if m.wcr:
                return False
            desc = sdfg.arrays[m.data]
            if desc not in outputs.keys():
                outputs[desc] = []
            outputs[desc].append(m.subset)

        for desc, accesses in outputs.items():
            if isinstance(desc, (dace.data.Array, dace.data.View)):
                for a in accesses:
                    if a.num_elements() > 1:
                        return False
                    indices = a.min_element()
                    unmatched_indices = set(params)
                    for idx in indices:
                        if isinstance(idx, sympy.Symbol):
                            bidx = idx
                        elif isinstance(idx, sympy.Add):
                            if len(idx.free_symbols) != 1:
                                return False
                            bidx = list(idx.free_symbols)[0]
                        else:
                            return False
                        if bidx in unmatched_indices:
                            unmatched_indices.remove(bidx)
                    if len(unmatched_indices) > 0:
                        return False
            else:
                return False

        map_scope = graph.scope_subgraph(map_entry)
        if len(map_scope.nodes()) > 3:
            return False

        return stencil_found

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        map_entry = graph.node(candidate[StencilDetection.map_entry])
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        state = sdfg.nodes()[self.state_id]
        map_entry = state.nodes()[self.subgraph[StencilDetection.map_entry]]
        map_exit = state.exit_node(map_entry)
        map_scope = state.scope_subgraph(map_entry)
        tasklet = next(n for n in map_scope.nodes()
                       if isinstance(n, nodes.Tasklet))

        # Find offsets
        offsets = [None] * len(map_entry.map.params)
        for i, p in enumerate(map_entry.map.params):
            sp = symbolic.pystr_to_symbolic(p)
            values = set()
            for e in state.out_edges(tasklet):
                sset = e.data.subset
                if isinstance(sset, subsets.Range):
                    for rng in sset:
                        assert (rng[0] == rng[1])
                        if sp in rng[0].free_symbols:
                            nval = rng[0].subs(sp, 0)
                            try:
                                nval = int(str(nval))
                                values.add(nval)
                            except:
                                pass
                elif isinstance(sset, subsets.Indices):
                    for idx in sset:
                        if sp in idx.free_symbols:
                            nval = idx.subs(sp, 0)
                            try:
                                nval = int(str(nval))
                                values.add(nval)
                            except:
                                pass
            if len(values) > 0:
                offsets[i] = min(values)
        print(offsets)

        # Get initial replacement dictionary
        rdict = {}
        for i, (p,
                r) in enumerate(zip(map_entry.map.params, map_entry.map.range)):
            if offsets[i]:
                rdict[symbolic.pystr_to_symbolic(
                    p)] = symbolic.pystr_to_symbolic(f'{r[0]} - {offsets[i]}')
            else:
                rdict[symbolic.pystr_to_symbolic(p)] = r[0]

        from dace.libraries.stencil import Stencil

        code = tasklet.code.as_string
        in_data = {}
        in_conns = set()
        out_data = {}
        out_conns = set()

        for e in state.in_edges(tasklet):
            print(e.data)
            sset = e.data.subset
            sset.replace(rdict)
            print(e.data)
            conn = f'__{e.data.data}'
            in_conns.add(conn)
            in_data[e.data.data] = conn
            code = code.replace(e.dst_conn, f'{conn}[{sset}]')

        for e in state.out_edges(tasklet):
            print(e.data)
            sset = e.data.subset
            sset.replace(rdict)
            print(e.data)
            conn = f'__{e.data.data}'
            out_conns.add(conn)
            out_data[e.data.data] = conn
            code = code.replace(e.src_conn, f'{conn}[{sset}]')

        stencil_node = Stencil(
            f'{map_entry.label}_stencil',
            code,
            # TODO: Generalize this
            iterator_mapping={
                '__A': (True, ),
                'B': (True, )
            },
            boundary_conditions={'__B': {
                'btype': 'shrink'
            }},
            # in_connectors=in_conns,
            # out_connectors=out_conns
        )
        state.add_node(stencil_node)

        inputs = {}
        outputs = {}
        for e in state.in_edges(map_entry):
            if e.data.data in in_data:
                inputs[e.data.data] = e.src
        for e in state.out_edges(map_exit):
            if e.data.data in out_data:
                outputs[e.data.data] = e.dst

        state.remove_nodes_from(map_scope.nodes())
        for d, c in in_data.items():
            state.add_memlet_path(inputs[d],
                                  stencil_node,
                                  memlet=dace.Memlet.from_array(
                                      d, sdfg.arrays[d]),
                                  dst_conn=c)
        for d, c in out_data.items():
            state.add_memlet_path(stencil_node,
                                  outputs[d],
                                  memlet=dace.Memlet.from_array(
                                      d, sdfg.arrays[d]),
                                  src_conn=c)
