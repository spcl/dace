""" This file implements the RegisterScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.dataflow import DeduplicateAccess
from dace.properties import make_properties, Property, TypeProperty
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView
from dace.sdfg.nodes import Node, Tasklet
from dace.config import Config
from dace.memlet import Memlet

import dace.sdfg.propagation as propagation
import dace.sdfg.nodes as nodes
import dace.symbolic as symbolic
import dace.dtypes as dtypes
import dace.subsets as subsets
import numpy as np
import sympy
import ast
import math
import networkx as nx

from typing import Set, Union, List, Callable, Dict, Type, Iterable

from dace.transformation.estimator.scoring import ScoringFunction, MemletScore

import json
import warnings
import os
import functools
import itertools
import collections
import sys


@make_properties
class RegisterScore(MemletScore):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''

    register_per_block = Property(desc="No of Registers per Block available",
                                  dtype=int,
                                  default=65536)
    #default=32768)

    max_registers_allocated = Property(
        desc="Maximum amount of registers available per thread",
        dtype=int,
        default=50)

    datatype = TypeProperty(desc="Datatype of Input", default=np.float32)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 io: Dict,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 transformation_function: Type = CompositeFusion,
                 **kwargs):

        super().__init__(sdfg=sdfg,
                         graph=graph,
                         subgraph=subgraph,
                         io=io,
                         gpu=gpu,
                         transformation_function=transformation_function,
                         **kwargs)

        self._datatype_multiplier = dtypes._BYTES[self.datatype] // 4

        # for debugging purposes
        self._i = 0

    def propagate_outward(self,
                          graph: SDFGState,
                          memlets: List[Memlet],
                          context: List[Union[nodes.MapEntry,
                                              nodes.NestedSDFG]],
                          penetrate_everything=False):
        '''
        Propagates Memlets outwards given map entry nodes in context vector. 
        Assumes that first entry is innermost entry closest to memlet 
        '''
        for ctx in context:
            if isinstance(ctx, nodes.MapEntry):
                if not penetrate_everything and ctx.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                    # we are at thread block level - do not propagate
                    break
                for i, memlet in enumerate(memlets):
                    memlets[i] = propagation.propagate_memlet(
                        graph, memlet, ctx, False)
            else:
                raise NotImplementedError("TODO")

        return memlets

    def tasklet_symbols(self, astobj, max_size=12):
        '''
        traverses a tasklet ast and primitively assigns 
        registers to symbols and constants 
        '''

        assignments = {}
        base_multiplier = dtypes._BYTES[self.datatype] / 4
        for n in ast.walk(astobj):
            if isinstance(n, ast.Name):
                assignments[n.id] = 1 * base_multiplier
            if isinstance(n, ast.Constant):
                assignments[n] = 1 * base_multiplier

        return assignments

    def evaluate_register_traffic(self,
                                  sdfg: SDFG,
                                  graph: SDFGState,
                                  scope_node: nodes.MapEntry,
                                  discarded_register_nodes: Set,
                                  scope_dict: dict = None):
        ''' 
        Evaluates traffic in a scope subgraph with given scope 
        node that flows out of and into registers 
        '''

        if not scope_dict:
            scope_dict = graph.scope_dict()

        register_traffic = 0

        for node in (graph.scope_subgraph(scope_node)
                     if scope_node else graph.nodes()):
            if node in discarded_register_nodes:
                # see whether scope is contained in outer_entry
                scope = scope_dict[node]
                current_context = []
                while scope: #and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                #assert scope == scope_node
                # add to traffic
                memlets = list(e.data for e in itertools.chain(
                    graph.out_edges(node), graph.in_edges(node)))
                
                self.propagate_outward(graph,
                                       memlets,
                                       current_context,
                                       penetrate_everything=True)
                for memlet in memlets:
                    register_traffic += memlet.volume

            if isinstance(node, nodes.NestedSDFG):
                for state in node.sdfg.nodes():
                    register_traffic += self.evaluate_register_traffic(
                        node.sdfg, state, None, discarded_register_nodes)

        return self.symbolic_evaluation(register_traffic)

    def evaluate_tasklet_output(self, sdfg: SDFG, graph: SDFGState,
                                node: Union[nodes.Tasklet, nodes.NestedSDFG],
                                outside_register_arrays: Set,
                                discarded_register_nodes: Set):

        result = 0
        if isinstance(node, nodes.Tasklet):
            for e in graph.out_edges(node):
                # see whether path comes from global map entry
                for path_e in graph.memlet_tree(e):
                    #if isinstance(path_e.dst, nodes.AccessNode) and path_e.dst.data in discarded_register_nodes:

                    if isinstance(path_e.dst, nodes.AccessNode):
                        # if data in outside_register_arrays: break immediately
                        if path_e.dst.data in outside_register_arrays:
                            break
                        # exclude rest of the actual register arrays
                        if sdfg.data(path_e.dst.data
                                     ).storage == dtypes.StorageType.Register:
                            if path_e.dst not in discarded_register_nodes:
                                break
                        # if discarded -> legit

                else:
                    # FORNOW: Propagation is still not working correctly
                    # use this as a fix
                    result_subset = self.symbolic_evaluation(
                        e.data.subset.num_elements())
                    result_volume = self.symbolic_evaluation(e.data.volume)
                    result += min(result_subset,
                                  result_volume) * self._datatype_multiplier

        return self.symbolic_evaluation(result) * self._datatype_multiplier

    def evaluate_tasklet_input(self, sdfg: SDFG, graph: SDFGState,
                               node: Union[nodes.Tasklet, nodes.NestedSDFG],
                               outside_register_arrays: Set,
                               discarded_register_nodes: Set):

        result = 0
        if isinstance(node, nodes.Tasklet):
            for e in graph.in_edges(node):
                # see whether path comes from register node
                # if so it already got covered in output
                for path_e in graph.memlet_path(e):
                    if isinstance(path_e.src, nodes.AccessNode):
                        if path_e.src.data in outside_register_arrays:
                            break
                        if sdfg.data(path_e.src.data
                                     ).storage == dtypes.StorageType.Register:
                            if path_e.src not in discarded_register_nodes:
                                break

                else:
                    result += e.data.subset.num_elements()

        return self.symbolic_evaluation(result) * self._datatype_multiplier

    def evaluate_tasklet_inner(self,
                               sdfg: SDFG,
                               graph: SDFGState,
                               node: Union[nodes.Tasklet, nodes.NestedSDFG],
                               discarded_register_nodes: Set,
                               array_register_size: int,
                               max_size=8):
        '''
        Evaluates a tasklet or nested sdfg node for a register size usage estimate 

        :param node:    tasklet or nsdfg node 
        :param context: list of map entries to propagate thru ending with toplevel 
        :param nested:  specify whether we are in a nested sdfg
        '''

        estimate_internal = 0
        if isinstance(node, nodes.Tasklet):
            if node.code.language == dtypes.Language.Python:
                names = set(n.id
                            for n in ast.walk(ast.parse(node.code.as_string))
                            if isinstance(n, ast.Name))
                names = self.tasklet_symbols(ast.parse(node.code.as_string))
                connector_symbols = set(
                    itertools.chain(node.in_connectors.keys(),
                                    node.out_connectors.keys()))
                estimate_internal = sum([
                    v for (k, v) in names.items() if k not in connector_symbols
                ])

            else:
                warnings.warn(
                    'WARNING: Register Score cannot evaluate non Python block')
                # just use some shady estimate
                estimate_internal = min(
                    8, math.ceil(node.code.as_string.count('\n') / 2))
        else:
            # nested sdfg
            # get an estimate for each nested sdfg state and max over these values

            # first add all connectors that belong to a register location to our register set

            known_registers = set()

            for e in graph.in_edges(node):
                for path_e in graph.memlet_path(e):
                    if isinstance(path_e.src, nodes.AccessNode):
                        if sdfg.data(path_e.src.data
                                     ).storage == dtypes.StorageType.Register:
                            if path_e.src not in discarded_register_nodes:
                                known_registers.add(e.dst_conn)
                                break

            for e in graph.out_edges(node):
                for path_e in graph.memlet_path(e):
                    if isinstance(path_e.dst, nodes.AccessNode):
                        if sdfg.data(path_e.dst.data
                                     ).storage == dtypes.StorageType.Register:
                            if path_e.dst not in discarded_register_nodes:
                                known_registers.add(e.src_conn)
                                break

            # NOTE: would have to extend unused_registers here
            # for simplicity I did not do this
            estimate_internal = max([
                self.evaluate_state(sdfg = node.sdfg, 
                                    graph = s, 
                                    scope_node = None, 
                                    discarded_register_nodes = discarded_register_nodes,
                                    outside_register_arrays = known_registers)[0]
                for s in node.sdfg.nodes()
            ])
            if estimate_internal == 0:
                warnings.warn(
                    'Detected a Nested SDFG where Tasklets were found inside for analysis'
                )

        estimate_internal = self.symbolic_evaluation(estimate_internal)

       
        return min(estimate_internal, max_size)

    def evaluate_state(self,
                       sdfg: SDFG,
                       graph: SDFGState,
                       scope_node: Node,
                       discarded_register_nodes: Set,
                       outside_register_arrays: Set,
                       outside_register_array_size=0,
                       scope_dict: dict = None,
                       scope_children: dict = None):
        '''
        Evaluates Register spill for a whole state where scope_node i
        ndicates the outermost scope in which spills should be analyzed 
        (within its inner scopes as well)
        If scope_node is None, then the whole graph will get analyzed.
        '''

        # get some variables if they haven not already been inputted
        if scope_dict is None:
            scope_dict = graph.scope_dict()
        if scope_children is None:
            scope_children = graph.scope_children()

        subgraph = graph.scope_subgraph(
            scope_node) if scope_node is not None else SubgraphView(
                graph, graph.nodes())

        # 1. Create a proxy graph to determine the topological order of all tasklets
        #    that reside inside our graph

        # loop over all tasklets
        context = dict()
        for node in subgraph.nodes():
            if isinstance(node, (nodes.Tasklet, nodes.NestedSDFG)):
                # see whether scope is contained in outer_entry
                scope = scope_dict[node]
                current_context = []
                while scope and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                assert scope == scope_node
                context[node] = current_context

        # similarly as in transient_reuse, create a proxy graph only
        # containing tasklets that are inside our fused map
        proxy = nx.MultiDiGraph()
        for n in subgraph.nodes():
            proxy.add_node(n)
        for n in subgraph.nodes():
            for e in graph.all_edges(n):
                if e.src in subgraph and e.dst in subgraph:
                    proxy.add_edge(e.src, e.dst)

        # remove all nodes that are not Tasklets
        for n in subgraph.nodes():
            if n not in context:
                for p in proxy.predecessors(n):
                    for c in proxy.successors(n):
                        proxy.add_edge(p, c)
                proxy.remove_node(n)

        # set up predecessor and successor array
        predecessors, successors = {}, {}
        for n in proxy.nodes():
            successors[n] = set(proxy.successors(n))
            predecessors[n] = set(proxy.predecessors(n))

        # 2. Create a list of active sets that represent the maximum
        #    number of active units of execution at a given timeframe.
        #    This serves as estimate later

        active_sets = list()
        for cc in nx.weakly_connected_components(proxy):
            queue = set([node for node in cc if len(predecessors[node]) == 0])
            while len(queue) > 0:
                active_sets.append(set(queue))
                for tasklet in queue:
                    # remove predecessor link (for newly added elements to queue)
                    for tasklet_successor in successors[tasklet]:
                        if tasklet in predecessors[tasklet_successor]:
                            predecessors[tasklet_successor].remove(tasklet)

                # if all predecessor in degrees in successors are 0
                # then tasklet is ready to be removed from the queue
                next_tasklets = set()
                tasklets_to_remove = set()
                for tasklet in queue:
                    remove_tasklet = True
                    for tasklet_successor in successors[tasklet]:
                        if len(predecessors[tasklet_successor]) > 0:
                            remove_tasklet = False
                        elif tasklet_successor not in queue:
                            next_tasklets.add(tasklet_successor)
                    if remove_tasklet:
                        tasklets_to_remove.add(tasklet)

                if len(next_tasklets) == 0 and len(tasklets_to_remove) == 0:
                    sdfg.save('error_here.sdfg')
                    print("ERROR")
                    sys.exit(0)
                queue |= next_tasklets
                queue -= tasklets_to_remove

                assert len(next_tasklets & tasklets_to_remove) == 0

        # 3. Determine all new arrays that reside inside the subgraph and that are
        #    allocated to registers.
        register_arrays = dict()
        for node in subgraph.nodes():
            if isinstance(node, nodes.AccessNode) and sdfg.data(
                    node.data).storage == dtypes.StorageType.Register:
                if node.data not in (register_arrays.keys()
                                     | outside_register_arrays):
                    register_arrays[node.data] = sdfg.data(node.data).total_size

        # 5. For each tasklet, determine inner, output and input register estimates
        #    and store them. Do not include counts from registers that are in
        #    pure_registers as they are counted seperately (else they would be
        #    counted twice in input and output)

        # 6. TODO: CHANGEDESC: For each active set, determine the number of 
        # registers that are used for storage
        registers_after_tasklet = collections.defaultdict(dict)
        for tasklet in context.keys():
            for oe in graph.out_edges(tasklet):
                for e in graph.memlet_tree(oe):
                    if isinstance(
                            e.dst,
                            nodes.AccessNode) and e.dst.data in register_arrays:
                        if e.dst in registers_after_tasklet[tasklet]:
                            raise NotImplementedError(
                                "Not implemented for multiple input edges at a pure register transient"
                            )

                        # FORNOW: just take subset after
                        #subset_before = self.symbolic_evaluation(e.data.subset.num_elements())
                        #subset_after = sum(self.symbolic_evaluation(ee.data.subset.num_elements()) for ee in graph.out_edges(e.dst))
                        subset_after = None
                        for ee in graph.out_edges(e.dst):
                            if ee.data.data == e.dst.data:
                                subset_after = subsets.union(
                                    subset_after, ee.data.subset)
                            elif ee.data.other_subset is not None:
                                subset_after = subsets.union(
                                    subset_after, ee.data.other_subset)

                        #subset_after = sum(ee.data.subset.num_elements() for ee in graph.out_edges(e.dst))
                        registers_after_tasklet[tasklet][e.dst] = subset_after

        # 7. TODO: Modify!! Loop over all active sets and choose the one with maximum
        #    register usage and return that estimate
        previous_tasklet_set = set()
        active_set_scores = list()
        used_register_arrays = dict()

        for tasklet_set in active_sets:
            # evaluate score for current tasklet set
            if scope_node:
                print(f"----- Active Set: {tasklet_set} -----")
            # 1. calculate score from active registers

            # NOTE: once we can calculate subset diffs, we can implement this much
            # more accurate and elegantly by adding / diffing subsets out to / of 
            # used_register_arrays at each iteration step

            # add used register arrays
            for tasklet in tasklet_set - previous_tasklet_set:
                for (reg_node,
                     reg_volume) in registers_after_tasklet[tasklet].items():
                    used_register_arrays[reg_node] = reg_volume

            # remove used register arrays
            for tasklet in previous_tasklet_set - tasklet_set:
                for reg_node in registers_after_tasklet[tasklet]:
                    del used_register_arrays[reg_node]

            mapping = collections.defaultdict(list)
            for (reg_node, volume) in used_register_arrays.items():
                mapping[reg_node.data].append(
                    reg_node
                )  # create mapping reg_name -> [reg_node, reg_node, .. ]

            total_max_size = self.max_registers_allocated
            total_size = outside_register_array_size
            # TODO: better sort key
            sort_key = lambda items: self.symbolic_evaluation(
                sum(used_register_arrays[r].num_elements() for r in items[1]
                    if r not in discarded_register_nodes))

            for (reg_name, reg_nodes) in sorted(mapping.items(), key=sort_key):
                # if any regs are already discarded, discard rest as well
                if any([r in discarded_register_nodes for r in reg_nodes]):
                    for reg_node in reg_nodes:
                        discarded_register_nodes.add(reg_node)
                else:
                    total_subset = None
                    for reg_node in reg_nodes:

                        total_subset = subsets.union(
                            total_subset, used_register_arrays[reg_node])

                    volume = self.symbolic_evaluation(
                        total_subset.num_elements()) * self._datatype_multiplier

                    if volume + total_size <= total_max_size:
                        total_size += volume
                    else:
                        # volume too large
                        print("Volume too large")
                        print("current_total=", total_size)
                        print("current_volume=", volume)
                        for reg_node in reg_nodes:
                            discarded_register_nodes.add(reg_node)

            # evaluate array score
            array_score = total_size

            # evaluate tasklet scores
            input_score = sum(
                self.evaluate_tasklet_input(
                    sdfg, graph, t, outside_register_arrays,
                    discarded_register_nodes) for t in tasklet_set)
            output_score = sum(
                self.evaluate_tasklet_output(
                    sdfg, graph, t, outside_register_arrays,
                    discarded_register_nodes) for t in tasklet_set)
            inner_score = sum(
                self.evaluate_tasklet_inner(
                    sdfg, graph, t, discarded_register_nodes, total_size)
                for t in tasklet_set)

            tasklet_score = input_score + output_score + inner_score
            set_score = tasklet_score + array_score

            if scope_node:
                print("DEBUG -> Input Score", input_score)
                print("DEBUG -> Output Score", output_score)
                print("DEBUG -> Inner Score", inner_score)
                print("DEBUG -> Tasklet Score", tasklet_score)
                print("DEBUG -> Array Score", array_score)
                print("DEBUG -> Score", set_score)
                print("DEBUG -> Discarded Register Nodes", discarded_register_nodes)

            active_set_scores.append(set_score)
            previous_tasklet_set = tasklet_set

        score = max(active_set_scores) if len(active_set_scores) > 0 else 0
       
        # return (registers_used, nodes_pushed_to_local)
        if scope_node:
            print(f"----- Score: {score} -----")

        return (score, discarded_register_nodes)

    def evaluate_available(self, outer_map):
        ''' FORNOW: get register available count '''

        threads = Config.get('compiler', 'cuda', 'default_block_size')
        threads = functools.reduce(lambda a, b: a * b,
                                   [int(e) for e in threads.split(',')])
        reg_count_available = self.register_per_block / (min(
            1028, 16 * threads))

        return reg_count_available

    def estimate_spill(self, sdfg, graph, scope_node):
        ''' 
        estimates spill for a fused subgraph that contains one outer map entry 
        '''
        # get some variables
        scope_dict = graph.scope_dict()
        outer_map_entry = scope_node
        outer_map = scope_node.map

        # get register used count
        state_evaluation = self.evaluate_state(sdfg=sdfg,
                                               graph=graph,
                                               scope_node=outer_map_entry,
                                               discarded_register_nodes = set(),
                                               outside_register_arrays = set(),
                                               scope_dict=scope_dict)

        (reg_count_required, discarded_nodes) = state_evaluation
        # get register provided count (FORNOW)
        reg_count_available = self.evaluate_available(outer_map=outer_map)
        # get register traffic count

        discarded_register_traffic = self.evaluate_register_traffic(
            sdfg=sdfg,
            graph=graph,
            scope_node=outer_map_entry,
            discarded_register_nodes=discarded_nodes,
            scope_dict=scope_dict,
        )

        return (discarded_register_traffic, reg_count_required)

    def score(self, subgraph: SubgraphView):
        '''
        Applies CompositeFusion to the Graph and compares Memlet Volumes
        with the untransformed SDFG passed to __init__().
        '''

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        if self.debug:
            print("ScoreMemlet::Debug::Subgraph to Score:",
                  subgraph_copy.nodes())

        transformation_function = self._transformation(subgraph_copy)
        # assign properties to transformation
        for (arg, val) in self._kwargs.items():
            try:
                setattr(transformation_function, arg, val)
            except AttributeError:
                warnings.warn(f"Attribute {arg} not found in transformation")
            except (TypeError, ValueError):
                warnings.warn(f"Attribute {arg} has invalid value {val}")

        transformation_function.apply(sdfg_copy)
        if self.deduplicate:
            sdfg_copy.apply_transformations_repeated(DeduplicateAccess,
                                                     states=[graph_copy])
        if self.propagate_all or self.deduplicate:
            propagation.propagate_memlets_scope(sdfg_copy, graph_copy,
                                                graph_copy.scope_leaves())

        self._i += 1
        sdfg_copy.save(f'inspect_{self._i}.sdfg')

        current_traffic = self.estimate_traffic(sdfg_copy, graph_copy)

        # NOTE: assume transformation function has
        # .global_map_entry field, this makes it much easier

        outer_entry = transformation_function._global_map_entry
        TEST = self.estimate_spill(sdfg_copy, graph_copy,
                                            outer_entry)[0]
        print("SPILL=", TEST)
        spill_traffic = TEST 

        return (current_traffic + spill_traffic) / self._base_traffic

    @staticmethod
    def name():
        return "Estimated Memlet Traffic plus Spill"
