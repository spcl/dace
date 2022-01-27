from collections import defaultdict
import copy
import sympy as sp
import networkx as nx
import graphviz
from numpy import nanargmax, outer
from dace.sdfg.sdfg import SDFG
from dace.transformation.estimator.soap.soap import AccessParams, SoapStatement
from dace.transformation.estimator.soap.utils import *
import dace
from dace.sdfg.graph import MultiConnectorEdge
# from soap.SOAP import *
from dace.codegen import control_flow
from dace.sdfg.nodes import *
from dace.subsets import Range
from dace.codegen.control_flow import structured_control_flow_tree
from typing import Optional, List, Set, Tuple
from dace.symbolic import pystr_to_symbolic, issymbolic
from dace import subsets
from warnings import warn

from dataclasses import dataclass, field


class SDGPath:
    def __init__(self, values):
        self.path = copy.copy(values)
        
    def __deepcopy__(self, memo):
        node = object.__new__(SDGPath)

        node.path = copy.copy(self.path)
        return node

    def append(self, value):
        self.path.append(value)
        
    def __getitem__(self, item):
        return self.path.__getitem__(item)
    
# ------------------------------------
# Symbolic Directed Graph abstraction
# ------------------------------------
class SdgScope:
    def __init__(self, sdfg_entry_node : Node = None):
        self.n_iterations = 1
        self.ranges = {}
        self.map_ranges = {}
        self.loop_ranges = {}
        self.sdfg_path = SDGPath([sdfg_entry_node])
        self.sdfg_mapping = {}
        self.SOAP_executions = 1
        self.SDFG_arrays = {}
        self.input_SDFG_arrays = []



class SdgNode:
    def __init__(self, scope : SdgScope = None, 
                sdfg_node : Node = None, 
                statement : SoapStatement = None):
        self.scope = scope
        self.sdfg_node = sdfg_node
        self.statement = statement



"""
Main SDG (Symbolic Directed Graph) class. 
The constuctor takes the input SDFG, and constructs SDG 
(every vertex is a SOAP statement, every edge is an access function vector).

It provides functionality both per-statement (per-vertex) I/O analysis, as well 
as the SDG subgraphing (finding optimal kernel fusions).
"""
@dataclass
class SDG:
    sdfg : SDFG = None
    solver : Solver = None
    graph : nx.MultiDiGraph = None
    node_relabeler : Dict = field(default_factory=dict)
    array_version_counter : Dict = None

    def __post_init__(self):
        self.array_version_counter = defaultdict(int)
        if self.graph is None:
            self.graph = nx.MultiDiGraph()
        if self.sdfg is not None:
            self.from_SDFG(self.sdfg)

    @property
    def nodes(self):
        return list(self.graph.nodes())

    @property
    def edges(self):
        return list(self.graph.edges())

    def in_edges(self, node : str) -> List[Tuple[str,str, Dict]]:
        return list(self.graph.in_edges(node, data = True))

    def out_edges(self, node : str) -> List[Tuple[str,str, Dict]]:
        return list(self.graph.out_edges(node, data = True))


    def add_edge(self, u : str, v : str, _base_access : str, 
                access_params : AccessParams, _st : SoapStatement, 
                is_wcr = False, in_transient = False, out_transient = False) -> None:
        
        if v in self.graph.nodes:
            self.graph.nodes[v]['st'] = _st
        else:
            self.graph.add_node(v, st = _st)
        phi_str = '[' + _base_access.replace('*', ',') + '] ' + str(access_params.offsets).replace(',)',')')
        if is_wcr:
            phi_str += '*'
        self.graph.add_edge(u, v, label = str(phi_str), 
                        base_access = _base_access, 
                        offsets = access_params,
                        wcr = is_wcr)        
        self.graph.nodes[u]['transient'] = in_transient                
        self.graph.nodes[v]['transient'] = out_transient         


    def get_node_label(self, u):
        basePhi = (u[0], u[1])
        if basePhi in self.node_relabeler.keys():
            previousInd = int(self.node_relabeler[basePhi].split('_')[-1])

        else:
            self.node_relabeler[basePhi] = u[0] + '[' + u[1] + ']_0'


    
    def from_SDFG(self, sdfg : dace.SDFG) -> None:     
        """
        Recursively builds the SDG from the input SDFG.
        The recursive calls are one of the following:
        1. _from_SDFG_sdfg - if the nested SDFG is encountered
        2. _from_SDFG_loop
        3. _from_SDFG_state
        4. _from_SDFG_node
        """   
        sdg_scope = SdgScope()
        dace.propagate_memlets_sdfg(sdfg)
        sdfg.save("tmp.sdfg", hash=False)        
        # get all output arrays within a given STATE (for SSA purposes)
        sdg_scope.output_arrays = self._get_output_arrays_sdfg(sdfg, sdg_scope)
        self._from_SDFG_sdfg(sdfg, sdg_scope)


    def _from_SDFG_sdfg(self, sdfg: dace.SDFG, sdg_scope : SdgScope) -> None:
        control_tree = structured_control_flow_tree(sdfg, lambda x: None)       
        # TODO: topological sort?
        for control_node in control_tree.children:
            if isinstance(control_node, control_flow.SingleState):
                state = control_node.state
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(state)
                sdg_scope = self._from_SDFG_scope(state, None, inner_scope)
            
            elif isinstance(control_node, control_flow.ForScope):
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(control_node.guard)    
                
                # TODO: when the loop annotation pass is fixed, we need to add the string parsing thing (below)
                # loop_ranges = {k: v for k, v in zip(control_node.guard.ranges.keys(),
                #            [x.ranges[0] for x in control_node.guard.ranges.values() ] ) if isinstance(k, str)}   
                loop_ranges = {str(k): v for k, v in zip(control_node.guard.ranges.keys(),
                            [x.ranges[0] for x in control_node.guard.ranges.values() ] )}
                inner_scope.ranges = {**sdg_scope.ranges, **loop_ranges}
                inner_scope.loop_ranges = {**sdg_scope.loop_ranges, **loop_ranges}
                inner_scope.innermost_ranges = loop_ranges
                self._from_SDFG_loop(control_node, inner_scope)
                
            elif isinstance(control_node, control_flow.IfScope):
                a = 1
                

    def _from_SDFG_loop(self, loop : control_flow.ForScope, 
                sdg_scope: SdgScope) -> None:
        # TODO: loop iterations are already propagated into the nested scopes?
        # num_executions = loop.guard.executions
        # sdg_scope.SOAP_executions *= num_executions
        
        # TODO: topological sort?
        for control_node in loop.body.children:
            if isinstance(control_node, control_flow.SingleState):
                state = control_node.state
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(state)              
                sdg_scope = self._from_SDFG_scope(state, None, inner_scope)
            
            elif isinstance(control_node, control_flow.ForScope):
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(control_node.guard)       
                # TODO: when the loop annotation pass is fixed, we need to add the string parsing thing (below)
                # loop_ranges = {k: v for k, v in zip(control_node.guard.ranges.keys(),
                #            [x.ranges[0] for x in control_node.guard.ranges.values() ] ) if isinstance(k, str)}   
                loop_ranges = {str(k): v for k, v in zip(control_node.guard.ranges.keys(),
                            [x.ranges[0] for x in control_node.guard.ranges.values() ] )}
                inner_scope.loop_ranges = {**sdg_scope.loop_ranges, **loop_ranges}
                inner_scope.ranges = {**sdg_scope.ranges, **loop_ranges}
                inner_scope.innermost_ranges = loop_ranges
                self._from_SDFG_loop(control_node, inner_scope)


    def _from_SDFG_scope(self, state: dace.SDFGState, 
                    scope: Optional[dace.nodes.EntryNode], 
                    sdg_scope : SdgScope) -> SdgScope:
        num_executions = state.executions
        sdg_scope.SOAP_executions *= num_executions
        sdg_scope.SDFG_arrays = {**sdg_scope.SDFG_arrays, **state.parent.arrays}
        sdg_scope.input_SDFG_arrays += [src_node.data for src_node in state.source_nodes() if isinstance(src_node, AccessNode)]
   
        snodes = [n for n in nx.topological_sort(state.nx) if n in state.scope_children()[scope]]
        for node in snodes:
            if isinstance(node, (dace.nodes.Tasklet, dace.nodes.LibraryNode)):
                inner_scope = copy.deepcopy(sdg_scope)
                self._from_SDFG_node(node, state, inner_scope)
            elif isinstance(node, (dace.nodes.AccessNode)):
                if node.label in ["__return_1", "__tmp4", "__tmp8"]:
                    a = 1
                # if the access node has a single in-edge coming from a different access node, 
                # then it is a projection. We then add this to the sdfg mapping.
                in_edges = state.in_edges(node)
                if len(in_edges) > 0:
                    if isinstance(sdg_scope.SDFG_arrays[node.label], dace.data.View):
                        # get the original node:
                        # src_node = state.memlet_path(in_edges[0])[0].data

                        src_node = in_edges[0].data
                        
                        # the view slice will give us a range sth like (j,j,1). We need to convert ti to (0, j-1,1)
                        # src_node.subset.ranges = [(0, y - 1, 1) if (x == y) else (x,y,z) for (x,y,z) in src_node.subset.ranges ]
                        # src_node.subset.ranges = [(list(y.free_symbols)[0], list(y.free_symbols)[0], 1) 
                        #                             if (x != y) else (x,y,z) for (x,y,z) in src_node.subset.ranges ]
                        self.add_projection(src_node, state.out_edges(node)[0].data, sdg_scope)
                    
                    # check if this is an initialization of transient (access_node -> tasklet -> transient access_node)
                    in_node = in_edges[0].src

                    #TODO: to check
                    if sdg_scope.SDFG_arrays[node.data].transient:
                    # if True:  
                        if isinstance(in_node, dace.nodes.Tasklet):
                            in_edges = state.in_edges(in_node)
                            if len(in_edges) > 0:
                                in_in_node = in_edges[0].src
                                if isinstance(in_in_node, dace.nodes.AccessNode):
                                    # now we need to propagate the updated sdfg_mapping one layer higher
                                    if node.data == "_x":
                                        a = 1
                                    tmp = SDG._fuse_mapping({node.data: in_edges[0].data}, sdg_scope.sdfg_mapping)
                                    if any([len(sp.sympify(i).free_symbols) > 0 for (i,j,k) in list(tmp.values())[0].subset.ranges]):
                                    # sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **{node.data: in_edges[0].data}}
                                        sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **tmp}
                                    if "__tmp6" in sdg_scope.sdfg_mapping.keys():
                                        a = 1

                        if isinstance(in_node, dace.nodes.MapEntry):
                            if node.data == "_x":
                                a = 1
                            tmp = SDG._fuse_mapping({node.data: in_edges[0].data}, sdg_scope.sdfg_mapping)
                            if any([len(sp.sympify(i).free_symbols) > 0 for (i,j,k) in list(tmp.values())[0].subset.ranges]):
                                sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **tmp}
                            if "__tmp6" in sdg_scope.sdfg_mapping.keys():
                                a = 1

                        # confront: npbench, cholesky_dace. __tmp2 access node should inherit loop ranges [i,j], otherwise,
                        # this would be an empty memlet
                        if isinstance(in_node, dace.nodes.NestedSDFG)  \
                                    and state.in_edges(node)[0].data.wcr is not None \
                                    and len(sdg_scope.loop_ranges) > 0:
                            if node.label == "__tmp6":
                                a = 1 
                            composed_memlet = copy.deepcopy(state.out_edges(node)[0].data)
                            # if the range of the memlet is empty (it is a scalar), add all ranges.
                            # if it's not empty, add only loop ranges
                            if len(composed_memlet.subset.ranges) == 0 or \
                                    (len(composed_memlet.subset.ranges) == 1 and composed_memlet.subset.ranges[0][0] == 0):
                                composed_memlet.subset.ranges += SDG._dict_ranges_to_list(sdg_scope.ranges)
                            else:        
                                composed_memlet.subset.ranges += SDG._dict_ranges_to_list(sdg_scope.loop_ranges)
                            composed_memlet.subset.ranges = [rng for rng in composed_memlet.subset.ranges 
                                            if len(sp.sympify(rng[0]).free_symbols) > 0]
                            sdg_scope.sdfg_mapping[composed_memlet.data] = composed_memlet

                        if isinstance(in_node, dace.nodes.AccessNode):
                            in_edges = state.in_edges(in_node)
                            if len(in_edges) > 0:
                                tmp = SDG._fuse_mapping({node.data: in_edges[0].data}, sdg_scope.sdfg_mapping)
                                if any([len(sp.sympify(i).free_symbols) > 0 for (i,j,k) in list(tmp.values())[0].subset.ranges]):
                                # sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **{node.data: in_edges[0].data}}
                                    sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **tmp}
                            else:
                                in_edges = state.in_edges(node)
                                tmp = SDG._fuse_mapping({node.data: in_edges[0].data}, sdg_scope.sdfg_mapping)
                                if any([len(sp.sympify(i).free_symbols) > 0 for (i,j,k) in list(tmp.values())[0].subset.ranges]):
                                # sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **{node.data: in_edges[0].data}}
                                    sdg_scope.sdfg_mapping = {**sdg_scope.sdfg_mapping, **tmp}

                        
            elif isinstance(node, dace.nodes.EntryNode): 
                if 'stateFOR31_map' in node.label:
                    a = 1
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(node)       
                map_ranges = {k: v for k, v in zip(node.params, node.range.ranges) if isinstance(k, str)}
                inner_scope.ranges = {**sdg_scope.ranges, **map_ranges}
                inner_scope.map_ranges = {**sdg_scope.map_ranges, **map_ranges}
                inner_scope.innermost_ranges = map_ranges
                num_executions = state.executions
                inner_scope.SOAP_executions = num_executions
                self._from_SDFG_scope(state, node, inner_scope)
            elif isinstance(node, dace.nodes.NestedSDFG):
                if node.label == '_MatMult_gemvt':
                    a = 1
                inner_scope = self._add_ranges_and_mappings(state, sdg_scope, node)
                if "__tmp2" in inner_scope.sdfg_mapping.keys():
                    a = 1
                self._from_SDFG_sdfg(node.sdfg, inner_scope)
                
        # return updated sdg_scope (e.g., sdfg_mapping updated by transients)
        return sdg_scope
        


    # the core SDG creation function. The elementary building block for SDG is the SDFG's tasklet
    def _from_SDFG_node(self, node : dace.nodes, 
                state: dace.SDFGState, sdg_scope : SdgScope) -> None:  
        if node.label in ['assign_16_12']:
            a = 1
                  
        if not state.out_edges(node) or not state.in_edges(node):
            return   
                           
        S = SoapStatement() 
        S.name = node.label
        S.tasklet = {node}
        S.daceRanges[sdg_scope.sdfg_path[-1].label] = sdg_scope.ranges      
        S.output_arrays = sdg_scope.output_arrays  

        # add WCR (write conflict resolution) edges to the input edges
        input_edges = state.in_edges(node)
        S.wcr_edges = []
        for e in state.out_edges(node):
            if e.data.wcr:  
                # leave transients alone. They will have empty ranges that will be added by
                # the _get_outer_memlet call
                if not sdg_scope.SDFG_arrays[e.data.data].transient:
                    e.data.src_subset = e.data.dst_subset
                input_edges.append(e)                
                        

        for e in input_edges:
            if node.label in ['_Mult_', '_USub_'] and e.data.data == "temp2":
                a = 1
            # resolve transients, inner scope mappings, etc. Retreive outermost ranges and array names.
            outer_memlet = self._get_outer_memlet(e.data, sdg_scope, False)
            
            # check if the resolved memlet exists, is not empty, and is not scalar
            if not outer_memlet or len(outer_memlet.subset.ranges) == 0 \
                 or len(outer_memlet.subset.ranges[0][0].free_symbols) == 0:
                     #                # or (len(outer_memlet.subset.ranges) == 1 and \
                continue
                      
            # if the edge is WCR, add it to the list
            SSA_dim = []
            
            # old, fishy version. This "e.data.src_subset is None" seems broken for the polybench/doitgen
            if e.data.wcr:
                if e.data.src_subset is None or e.data.dst_subset.ranges != e.data.src_subset.ranges:
                    # SSA_dim = e.data.dst_subset 
                    a = e.data.dst_subset 
                else:
                    a = 1  
            
            if e.data.wcr:
                S.wcr_edges.append(outer_memlet)
                if e.data.src_subset is not None and (e.data.dst_subset.ranges != e.data.src_subset.ranges):
                    SSA_dim = e.data.dst_subset    
                    a = 1
                else:
                    a = 1          
                             
            if not SSA_dim:
                # determine the SSA dimension if we have the same input and output access (overwriting the data)
                # if the memlet is coming from the outside of the map, we don't include them
                if isinstance(e.src, dace.nodes.MapEntry):
                    SSA_dim = []
                    if  SDG._get_SSA_dim(outer_memlet, sdg_scope) != []:
                        a = 1
                        SSA_dim = SDG._get_SSA_dim(outer_memlet, sdg_scope)
                else:
                    SSA_dim = SDG._get_SSA_dim(outer_memlet, sdg_scope)
                
            # add this input access to the elementary SOAP statement
            S.add_edge_to_statement(outer_memlet, SSA_dim)            
            

        # # if we have a tasklet with a single transient in edge memlet and single transient out edge memlet,
        #     # and the in_memlet is in the sdfg mapping, then we also map this to the out memlet   
        # if len(input_edges) == 1 and  input_edges[0].data.data in sdg_scope.SDFG_arrays.keys() \
        #             and sdg_scope.SDFG_arrays[input_edges[0].data.data].transient \
        #             and input_edges[0].data.data in sdg_scope.sdfg_mapping.keys():
        #     single_in_transient_mapped = sdg_scope.sdfg_mapping[input_edges[0].data.data]
        # else:
        #     single_in_transient_mapped = None


        for e in state.out_edges(node):           
            # resolve transients, inner scope mappings, etc. Retreive outermost ranges and array names.
            outer_memlet = self._get_outer_memlet(e.data, sdg_scope, True)
            if not outer_memlet or len(outer_memlet.subset.ranges) == 0:
                continue
            # deterimne the SSA dimension if we have the same input and output access (overwriting the data)
            SSA_dim = SDG._get_SSA_dim(outer_memlet, sdg_scope)
                   
            # add this output access to the elementary SOAP statement
            S.add_output_edge_to_statement(outer_memlet, SSA_dim)    
            

        if not S.output_accesses:
            return
        if not any(S.daceRanges.values()):
            return
        S.numExecutions = sdg_scope.SOAP_executions 

        S.clean_iter_vars()

        if len(S.phis) == 0:
            return
        if Config.get("soap", "analysis", "io_analysis") :
            S.loop_ranges = sdg_scope.loop_ranges
            S.map_ranges = sdg_scope.map_ranges
            S.solve(self.solver)
            if S.Dom_size == 0 or len(S.phis) == 0 or S.Q == 0:
                return

        # perform WD analysis
        if Config.get("soap", "analysis", "wd_analysis") :
            S.reductionRanges = node.reductionRanges
            S.update_ranges()        
            S.calculate_dominator_size()  
            S.count_V()
            S.count_D()

        # everything is fine with this statement, it is well defined and ready for the further SDG analysis
        node.soap_statement = S
        
        # now we update the SDG
        self._add_statement_to_SDG(S, sdg_scope)
        
        self.plot_SDG()
        return
          
    
    def _add_ranges_and_mappings(self, state : dace.SDFGState, sdg_scope : SdgScope, 
                    node : dace.nodes) -> SdgScope:
        inner_scope = copy.deepcopy(sdg_scope)
        inner_scope.sdfg_path.append(node) 
        
        # # First, we update the current mapping and check
        # # if we map something to transient arrays. 
        # # If yes, we need to add the outer ranges to them
        # for inner_mem, outer_mem in inner_scope.sdfg_mapping.items():
        #     if sdg_scope.SDFG_arrays[outer_mem.data].transient:
        #         inner_scope.sdfg_mapping[inner_mem].subset.ranges += \
        #             [(dace.symbol(i), dace.symbol(i), 1) 
        #              for i in sdg_scope.innermost_ranges.keys()]
        
        src_sdfg_mapping = {e.dst_conn : e.data for e in state.in_edges(node)}                
        dst_sdfg_mapping = {e.src_conn : e.data for e in state.out_edges(node)}                
        # if the arrays are created inside the map scopes, we need to add the map ranges to
        # the mapped memlets' ranges.
        # e.g., polybench's doitgen. If we create a "sum[p]" array within the two nested maps q and r,
        # the resulting array should be "sum[p,q,r]"                
        if len(sdg_scope.ranges) > 0:            
            vars_to_add = list(map(dace.symbol, [v for v in sdg_scope.map_ranges.keys()]))
            
            # If the array is initialized inside a scope (meaning it is NOT a source node in the nested SDFG,
            # but rather it has an initialization tasklet), then we add also loop ranges, not only the map ones            
            for arr, mem in dst_sdfg_mapping.items():
                tmp_mem = copy.deepcopy(mem)
                if tmp_mem.data in sdg_scope.sdfg_mapping.keys():
                    tmp_mem = sdg_scope.sdfg_mapping[mem.data]                  
                if tmp_mem.data not in sdg_scope.input_SDFG_arrays:
                    vars_to_add_in = vars_to_add + list(map(dace.symbol, [v for v in sdg_scope.loop_ranges.keys()]))
                else:
                    vars_to_add_in = vars_to_add
                
                # TODO: controversial new change
                mem.subset.ranges += [(x, x, 1) for x in vars_to_add_in ]
                # mem.subset.ranges += [(x, x, 1) for x in vars_to_add_in if str(x) not in mem.subset.free_symbols ]
                mem.subset = Range(mem.subset.ranges)                
                
                # # BUT, we remove empty ranges (0,0,1)
                # # TODO: not sure what is the proper way how to filter them correctly.
                # # we DON'T want to filter ranges like (0, N, 1), as they are needed for memlet.subset.compose
                # # test1 = [rng for rng in mem.subset.ranges if len(rng[0].free_symbols) > 0]
                # test2 = [rng for rng in mem.subset.ranges if len(rng[1].free_symbols) > 0]
                # # if test1 != test2:
                # #     a = 1
                # mem.subset.ranges = test2
        
        cur_scope_mapping = {**src_sdfg_mapping, ** dst_sdfg_mapping}
        
        #inner_scope.sdfg_mapping = cur_scope_mapping
        
        # # transitive mapping
        # fused_mapping = {}
        # for inner_mem, outer_mem in cur_scope_mapping.items():
        #     # resolve dynamic arrays and like scalars. If the memlet is dynamic, inherit ranges from 
        #     # the tasklet it is assigned in
        #     if outer_mem.dynamic:
        #         outer_mem.subset.ranges = self._find_transient_assignement(state, sdg_scope, node)
            
        #     if outer_mem.data in inner_scope.sdfg_mapping.keys():
        #         outmost_mem = copy.deepcopy(inner_scope.sdfg_mapping[outer_mem.data])
        #         # TODO: can we remove the following line?
        #         # outmost_mem.subset = SDG._proj_compose(outmost_mem, outer_mem) #outmost_mem.subset.compose(outer_mem.subset)
        #         fused_mapping[inner_mem] = outmost_mem
        #     else:
        #         fused_mapping[inner_mem] = outer_mem
        
        # inner_scope.sdfg_mapping = fused_mapping
        inner_scope.sdfg_mapping = SDG._fuse_mapping(cur_scope_mapping, inner_scope.sdfg_mapping)

        if "__tmp2" in inner_scope.sdfg_mapping.keys():
            a = 1
        return inner_scope
    
    
    def _find_transient_assignement(self, state : dace.SDFGState, sdg_scope : SdgScope, node : dace.nodes):
        a = 1
    
    
    @staticmethod
    def _fuse_mapping(inner_mapping : Dict, outer_mapping : Dict) -> Dict:
        # transitive mapping
        fused_mapping = {}
        for inner_mem, outer_mem in inner_mapping.items():
            if outer_mem.data in outer_mapping.keys():
                outmost_mem = copy.deepcopy(outer_mapping[outer_mem.data])
                # TODO: can we remove the following line?
                # outmost_mem.subset = SDG._proj_compose(outmost_mem, outer_mem) #outmost_mem.subset.compose(outer_mem.subset)
                fused_mapping[inner_mem] = outmost_mem
            else:
                fused_mapping[inner_mem] = outer_mem
        return fused_mapping

    @staticmethod
    def _proj_compose(outer_mem : dace.Memlet, inner_mem : dace.Memlet) -> Range:
        outer_rng = outer_mem.subset.ranges
        inner_rng = inner_mem.subset.ranges
        
        if np.prod(outer_mem.subset.size()) == np.prod(inner_mem.subset.size()):
            # return len(outer_rng) > len(inner_rng) ? outer_rng : inner_rng
            return Range(outer_rng if len(outer_rng) < len(inner_rng) else inner_rng)
        
        if len(outer_rng) < len(inner_rng):
            # injective
            a = 1
            # inner_mem.subset.ranges = outer_mem.subset.ranges
        elif len(outer_rng) > len(inner_rng):
            # surjective
            a = 1
            # outer_mem.subset.ranges = inner_mem.subset.ranges
        elif not set(inner_mem.subset.ranges) == set(outer_mem.subset.ranges):
            # Example situation:
            # outer_rng == [(r, r, 1), (0, NQ - 1, 1), (0, NP - 1, 1)]
            # inner_rng == [(0, NQ - 1, 1), (0, 0, 1), (0, NP - 1, 1)]
            # expected output:
            # [(r, r, 1), (0, NQ - 1, 1), (0, NP - 1, 1)]
            # reference example:
            # npbench/polybench/doitgen
            return outer_mem.subset
                
        return outer_mem.subset.compose(inner_mem.subset)

    
    @staticmethod
    def _dict_ranges_to_list(ranges : Dict) -> List:
        return [(dace.symbol(i), dace.symbol(i), 1) for i in ranges.keys()]



    def _add_statement_to_SDG(self, S : SoapStatement, sdg_scope : SdgScope) -> None:
        output_arrays_vers = []
        wcr_arrays = [e.data for e in S.wcr_edges]
        for output_array in S.output_accesses.keys():
            # !!!! CAREFUL !!!!
            # THIS SHOULD BE THE ONLY PLACE WHERE THIS COUNTER IS INCREMENTED
            self.array_version_counter[output_array] += 1
            output_arrays_vers.append(output_array + "_" + str(self.array_version_counter[output_array]))
            
            # check if it's transient
            if output_array in sdg_scope.SDFG_arrays.keys() and  \
                        sdg_scope.SDFG_arrays[output_array].transient:  
                out_transient = True
            else:
                out_transient = False
            
        # iterate over input accesses     
        for array_name, array_accesses in S.phis.items():
            if array_name in S.output_accesses.keys():                
                input_array_ver = array_name + "_" + str(self.array_version_counter[array_name] - 1)
            else:
                input_array_ver = array_name + "_" + str(self.array_version_counter[array_name])
            
            # check if it's transient
            if array_name in sdg_scope.SDFG_arrays.keys() and  \
                        sdg_scope.SDFG_arrays[array_name].transient:  
                in_transient = True
            else:
                in_transient = False
                
            # check if it's wcr
            if array_name in wcr_arrays:
                wcr = True
            else:
                wcr = False
            
            # iterate over different base accesses to the same array 
            for base_access, access_params in array_accesses.items():       
                for output_array in output_arrays_vers:    
                    # another check if the edge is wcr. If the dimension of the output is smaller than the input,
                    # it is a wcr
                    out_dim = len(S.output_accesses[strip(output_array)].baseAccess.split('*'))
                    in_dim = len(base_access.split('*'))
                    # if (out_dim < in_dim) and wcr == False:
                    #     wcr = True
                    self.add_edge(input_array_ver, output_array, base_access, access_params, S,
                                  wcr, in_transient, out_transient)



    def add_projection(self,
                       project_from : dace.memlet,
                       project_to : dace.memlet,
                       sdg_scope : SdgScope) -> None:
        """
        Handling projections between different dimensions of input/output memlets
        """
        from_ranges = project_from.subset.ranges
        to_ranges = project_to.subset.ranges
        
        if len(from_ranges) < len(to_ranges):
            # injective
            src_memlet = copy.deepcopy(project_from)
            src_memlet.subset.ranges += [(0, 0, 1)] * (len(to_ranges) - len(from_ranges))
            sdg_scope.sdfg_mapping[project_to.data] = src_memlet
        if len(from_ranges) > len(to_ranges):
            # surjective
            # to_expr = sp.prod([y for (x,y,z) in project_to.subset.ranges])
            # from_expr = sp.prod([y for (x,y,z) in project_from.subset.ranges])
            # a = int_to_real(to_expr)
            dst_memlet = copy.deepcopy(project_to)
            # dst_memlet.subset.ranges += [(0, 0, 1)] * (len(from_ranges) - len(to_ranges))
            sdg_scope.sdfg_mapping[dst_memlet.data] = project_from
        if len(from_ranges) == len(to_ranges):
            # bijective
            sdg_scope.sdfg_mapping[project_to.data] = project_from
           

    # ------------------------------------
    # SDFG preprocessing
    # ------------------------------------
    
    def _get_output_arrays_sdfg(self, sdfg: dace.SDFG, 
                    sdg_scope : SdgScope) -> Dict[str, Tuple[str, Tuple[int]]]:
        """
        Before we create SDG, we need to know which arrays are EVENTUALLY updated,
        to differentiate between read-only arrays (no SSA dimension), and the ones
        for which the SSA dimension must be determined.

        It is a recursive function, calling _get_output_arrays_state and _get_output_arrays_loop  
        """
        output_arrays = {}
        control_tree = structured_control_flow_tree(sdfg, lambda x: None) 
        
        # [n for n in nx.topological_sort(state.nx) if n in state.scope_children()[scope]]
        for control_node in control_tree.children:
            if isinstance(control_node, control_flow.SingleState):
                state = control_node.state
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(state)              
                output_arrays = {**output_arrays, **self._get_output_arrays_state(state, None, inner_scope)}
            
            elif isinstance(control_node, control_flow.ForScope):
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(control_node.guard)       
                loop_ranges = {str(k): v for k, v in zip(control_node.guard.ranges.keys(),
                            [x.ranges[0] for x in control_node.guard.ranges.values() ] )}
                inner_scope.ranges = {**sdg_scope.ranges, **loop_ranges}
                inner_scope.loop_ranges = {**sdg_scope.loop_ranges, **loop_ranges}
                inner_scope.innermost_ranges = loop_ranges
                output_arrays = {**output_arrays, **self._get_output_arrays_loop(control_node, inner_scope)}
                
        return output_arrays
                

    def _get_output_arrays_loop(self, loop : control_flow.ForScope, 
                sdg_scope: SdgScope) -> Dict[str, Tuple[str, Tuple[int]]]:
        output_arrays = {}
        
        #[n for n in nx.topological_sort(state.nx) if n in state.scope_children()[scope]]
        for control_node in loop.body.children:
            if isinstance(control_node, control_flow.SingleState):
                state = control_node.state
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(state)              
                output_arrays = {**output_arrays, **self._get_output_arrays_state(state, None, inner_scope)}
            
            elif isinstance(control_node, control_flow.ForScope):
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(control_node.guard)       
                loop_ranges = {k: v for k, v in zip(control_node.guard.ranges.keys(),
                            [x.ranges[0] for x in control_node.guard.ranges.values()]) if isinstance(k, str)}
                inner_scope.ranges = {**sdg_scope.ranges, **loop_ranges}
                output_arrays = {**output_arrays, **self._get_output_arrays_loop(control_node, inner_scope)}
                
        return output_arrays

    
    def _get_output_arrays_state(self, state: dace.SDFGState, 
            scope: Optional[dace.nodes.EntryNode],
            sdg_scope : SdgScope) -> Dict[str, Tuple[str, Tuple[int]]]:            
        output_arrays = {}
        if len(sdg_scope.ranges) == 0:
            # then we are not in any parametric range scope            
            iter_vars = set()
        else:
            iter_vars = set(map(dace.symbol, [v for v in sdg_scope.ranges.keys()]))
        snodes = state.scope_children()[scope]    
        sdg_scope.SDFG_arrays = {**sdg_scope.SDFG_arrays, **state.parent.arrays}
        sdg_scope.input_SDFG_arrays += [src_node.data for src_node in state.source_nodes() if isinstance(src_node, AccessNode)]
        for node in snodes:
            if isinstance(node, (dace.nodes.Tasklet, dace.nodes.LibraryNode)):      
                inner_scope = copy.deepcopy(sdg_scope)  
                for e in state.out_edges(node):
                    # resolve transients, inner scope mappings, etc. Resolve outermost ranges and array names
                    outer_memlet = self._get_outer_memlet(e.data, inner_scope, True)
                    if not outer_memlet or not iter_vars:
                        continue                    
                    (arrayName, baseAccess, offsets) =  get_access_from_memlet(outer_memlet, iter_vars)
                    
                    # This check is needed to exclude output arrays that do not have any SSA dim. 
                    # That is, if the arrays dimension (baseAccess) is the same as iteration dimension
                    # (sdg_scope.ranges), even though it is an output array, we don't include it.
                    # Otherwise, we might attach to it a wrong SSA_dim later.
                    if set(sdg_scope.ranges.keys()) != set(baseAccess.split('*')):
                        output_arrays[arrayName] = (baseAccess, offsets)
                                        
            elif isinstance(node, dace.nodes.EntryNode):       
                inner_scope = copy.deepcopy(sdg_scope)
                inner_scope.sdfg_path.append(node)       
                map_ranges = {k: v for k, v in zip(node.params, node.range.ranges) if isinstance(k, str)}
                if "stateFOR115" in node.label:
                    a = 1
                inner_scope.ranges = {**sdg_scope.ranges, **map_ranges}
                inner_scope.map_ranges = {**sdg_scope.map_ranges, **map_ranges}
                num_executions = state.executions
                inner_scope.SOAP_executions = num_executions
                output_arrays = {**output_arrays, **self._get_output_arrays_state(state, node, inner_scope)}
                
            elif isinstance(node, dace.nodes.NestedSDFG):
                inner_scope = self._add_ranges_and_mappings(state, sdg_scope, node)
                output_arrays = {**output_arrays, **self._get_output_arrays_sdfg(node.sdfg, inner_scope)}

            elif isinstance(node, dace.nodes.AccessNode):
                if len(state.in_edges(node)) > 0 and len(state.out_edges(node)) == 0:

                    (arrayName, baseAccess, offsets) =  get_access_from_memlet(
                            SDG._memlet_ranges_to_iters(state.in_edges(node)[0].data), iter_vars)
                    
                    # This check is needed to exclude output arrays that do not have any SSA dim. 
                    # That is, if the arrays dimension (baseAccess) is the same as iteration dimension
                    # (sdg_scope.ranges), even though it is an output array, we don't include it.
                    # Otherwise, we might attach to it a wrong SSA_dim later.
                    if set(sdg_scope.ranges.keys()) != set(baseAccess.split('*')):
                        if arrayName in output_arrays.keys():
                            if output_arrays[arrayName][0] == "":
                                output_arrays[arrayName] = (baseAccess, offsets)    
                        else:
                            output_arrays[arrayName] = (baseAccess, offsets)
                                   
        return output_arrays
    
    
    

    def _get_outer_memlet(self, inner_memlet : dace.Memlet, sdg_scope : SdgScope, 
                output : bool, single_in_transient_mapped : dace.Memlet = None) -> dace.Memlet:
        """
        Returns an outermost memlet which can be traced from the given inner_memlet
        """
        new_memlet = copy.deepcopy(inner_memlet)
        new_memlet._state = inner_memlet._state

        if new_memlet.data in sdg_scope.sdfg_mapping:
            outer_memlet = copy.deepcopy(sdg_scope.sdfg_mapping[new_memlet.data])

            # needed for transient mapping. Sometimes we add transients from output edges to the SDG,
            # but then, because of the fused mapping, the input memlets are propagated already to the outer (non-transient) memlets.
            # In this case, to keep the correct connectivity of the SDG, we keep the node name as the original transient
            if new_memlet.data in ['_'.join(node.split('_')[:-1]) for node in self.graph.nodes._nodes.keys()]:
                outer_memlet.data = new_memlet.data

            # # The following line is needed if the outer memlet is a View. If so, we need to compose slice ranges
            # outer_memlet.subset = outer_memlet.subset.compose(new_memlet.subset)

            # if sdg_scope.SDFG_arrays[new_memlet.data].transient:
            #     return outer_memlet
            
            # a controversial trick for triangular dimensions. If the outer memlet ranges
            # are, e.g., [(0:N),( i + 1:M)], then we cut out this variable offset i and leave
            # only [(0:N),(0:M)]. Otherwise, we will have problems later. However, we also
            # have to keep track which iteraton variable now became the offsets
            # instead of free variable     
            offset_variables = set.union(*[sp.sympify(x).free_symbols for (x,y,z) 
                        in outer_memlet.subset.ranges if (x != y)] + [set()])
            outer_memlet.subset.ranges = [(0,y,z) if (x != y) else (x,y,z)  
                                          for (x,y,z) in outer_memlet.subset.ranges]
            
            if len(offset_variables) > 0:
                # if we have any offset variables, we remove them from new_memlet.subsets                
                subs_list = [(iter,0) for iter in offset_variables]

                if new_memlet.src_subset is not None:
                    new_memlet.src_subset.ranges = [(rng[0].subs(subs_list), rng[1].subs(subs_list), rng[2]) 
                                                for rng in new_memlet.src_subset]
                if new_memlet.dst_subset is not None:
                    new_memlet.dst_subset.ranges = [(rng[0].subs(subs_list), rng[1].subs(subs_list), rng[2]) 
                                                for rng in new_memlet.dst_subset]

                        
        else:
            # if we have a tasklet with a single transient in edge memlet and single transient out edge memlet,
            # and the in_memlet is in the sdfg mapping, then we also map this to the out memlet
            if output and sdg_scope.SDFG_arrays[new_memlet.data].transient and single_in_transient_mapped:
                outer_memlet = single_in_transient_mapped
            else:
                outer_memlet = new_memlet  

        if output:
            a = 1
            outer_memlet.subset = outer_memlet.subset.compose(new_memlet.dst_subset)
        else:
            try:
                outer_memlet.subset = outer_memlet.subset.compose(new_memlet.src_subset)
            except:
                try:
                    outer_memlet.subset = outer_memlet.dst_subset.compose(new_memlet.src_subset)
                except:
                    pass

        if outer_memlet.is_empty():
            return None

        # Remove empty ranges ([0,0,1]). These show up usually if we have a temporary 
        # scalar memlet (e.g.,. tmp[0]). Instead, we will add "proper" ranges from outer maps.
        outer_memlet.subset.ranges = [rng for rng in outer_memlet.subset.ranges 
                                            if len(sp.sympify(rng[0]).free_symbols) > 0]

        # if memlet is declared within a map scope (e.g., we have a "sum" memlet inside an outer map),
        # we add the ranges of the outer map to the memlet.
        # To filter which are the "outer maps", we need to compare outer_memlet._state with 
        # the sdg_scope.sdfg_path, and take all the nodes in the path preceeding it.
        # E.g.: doitgen in polybench kernels. sum[p] is declared inside an outer map[r,q].
        # The resulting memlet will be sum[p,r,q] 
        mem_scope_ind = sdg_scope.sdfg_path.path.index(outer_memlet._state) + 1    
        outer_maps = [map_node for map_node in 
                    sdg_scope.sdfg_path[:mem_scope_ind] 
                    if isinstance(map_node, MapEntry)]
         
        #  We need to  know if the memlet is definied inside a map of a given state
        if outer_memlet._state is not None:
            state = outer_memlet._state
            mem_acc_node = [n for n in state.nx.nodes if isinstance(n, AccessNode) and n.data == outer_memlet.data][0]
            scope = 1
            while scope is not None:                
                scope = state.scope_dict()[mem_acc_node]
                if isinstance(scope, MapEntry):
                    outer_maps.append(scope)
                    mem_acc_node = scope
        
        #if new_memlet != outer_memlet:
        for outer_map in outer_maps:
            map_ranges = {k: v for k, v in zip(outer_map.params, outer_map.range.ranges) if isinstance(k, str)}
            outer_memlet.subset.ranges += rng_to_subset(map_ranges)
            
        if outer_memlet.subset is None or \
                        len(outer_memlet.subset.ranges) == 0 or \
                        len(outer_memlet.subset.ranges[0][0].free_symbols) == 0:
                            
            # TODO: New. We now always add these ranges. Before, we filtered out suspicious memlets            
            # I really don't like the following part... Looks hacky
            if outer_memlet.data in sdg_scope.SDFG_arrays.keys() and  \
                        sdg_scope.SDFG_arrays[outer_memlet.data].transient:  
                if len(sdg_scope.map_ranges) == 0:
                    outer_memlet.subset.ranges = rng_to_subset(sdg_scope.loop_ranges)
                else:
                    outer_memlet.subset.ranges = rng_to_subset(sdg_scope.map_ranges)
            elif outer_memlet.dynamic:
                outer_memlet.subset.ranges = rng_to_subset(sdg_scope.loop_ranges)
            else:  
                warn('Unable to parse memlet ' + str(outer_memlet) 
                    + " in state "  + str(sdg_scope.sdfg_path.path[-1]) + "\n")
                # I SUPER dislike this
                return None           
            if len(outer_memlet.subset.ranges) > 0 and \
                len(sp.sympify(outer_memlet.subset.ranges[0][0]).free_symbols) > 0:
                a = 1
            
        return outer_memlet
    
    # ------------------------------------
    # static SDG functions
    # ------------------------------------
    
    @staticmethod
    def _get_SSA_dim(memlet : dace.Memlet, sdg_scope : SdgScope, 
            preliminary_check = False) -> List[dace.symbol]:
        SSA_dim = []
        if len(sdg_scope.ranges) == 0:
            iter_vars = []
        else:
            iter_vars = set(map(dace.symbol, [v for v in sdg_scope.ranges.keys()]))
        # check if the memlet's array is the same as one of the outputs:
        if preliminary_check or memlet.data in sdg_scope.output_arrays.keys():
            # the same array is used for input and output. Determining the input's base access
            (array_name, base_access, offsets) = get_access_from_memlet(memlet, iter_vars)
            
            if preliminary_check or eq_accesses(base_access, sdg_scope.output_arrays[memlet.data][0]):
                # now we need to determine the SSA dimension
                memlet_iter_vars = set(
                        [str(rng[0].free_symbols.pop())
                        for rng in memlet.subset.ranges
                            if len(rng[0].free_symbols) > 0])
                
                if len(memlet_iter_vars) == 0 and not hasattr(sdg_scope, 'inntermost_ranges'):
                    return SSA_dim
                # TODO: experimental. We need have four pools to choose from to find the SSA_dim.
                # 1. innermost_ranges: these are ranges, either from a map or a loop, that is closest to this statement
                # 2. loop_ranges
                # 3. map_ranges
                # 4. ranges : these are all the ranges in the scope.
                # It is unclear from which pool we should take it
                potential_ssa_pool = set(sdg_scope.innermost_ranges) # set(sdg_scope.ranges) - set(sdg_scope.innermost_ranges) 
                SSA_dim =  [ [dace.symbol(k), dace.symbol(k), 1] \
                                 for k in potential_ssa_pool - memlet_iter_vars]
                
                # we first try to add the SSA_dim as the innermost range (the one from the innermost loop).
                # If it fails, we then look for the SSA dim in the outer ranges.
                if not SSA_dim:
                    SSA_dim =  [ [dace.symbol(k), dace.symbol(k), 1] \
                            for k in set(sdg_scope.loop_ranges) - memlet_iter_vars]
        
        return SSA_dim

    @staticmethod
    def _memlet_ranges_to_iters(memlet : dace.Memlet) -> dace.Memlet:
        iter_memlet = copy.deepcopy(memlet)
        iter_ranges = [(y, y, 1) for (x,y,z) in iter_memlet.subset.ranges]
        iter_memlet.subset.ranges = iter_ranges
        return iter_memlet
            

    
    # ------------------------------------
    # helper functions
    # ------------------------------------

    
    def plot_SDG(self, filename : str = 'SDG.dot') -> None:
        nx.nx_pydot.write_dot(self.graph, filename)



    # ------------------------------------
    # SDG PARTITIONING
    # ------------------------------------
    def perform_loop_swapping(self, node : str, swaplist : Dict[str, str], 
                              visited: Set[str],
                              first : bool) -> bool:
        """
        Propagates loop variables swapping. E.g., if we have a transient, whose 
        input edge is [i, j], but the output edge is [i, k], we try to propagate it
        to all consecutive nodes in the graph, swapping j with k
        """
        if node in visited:
            return True
        visited |= {node}
        inv_swaplist = {v : k for k,v in swaplist.items()}
        
        if not first:
            # if not, then the node is NOT a SOAP statement, but some input array
            if 'st' in self.graph.nodes[node].keys():
                statement = self.graph.nodes[node]['st']
                
                # if '__tmp7' in self.graph.nodes['__tmp8_1']['st'].phis.keys() and \
                #         list(self.graph.nodes['__tmp8_1']['st'].phis['__tmp8'].keys())[0] == list(self.graph.nodes['__tmp8_1']['st'].phis['__tmp7'].keys())[0]:
                #     a = 1
                if node == "__return_1":
                    self.plot_SDG()
                    a = 1
                statement.swap_iter_vars(swaplist, inv_swaplist, self.solver)
                
                # if '__tmp7' in self.graph.nodes['__tmp8_1']['st'].phis.keys() and \
                #         list(self.graph.nodes['__tmp8_1']['st'].phis['__tmp8'].keys())[0] == list(self.graph.nodes['__tmp8_1']['st'].phis['__tmp7'].keys())[0]:
                #     a = 1
                 
            
        valid_swapping = True
        for e in (list(self.out_edges(node)) + list(self.in_edges(node))):            
            in_node = e[0]                
            out_node = e[1]
            if in_node in visited and out_node in visited:
                continue
            _base_access = e[2]["base_access"]
            if any(((iter in swaplist.keys()) or (iter in inv_swaplist.keys())) 
                    for iter in _base_access.split('*')):
                    
                e[2]["base_access"] = swap_in_string( e[2]["base_access"], swaplist, inv_swaplist)
                e[2]["label"] = "[" + ",".join(e[2]["base_access"].split('*')) + "]" + \
                                e[2]["label"].split(']')[1]
            
            valid_swapping = valid_swapping and (self.perform_loop_swapping(in_node, swaplist, visited, False)
                and self.perform_loop_swapping(out_node, swaplist, visited, False))
        
        return valid_swapping
                                
                
                            
                            

            
    
    
    
    # --- preprocessing ---
    def remove_transient_arrays(self) -> None:
        """
        removes transient nodes in the sdfg if there is only a single
        in-edge and single out-edge
        """
        all_nodes = list(nx.topological_sort(self.graph))
        # since we will be removing elemnts from the list, we iterate over the indices
        num_nodes = len(all_nodes)
        i = 0
        while (i < num_nodes):
            node = all_nodes[i]
            if len(self.in_edges(node)) == 1 and len(self.out_edges(node)) == 1:
                if self.graph.nodes[node]['transient'] == True:
                    # check if the accesses match
                    in_access = self.in_edges(node)[0]
                    out_access = self.out_edges(node)[0]
                    # remove this node
                    pred = self.in_edges(node)[0][0]
                    succ = self.out_edges(node)[0][1]
                    
                    edge = self.in_edges(node)[0][2]
                    _label  = edge["label"]
                    _base_access = edge["base_access"]
                    _offsets = edge["offsets"]
                    _wcr = edge["wcr"]
                    base_st = self.graph.nodes[node]['st']
                    next_st = self.graph.nodes[succ]['st']
                    
                    # check if the input base access is the same as the output base access.
                    # If not, we need to propagate loop variable swap:
                    out_base_access = self.out_edges(node)[0][2]["base_access"]
                    
                    invalid_swapping = False
                    if out_base_access != _base_access:
                        if len(out_base_access.split('*')) != len(_base_access.split('*')):
                            invalid_swapping = True
                        else:
                            swaplist = {k : v for k,v in 
                                        dict(zip(out_base_access.split('*'), _base_access.split('*'))).items()
                                        if k != v}
                            visited_nodes = set(nx.single_source_shortest_path(
                                self.graph.reverse(), node).keys())
                            visited_nodes.remove(node)
                            if node == "fc2_1":
                                a = 1
                            if not self.perform_loop_swapping(node, swaplist, visited_nodes, first = True):
                                invalid_swapping = True                            
                            self.plot_SDG()
                            
                            
                        
                    if invalid_swapping:
                        i+=1
                        continue
                    
                    # we need to remove the transient from phis of the next_st,
                    # and instead, add the phis from the base_st
                    del next_st.phis[strip(node)]
                    prev_phi = list(base_st.phis.items())[0]
                    if prev_phi[0] in next_st.phis:
                        next_st.phis[prev_phi[0]] = {**next_st.phis[prev_phi[0]], **prev_phi[1]}
                    else:
                        next_st.phis[prev_phi[0]] = prev_phi[1]
                    
                    self.graph.remove_node(node)
                    self.graph.add_edge(pred, succ, label = _label, 
                        base_access = _base_access, 
                        offsets = _offsets,
                        wcr = _wcr)
                    all_nodes.remove(node)
                    num_nodes -= 1
                    i -= 1
                   
            i+=1
                    
        self.plot_SDG()
    
        

    # we don't merge A and B if there is an edge (A,B) but also (A,C), (C,B)
    # this will be taken care of once we are in C
    def is_shortest_pred(self, source_node: str, pred_node: str) -> bool:
        shortest_pred = True
        for sibling in self.graph.successors(pred_node):   
            if sibling != source_node:
                # our brother has to diverge from our recursive path
                if any([source_node == n for n in nx.descendants(self.graph, sibling)]):
                #any([source_node == n for n in nx.ancestors(self.graph, sibling)]) or \                    
                    shortest_pred = False
                    return shortest_pred
        return shortest_pred

    
    # we don't add w to existing subgraph SG = (SV,SE), V = {B,C,D...} if there is a path
    # from w to any v in SV such that any of the vertices in the path is not in SV.
    # That is, V is a compact set in G (any path between the vertices is in SV is in V)
    def is_complete_tree(self, new_node: str, existing_nodes: Set[str]) -> bool:
        new_subgraph = copy.deepcopy(existing_nodes)
        new_subgraph.add(new_node)
        paths = list(nx.all_simple_paths(self.graph, new_node, existing_nodes))
        return all([all([v in new_subgraph for v in path]) for path in paths ])


    def recursive_SDG_subgraphing(self, node : str, nodes_so_far : Set[str], checked_subgraphs : set) -> List[SoapStatement]:
        if "all_subgraphs" in self.graph.nodes[node].keys():
            return self.graph.nodes[node]["all_subgraphs"]
        base_st = self.graph.nodes[node]['st']
        S = copy.deepcopy(base_st)
        S.name = node
        S.subgraph = set([node])
        sdg_statements = []
        sdg_statements.append(S)
        inner_nodes = copy.deepcopy(nodes_so_far)
        inner_nodes.add(node)

        if node == "__tmp6_1":
            a = 1

        for pred in self.graph.predecessors(node): 
            # merging horizontally - input reuse
            for sibling in self.graph.successors(pred):   
                if sibling != node:
                    # our brother has to diverge from our recursive path
                    # is_divergent = not any([node == n for n in nx.ancestors(self.graph, sibling)]) \
                    #             and not any([node == n for n in nx.descendants(self.graph, sibling)])
                    is_divergent = self.is_complete_tree(sibling, inner_nodes)
                    if is_divergent:
                        s_st = self.graph.nodes[sibling]['st']
                        s_st.name = sibling
                        s_st.subgraph = set([sibling])
                        if "transient" not in self.graph.nodes[sibling].keys():
                            self.graph.nodes[sibling]['transient'] = False

                        # if self.graph.nodes[sibling]['transient'] == True: 
                        #     continue
                        if sibling == "__tmp4_1": # '__tmp9_1;__tmp7_1;__tmp5_1;__tmp4_1'
                            a = 1
                        
                        # if the brother has no other parents than our shared parent, always add - do not branch
                        if len(list(self.graph.predecessors(sibling))) == 1:
                            for cur_stat in sdg_statements:  
                                S = copy.deepcopy(cur_stat)
                                status = S.concatenate_sdg_statements(None, s_st)
                                if status == 0:                                
                                    cur_stat.concatenate_sdg_statements(None, s_st)                            
                        else:
                            sibling_statements = copy.deepcopy(sdg_statements)
                            for sib_stat in sibling_statements:                                 
                                S = copy.deepcopy(sib_stat)
                                status = S.concatenate_sdg_statements(None, s_st)
                                if status == 0 and S.subgraph not in checked_subgraphs.union(set([frozenset(sg.subgraph) for sg in sdg_statements])):
                                    sdg_statements.append(S)    

            pred_arr_name = strip(pred)
            # the first condition catches the case where the scope dimension changed, and the array is no longer transient
            # if S accesses array pred with different base accesses (e.g., A[i,k], A[k,j]), we don't go any deeper
            if pred_arr_name not in base_st.phis.keys() or \
                len(base_st.phis[pred_arr_name]) > 1:
                    continue
                
            # don't fuse over the WCR edge
            edge = self.graph.edges[pred, node, 0]
            if edge['wcr'] == True:
                continue
            
            if len(self.in_edges(pred)) > 0:            
                # TODO: pruning. What's the best strategy here?            
                pred_st = self.graph.nodes[pred]['st']
                if len(pred_st.rhoOpts.free_symbols) > 0 and any([pred_arr_name in in_arr for in_arr in pred_st.phis.keys()]):
                    continue
                if any([len(base_accesses) > 1 for base_accesses in pred_st.phis.values()]):
                    continue
                
                # if len(par_st.rhoOpts.free_symbols) == 0: 
                # if not e[2]['wcr']:           # e[2] is the dict of properties of the edge     
                # if all([self.is_shortest_pred(n, pred) for n in inner_nodes]):
                if self.is_complete_tree(pred, inner_nodes):
                    pred_statements = self.recursive_SDG_subgraphing(pred, inner_nodes, checked_subgraphs)   

                    # perform all-to-all possible mergings
                    sibling_statements = copy.deepcopy(sdg_statements)
                    for sib_stat in sibling_statements:
                        for pred_stat in pred_statements:      
                            # check if the concatenation crosses a WCR edge
                            if not self.crosses_wcr(pred_stat.name, sib_stat.name):
                                S = copy.deepcopy(sib_stat)
                                status = S.concatenate_sdg_statements(pred, pred_stat)
                                if status == 0 and S.subgraph not in checked_subgraphs.union(set([frozenset(sg.subgraph) for sg in sdg_statements])):
                                    sdg_statements.append(S)    
  

        self.graph.nodes[node]["all_subgraphs"] = sdg_statements
        
        return sdg_statements



    def crosses_wcr(self, subgraph_1 : str, subgraph_2 : str) -> bool:
        nodes_1 = subgraph_1.split(';')
        nodes_2 = subgraph_2.split(';')
        for node_1 in nodes_1:
            for node_2 in nodes_2:
                if [node_1, node_2, 0] in self.graph.edges:
                    edge = self.graph.edges[node_1, node_2, 0]
                    if edge['wcr'] == True:
                        return True
                if [node_2, node_1, 0] in self.graph.edges:
                    edge = self.graph.edges[node_2, node_1, 0]
                    if edge['wcr'] == True:
                        return True
        return False


    def propagate_SDG_rho(self, S : SoapStatement) -> None:    
        SDG_subgraph  = S.name.split(';')
        for SDG_node in SDG_subgraph:
            node_S = self.graph.nodes[SDG_node]['st']
            if not node_S.parent_subgraph:
                node_S.parent_subgraph = S
            else:
                # find the best embedding subgraph
                [better_subgraph, Q_val] = compare_st(S, node_S.parent_subgraph)
                node_S.parent_subgraph = better_subgraph


    # structure-aware partitioning
    def calculate_IO_of_SDG(self) -> Tuple[sp.core.Expr, list[SoapStatement]]:
        """
        Exhaustively creates all possible subgraphs (using recursive_SDG_subgraphing),
        and then chooses the best (with the lowest I/O cost Q) SDG partition (using compare_st).
        """
        # clean up the SDG.
        # 1. remove intermediate transients:
        self.remove_transient_arrays()
        # 2. remove output transients:
        sinks = [u for u, deg in self.graph.out_degree() if not deg]
        final_sinks = [u for u in sinks if not self.graph.nodes[u]['transient']]
        if len(final_sinks) > 0:
            for v in sinks:
                if self.graph.nodes[v]['transient']:
                    self.graph.remove_node(v)
        self.plot_SDG()   
        
        Q_total = sp.sympify(0)
        ret_subgraphs = []
        checked_subgraphs = set()
        all_nodes = list(reversed(list(nx.topological_sort(self.graph))))
        processed_nodes = []
        for node in all_nodes: #final_sinks: #all_nodes:
            # input node
            if len(self.in_edges(node)) == 0:
                continue

            # optimal rho already found for the node
            if node in processed_nodes:
                continue

            subgraph_opt = SoapStatement()
            subgraph_opt.Q = 0        
            Q_opt_val = 0
            sdg_subgraphs_statements = self.recursive_SDG_subgraphing(node, set(), checked_subgraphs)
            # append checked sugraphs from this node to the global set
            checked_subgraphs = checked_subgraphs.union(set([frozenset(sg.subgraph) for sg in sdg_subgraphs_statements]))

            for subgraph_st in sdg_subgraphs_statements:    
                if subgraph_st.name == 'A_1;__tmp8_1;__return_1_2':
                    a = 1
                subgraph_st.solve(self.solver)     
                if subgraph_st.parent_subgraph:
                    [better_subgraph, Q_val] = compare_st(subgraph_st, subgraph_st.parent_subgraph)
                    subgraph_st.rhoOpts = better_subgraph.rhoOpts
                    subgraph_st.name = better_subgraph.name
                    subgraph_st.Q = subgraph_st.V / subgraph_st.rhoOpts                
                [subgraph_opt, Q_opt_val] = compare_st(subgraph_st, subgraph_opt, Q_opt_val)

            self.graph.nodes[node]['st'] = subgraph_opt
            processed_nodes += subgraph_opt.name.split(';')
            
            Q_total = sp.simplify(Q_total + subgraph_opt.Q)
            ret_subgraphs.append(subgraph_opt)       

        return Q_total, ret_subgraphs


    # structure-aware WD analysis
    def CalculateWDofSDG(self) -> Tuple[sp.Expr, sp.Expr]:
        # set edge weights to match destination node weights. We need numerical values
        # potentialParams = ['n']                
        potentialParams = ['n', 'm', 'w', 'h', 'N', 'M', 'W', 'H', 'NI', 'NJ', 'NK', 'NP', 'NQ', 'NR', 'step']  
        for v in self.graph.nodes:
            if "st" in self.graph.nodes[v].keys():
                subsList = []
                v_D = self.graph.nodes[v]['st'].D
                for symbol in v_D.free_symbols:
                    if any([(param in str(symbol)) for param in potentialParams]):
                        subsList.append([symbol, 1000])                           
                v_D = v_D.subs(subsList)  
                for e in self.graph.in_edges(v):
                    self.graph[e[0]][e[1]][0]['weight'] = v_D
        critical_path = nx.dag_longest_path(self.graph)
        D = sum([self.graph.nodes[v]['st'].D for v in critical_path if "st" in self.graph.nodes[v].keys()])
        W = sum([self.graph.nodes[v]['st'].W for v in self.graph.nodes if "st" in self.graph.nodes[v].keys()])
        return [W,D]