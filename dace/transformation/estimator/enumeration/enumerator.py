""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.transformation.estimator import ScoringFunction
import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools
import warnings 


@make_properties
class Enumerator:
    '''
    Base Enumerator Class
    '''
    mode = Property(desc="Data type the Iterator should return. "
                    "Choice between Subgraph and List of Map Entries.",
                    default="map_entries",
                    choices=["subgraph", "map_entries"],
                    dtype=str)

    debug = Property(desc="Debug mode", default=True, dtype=bool)


    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = None):

        self._sdfg = sdfg
        self._graph = graph
        self._scope_children = graph.scope_children()
        self._condition_function = condition_function

        # get hightest scope maps
        self._map_entries = helpers.get_outermost_scope_maps(
            sdfg, graph, subgraph)
        self._max_length = len(self._map_entries)

      
        # for memorization purposes
        self._histogram = None
       
    def iterator(self):
        '''
        iterator interface to implement
        '''
        # Interface to implement
        raise NotImplementedError
    
 
    def list(self):
        return list(e for e in self.iterator())
       
    def __iter__(self):
        yield from self.iterator()

    def calculate_topology(self, subgraph):
        ''' Calculates topology information of the graph 
        self._adjacency_list: neighbors dict of outermost scope maps  
        self._source_maps: outermost scope maps that have in_degree 0 in the subgraph / graph 
        self._labels: assigns index according to topological ordering (1) + node ID (2) with priorities (1) and (2)
        '''
        sdfg = self._sdfg 
        graph = self._graph

        self._adjacency_list = {m: set() for m in self._map_entries}
        # helper dict needed for a quick build
        exit_nodes = {graph.exit_node(me): me for me in self._map_entries}
        if subgraph:
            proximity_in = set(ie.src for me in self._map_entries for ie in graph.in_edges(me))
            proximity_out = set(ie.dst for me in exit_nodes for ie in graph.out_edges(me))
            extended_subgraph = SubgraphView(graph, set(itertools.chain(subgraph.nodes(), proximity_in, proximity_out)))

        
        for node in (extended_subgraph.nodes() if subgraph else graph.nodes()):
            if isinstance(node, nodes.AccessNode):
                adjacent_entries = set()
                for e in graph.in_edges(node):
                    if isinstance(e.src, nodes.MapExit) and e.src in exit_nodes:
                        adjacent_entries.add(exit_nodes[e.src])
                for e in graph.out_edges(node):
                    if isinstance(
                            e.dst,
                            nodes.MapEntry) and e.dst in self._map_entries:
                        adjacent_entries.add(e.dst)

                # bidirectional mapping
                for entry in adjacent_entries:
                    for other_entry in adjacent_entries:
                        if entry != other_entry:
                            self._adjacency_list[entry].add(other_entry)
                            self._adjacency_list[other_entry].add(entry)
        
        

        # get DAG children and parents 
        children_dict = defaultdict(set)
        parent_dict = defaultdict(set)

        for map_entry in self._map_entries:
            map_exit = graph.exit_node(map_entry)
            for e in graph.out_edges(map_exit):
                if isinstance(e.dst, nodes.AccessNode):
                    for oe in graph.out_edges(e.dst):
                        if oe.dst in self._map_entries:
                            other_entry = oe.dst
                            children_dict[map_entry].add(other_entry)
                            parent_dict[other_entry].add(map_entry)
   

        # find out source nodes 
        self._source_maps = [me for me in self._map_entries if len(parent_dict[me]) == 0]
        # assign a unique id to each map entry according to topological
        # ordering. If on same level, sort according to ID for determinism

        self._labels = {} # map -> ID 
        current_id = 0
        while current_id < len(self._map_entries):
            # get current ids whose in_degree is 0 
            candidates = list(me for (me, s) in parent_dict.items() if len(s) == 0 and me not in self._labels)
            candidates.sort(key = lambda me: self._graph.node_id(me))
            for c in candidates:
                self._labels[c] = current_id 
                current_id += 1
                # remove candidate for each players adjacency list 
                for c_child in children_dict[c]:
                    parent_dict[c_child].remove(c)
        


    def histogram(self, visual=True, cached=True):
        ''' Prints a pretty (close enough) histogram
        that lists the number of subgraphs 
        and the frequency of subgraphs with a 
        certain amount of maps
        '''
        if self._histogram is None and cached:
            old_mode = self.mode
            self.mode = 'map_entries'
            lst = self.list()
            self._histogram = {}
            for i in range(1, 1 + self._max_length):
                no_elements = sum([len(sg) == i for sg in lst])
                self._histogram[i] = no_elements
            self.mode = old_mode

        if visual:
            print("*** Subgraph Statistics ***")
            for (i, no_elements) in sorted(self._histogram.items(),
                                           key=lambda a: a[0]):
                print(i, no_elements, "*" * no_elements)

        return self._histogram
