from typing import List
import sympy

import dace
from dace.sdfg import graph as gr
from dace.transformation import transformation as xf
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
# from dace.transformation.helpers import redirect_edge
from dace.sdfg.utils import change_edge_src, change_edge_dest


class LoopStates():
    guard_state: dace.SDFGState
    entry_state: dace.SDFGState
    exit_state: dace.SDFGState
    itervar: str
    iter_start: int

    def __init__(self, guard_state: dace.SDFGState, entry_state: dace.SDFGState, exit_state: dace.SDFGState,
                 itervar: str, iter_start: int):
        self.guard_state = guard_state
        self.entry_state = entry_state
        self.exit_state = exit_state
        self.itervar = itervar
        self.iter_start = iter_start

    def __str__(self) -> str:
        return f"guard: {self.guard_state} entry: {self.entry_state} exit: {self.exit_state}"

    def get_init_edge(self, graph: gr.OrderedDiGraph) -> dace.sdfg.graph.Edge:
        # Assume each guard has 2 incoming edges. One from end of loop and one from outside
        assert len(graph.in_edges(self.guard_state)) == 2
        edge = graph.in_edges(self.guard_state)[0]
        assignment = edge.data.assignments
        if len(assignment) != 1 or self.itervar not in assignment or assignment[self.itervar] != str(self.iter_start):
            edge = graph.in_edges(self.guard_state)[1]
        return edge

    def get_end_loop_body_guard_edge(self, graph: gr.OrderedDiGraph) -> dace.sdfg.graph.Edge:
        # Assume each guard has 2 incoming edges. One from end of loop and one from outside
        assert len(graph.in_edges(self.guard_state)) == 2
        edge = graph.in_edges(self.guard_state)[0]
        assignment = edge.data.assignments
        if len(assignment) != 1 or self.itervar not in assignment or assignment[self.itervar] == str(self.iter_start):
            edge = graph.in_edges(self.guard_state)[1]
        return edge

    def get_guard_entry_edge(self, graph: gr.OrderedDiGraph) -> dace.sdfg.graph.Edge:
        edges = graph.edges_between(self.guard_state, self.entry_state)
        assert len(edges) == 1
        return edges[0]

    def get_guard_exit_edge(self, graph: gr.OrderedDiGraph) -> dace.sdfg.graph.Edge:
        edges = graph.edges_between(self.guard_state, self.exit_state)
        assert len(edges) == 1
        return edges[0]


def swap_edge_dst(graph: gr.OrderedDiGraph, edge1: dace.sdfg.graph.Edge, node1: dace.SDFGState,
                  edge2: dace.sdfg.graph.Edge, node2: dace.SDFGState):
    """
    Swaps the destinations of the two given edges. Makes sure that the data of the edge changes with the destination
    node

    :param graph: The graph to act upon
    :type graph: gr.OrderedDiGraph
    :param edge1: The first edge
    :type edge1: dace.sdfg.graph.Edge
    :param node1: The original destination of the first edge
    :type node1: dace.SDFGState
    :param edge2: The second edge
    :type edge2: dace.sdfg.graph.Edge
    :param node2: The original destination of the second edge
    :type node2: dace.SDFGState
    """

    graph.remove_edge(edge1)
    graph.remove_edge(edge2)
    print(f"Change: {edge1.src} -> {edge1.dst} ({edge1.data.condition.as_string}) to "
          f"{edge1.src} -> {node2} ({edge2.data.condition.as_string})")
    print(f"Change: {edge2.src} -> {edge2.dst} ({edge2.data.condition.as_string}) to "
          f"{edge2.src} -> {node1} ({edge1.data.condition.as_string})")
    graph.add_edge(edge1.src, node2, edge2.data)
    graph.add_edge(edge2.src, node1, edge1.data)


def swap_edge_src(graph: gr.OrderedDiGraph, edge1: dace.sdfg.graph.Edge, node1: dace.SDFGState,
                  edge2: dace.sdfg.graph.Edge, node2: dace.SDFGState):
    """
    Swaps the sources of the two given edges. Makes sure that the data of the edge changes with the source
    node

    :param graph: The graph to act upon
    :type graph: gr.OrderedDiGraph
    :param edge1: The first edge
    :type edge1: dace.sdfg.graph.Edge
    :param node1: The original destination of the first edge
    :type node1: dace.SDFGState
    :param edge2: The second edge
    :type edge2: dace.sdfg.graph.Edge
    :param node2: The original destination of the second edge
    :type node2: dace.SDFGState
    """

    graph.remove_edge(edge1)
    graph.remove_edge(edge2)
    print(f"Change: {edge1.src} -> {edge1.dst} ({edge1.data.condition.as_string}) to "
          f"{node2} -> {edge1.dst} ({edge2.data.condition.as_string})")
    print(f"Change: {edge2.src} -> {edge2.dst} ({edge2.data.condition.as_string}) to "
          f"{node1} -> {edge2.dst} ({edge1.data.condition.as_string})")
    graph.add_edge(node2, edge1.dst, edge2.data)
    graph.add_edge(node1, edge2.dst, edge1.data)


def change_dst_of_edge(graph: gr.OrderedDiGraph, edge: dace.sdfg.graph.Edge, new_dst: dace.SDFGState):
    print(f"Change destination of edge {edge.src} -> {edge.dst} to {new_dst} with condition: "
          f"{edge.data.condition.as_string}")
    graph.remove_edge(edge)
    if isinstance(edge, gr.MultiConnectorEdge):
        graph.add_edge(edge.src, edge.src_conn, new_dst, edge.dst_conn, edge.data)
    else:
        graph.add_edge(edge.src, new_dst, edge.data)


def change_src_of_edge(graph: gr.OrderedDiGraph, edge: dace.sdfg.graph.Edge, new_src: dace.SDFGState):
    print(f"Change source of edge {edge.src} -> {edge.dst} to {new_src} with condition: "
          f"{edge.data.condition.as_string}")
    graph.remove_edge(edge)
    if isinstance(edge, gr.MultiConnectorEdge):
        graph.add_edge(new_src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
    else:
        graph.add_edge(new_src, edge.dst, edge.data)


def swap_nodes(node1: dace.SDFGState, node2: dace.SDFGState, sdfg: dace.SDFG):
    print(f"[SwapLoopOrder::swap_nodes] swap {node1} with {node2}")
    node1_in_edges = sdfg.in_edges(node1)
    node1_out_edges = sdfg.out_edges(node1)
    node2_in_edges = sdfg.in_edges(node2)
    node2_out_edges = sdfg.out_edges(node2)

    # change_edge_dest(sdfg, node2, node1)
    # change_edge_src(sdfg, node1, node2)

    # TODO: Can't use redirect_edge as it assumes edge has connectors
    for edge in node1_in_edges:
        change_dst_of_edge(sdfg, edge, node2)
    for edge in node1_out_edges:
        print(f"Change source of edge {edge.src} -> {edge.dst} to {node2}")
        change_src_of_edge(sdfg, edge, node2)
    for edge in node2_in_edges:
        change_dst_of_edge(sdfg, edge, node1)
    for edge in node2_out_edges:
        print(f"Change source of edge {edge.src} -> {edge.dst} to {node1}")
        change_src_of_edge(sdfg, edge, node1)


def loop_to_move(this_end: dace.symbolic.SymbolicType):
    if isinstance(this_end, sympy.core.add.Add):
        if str(this_end) == 'KLEV - 1':
            return True
    return False


class SwapLoopOrder(DetectLoop, xf.MultiStateTransformation):
    # TODO: Is it possible to have multiple inner loops?
    inner_loops: List[LoopStates]
    outer_loop: LoopStates

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        loop_info = find_for_loop(sdfg, self.loop_guard, self.loop_begin)
        if not loop_info:
            return False

        this_itervar, (this_start, this_end, this_step), _ = loop_info
        self.outer_loop = LoopStates(self.loop_guard, self.loop_begin, self.exit_state, this_itervar, this_start)
        print(f"[SwapLoopOrder::can_be_applied] guard: {self.loop_guard}, begin: {self.loop_begin}, "
              f"exit: {self.exit_state} this_itervar: {this_itervar}")
        new_guard = self.loop_begin
        self.inner_loops = []
        while len(sdfg.out_edges(new_guard)) == 1:
            new_guard = sdfg.out_edges(new_guard)[0].dst
        for edge in sdfg.out_edges(new_guard):
            # print(f"[SwapLoopOrder::can_be_applied] Try to find next loop: guard: {new_guard}, entry: {edge.dst}")
            next_loop_info = find_for_loop(sdfg, new_guard, edge.dst)
            # The selection of the loop to have as the outer one is very hacky
            if next_loop_info is not None:
                next_itervar, (next_start, next_end, _), (_, _) = next_loop_info
                if this_itervar != next_itervar \
                        and loop_to_move(next_end) \
                        and new_guard != self.loop_guard:
                    # A loop guard has two outgoing edges, one to the entry and one to the exit state
                    new_guard_out_edges = list(sdfg.out_edges(new_guard))
                    # Assign entry and exit accordingly
                    assert len(new_guard_out_edges) == 2
                    new_exit = new_guard_out_edges[0].dst
                    if new_exit == edge.dst:
                        new_exit = new_guard_out_edges[1].dst
                    new_entry = edge.dst

                    inner_loop = LoopStates(new_guard, new_entry, new_exit, next_itervar, next_start)
                    print(f"[SwapLoopOrder::can_be_applied] Found next loop {inner_loop}")
                    self.inner_loops.append(inner_loop)
                # else:
                #     print(f"[SwapLoopOrder::can_be_applied] Not next loop. loop_to_move: {loop_to_move(next_end)} "
                #           f"next_end: {next_end}, new_guard: {new_guard}, next_itervar: {next_itervar}")

        return len(self.inner_loops) > 0

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        # Swap the two loops
        for inner_loop in self.inner_loops:
            print("[SwapLoopOrder::apply] Swapping loops with:")
            print(f"[SwapLoopOrder::apply] Outer loop: {self.outer_loop}")
            print(f"[SwapLoopOrder::apply] Inner loop: {inner_loop}")

            # Swap edges going from init to guard
            print("1. Swap Edges going from init to guard")
            swap_edge_dst(sdfg, inner_loop.get_init_edge(sdfg), inner_loop.guard_state,
                          self.outer_loop.get_init_edge(sdfg), self.outer_loop.guard_state)
            # Swap edges going from guard to the loop begin/entry
            print("2. Swap Edges going from guard to the loop begin")
            swap_edge_src(sdfg, inner_loop.get_guard_entry_edge(sdfg), inner_loop.guard_state,
                          self.outer_loop.get_guard_entry_edge(sdfg), self.outer_loop.guard_state)
            # Swap edges going from guard to loop exit
            print("3. Swap Edges going from guard to the loop exit")
            swap_edge_dst(sdfg, inner_loop.get_guard_exit_edge(sdfg), inner_loop.exit_state,
                          self.outer_loop.get_guard_exit_edge(sdfg), self.outer_loop.exit_state)
            # Swap edges going from end of loop body to guard
            print("4. Swap Edges going from end of loop body to the guard")
            swap_edge_dst(sdfg, inner_loop.get_end_loop_body_guard_edge(sdfg), inner_loop.guard_state,
                          self.outer_loop.get_end_loop_body_guard_edge(sdfg), self.outer_loop.guard_state)

        print("[SwapLoopOrder::apply] done")
        return sdfg
