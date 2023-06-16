from typing import List
import sympy

import dace
from dace.sdfg import graph as gr
from dace.transformation import transformation as xf
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
# from dace.transformation.helpers import redirect_edge
from dace.sdfg.utils import change_edge_src, change_edge_dest


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
    inner_loop_guards: List[dace.SDFGState]
    inner_loop_entries: List[dace.SDFGState]
    inner_loop_exits: List[dace.SDFGState]

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        loop_info = find_for_loop(sdfg, self.loop_guard, self.loop_begin)
        if not loop_info:
            return False

        this_itervar, (this_start, this_end, this_step), _ = loop_info
        print(f"[SwapLoopOrder::can_be_applied] guard: {self.loop_guard}, begin: {self.loop_begin}, "
              f"exit: {self.exit_state} this_itervar: {this_itervar}")
        new_guard = self.loop_begin
        self.inner_loop_guards = []
        self.inner_loop_entries = []
        self.inner_loop_exits = []
        while len(sdfg.out_edges(new_guard)) == 1:
            new_guard = sdfg.out_edges(new_guard)[0].dst
        for edge in sdfg.out_edges(new_guard):
            # print(f"[SwapLoopOrder::can_be_applied] Try to find next loop: guard: {new_guard}, entry: {edge.dst}")
            next_loop_info = find_for_loop(sdfg, new_guard, edge.dst)
            # The selection of the loop to have as the outer one is very hacky
            if next_loop_info is not None:
                next_itervar, (_, next_end, _), (_, _) = next_loop_info
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

                    print(f"[SwapLoopOrder::can_be_applied] Found next loop with guard: {new_guard} "
                          f"entry: {new_entry} and exit: {new_exit}")
                    self.inner_loop_guards.append(new_guard)
                    self.inner_loop_entries.append(new_entry)
                    self.inner_loop_exits.append(new_exit)
                # else:
                #     print(f"[SwapLoopOrder::can_be_applied] Not next loop. loop_to_move: {loop_to_move(next_end)} "
                #           f"next_end: {next_end}, new_guard: {new_guard}, next_itervar: {next_itervar}")

        return len(self.inner_loop_guards) > 0

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        # Swap the two loops
        for inner_guard, inner_entry, inner_exit in \
             zip(self.inner_loop_guards, self.inner_loop_entries, self.inner_loop_exits):
            # swap_nodes(inner_guard, self.loop_guard, sdfg)
            # Don't want to completely swap nodes, only edge to/from body
            outer_in_edges = sdfg.in_edges(self.loop_guard)
            inner_in_edges = sdfg.in_edges(inner_guard)

            # Change all incoming edges
            # TODO: Need to include edges coming from exit node
            print("Swap incoming edges of the guards")
            # Assume guards have two incoming edges
            assert len(outer_in_edges)  == 2 and len(inner_in_edges) == 2

            # Get the edges going to the guard not comming from the exit state
            outer_init_edge = outer_in_edges[0]
            inner_init_edge = inner_in_edges[0]
            if outer_init_edge.src == self.exit_state:
                outer_in_edges = outer_in_edges[1]
            if inner_init_edge.src == inner_exit:
                inner_in_edges = inner_in_edges[1]

            swap_edge_dst(graph, outer_init_edge, self.loop_guard, inner_init_edge, inner_guard)


            # for edge in outer_in_edges:
            #     if edge.src != inner_exit:
            #         change_dst_of_edge(sdfg, edge, inner_guard)
            # for edge in inner_in_edges:
            #     if edge.src != self.exit_state:
            #         change_dst_of_edge(sdfg, edge, self.loop_guard)

            # Change edges to the loop entry/body
            outer_body_edges_in = sdfg.edges_between(self.loop_guard, self.loop_begin)
            outer_body_edges_out = sdfg.edges_between(self.loop_begin, self.loop_guard)
            inner_body_edges_in = sdfg.edges_between(inner_guard, inner_entry)
            inner_body_edges_out = sdfg.edges_between(inner_entry, inner_guard)
            
            print("Swap edges with regard to the loop begins")
            for edge in outer_body_edges_in:
                change_src_of_edge(sdfg, edge, inner_guard)
            # for edge in outer_body_edges_out:
            #     change_dst_of_edge(sdfg, edge, inner_guard)
            for edge in inner_body_edges_in:
                change_src_of_edge(sdfg, edge, self.loop_guard)
            # for edge in inner_body_edges_out:
            #     change_dst_of_edge(sdfg, edge, self.loop_guard)

            # Need to swap source of the exit nodes, but "keep" the condition
            inner_guard_exit_edge = sdfg.edges_between(inner_guard, inner_exit)[0]
            outer_guard_exit_edge = sdfg.edges_between(self.loop_guard, self.exit_state)[0]
            sdfg.remove_edge(inner_guard_exit_edge)
            sdfg.remove_edge(outer_guard_exit_edge)
            sdfg.add_edge(self.loop_guard, inner_guard_exit_edge.dst, outer_guard_exit_edge.data)
            sdfg.add_edge(inner_guard, outer_guard_exit_edge.dst, inner_guard_exit_edge.data)
            # Edges away from inner exit
            for edge in sdfg.out_edges(inner_exit):
                change_dst_of_edge(sdfg, edge, inner_guard)

        print("[SwapLoopOrder::apply] done")
        return sdfg
