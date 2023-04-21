import argparse
from collections import deque
from dace.sdfg import nodes as nd, propagation, InterstateEdge, utils as sdutil
from dace import SDFG, SDFGState
from dace.subsets import Range
from typing import Tuple, Dict
import os
import sympy as sp
import networkx as nx


def sdfg_work_depth(sdfg: SDFG, sym_map: Dict[str, int]) -> Tuple[sp.Expr, sp.Expr]:
    print('Analyzing work and depth of SDFG', sdfg.name)
    print('SDFG has', len(sdfg.nodes()), 'states')
    print('Calculating work and depth for all states individually...')

    state_depths: Dict[SDFGState, sp.Expr] = {}
    state_works: Dict[SDFGState, sp.Expr] = {}
    for state in sdfg.nodes():
        state_work, state_depth = state_work_depth(state, sym_map)
        if state.executions == 0:
            print('State executions must be statically known exactly or with an upper bound. Offender:', state)
        state_works[state] = state_work * state.executions
        state_depths[state] = state_depth * state.executions

    print('Calculating work and depth of the SDFG...')

    # Prepare the SDFG for a detph analysis by 'inlining' loops. This removes the edge between the guard and the exit
    # state and the edge between the last loop state and the guard, and instead places an edge between the last loop
    # state and the exit state. Additionally, construct a dummy exit state and connect every state that has no outgoing
    # edges to it.
    dummy_exit = sdfg.add_state('dummy_exit')
    for state in sdfg.nodes():
        if hasattr(state, 'condition_edge') and hasattr(state, 'is_loop_guard') and state.is_loop_guard:
            loop_begin = state.condition_edge.dst
            loop_states = set(sdutil.dfs_conditional(sdfg, sources=[loop_begin], condition=lambda _, s: s != state))
            loop_exit = None
            exit_edge = None
            loop_end = None
            end_edge = None
            for iedge in sdfg.in_edges(state):
                if iedge.src in loop_states:
                    end_edge = iedge
                    loop_end = iedge.src
            for oedge in sdfg.out_edges(state):
                if oedge.dst not in loop_states:
                    loop_exit = oedge.dst
                    exit_edge = oedge

            if loop_exit is None or loop_end is None:
                raise RuntimeError('Failed to analyze the depth of a loop starting at', state)

            sdfg.remove_edge(exit_edge)
            sdfg.remove_edge(end_edge)
            sdfg.add_edge(loop_end, loop_exit, InterstateEdge())

        if len(sdfg.out_edges(state)) == 0 and state != dummy_exit:
            sdfg.add_edge(state, dummy_exit, InterstateEdge())

    depth_map: Dict[SDFGState, sp.Expr] = {}
    work_map: Dict[SDFGState, sp.Expr] = {}
    state_depths[dummy_exit] = sp.sympify(0)
    state_works[dummy_exit] = sp.sympify(0)

    # Perform a BFS traversal of the state machine and calculate the maximum depth at each state. Only advance to the
    # next state in the BFS if all incoming edges have been visited, to ensure the maximum depth expression has been
    # calculated.
    traversal_q = deque()
    traversal_q.append((sdfg.start_state, sp.sympify(0), sp.sympify(0), None))
    visited = set()
    while traversal_q:
        state, depth, work, ie = traversal_q.popleft()

        if ie is not None:
            visited.add(ie)

        n_depth = sp.simplify(depth + state_depths[state]).subs(sym_map)
        n_work = sp.simplify(work + state_works[state]).subs(sym_map)

        if state in depth_map:
            depth_map[state] = sp.Max(depth_map[state], n_depth)
        else:
            depth_map[state] = n_depth

        if state in work_map:
            work_map[state] = sp.Max(work_map[state], n_work)
        else:
            work_map[state] = n_work

        out_edges = sdfg.out_edges(state)
        if any(iedge not in visited for iedge in sdfg.in_edges(state)):
            pass
        else:
            for oedge in out_edges:
                traversal_q.append((oedge.dst, n_depth, n_work, oedge))

    max_depth = depth_map[dummy_exit]
    max_work = work_map[dummy_exit]

    print('SDFG', sdfg.name, 'processed')

    return sp.simplify(max_work).subs(sym_map), sp.simplify(max_depth).subs(sym_map)


def scope_work_depth(state: SDFGState, sym_map: Dict[str, int], entry: nd.EntryNode = None) -> Tuple[sp.Expr, sp.Expr]:
    proxy = nx.DiGraph()
    source = nd.Node()
    sink = nd.Node()
    node_depths = {}
    proxy.add_node(source)
    proxy.add_node(sink)

    # TODO: Document this. SC deadline..
    work = sp.sympify(0)
    max_depth = sp.sympify(0)
    scope_nodes = state.scope_children()[entry]
    scope_exit = None if entry is None else state.exit_node(entry)
    for node in scope_nodes:
        if isinstance(node, nd.EntryNode):
            s_work, s_depth = scope_work_depth(state, sym_map, node)
            work += s_work
            node_depths[node] = s_depth

            iedges = state.in_edges(node)
            if len(iedges) == 0 and not proxy.has_edge(source, node):
                proxy.add_edge(source, node)
            else:
                for iedge in iedges:
                    if iedge.src == entry:
                        if not proxy.has_edge(source, node):
                            proxy.add_edge(source, node)
                    elif not proxy.has_edge(iedge.src, node) and not isinstance(iedge.src, nd.ExitNode):
                        proxy.add_edge(iedge.src, node)
            exit_node = state.exit_node(node)
            oedges = state.out_edges(exit_node)
            if len(oedges) == 0 and not proxy.has_edge(node, sink):
                proxy.add_edge(node, sink)
            else:
                for oedge in oedges:
                    if not proxy.has_edge(node, oedge.dst) and not isinstance(oedge.dst, nd.ExitNode):
                        proxy.add_edge(node, oedge.dst)
        elif node == scope_exit:
            iedges = state.in_edges(node)
            if len(iedges) == 0 and not proxy.has_edge(source, node):
                proxy.add_edge(source, node)
            else:
                for iedge in state.in_edges(node):
                    if iedge.src == entry:
                        if not proxy.has_edge(source, sink):
                            proxy.add_edge(source, sink)
                    elif not proxy.has_edge(iedge.src, sink) and not isinstance(iedge.src, nd.ExitNode):
                        proxy.add_edge(iedge.src, sink)
        else:
            if isinstance(node, nd.Tasklet):
                work += 1
                node_depths[node] = sp.sympify(1)
            elif isinstance(node, nd.NestedSDFG):
                nsdfg_work, nsdfg_depth = sdfg_work_depth(node.sdfg, sym_map)
                work += nsdfg_work
                node_depths[node] = nsdfg_depth

            iedges = state.in_edges(node)
            if len(iedges) == 0 and not proxy.has_edge(source, node):
                proxy.add_edge(source, node)
            else:
                for iedge in state.in_edges(node):
                    if iedge.src == entry:
                        if not proxy.has_edge(source, node):
                            proxy.add_edge(source, node)
                    elif not proxy.has_edge(iedge.src, node) and not isinstance(iedge.src, nd.ExitNode):
                        proxy.add_edge(iedge.src, node)
            oedges = state.out_edges(node)
            if len(oedges) == 0 and not proxy.has_edge(node, sink):
                proxy.add_edge(node, sink)
            else:
                for oedge in oedges:
                    if not proxy.has_edge(node, oedge.dst) and not isinstance(oedge.dst, nd.ExitNode):
                        proxy.add_edge(node, oedge.dst)
    if entry is not None:
        if isinstance(entry, nd.MapEntry):
            nmap: nd.Map = entry.map
            range: Range = nmap.range
            n_exec = range.num_elements_exact()
            work = work * sp.simplify(n_exec).subs(sym_map)
        else:
            print('WARNING: Only Map scopes are supported in work analysis for now. Assuming 1 iteration.')

    for path in nx.all_simple_paths(proxy, source, sink):
        depth = sp.sympify(0)
        for node in path:
            if node in node_depths:
                depth += node_depths[node]
        max_depth = sp.Max(max_depth, depth)

    return sp.simplify(work).subs(sym_map), sp.simplify(max_depth).subs(sym_map)

def state_work_depth(state: SDFGState, sym_map: Dict[str, int]) -> Tuple[sp.Expr, sp.Expr]:
    work, depth = scope_work_depth(state, sym_map, None)
    return sp.simplify(work).subs(sym_map), sp.simplify(depth).subs(sym_map)


def analyze_sdfg(sdfg: SDFG, sym_map: Dict[str, int]) -> Tuple[sp.Expr, sp.Expr]:
    print('Propagating states...')
    for sd in sdfg.all_sdfgs_recursive():
        propagation.propagate_states(sd)

    print('Checking for dynamically unbounded executions...')
    for sd in sdfg.all_sdfgs_recursive():
        if any([s.executions == 0 and s.dynamic_executions for s in sd.nodes()]):
            print('WARNING: SDFG has dynamic executions. The analysis may fail in unexpected ways or be inaccurate.')

    print('Analyzing SDFG...')
    work, depth = sdfg_work_depth(sdfg, sym_map)

    simplified_work = sp.simplify(work).subs(sym_map)
    simplified_depth = sp.simplify(depth).subs(sym_map)

    return simplified_work, simplified_depth


################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################

class keyvalue(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for v in values:
            k, v = v.split('=')
            getattr(namespace, self.dest)[k] = v


def main() -> None:
    parser = argparse.ArgumentParser(
        'work_depth_analysis',
        usage='python work_depth_analysis.py [-h] filename',
        description='Analyze the work/depth of an SDFG.'
    )

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--kwargs', nargs='*', help='Define symbols.', action=keyvalue)

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    print('Loading SDFG...')
    sdfg = SDFG.from_file(args.filename)

    symbols_map = {}
    for k, v in args.kwargs.items():
        symbols_map[k] = int(v)

    work, depth = analyze_sdfg(sdfg, symbols_map)

    print('=' * 80)
    print('Work:', work)
    print('Free symbols:', work.free_symbols)

    print('-'* 80)
    print('Depth:', depth)
    print('Free symbols:', depth.free_symbols)
    print('=' * 80)


if __name__ == '__main__':
    main()
