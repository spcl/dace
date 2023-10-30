# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformations to convert subgraphs to write-conflict resolutions. """
import ast
import re
import copy
from dace import registry, nodes, dtypes, Memlet
from dace.transformation import transformation, helpers as xfh
from dace.sdfg import graph as gr, utils as sdutil
from dace import SDFG, SDFGState
from dace.sdfg.state import StateSubgraphView
from dace.transformation import helpers
from dace import propagate_memlet


class AugAssignToWCR(transformation.SingleStateTransformation):
    """
    Converts an augmented assignment ("a += b", "a = a + b") into a tasklet
    with a write-conflict resolution.
    """
    input = transformation.PatternNode(nodes.AccessNode)
    tasklet = transformation.PatternNode(nodes.Tasklet)
    output = transformation.PatternNode(nodes.AccessNode)
    map_entry = transformation.PatternNode(nodes.MapEntry)
    map_exit = transformation.PatternNode(nodes.MapExit)

    _EXPRESSIONS = ['+', '-', '*', '^', '%']  #, '/']
    _FUNCTIONS = ['min', 'max']
    _EXPR_MAP = {'-': ('+', '-({expr})'), '/': ('*', '((decltype({expr}))1)/({expr})')}
    _PYOP_MAP = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.BitXor: '^', ast.Mod: '%', ast.Div: '/'}

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.input, cls.tasklet, cls.output),
            sdutil.node_path_graph(cls.input, cls.map_entry, cls.tasklet, cls.map_exit, cls.output)
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        inarr = self.input
        tasklet = self.tasklet
        outarr = self.output
        if inarr.data != outarr.data:
            return False

        # Free tasklet
        if expr_index == 0:
            if graph.entry_node(tasklet) is not None:
                return False

            inedges = graph.edges_between(inarr, tasklet)
            if len(graph.edges_between(tasklet, outarr)) > 1:
                return False

            # Make sure augmented assignment can be fissioned as necessary
            if any(not isinstance(e.src, nodes.AccessNode) for e in graph.in_edges(tasklet)):
                return False

            outedge = graph.edges_between(tasklet, outarr)[0]
        else:  # Free map
            me = self.map_entry
            mx = self.map_exit

            # Only free maps supported for now
            if graph.entry_node(me) is not None:
                return False

            inedges = graph.edges_between(me, tasklet)
            if len(graph.edges_between(tasklet, mx)) > 1:
                return False

            # Make sure augmented assignment can be fissioned as necessary
            if any(e.src is not me and not isinstance(e.src, nodes.AccessNode)
                   for e in graph.in_edges(me) + graph.in_edges(tasklet)):
                return False

            outedge = graph.edges_between(tasklet, mx)[0]

        # Get relevant output connector
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)
        funcs = '|'.join(re.escape(o) for o in AugAssignToWCR._FUNCTIONS)

        if tasklet.language is dtypes.Language.Python:
            # Match a single assignment with a binary operation as RHS
            if len(tasklet.code.code) > 1:
                return False
            if not isinstance(tasklet.code.code[0], ast.Assign):
                return False
            ast_node: ast.Assign = tasklet.code.code[0]
            if len(ast_node.targets) > 1:
                return False
            if not isinstance(ast_node.targets[0], ast.Name):
                return False
            lhs: ast.Name = ast_node.targets[0]
            if lhs.id != outconn:
                return False
            if not isinstance(ast_node.value, ast.BinOp):
                return False
            rhs: ast.BinOp = ast_node.value
            if not isinstance(rhs.op, tuple(AugAssignToWCR._PYOP_MAP.keys())):
                return False
            inconns = tuple(edge.dst_conn for edge in inedges)
            for n in (rhs.left, rhs.right):
                if isinstance(n, ast.Name) and n.id in inconns:
                    return True
        elif tasklet.language is dtypes.Language.CPP:
            cstr = tasklet.code.as_string.strip()
            for edge in inedges:
                # Try to match a single C assignment that can be converted to WCR
                inconn = edge.dst_conn
                lhs = r'^\s*%s\s*=\s*%s\s*%s.*;$' % (re.escape(outconn), re.escape(inconn), ops)
                # rhs: a = (...) op b
                rhs = r'^\s*%s\s*=\s*\(.*\)\s*%s\s*%s;$' % (re.escape(outconn), ops, re.escape(inconn))
                func_lhs = r'^\s*%s\s*=\s*(%s)\(\s*%s\s*,.*\)\s*;$' % (re.escape(outconn), funcs, re.escape(inconn))
                func_rhs = r'^\s*%s\s*=\s*(%s)\(.*,\s*%s\s*\)\s*;$' % (re.escape(outconn), funcs, re.escape(inconn))
                if re.match(lhs, cstr) is None and re.match(rhs, cstr) is None:
                    if re.match(func_lhs, cstr) is None and re.match(func_rhs, cstr) is None:
                        inconns = list(self.tasklet.in_connectors)
                        if len(inconns) != 2:
                            continue

                        # Special case: a = <other> op b
                        other_inconn = inconns[0] if inconns[0] != inconn else inconns[1]
                        rhs2 = r'^\s*%s\s*=\s*%s\s*%s\s*%s;$' % (re.escape(outconn), re.escape(other_inconn), ops,
                                                                 re.escape(inconn))
                        if re.match(rhs2, cstr) is None:
                            continue

                # Same memlet
                if edge.data.subset != outedge.data.subset:
                    continue

                # If in map, only match if the subset is independent of any
                # map indices (otherwise no conflict)
                if expr_index == 1:
                    if not permissive and len(outedge.data.subset.free_symbols & set(me.map.params)) == len(
                            me.map.params):
                        continue

                return True
        else:
            # Only Python/C++ tasklets supported
            return False

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        input: nodes.AccessNode = self.input
        tasklet: nodes.Tasklet = self.tasklet
        output: nodes.AccessNode = self.output
        if self.expr_index == 1:
            me = self.map_entry
            mx = self.map_exit

        # If state fission is necessary to keep semantics, do it first
        if state.in_degree(input) > 0:
            subgraph_nodes = set([e.src for e in state.bfs_edges(input, reverse=True)])
            subgraph_nodes.add(input)

            subgraph = StateSubgraphView(state, subgraph_nodes)
            helpers.state_fission(sdfg, subgraph)

        if self.expr_index == 0:
            inedges = state.edges_between(input, tasklet)
            outedge = state.edges_between(tasklet, output)[0]
        else:
            inedges = state.edges_between(me, tasklet)
            outedge = state.edges_between(tasklet, mx)[0]

        # Get relevant output connector
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)
        funcs = '|'.join(re.escape(o) for o in AugAssignToWCR._FUNCTIONS)

        # Change tasklet code
        if tasklet.language is dtypes.Language.Python:
            # Match a single assignment with a binary operation as RHS
            ast_node: ast.Assign = tasklet.code.code[0]
            lhs: ast.Name = ast_node.targets[0]
            rhs: ast.BinOp = ast_node.value
            op = AugAssignToWCR._PYOP_MAP[type(rhs.op)]
            inconns = list(edge.dst_conn for edge in inedges)
            for n in (rhs.left, rhs.right):
                if isinstance(n, ast.Name) and n.id in inconns:
                    inedge = inedges[inconns.index(n.id)]
                else:
                    new_rhs = n
            new_node = ast.copy_location(ast.Assign(targets=[lhs], value=new_rhs), ast_node)
            tasklet.code.code = [new_node]

        elif tasklet.language is dtypes.Language.CPP:
            cstr = tasklet.code.as_string.strip()
            for edge in inedges:
                inconn = edge.dst_conn
                match = re.match(r'^\s*%s\s*=\s*%s\s*(%s)(.*);$' % (re.escape(outconn), re.escape(inconn), ops), cstr)
                if match is None:
                    match = re.match(
                            r'^\s*%s\s*=\s*\((.*)\)\s*(%s)\s*%s;$' % (re.escape(outconn), ops, re.escape(inconn)), cstr)
                    if match is None:
                        func_rhs = r'^\s*%s\s*=\s*(%s)\((.*),\s*%s\s*\)\s*;$' % (re.escape(outconn), funcs,
                                                                                 re.escape(inconn))
                        match = re.match(func_rhs, cstr)
                        if match is None:
                            func_lhs = r'^\s*%s\s*=\s*(%s)\(\s*%s\s*,(.*)\)\s*;$' % (re.escape(outconn), funcs,
                                                                                     re.escape(inconn))
                            match = re.match(func_lhs, cstr)
                            if match is None:
                                inconns = list(self.tasklet.in_connectors)
                                if len(inconns) != 2:
                                    continue

                                # Special case: a = <other> op b
                                other_inconn = inconns[0] if inconns[0] != inconn else inconns[1]
                                rhs2 = r'^\s*%s\s*=\s*(%s)\s*(%s)\s*%s;$' % (
                                    re.escape(outconn), re.escape(other_inconn), ops, re.escape(inconn))
                                match = re.match(rhs2, cstr)
                                if match is None:
                                    continue
                                else:
                                    op = match.group(2)
                                    expr = match.group(1)
                            else:
                                op = match.group(1)
                                expr = match.group(2)
                        else:
                            op = match.group(1)
                            expr = match.group(2)
                    else:
                        op = match.group(2)
                        expr = match.group(1)
                else:
                    op = match.group(1)
                    expr = match.group(2)

                if edge.data.subset != outedge.data.subset:
                    continue

                # Map asymmetric WCRs to symmetric ones if possible
                if op in AugAssignToWCR._EXPR_MAP:
                    op, newexpr = AugAssignToWCR._EXPR_MAP[op]
                    expr = newexpr.format(expr=expr)

                tasklet.code.code = '%s = %s;' % (outconn, expr)
                inedge = edge
                break
        else:
            raise NotImplementedError

        # Change output edge
        if op in AugAssignToWCR._FUNCTIONS:
            outedge.data.wcr = f'lambda a,b: {op}(a, b)'
        else:
            outedge.data.wcr = f'lambda a,b: a {op} b'

        # Remove input node and connector
        state.remove_memlet_path(inedge)

        # If outedge leads to non-transient, and this is a nested SDFG,
        # propagate outwards
        sd = sdfg
        while (not sd.arrays[outedge.data.data].transient and sd.parent_nsdfg_node is not None):
            nsdfg = sd.parent_nsdfg_node
            nstate = sd.parent
            sd = sd.parent_sdfg
            outedge = next(iter(nstate.out_edges_by_connector(nsdfg, outedge.data.data)))
            for outedge in nstate.memlet_path(outedge):
                if op in AugAssignToWCR._FUNCTIONS:
                    outedge.data.wcr = f'lambda a,b: {op}(a, b)'
                else:
                    outedge.data.wcr = f'lambda a,b: a {op} b'
            # At this point we are leading to an access node again and can
            # traverse further up
