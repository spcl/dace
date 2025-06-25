# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformations to convert subgraphs to write-conflict resolutions. """
import ast
import copy
import re
import copy
from dace import nodes, dtypes, Memlet
from dace.frontend.python import astutils
from dace.transformation import transformation
from dace.sdfg import utils as sdutil
from dace import Memlet, SDFG, SDFGState
from dace.transformation import helpers
from dace.sdfg.propagation import propagate_memlets_state


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

            # If in map, only match if the subset is independent of any
            # map indices (otherwise no conflict)
            if not permissive and len(outedge.data.subset.free_symbols & set(me.map.params)) == len(me.map.params):
                return False

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

                return True

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
            new_state = self.isolate_tasklet(state)
        else:
            new_state = state

        if self.expr_index == 0:
            inedges = new_state.edges_between(input, tasklet)
            outedge = new_state.edges_between(tasklet, output)[0]
        else:
            inedges = new_state.edges_between(me, tasklet)
            outedge = new_state.edges_between(tasklet, mx)[0]

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
            if isinstance(rhs.left, ast.Name) and rhs.left.id in inconns:
                inedge = inedges[inconns.index(rhs.left.id)]
                new_rhs = rhs.right
            else:
                inedge = inedges[inconns.index(rhs.right.id)]
                new_rhs = rhs.left

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
                                inconns = list(tasklet.in_connectors)
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
        new_state.remove_memlet_path(inedge)
        propagate_memlets_state(sdfg, new_state)

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

    def isolate_tasklet(
        self,
        state: SDFGState,
    ) -> SDFGState:
        tlet: nodes.Tasklet = self.tasklet
        newstate = state.parent_graph.add_state_after(state)

        # Bookkeeping
        nodes_to_move = set([tlet])
        boundary_nodes = set()
        orig_edges = set()

        for edge in state.in_edges(tlet):
            for e in state.memlet_path(edge):
                nodes_to_move.add(e.src)
                orig_edges.add(e)

        # Find all consumer nodes of `tlet`.
        for edge in state.edge_bfs(tlet):
            nodes_to_move.add(edge.dst)
            orig_edges.add(edge)

            # If a consumer is not an AccessNode we also have to relocate its dependencies.
            if not isinstance(edge.dst, nodes.AccessNode):
                for iedge in state.in_edges(edge.dst):
                    if iedge == edge:
                        continue
                    for e in state.memlet_path(iedge):
                        nodes_to_move.add(e.src)
                        orig_edges.add(e)

        # Define boundary nodes
        for node in nodes_to_move:
            if isinstance(node, nodes.AccessNode):
                for iedge in state.in_edges(node):
                    if iedge.src not in nodes_to_move:
                        boundary_nodes.add(node)
                        break
                if node in boundary_nodes:
                    continue
                for oedge in state.out_edges(node):
                    if oedge.dst not in nodes_to_move:
                        boundary_nodes.add(node)
                        break

        # Duplicate boundary nodes
        new_nodes = {}
        for node in boundary_nodes:
            node_ = copy.deepcopy(node)
            state.add_node(node_)
            new_nodes[node] = node_

        for edge in state.edges():
            if edge.src in boundary_nodes and edge.dst in boundary_nodes:
                state.add_edge(new_nodes[edge.src], edge.src_conn, new_nodes[edge.dst], edge.dst_conn,
                               copy.deepcopy(edge.data))
            elif edge.src in boundary_nodes:
                state.add_edge(new_nodes[edge.src], edge.src_conn, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))
            elif edge.dst in boundary_nodes:
                state.add_edge(edge.src, edge.src_conn, new_nodes[edge.dst], edge.dst_conn, copy.deepcopy(edge.data))

        state.remove_nodes_from(nodes_to_move)

        # Set the new parent state
        # TODO: Note sure if `add_node()` does it on its own?
        for node in nodes_to_move:
            if isinstance(node, nodes.NestedSDFG):
                node.sdfg.parent = newstate

        newstate.add_nodes_from(nodes_to_move)
        for e in orig_edges:
            newstate.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, e.data)

        return newstate


class WCRToAugAssign(transformation.SingleStateTransformation):
    """
    Converts a tasklet with a write-conflict resolution to an augmented assignment subgraph (e.g., "a = a + b").
    """
    tasklet = transformation.PatternNode(nodes.Tasklet)
    output = transformation.PatternNode(nodes.AccessNode)
    map_exit = transformation.PatternNode(nodes.MapExit)

    _EXPRESSIONS = ['+', '-', '*', '^', '%']  #, '/']
    _EXPR_MAP = {'-': ('+', '-({expr})'), '/': ('*', '((decltype({expr}))1)/({expr})')}
    _PYOP_MAP = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.BitXor: '^', ast.Mod: '%', ast.Div: '/'}

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.tasklet, cls.output),
            sdutil.node_path_graph(cls.tasklet, cls.map_exit, cls.output)
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if expr_index == 0:
            edges = graph.edges_between(self.tasklet, self.output)
        else:
            edges = graph.edges_between(self.tasklet, self.map_exit)
        if len(edges) != 1:
            return False
        if edges[0].data.wcr is None:
            return False

        # If the access subset on the WCR edge is overapproximated (i.e., the access may be dynamic), we do not support
        # swapping to an augmented assignment pattern with this transformation.
        if edges[0].data.subset.num_elements() > edges[0].data.volume or edges[0].data.dynamic is True:
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        if self.expr_index == 0:
            edge = state.edges_between(self.tasklet, self.output)[0]
            wcr = ast.parse(edge.data.wcr).body[0].value.body
            if isinstance(wcr, ast.BinOp):
                wcr.left.id = '__in1'
                wcr.right.id = '__in2'
                code = astutils.unparse(wcr)
            else:
                raise NotImplementedError
            edge.data.wcr = None
            in_access = state.add_access(self.output.data)
            new_tasklet = state.add_tasklet('augassign', {'__in1', '__in2'}, {'__out'}, f"__out = {code}")
            scal_name, scal_desc = sdfg.add_scalar('tmp',
                                                   sdfg.arrays[self.output.data].dtype,
                                                   transient=True,
                                                   find_new_name=True)
            state.add_edge(self.tasklet, edge.src_conn, new_tasklet, '__in1', Memlet.from_array(scal_name, scal_desc))
            state.add_edge(in_access, None, new_tasklet, '__in2', copy.deepcopy(edge.data))
            state.add_edge(new_tasklet, '__out', self.output, edge.dst_conn, edge.data)
            state.remove_edge(edge)
        else:
            edge = state.edges_between(self.tasklet, self.map_exit)[0]
            map_entry = state.entry_node(self.map_exit)
            wcr = ast.parse(edge.data.wcr).body[0].value.body
            if isinstance(wcr, ast.BinOp):
                wcr.left.id = '__in1'
                wcr.right.id = '__in2'
                code = astutils.unparse(wcr)
            else:
                raise NotImplementedError
            for e in state.memlet_path(edge):
                e.data.wcr = None
            in_access = state.add_access(self.output.data)
            new_tasklet = state.add_tasklet('augassign', {'__in1', '__in2'}, {'__out'}, f"__out = {code}")
            scal_name, scal_desc = sdfg.add_scalar('tmp',
                                                   sdfg.arrays[self.output.data].dtype,
                                                   transient=True,
                                                   find_new_name=True)
            state.add_edge(self.tasklet, edge.src_conn, new_tasklet, '__in1', Memlet.from_array(scal_name, scal_desc))
            state.add_memlet_path(in_access, map_entry, new_tasklet, memlet=copy.deepcopy(edge.data), dst_conn='__in2')
            state.add_edge(new_tasklet, '__out', self.map_exit, edge.dst_conn, edge.data)
            state.remove_edge(edge)
