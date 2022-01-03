# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformations to convert subgraphs to write-conflict resolutions. """
import ast
import re
from dace import registry, nodes, dtypes
from dace.transformation import transformation, helpers as xfh
from dace.sdfg import graph as gr, utils as sdutil
from dace import SDFG, SDFGState


@registry.autoregister_params(singlestate=True)
class AugAssignToWCR(transformation.Transformation):
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
    _EXPR_MAP = {
        '-': ('+', '-({expr})'),
        '/': ('*', '((decltype({expr}))1)/({expr})')
    }
    _PYOP_MAP = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.BitXor: '^',
        ast.Mod: '%',
        ast.Div: '/'
    }

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(AugAssignToWCR.input, AugAssignToWCR.tasklet,
                                   AugAssignToWCR.output),
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        inarr = graph.node(candidate[AugAssignToWCR.input])
        tasklet: nodes.Tasklet = graph.node(candidate[AugAssignToWCR.tasklet])
        outarr = graph.node(candidate[AugAssignToWCR.output])
        if inarr.data != outarr.data:
            return False

        # Free tasklet
        if expr_index == 0:
            # Only free tasklets supported for now
            if graph.entry_node(tasklet) is not None:
                return False

            inedges = graph.edges_between(inarr, tasklet)
            if len(graph.edges_between(tasklet, outarr)) > 1:
                return False

            # Make sure augmented assignment can be fissioned as necessary
            if any(not isinstance(e.src, nodes.AccessNode)
                   for e in graph.in_edges(tasklet)):
                return False
            if graph.in_degree(inarr) > 0 and graph.out_degree(outarr) > 0:
                return False

            outedge = graph.edges_between(tasklet, outarr)[0]
        else:  # Free map
            me: nodes.MapEntry = graph.node(candidate[AugAssignToWCR.map_entry])
            mx = graph.node(candidate[AugAssignToWCR.map_exit])

            # Only free maps supported for now
            if graph.entry_node(me) is not None:
                return False

            inedges = graph.edges_between(me, tasklet)
            if len(graph.edges_between(tasklet, mx)) > 1:
                return False

            # Currently no fission is supported
            if any(e.src is not me and not isinstance(e.src, nodes.AccessNode)
                   for e in graph.in_edges(me) + graph.in_edges(tasklet)):
                return False
            if graph.in_degree(inarr) > 0:
                return False

            outedge = graph.edges_between(tasklet, mx)[0]

        # Get relevant output connector
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(
            re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)

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
                lhs = r'^\s*%s\s*=\s*%s\s*%s.*;$' % (re.escape(outconn),
                                                     re.escape(inconn), ops)
                rhs = r'^\s*%s\s*=\s*.*%s\s*%s;$' % (re.escape(outconn), ops,
                                                     re.escape(inconn))
                if re.match(lhs, cstr) is None:
                    continue
                # Same memlet
                if edge.data.subset != outedge.data.subset:
                    continue

                # If in map, only match if the subset is independent of any
                # map indices (otherwise no conflict)
                if (expr_index == 1
                        and len(outedge.data.subset.free_symbols
                                & set(me.map.params)) == len(me.map.params)):
                    continue

                return True
        else:
            # Only Python/C++ tasklets supported
            return False

        return False

    def apply(self, sdfg: SDFG):
        input: nodes.AccessNode = self.input(sdfg)
        tasklet: nodes.Tasklet = self.tasklet(sdfg)
        output: nodes.AccessNode = self.output(sdfg)
        state: SDFGState = sdfg.node(self.state_id)

        # If state fission is necessary to keep semantics, do it first
        if (self.expr_index == 0 and state.in_degree(input) > 0
                and state.out_degree(output) == 0):
            newstate = sdfg.add_state_after(state)
            newstate.add_node(tasklet)
            new_input, new_output = None, None

            # Keep old edges for after we remove tasklet from the original state
            in_edges = list(state.in_edges(tasklet))
            out_edges = list(state.out_edges(tasklet))

            for e in in_edges:
                r = newstate.add_read(e.src.data)
                newstate.add_edge(r, e.src_conn, e.dst, e.dst_conn, e.data)
                if e.src is input:
                    new_input = r
            for e in out_edges:
                w = newstate.add_write(e.dst.data)
                newstate.add_edge(e.src, e.src_conn, w, e.dst_conn, e.data)
                if e.dst is output:
                    new_output = w

            # Remove tasklet and resulting isolated nodes
            state.remove_node(tasklet)
            for e in in_edges:
                if state.degree(e.src) == 0:
                    state.remove_node(e.src)
            for e in out_edges:
                if state.degree(e.dst) == 0:
                    state.remove_node(e.dst)

            # Reset state and nodes for rest of transformation
            input = new_input
            output = new_output
            state = newstate
        # End of state fission

        if self.expr_index == 0:
            inedges = state.edges_between(input, tasklet)
            outedge = state.edges_between(tasklet, output)[0]
        else:
            me = self.map_entry(sdfg)
            mx = self.map_exit(sdfg)

            inedges = state.edges_between(me, tasklet)
            outedge = state.edges_between(tasklet, mx)[0]

        # Get relevant output connector
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(
            re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)

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
            new_node = ast.copy_location(
                ast.Assign(targets=[lhs], value=new_rhs), ast_node)
            tasklet.code.code = [new_node]

        elif tasklet.language is dtypes.Language.CPP:
            cstr = tasklet.code.as_string.strip()
            for edge in inedges:
                inconn = edge.dst_conn
                match = re.match(
                    r'^\s*%s\s*=\s*%s\s*(%s)(.*);$' %
                    (re.escape(outconn), re.escape(inconn), ops), cstr)
                if match is None:
                    # match = re.match(
                    #     r'^\s*%s\s*=\s*(.*)\s*(%s)\s*%s;$' %
                    #     (re.escape(outconn), ops, re.escape(inconn)), cstr)
                    # if match is None:
                    continue
                    # op = match.group(2)
                    # expr = match.group(1)
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
        outedge.data.wcr = f'lambda a,b: a {op} b'

        if self.expr_index == 0:
            # Remove input node and connector
            state.remove_edge_and_connectors(inedge)
            if state.degree(input) == 0:
                state.remove_node(input)
        else:
            # Remove input edge and dst connector, but not necessarily src
            state.remove_memlet_path(inedge)

        # If outedge leads to non-transient, and this is a nested SDFG,
        # propagate outwards
        sd = sdfg
        while (not sd.arrays[outedge.data.data].transient
               and sd.parent_nsdfg_node is not None):
            nsdfg = sd.parent_nsdfg_node
            nstate = sd.parent
            sd = sd.parent_sdfg
            outedge = next(
                iter(nstate.out_edges_by_connector(nsdfg, outedge.data.data)))
            for outedge in nstate.memlet_path(outedge):
                outedge.data.wcr = f'lambda a,b: a {op} b'
            # At this point we are leading to an access node again and can
            # traverse further up
