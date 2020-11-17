# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformations to convert subgraphs to write-conflict resolutions. """
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

    _EXPRESSIONS = ['+', '-', '*', '^', '%', '/']
    _EXPR_MAP = {
        '-': ('+', '-({expr})'),
        '/': ('*', '((decltype({expr}))1)/({expr})')
    }

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(AugAssignToWCR.input, AugAssignToWCR.tasklet,
                                   AugAssignToWCR.output)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        inarr = graph.node(candidate[AugAssignToWCR.input])
        tasklet: nodes.Tasklet = graph.node(candidate[AugAssignToWCR.tasklet])
        outarr = graph.node(candidate[AugAssignToWCR.output])
        if inarr.data != outarr.data:
            return False

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

        # Get relevant output connector
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(
            re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)

        if tasklet.language is dtypes.Language.Python:
            # Expect ast.Assign(ast.Expr())
            return False
        elif tasklet.language is dtypes.Language.CPP:
            cstr = tasklet.code.as_string.strip()
            for edge in inedges:
                # Try to match a single C assignment that can be converted to WCR
                inconn = edge.dst_conn
                if re.match(
                        r'^\s*%s\s*=\s*%s\s*%s.*;$' %
                    (re.escape(outconn), re.escape(inconn), ops), cstr) is None:
                    continue
                # Same memlet
                if edge.data.subset != outedge.data.subset:
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
        if state.in_degree(input) > 0 and state.out_degree(output) == 0:
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

        inedges = state.edges_between(input, tasklet)
        outedge = state.edges_between(tasklet, output)[0]
        # Get relevant output connector
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(
            re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)

        # Change tasklet code
        if tasklet.language is dtypes.Language.Python:
            raise NotImplementedError
        elif tasklet.language is dtypes.Language.CPP:
            cstr = tasklet.code.as_string.strip()
            for edge in inedges:
                inconn = edge.dst_conn
                match = re.match(
                    r'^\s*%s\s*=\s*%s\s*(%s)(.*);$' %
                    (re.escape(outconn), re.escape(inconn), ops), cstr)
                if match is None:
                    continue
                if edge.data.subset != outedge.data.subset:
                    continue

                op = match.group(1)
                expr = match.group(2)

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

        # Remove input node and connector
        state.remove_edge_and_connectors(inedge)
        if state.degree(input) == 0:
            state.remove_node(input)

        # If outedge leads to non-transient, and this is a nested SDFG,
        # propagate outwards
        sd = sdfg
        while (not sd.arrays[outedge.dst.data].transient
               and sd.parent_nsdfg_node is not None):
            nsdfg = sd.parent_nsdfg_node
            nstate = sd.parent
            sd = sd.parent_sdfg
            outedge = next(
                iter(nstate.out_edges_by_connector(nsdfg, outedge.dst.data)))
            for outedge in nstate.memlet_path(outedge):
                outedge.data.wcr = f'lambda a,b: a {op} b'
            # At this point we are leading to an access node again and can
            # traverse further up
