# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformations to convert subgraphs to write-conflict resolutions. """
import re
from dace import registry, nodes, dtypes
from dace.transformation import transformation
from dace.sdfg import utils as sdutil
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

    _EXPRESSIONS = ['+', '-', '*', '/', '^', '%']

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

        if graph.degree(inarr) != 1:
            return False

        inedge = graph.edges_between(inarr, tasklet)[0]
        outedge = graph.edges_between(tasklet, outarr)[0]

        # Same memlet
        if inedge.data.subset != outedge.data.subset:
            return False

        # Get relevant input/output connectors
        inconn = inedge.dst_conn
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(
            re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)

        if tasklet.language is dtypes.Language.Python:
            # Expect ast.Assign(ast.Expr())
            return False
        elif tasklet.language is dtypes.Language.CPP:
            # Try to match a single C assignment that can be converted to WCR
            cstr = tasklet.code.as_string.strip()
            if re.match(
                    r'^\s*%s\s*=\s*%s\s*%s.*;$' %
                (re.escape(outconn), re.escape(inconn), ops), cstr) is None:
                return False
        else:
            # Only Python/C++ tasklets supported
            return False

        return True

    def apply(self, sdfg: SDFG):
        input: nodes.AccessNode = self.input(sdfg)
        tasklet: nodes.Tasklet = self.tasklet(sdfg)
        output: nodes.AccessNode = self.output(sdfg)
        state: SDFGState = sdfg.node(self.state_id)

        inedge = state.edges_between(input, tasklet)[0]
        outedge = state.edges_between(tasklet, output)[0]
        # Get relevant input/output connectors
        inconn = inedge.dst_conn
        outconn = outedge.src_conn

        ops = '[%s]' % ''.join(
            re.escape(o) for o in AugAssignToWCR._EXPRESSIONS)

        # Change tasklet code
        if tasklet.language is dtypes.Language.Python:
            raise NotImplementedError
        elif tasklet.language is dtypes.Language.CPP:
            match = re.match(
                r'^\s*%s\s*=\s*%s\s*(%s)(.*);$' %
                (re.escape(outconn), re.escape(inconn), ops),
                tasklet.code.as_string.strip())
            op = match.group(1)
            expr = match.group(2)
            tasklet.code.code = '%s = %s;' % (outconn, expr)
        else:
            op = ''

        # Change output edge
        outedge.data.wcr = f'lambda a,b: a {op} b'

        # Remove input node and connector
        state.remove_edge_and_connectors(inedge)
        state.remove_node(input)
