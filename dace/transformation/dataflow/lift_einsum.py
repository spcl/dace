# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from typing import Dict

import sympy

from dace import SDFG, SDFGState, dtypes, nodes, subsets
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python import astutils
from dace.sdfg import utils as sdutil
from dace.symbolic import pystr_to_symbolic
from dace.transformation import transformation as xf


class LiftEinsum(xf.SingleStateTransformation):
    """
    Detects a tensor operation that can be represented by an Einstein-notation sum (einsum, e.g., matrix
    multiplication) and replaces the pattern with an ``Einsum`` library node.
    """
    EINSUM_CHARS = 'ijklmnopqrstuvwxyzabcdefgh'

    map_entry = xf.PatternNode(nodes.MapEntry)
    tasklet = xf.PatternNode(nodes.Tasklet)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.tasklet)]

    def can_be_applied(self, state: SDFGState, _, sdfg: SDFG, permissive=False):
        contents = state.scope_subgraph(self.map_entry, False, False)
        # Ensure map body contains only the tasklet
        if len(contents.nodes()) != 1:
            return False

        # Check that indices match map indices
        unique_chars = set()
        input_chars = set()
        output_chars = set()
        for e in state.all_edges(self.tasklet):
            memlet = e.data
            if memlet.is_empty():
                continue
            if memlet.dynamic:
                return False
            if memlet.volume != 1 or memlet.subset.num_elements() != 1:
                return False
            ind = set(str(rb) for rb, _, _ in memlet.subset.ndrange())
            unique_chars |= ind
            if any(i != '0' and i not in self.map_entry.map.params for i in ind):
                return False

            # Keep track of input/output indices for WCR check
            if e.dst is self.tasklet:
                input_chars |= ind
            else:
                output_chars |= ind

        # If there are too many characters for an einsum expression, fail
        if len(unique_chars) == 0 or len(unique_chars) > len(self.EINSUM_CHARS):
            return False

        # Test outputs
        oe = state.out_edges(self.tasklet)
        if len(oe) != 1:
            return False
        oe = oe[0]
        # Check for WCR if relevant
        if input_chars - output_chars:
            if oe.data.wcr is None or detect_reduction_type(oe.data.wcr) != dtypes.ReductionType.Sum:
                return False
        elif oe.data.wcr is not None:  # if, e.g., outer product, no WCR
            return False

        # Check tasklet contents
        if self.tasklet.code.language != dtypes.Language.Python:
            return False
        if len(self.tasklet.code.code) != 1:
            return False
        expr = self.tasklet.code.code[0]
        if not isinstance(expr, (ast.Assign, ast.AnnAssign)):
            return False
        if expr.value is None:
            return False

        try:
            symexpr = pystr_to_symbolic(astutils.unparse(expr.value))
        except (TypeError, sympy.SympifyError):
            return False

        expected = 1
        for iconn in self.tasklet.in_connectors:
            expected *= pystr_to_symbolic(iconn)
        if symexpr != expected:
            return False
        # End of tasklet content check

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        import dace.libraries.blas as blas

        map_exit = state.exit_node(self.map_entry)

        scope = state.scope_subgraph(self.map_entry)
        einsum = blas.Einsum('einsum')

        # Map connectors from tasklet to library node
        connectors: Dict[str, str] = {}
        for e in state.in_edges(self.tasklet):
            connectors[state.memlet_path(e)[-2].dst_conn] = e.dst_conn
            einsum.add_in_connector(e.dst_conn, self.tasklet.in_connectors[e.dst_conn])
        for e in state.out_edges(self.tasklet):
            connectors[state.memlet_path(e)[1].src_conn] = e.src_conn
            einsum.add_out_connector(e.src_conn, self.tasklet.out_connectors[e.src_conn])

        # Collect einsum string from sorted memlets
        param_mapping: Dict[str, str] = {}
        # letter_to_range: Dict[str, subsets.Range] = {}
        in_edges = list(sorted(state.in_edges(self.tasklet), key=lambda k: k.dst_conn))
        out_edge = state.out_edges(self.tasklet)[0]
        einsum_inputs = []
        einsum_output = ''
        for e in (in_edges + [out_edge]):
            # Create parameter mapping
            ind = [str(rb) for rb, _, _ in e.data.subset.ndrange()]
            expr = ''
            for i in ind:
                if i != '0' and i not in param_mapping:
                    param_mapping[i] = self.EINSUM_CHARS[len(param_mapping)]
                    # pind = self.map_entry.map.params.index(i)
                    # letter_to_range[i] = subsets.Range(self.map_entry.map.range[pind])
                expr += '' if i == '0' else param_mapping[i]
            if e is out_edge:
                einsum_output = expr
            else:
                einsum_inputs.append(expr)

        # Make einsum string
        einsum.einsum_str = f"{','.join(einsum_inputs)}->{einsum_output}"

        # Add new subgraph
        state.add_node(einsum)
        for e in state.in_edges(self.map_entry):
            state.add_edge(e.src, e.src_conn, einsum, connectors[e.dst_conn], e.data)
        for e in state.out_edges(map_exit):
            e.data.wcr = None  # Cancel WCR now that it is nested in the einsum
            state.add_edge(einsum, connectors[e.src_conn], e.dst, e.dst_conn, e.data)

        # Remove old scope
        state.remove_nodes_from(scope.nodes())
