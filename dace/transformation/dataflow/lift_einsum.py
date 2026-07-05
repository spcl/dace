# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import copy
from typing import Dict

import sympy

from dace import SDFG, SDFGState, dtypes, nodes, symbolic
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

    def _partial_range(self) -> bool:
        """True iff the map iterates a strict sub-range of its operands -- any
        dimension whose lower bound is not 0 or whose step is not 1. Such a range
        does not span the full operand extent the einsum contraction assumes."""
        for rng in self.map_entry.map.range:
            if str(rng[0]) != '0' or str(rng[2]) != '1':
                return True
        return False

    def can_be_applied(self, state: SDFGState, _, sdfg: SDFG, permissive=False):
        contents = state.scope_subgraph(self.map_entry, False, False)
        # Ensure map body contains only the tasklet
        if len(contents.nodes()) != 1:
            return False

        # The einsum/GEMM contracts over the FULL rectangular product of the map
        # ranges. Refuse a non-rectangular (parameter-dependent) iteration space --
        # e.g. a triangular ``j: 0:i+1`` (syrk/syr2k) -- whose dense-GEMM lowering
        # would compute the wrong (full instead of triangular) result.
        params = set(self.map_entry.map.params)
        for rng in self.map_entry.map.range:
            for bound in rng:
                if set(map(str, pystr_to_symbolic(bound).free_symbols)) & params:
                    return False

        # A PARTIAL iteration range (lower bound != 0 or non-unit step) touches only a
        # sub-range of its operands, but the einsum expansion runs over the full
        # [0, extent) implied by the operand shapes -- so a fissioned ``b[i]=y[i]*z[i]``
        # over ``2:LEN`` lifted as-is writes a whole-array einsum that clobbers
        # ``b[0], b[1]``. A 1-D partial range over purely 1-D operands is corrected in
        # apply() by restricting the operand memlets to the map range; a >1-D partial
        # range (offsets interacting across contracted axes) or multi-dimensional
        # operands under a 1-D partial map are refused (left as a correct map).
        if self._partial_range():
            if len(self.map_entry.map.params) > 1:
                return False
            for e in state.all_edges(self.tasklet):
                if not e.data.is_empty() and e.data.subset is not None and len(e.data.subset) != 1:
                    return False

        # Check that indices match map indices
        unique_chars = set()
        input_chars = set()
        output_chars = set()
        num_coeffs = 0  # scalar (map-parameter-free) inputs, e.g. a runtime alpha
        num_tensor_inputs = 0  # indexed (non-scalar) tensor operands, e.g. gemm's a, b
        tensor_input_chars = []  # per-tensor-input index set (for the elementwise guard)
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
                if not (ind - {'0'}):
                    num_coeffs += 1
                else:
                    num_tensor_inputs += 1
                    tensor_input_chars.append(ind - {'0'})
            else:
                output_chars |= ind

        if len(input_chars) == 0:
            return False

        # Require a genuine multi-operand contraction (matmul: a chain of >=2 tensor
        # operands). A single tensor operand is a unary copy / transpose / reduction
        # (einsum 'i->i', 'ij->ji', 'ij->i'), NOT a matmul -- lifting it to an Einsum
        # node is pointless and miscompiles (durbin's ``y[i] = z[i]`` copy -> 'i->i'
        # silently corrupts the solver). Those belong to the copy / reduction passes.
        if num_tensor_inputs < 2:
            return False

        # A scalar / full-reduction output (no free output index) is the reduction
        # pass's domain in general -- a MULTI-dimensional scalar contraction mis-lowers
        # whenever the operand index ORDER differs (``ij,ji->`` folds to the wrong 1x1
        # GEMM), and an AFFINE reversed access (durbin's ``sum += r[k-i-1]*y[i]``) is
        # already refused by the map-parameter index check above. The ONE exception is
        # a genuine 1-D dot product ``acc = sum_i a[i]*b[i]`` (``i,i->``) over the FULL
        # operand extent: a single map parameter forces every tensor operand to index
        # ``i``, so the einsum can only be ``i,...,i->`` -- which lowers correctly (2
        # operands -> degenerate 1x1 GEMM; more -> pure contraction). Lift only that
        # (its WCR is a Sum, checked below); refuse every wider scalar output.
        if not (output_chars - {'0'}):
            if len(self.map_entry.map.params) != 1 or self._partial_range():
                return False

        # Reject an ELEMENTWISE op: NO contracted index (every input index also appears in
        # the output) AND some tensor input accesses the SAME unit subset as the output --
        # e.g. ``Z[i]*Z[i]`` -> ``i,i->i`` (whose same-array unit operands mis-lower to a
        # dangling einsum connector). That belongs to the elementwise tile ops, not a GEMM.
        # A dot product / full reduction (scalar output) is already refused above, and a
        # matvec/matmul HAS a contracted index (``input_chars - output_chars`` non-empty,
        # so this guard is skipped) -- neither is caught. A genuine outer product
        # (``a[i]*b[j]`` -> ``i,j->ij``, no input matching the FULL output index) is preserved.
        out_idx = output_chars - {'0'}
        if not (input_chars - output_chars) and any(ci == out_idx for ci in tensor_input_chars):
            return False

        # At most one runtime scalar coefficient is supported (wired as the
        # einsum's single ``_alpha`` connector); refuse multi-scalar products.
        if num_coeffs > 1:
            return False

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
            # alpha != 1
            ratio = symexpr / expected
            if not ratio.is_Number and not isinstance(ratio, symbolic.symbol):
                return False
        # End of tasklet content check

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        import dace.libraries.blas as blas

        map_exit = state.exit_node(self.map_entry)

        scope = state.scope_subgraph(self.map_entry)
        einsum = blas.Einsum('einsum')

        connector_product = 1  # product of TENSOR-operand connectors (for alpha)

        # Map connectors from tasklet to library node. A tasklet input whose subset
        # carries a map parameter is an einsum tensor OPERAND; a single-element,
        # map-parameter-free input (e.g. ``alpha[0]``) is a runtime scalar
        # COEFFICIENT, wired as the einsum's explicit ``_alpha`` scalar input
        # connector (the data read stays explicit; the expansion consumes it).
        connectors: Dict[str, str] = {}
        in_edges = []
        out_edge = None
        coeff_edges = []  # scalar-coefficient input edges (alpha-like)
        coeff_map_conns: Dict[str, str] = {}  # map-entry IN-conn -> einsum coeff conn
        for e in state.in_edges(self.tasklet):
            if e.data.is_empty():
                continue
            ind = set(str(rb) for rb, _, _ in e.data.subset.ndrange())
            if not (ind - {'0'}):  # scalar coefficient: no map parameter
                coeff_edges.append(e)
                coeff_map_conns[state.memlet_path(e)[-2].dst_conn] = '_alpha'
                continue
            connectors[state.memlet_path(e)[-2].dst_conn] = e.dst_conn
            connector_product *= symbolic.symbol(e.dst_conn)
            einsum.add_in_connector(e.dst_conn, self.tasklet.in_connectors[e.dst_conn])
            in_edges.append(e)
        for e in state.out_edges(self.tasklet):
            if not e.data.is_empty():
                connectors[state.memlet_path(e)[1].src_conn] = e.src_conn
                einsum.add_out_connector(e.src_conn, self.tasklet.out_connectors[e.src_conn])
                out_edge = e

        # Compute the coefficient from the tasklet code. Dividing by the product of
        # the TENSOR connectors leaves the scalar-coefficient connectors (and any
        # numeric factor) as the coefficient.
        expr = self.tasklet.code.code[0]
        symexpr = pystr_to_symbolic(astutils.unparse(expr.value))
        alpha = symexpr / connector_product

        # Wire each runtime scalar coefficient as the einsum's ``_alpha`` input
        # connector and remove it from the symbolic alpha (set to 1); the residual
        # numeric factor stays on ``einsum.alpha`` so a mixed ``2.0 * alpha_data``
        # coefficient composes (the expansion multiplies property by connector).
        for e in coeff_edges:
            alpha = alpha.subs(symbolic.symbol(e.dst_conn), 1)
            einsum.add_in_connector('_alpha', self.tasklet.in_connectors[e.dst_conn])
        einsum.alpha = alpha

        # Collect the einsum string. CRITICAL: the expansion feeds operands in
        # SORTED connector-name order (``*sorted(inputs)`` in SpecializeEinsum), so
        # the input TERMS must be emitted in that SAME order -- otherwise the
        # operand<->term pairing (and hence the contracted dimensions) is wrong
        # whenever the tasklet's edge order differs from alphabetical (e.g. 2mm/3mm,
        # where it manifested as a dimension mismatch / silently wrong result).
        param_mapping: Dict[str, str] = {}
        input_terms: Dict[str, str] = {}  # einsum in-connector -> index string
        einsum_output = ''
        for e in (in_edges + [out_edge]):
            # Create parameter mapping
            ind = [str(rb) for rb, _, _ in e.data.subset.ndrange()]
            expr = ''
            for i in ind:
                if i != '0' and i not in param_mapping:
                    param_mapping[i] = self.EINSUM_CHARS[len(param_mapping)]
                expr += '' if i == '0' else param_mapping[i]
            if e is out_edge:
                einsum_output = expr
            else:
                input_terms[e.dst_conn] = expr

        # Make einsum string (input terms in the same sorted order the expansion uses)
        einsum_inputs = [input_terms[c] for c in sorted(input_terms)]
        einsum.einsum_str = f"{','.join(einsum_inputs)}->{einsum_output}"

        # Set beta (the output coefficient): the einsum computes
        # ``out = alpha * contraction + beta * out_prior``. beta=1 folds onto the
        # output's prior value; beta=0 overwrites. The prior value is meaningful --
        # so beta MUST be 1 -- exactly when it is well-defined; folding onto
        # undefined memory (beta=1 on a fresh slot) or discarding a meaningful value
        # (beta=0 on a pre-filled one) both miscompile. Decide from three signals,
        # in order:
        #   1. The WCR carries an identity (a Sum WCR's identity is 0, materialized
        #      as ``setzero`` on the accumulator AccessNode, e.g. 3mm ``E(1,+,0)``)
        #      -> the slot is initialized to 0 -> OVERWRITE (beta=0).
        #   2. Else there is an in-SDFG prior writer (e.g. gemm's ``C = beta*C``
        #      map, or a materialized zero-init) -> FOLD onto it (beta=1).
        #   3. Else the output is a non-transient argument -> the CALLER provides
        #      the prior value (a bare ``C += A@B`` over an input) -> FOLD (beta=1).
        #   4. Else it is a fresh, uninitialized transient -> OVERWRITE (beta=0)
        #      rather than fold garbage (the AccumulatorToMapAndReduce seed hazard).
        if out_edge.data.wcr is not None:
            out_data = out_edge.data.data
            out_node = state.memlet_path(out_edge)[-1].dst
            has_identity = isinstance(out_node, nodes.AccessNode) and out_node.setzero
            # Scoped to THIS sdfg (not all_sdfgs_recursive) so a same-named array in
            # an unrelated nested SDFG cannot false-positive a prior writer.
            has_prior_writer = any(n.data == out_data and st.in_degree(n) > 0 and n is not out_node
                                   for st in sdfg.states() for n in st.data_nodes())
            if has_identity:
                einsum.beta = 0.0
            elif has_prior_writer or not sdfg.arrays[out_data].transient:
                einsum.beta = 1.0
            else:
                einsum.beta = 0.0

        # A 1-D partial range (can_be_applied guarantees 1-D map + 1-D operands here)
        # propagated its operand memlets to the whole array, but the map touches only
        # its range; restrict each tensor operand + the output to the map range so the
        # einsum extent matches (else the elements outside the range are clobbered).
        restrict = self._partial_range()
        maprange = self.map_entry.map.range

        # Add new subgraph
        state.add_node(einsum)
        for e in state.in_edges(self.map_entry):
            if e.dst_conn in connectors:  # einsum tensor operand
                data = e.data
                if restrict:
                    data = copy.deepcopy(e.data)
                    data.subset = copy.deepcopy(maprange)
                state.add_edge(e.src, e.src_conn, einsum, connectors[e.dst_conn], data)
            elif e.dst_conn in coeff_map_conns:  # runtime scalar coefficient (not restricted)
                state.add_edge(e.src, e.src_conn, einsum, coeff_map_conns[e.dst_conn], e.data)
        for e in state.out_edges(map_exit):
            e.data.wcr = None  # Cancel WCR now that it is nested in the einsum
            data = e.data
            if restrict:
                data = copy.deepcopy(e.data)
                data.subset = copy.deepcopy(maprange)
            state.add_edge(einsum, connectors[e.src_conn], e.dst, e.dst_conn, data)

        # Remove old scope
        state.remove_nodes_from(scope.nodes())
