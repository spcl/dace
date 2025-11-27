# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Linear algebra operations for ONNX.

This module contains implementations of linear algebra operations including:
- MatMul: Matrix multiplication with broadcasting
- Gemm: General matrix multiplication (alpha*A*B + beta*C)
- Einsum: Einstein summation notation for tensor operations

"""

import copy
import itertools
import typing

import dace
from dace import SDFG, SDFGState, nodes
from dace.sdfg.nodes import Node
from dace.sdfg.utils import in_desc_with_name, out_desc_with_name

from dace import config
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.utils import in_desc_with_name, op_implementation, out_desc_with_name
from dace.frontend.common import create_einsum_sdfg

# ============================================================================
# Matrix Multiplication
# ============================================================================


@op_implementation(op="MatMul", name="pure")
class PureMatMul(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 1 or input1_dim == 1:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXEinsum  # avoid import loop

        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    if config.Config.get_bool('debugprint'):
                        print(
                            f"Warning: overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                        )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    if config.Config.get_bool('debugprint'):
                        print(
                            f"Warning: overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                        )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim), reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to
        # make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        einsum_node: onnx_op.ONNXOp = ONNXEinsum(node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0", nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1", nsdfg.make_array_memlet("B"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("Y"), None, nsdfg.make_array_memlet("Y"))

        return nsdfg


# ============================================================================
# Einstein Summation
# ============================================================================


@op_implementation(op="Einsum", name="pure")
class PureEinsum(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        if "..." in node.equation:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        for e in node.iter_inputs_in_onnx_order(state):
            desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, e.dst_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, desc)
        for e in node.iter_outputs_in_onnx_order(state):
            desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, e.src_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.src_conn, desc)

        # Check if there is a wcr sum to accumulate the result instead of initialization the output
        # This is necessary for gradient accumulation to be consistent
        output_edge = state.out_edges(node)
        assert len(output_edge) == 1, "Einsum node should have exactly one output edge"
        output_edge = output_edge[0]
        beta = 1 if output_edge.data.wcr else 0
        create_einsum_sdfg(nsdfg,
                           nstate,
                           node.equation.replace(" ", ""),
                           *(e.dst_conn for e in node.iter_inputs_in_onnx_order(state)),
                           output="Output",
                           beta=beta)
        return nsdfg


# ============================================================================
# General Matrix Multiplication (Gemm)
# ============================================================================


@op_implementation(op="Gemm", name="pure")
class PureGemm(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXEinsum  # avoid import loop
        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    if config.Config.get_bool('debugprint'):
                        print(
                            f"Warning: overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                        )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    if config.Config.get_bool('debugprint'):
                        print(
                            f"Warning: overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                        )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim), reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        if node.transA == 1:
            arg1 = ''.join(reversed(arg1))
        if node.transB == 1:
            arg2 = ''.join(reversed(arg2))

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to
        # make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Einsum: "A", "B" -> mm_result
        einsum_node: nodes.LibraryNode = ONNXEinsum(node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Decide on array names based on alpha and beta
        uid = state.node_id(node)
        mm_result = "Y"
        if node.alpha != 1 or node.beta != 0:
            mm_result = f"Ytmp_{uid}"
        scal_result = mm_result
        if node.alpha != 1:
            scal_result = f"scaled_{uid}"

        # Create arrays according to alpha and beta
        if node.alpha != 1 or node.beta != 0:
            Ytmp_desc = out_desc_with_name(node, state, sdfg, "Y")
            nsdfg.add_datadesc(f"Ytmp_{uid}", copy.deepcopy(Ytmp_desc))
            nsdfg.arrays[f"Ytmp_{uid}"].transient = True
        if node.beta != 0:
            beta_desc = out_desc_with_name(node, state, sdfg, "Y")
            nsdfg.add_datadesc(f"scaled_{uid}", copy.deepcopy(beta_desc))
            nsdfg.arrays[f"scaled_{uid}"].transient = True

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0", nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1", nsdfg.make_array_memlet("B"))
        mm_result_node = nstate.add_write(mm_result)
        nstate.add_edge(einsum_node, "Output", mm_result_node, None, nsdfg.make_array_memlet(mm_result))

        # Multiply by alpha: mm_result -> scal_result
        if node.alpha != 1:
            nstate.add_mapped_tasklet(
                node.label + '_alphascale',
                {
                    k: f'0:{Ytmp_desc.shape[i]}'
                    for i, k in enumerate(result)
                },
                dict(a=dace.Memlet(data=mm_result, subset=','.join(result))),
                f'o = a * dace.{Ytmp_desc.dtype}({node.alpha})',
                dict(o=dace.Memlet(data=scal_result, subset=','.join(result))),
                external_edges=True,
                input_nodes=dict(a=mm_result_node),
            )

        # Multiply by beta: scal_result, "C" -> "Y"
        if node.beta != 0:
            C_desc = in_desc_with_name(node, state, sdfg, "C")
            nsdfg.add_datadesc("C", copy.deepcopy(C_desc))
            nsdfg.arrays["C"].transient = False
            scal_result_node = next(n for n in nstate.sink_nodes()
                                    if isinstance(n, dace.nodes.AccessNode) and n.data == scal_result)
            beta_scale_code = f'o = s + c * dace.{C_desc.dtype}({node.beta})'
            if node.beta == 1:
                beta_scale_code = f'o = s + c'

            # Support broadcasting in C -> Y
            c_index = result[-len(C_desc.shape):]
            for c_shp, y_shp in zip(reversed(C_desc.shape), reversed(Y_desc.shape)):
                if c_shp != y_shp:
                    raise ValueError('Could not broadcast dimensions from C '
                                     'to Y in ONNXGemm')

            nstate.add_mapped_tasklet(
                node.label + '_betascale',
                {
                    k: f'0:{Y_desc.shape[i]}'
                    for i, k in enumerate(result)
                },
                dict(s=dace.Memlet(data=scal_result, subset=','.join(result)),
                     c=dace.Memlet(data="C", subset=','.join(c_index))),
                beta_scale_code,
                dict(o=dace.Memlet(data="Y", subset=','.join(result))),
                external_edges=True,
                input_nodes={scal_result: scal_result_node},
            )

        return nsdfg
