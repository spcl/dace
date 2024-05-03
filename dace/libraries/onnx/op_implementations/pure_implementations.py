import copy
import itertools
import logging
import typing

import dace
import numpy as np
from dace import SDFGState, SDFG, nodes, subsets, data
from dace.frontend.common import create_einsum_sdfg
from dace.sdfg.nodes import Node

from dace.libraries.onnx import converters
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.utils import op_implementation, program_for_node, empty_sdfg_for_node, \
    python_pure_op_implementation
from dace.transformation.onnx import constant_folding
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.util import iterables_equal, in_desc_with_name, out_desc_with_name, in_edge_with_name

log = logging.getLogger(__name__)


@python_pure_op_implementation
def Log(input, output):
    output[:] = np.log(input)


@python_pure_op_implementation
def Exp(input, output):
    output[:] = np.exp(input)


@python_pure_op_implementation
def Sqrt(X, Y):
    Y[:] = dace.elementwise(lambda x: sqrt(x), X)


@python_pure_op_implementation
def Pow(X, Y, Z):
    Z[:] = X**Y


@op_implementation(op="Clip", name="pure")
class PureClip(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        min_node = next(state.in_edges_by_connector(node, 'min')).src
        max_node = next(state.in_edges_by_connector(node, 'max')).src
        # TODO other cases
        return (onnx_constant_or_none(sdfg, min_node) is not None
                and onnx_constant_or_none(sdfg, max_node) is not None)

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        min_node = next(state.in_edges_by_connector(node, 'min')).src
        max_node = next(state.in_edges_by_connector(node, 'max')).src
        minval = onnx_constant_or_none(sdfg, min_node)
        maxval = onnx_constant_or_none(sdfg, max_node)

        input_dtype = in_desc_with_name(node, state, sdfg, "input").dtype
        minstr = f"dace.{input_dtype.to_string()}({minval})"
        maxstr = f"dace.{input_dtype.to_string()}({maxval})"

        lfunc = f"lambda x: min(max(x, {minstr}), {maxstr})"

        def prog(input, output):
            output[:] = dace.elementwise(lfunc, input)

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation
def Add(A, B, C):
    C[:] = A + B


@python_pure_op_implementation
def Sub(A, B, C):
    C[:] = A - B


@python_pure_op_implementation
def Mul(A, B, C):
    C[:] = A * B


@python_pure_op_implementation
def Div(A, B, C):
    C[:] = A / B


@python_pure_op_implementation
def Where(condition, X, Y, output):
    output[:] = np.where(condition, X, Y)


@python_pure_op_implementation(axes=lambda node: node.axes)
def ReduceMean(data, reduced):
    reduced[:] = np.mean(data, axis=axes)


@python_pure_op_implementation
def Erf(input, output):
    output[:] = dace.elementwise(lambda x: erf(x), input)


@op_implementation(op="MatMul", name="pure")
class PureMatMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 1 or input1_dim == 1:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
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
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim),
                                                    reversed(input1_dim)):
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

        # we lower to an ONNXEinsum node instead straight to the dace einsum to make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0",
                        nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1",
                        nsdfg.make_array_memlet("B"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("Y"), None,
                        nsdfg.make_array_memlet("Y"))

        return nsdfg


@op_implementation(op="Einsum", name="pure")
class PureEinsum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if "..." in node.equation:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        for e in node.iter_inputs_in_onnx_order(state):
            desc = copy.deepcopy(
                in_desc_with_name(node, state, sdfg, e.dst_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, desc)
        for e in node.iter_outputs_in_onnx_order(state):
            desc = copy.deepcopy(
                out_desc_with_name(node, state, sdfg, e.src_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.src_conn, desc)

        create_einsum_sdfg(None,
                           nsdfg,
                           nstate,
                           node.equation.replace(" ", ""),
                           *(e.dst_conn
                             for e in node.iter_inputs_in_onnx_order(state)),
                           output="Output")
        return nsdfg


@python_pure_op_implementation
def Identity(input, output):
    output[:] = input


@op_implementation(op="Expand", name="pure")
class PureExpand(ONNXForward):
    """ Handle no-op case for Expand """
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return iterables_equal(
            in_desc_with_name(node, state, sdfg, "input").shape,
            out_desc_with_name(node, state, sdfg, "output").shape)

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        constant_folding.remove_node_and_computation(sdfg, state, node,
                                                     "shape")

        def prog(input, output):
            output[:] = input

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation(
    string=lambda X: "lambda x: dace.{}(1) / x".format(X.dtype.to_string()))
def Reciprocal(X, Y):
    Y[:] = dace.elementwise(string, X)


@python_pure_op_implementation
def Tanh(input, output):
    output[:] = dace.elementwise(lambda x: tanh(x), input)


@python_pure_op_implementation(axes=lambda node: node.axes)
def ReduceSum(data, reduced):
    reduced[:] = np.sum(data, axis=axes)


@python_pure_op_implementation(axes=lambda node: node.axes)
def ReduceMax(data, reduced):
    reduced[:] = np.max(data, axis=axes)


@python_pure_op_implementation(axes=lambda node: node.axes)
def ReduceMin(data, reduced):
    reduced[:] = np.min(data, axis=axes)


softmax_compute = dict(
    axis=lambda node, input: list(range(len(input.shape)))[node.axis:],
    keepdims=lambda node: node.axis != 0)


@python_pure_op_implementation(**softmax_compute)
def Softmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=keepdims)
    exponent = np.exp(input - maximum)
    sum = np.add.reduce(exponent, axis=axis, keepdims=keepdims)
    output[:] = exponent / sum


@python_pure_op_implementation(
    perm=lambda node, data: node.perm
    if node.perm is not None else list(reversed(range(len(data.shape)))))
def Transpose(data, transposed):
    transposed[:] = np.transpose(data, axes=perm)


@op_implementation(op="Cast", name="pure")
class PureCast(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        if (in_desc_with_name(node, state, sdfg,
                              "input").dtype == out_desc_with_name(
                                  node, state, sdfg, "output").dtype):
            return True

        target_type = node.to
        try:
            converters.onnx_tensor_type_to_typeclass(target_type)
        except ValueError:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")
        if (input_desc.dtype == output_desc.dtype):

            def prog(input, output):
                output[:] = input

            return program_for_node(prog, sdfg, state, node)
        else:

            nsdfg, nstate, _, _ = empty_sdfg_for_node(sdfg,
                                                      state,
                                                      node,
                                                      add_access_nodes=False)

            shape = out_desc_with_name(node, state, sdfg, "output").shape
            map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
            index_str = f"{', '.join(map_ranges.keys())}"
            tasklet, _, _ = nstate.add_mapped_tasklet(
                node.label + "_tasklet",
                map_ranges=map_ranges,
                inputs={f"__input": dace.Memlet(f"input[{index_str}]")},
                code=f"__output = __input",
                outputs={"__output": dace.Memlet(f"output[{index_str}]")},
                external_edges=True)

            return nsdfg


@op_implementation(op="Gemm", name="pure")
class PureGemm(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
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
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim),
                                                    reversed(input1_dim)):
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
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=einsum_str)

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

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0",
                        nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1",
                        nsdfg.make_array_memlet("B"))
        mm_result_node = nstate.add_write(mm_result)
        nstate.add_edge(einsum_node, "Output", mm_result_node, None,
                        nsdfg.make_array_memlet(mm_result))

        # Multiply by alpha: mm_result -> scal_result
        if node.alpha != 1:
            nstate.add_mapped_tasklet(
                node.label + '_alphascale',
                {k: f'0:{Ytmp_desc.shape[i]}'
                 for i, k in enumerate(result)},
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
                                    if isinstance(n, dace.nodes.AccessNode)
                                    and n.data == scal_result)
            beta_scale_code = f'o = s + c * dace.{C_desc.dtype}({node.beta})'
            if node.beta == 1:
                beta_scale_code = f'o = s + c'

            # Support broadcasting in C -> Y
            c_index = result[-len(C_desc.shape):]
            for c_shp, y_shp in zip(reversed(C_desc.shape),
                                    reversed(Y_desc.shape)):
                if c_shp != y_shp:
                    raise ValueError('Could not broadcast dimensions from C '
                                     'to Y in ONNXGemm')

            nstate.add_mapped_tasklet(
                node.label + '_betascale',
                {k: f'0:{Y_desc.shape[i]}'
                 for i, k in enumerate(result)},
                dict(s=dace.Memlet(data=scal_result, subset=','.join(result)),
                     c=dace.Memlet(data="C", subset=','.join(c_index))),
                beta_scale_code,
                dict(o=dace.Memlet(data="Y", subset=','.join(result))),
                external_edges=True,
                input_nodes={scal_result: scal_result_node},
            )

        return nsdfg


@python_pure_op_implementation(
    cast_lambda=lambda X: "lambda x: max(x, dace.{}(0))".format(X.dtype.
                                                                to_string()))
def Relu(X, Y):
    Y[:] = dace.elementwise(cast_lambda, X)


@python_pure_op_implementation(
    cast_lambda=lambda node, X:
    "lambda x: (max(x, dace.{dtype}(0)) + {alpha} * min(x, dace.{dtype}(0)))".
    format(dtype=X.dtype.to_string(), alpha=node.alpha))
def LeakyRelu(X, Y):
    Y[:] = dace.elementwise(cast_lambda, X)


@python_pure_op_implementation(shape=lambda reshaped: reshaped.shape)
def Reshape(data, reshaped):
    reshaped[:] = np.reshape(data, shape)


@python_pure_op_implementation(
    shape=lambda input, node:
    [prod(input.shape[:node.axis]),
     prod(input.shape[node.axis:])])
def Flatten(input, output):
    output[:] = input.reshape(shape)


@op_implementation(op="Sum", name="pure")
class PureSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # check that all shapes are arrays, and that the shapes are all equal
        shape = None
        for edge in node.iter_inputs_in_onnx_order(state):
            desc = in_desc_with_name(node, state, sdfg, edge.dst_conn)
            if shape is None:
                shape = desc.shape

            if not iterables_equal(shape, desc.shape):
                return False

        if not iterables_equal(
                shape,
                out_desc_with_name(node, state, sdfg, "sum").shape):
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.name)
        input_names = []
        for e in node.iter_inputs_in_onnx_order(state):
            new_desc = copy.deepcopy(
                in_desc_with_name(node, state, sdfg, e.dst_conn))
            new_desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, new_desc)
            input_names.append(e.dst_conn)

        new_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "sum"))
        new_desc.transient = False
        nsdfg.add_datadesc("sum", new_desc)

        nstate = nsdfg.add_state()
        # we know all shapes are equal to the output shape
        shape = out_desc_with_name(node, state, sdfg, "sum").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        tasklet, _, _ = nstate.add_mapped_tasklet(
            node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs={
                f"__{inp}": dace.Memlet(f"{inp}[{index_str}]")
                for inp in input_names
            },
            code=f"__sum = {' + '.join(f'__{inp}' for inp in input_names)}",
            outputs={"__sum": dace.Memlet(f"sum[{index_str}]")},
            external_edges=True)

        tasklet.in_connectors = {
            f"__{inp}": in_desc_with_name(node, state, sdfg, inp).dtype
            for inp in input_names
        }
        tasklet.out_connectors = {
            "__sum": out_desc_with_name(node, state, sdfg, "sum").dtype
        }
        return nsdfg


@python_pure_op_implementation(**softmax_compute)
def LogSoftmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=keepdims)
    max_sub = input - maximum
    exponent = np.exp(max_sub)
    sum = np.add.reduce(exponent, axis=axis, keepdims=keepdims)
    log_sum = np.log(sum)
    output[:] = max_sub - log_sum


@op_implementation(op="Slice", name="pure")
class PureSlice(ONNXForward):
    '''
        Slice expansion
    '''
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # check that all the inputs (even the optional ones) are present and constant

        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if in_edge_with_name(
                node, state, "starts"
        ).src.data not in sdfg._parent_onnx_model.clean_weights:
            return False
        if in_edge_with_name(
                node, state,
                "ends").src.data not in sdfg._parent_onnx_model.clean_weights:
            return False

        # optional inputs
        is_axes_present = True
        try:
            if in_edge_with_name(
                    node, state, "axes"
            ).src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        is_steps_present = True
        try:
            if in_edge_with_name(
                    node, state, "steps"
            ).src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_steps_present = False

        # Current constraints: axes and steps must be explict. Axes must be zero and steps must be 1
        if not is_axes_present or not is_steps_present:
            return False

        step = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "steps").src.data].numpy()[0]
        axis = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "axes").src.data].numpy()[0]

        if step != 1 or axis != 0:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        start = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "starts").src.data].numpy()[0]
        end = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "ends").src.data].numpy()[0]

        output_shape = out_desc_with_name(node, state, sdfg, "output").shape
        if end == np.iinfo(np.int64).max:
            # Pytorch exporter artifact
            end = start + output_shape[0]

        def prog(data, output):
            tmp = data[start:end:1, :]
            # We need reshape to avoid Invalid Edge errors
            output[:] = np.reshape(tmp, output.shape)

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation
def Softplus(X, Y):
    Y[:] = np.log(1 + np.exp(X))


@python_pure_op_implementation(dtype=lambda X: X.dtype)
def Sigmoid(X, Y):
    Y[:] = dace.elementwise(lambda x: dtype(1) / (dtype(1) + exp(-x)), X)


@op_implementation(op="Transpose", name="einsum")
class EinsumTranspose(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        perm = node.perm
        input_desc = in_desc_with_name(node, state, sdfg, "data")
        output_desc = out_desc_with_name(node, state, sdfg, "transposed")

        letters = [chr(ord('z') - i) for i in range(26)]
        input_letters = "".join(letters[i]
                                for i, _ in enumerate(input_desc.shape))
        output_letters = "".join(letters[i] for i in perm)
        equation_str = f"{input_letters}->{output_letters}"

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=equation_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        nsdfg.add_datadesc("data", copy.deepcopy(input_desc))
        nsdfg.add_datadesc("transposed", copy.deepcopy(output_desc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["transposed"].transient = False

        nstate.add_edge(nstate.add_read("data"), None, einsum_node,
                        "Inputs__0", nsdfg.make_array_memlet("data"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("transposed"),
                        None, nsdfg.make_array_memlet("transposed"))

        return nsdfg


@op_implementation(op="Split", name="pure")
class SplitPure(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        split_dim = node.axis
        sizes = node.split
        idesc = in_desc_with_name(node, state, sdfg, "input")
        nsdfg.add_datadesc("input", copy.deepcopy(idesc))
        nsdfg.arrays["input"].transient = False

        rnode = nstate.add_read("input")

        offset = 0
        for i, odim in enumerate(sizes):
            # Set up new node shape and memlet
            new_shape = list(idesc.shape)
            new_shape[split_dim] = odim
            rng = subsets.Range([(0, s - 1, 1) if j != split_dim else
                                 (offset, offset + odim - 1, 1)
                                 for j, s in enumerate(new_shape)])
            offset += odim

            # Set up data descriptor
            oname = f"outputs__{i}"
            odesc = copy.deepcopy(out_desc_with_name(node, state, sdfg, oname))
            odesc.transient = False
            nsdfg.add_datadesc(oname, odesc)
            wnode = nstate.add_write(oname)

            # Perform copy (view)
            nstate.add_nedge(
                rnode, wnode,
                dace.Memlet(data="input",
                            subset=rng,
                            other_subset=subsets.Range.from_array(odesc)))

        return nsdfg


@op_implementation(op="Slice", name="pure")
class PureSliceAllConstant(ONNXForward):
    @staticmethod
    def _get_constant(conn: str, node: onnx_op.ONNXOp, state: SDFGState,
                      sdfg: SDFG):
        try:
            srcnode = next(state.in_edges_by_connector(node, conn)).src
        except StopIteration:
            return None
        # Scalar copied to GPU
        if 'gpu_' in srcnode.data:
            srcnode = state.predecessors(srcnode)[0]
        return onnx_constant_or_none(sdfg, srcnode)

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        for inconn in ("axes", "ends", "starts", "steps"):
            if PureSliceAllConstant._get_constant(inconn, node, state,
                                                  sdfg) is None:
                return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = PureSliceAllConstant._get_constant('axes', node, state, sdfg)
        ends = PureSliceAllConstant._get_constant('ends', node, state, sdfg)
        starts = PureSliceAllConstant._get_constant('starts', node, state,
                                                    sdfg)
        steps = PureSliceAllConstant._get_constant('steps', node, state, sdfg)

        constant_folding.remove_node_and_computation(sdfg, state, node, "axes")
        constant_folding.remove_node_and_computation(sdfg, state, node, "ends")
        constant_folding.remove_node_and_computation(sdfg, state, node,
                                                     "starts")
        constant_folding.remove_node_and_computation(sdfg, state, node,
                                                     "steps")

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        idesc = in_desc_with_name(node, state, sdfg, "data")
        odesc = out_desc_with_name(node, state, sdfg, "output")
        nsdfg.add_datadesc("data", copy.deepcopy(idesc))
        nsdfg.add_datadesc("output", copy.deepcopy(odesc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["output"].transient = False

        if not isinstance(axes, (tuple, list)):
            axes = [axes]
            ends = [ends]
            starts = [starts]
            steps = [steps]

        # Set up slicing memlet
        rng = [(0, s - 1, 1) for s in idesc.shape]
        for axis, start, end, step in zip(axes, starts, ends, steps):
            s = idesc.shape[axis]
            if end > s:
                end = s
            rng[axis] = (start, end - 1, step)

        sbs = subsets.Range(rng)
        osbs = subsets.Range.from_array(odesc)

        # Make copy / view
        rnode = nstate.add_read("data")
        wnode = nstate.add_write("output")

        nstate.add_nedge(
            rnode, wnode,
            dace.Memlet(data="data", subset=sbs, other_subset=osbs))

        return nsdfg


@op_implementation(op="Shape", name="pure")
class PureShape(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        data_desc = in_desc_with_name(node, state, sdfg, "data")

        try:
            np.array(data_desc.shape, np.int64)
        except Exception:
            # this happens if the shape is symbolic, for example
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        data_desc = in_desc_with_name(node, state, sdfg, "data")
        shape_val = np.array(data_desc.shape, np.int64)

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        nsdfg.add_datadesc(
            "data",
            copy.deepcopy(data_desc),
        )
        nsdfg.arrays["data"].transient = False
        nsdfg.add_array("shape", shape_val.shape, dtype=dace.int64)
        s = nstate.add_write("shape")

        for i, v in enumerate(shape_val):
            tasklet = nstate.add_tasklet("write_shape", {},
                                         {'shape_scalar': dace.int64},
                                         f"shape_scalar = {v}")
            nstate.add_edge(tasklet, "shape_scalar", s, None,
                            dace.Memlet("shape[{}]".format(i)))

        return nsdfg


@op_implementation(op="Gather", name="pure")
class PureGather(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # To understand this operator, read the docs for np.take.
        # The ONNX docs are not easy to understand (and are incorrect in opset 11)

        nsdfg, nstate, _, _ = empty_sdfg_for_node(sdfg,
                                                  state,
                                                  node,
                                                  add_access_nodes=False)
        out_shape = out_desc_with_name(node, state, sdfg, "output").shape
        idx_desc = in_desc_with_name(node, state, sdfg, "indices")
        idx_shape = idx_desc.shape
        data_shape = in_desc_with_name(node, state, sdfg, "data").shape

        # FIXME: we can sometimes generate views

        # Generate a copy kernel that loops over every element in the output
        # and read the correct element according to the indices

        axis = node.axis

        map_ranges = [(f"i{i}", f"0:{s}") for i, s in enumerate(out_shape)]
        # the map ranges can be partitioned into two parts.
        # the first part is the range over the indices, the second part is the
        # range over the data
        if isinstance(idx_desc, data.Scalar):
            # handle the edgecase here because the shape of a scalar in dace is
            # (1,) not ()
            idx_len = 0
        else:
            idx_len = len(idx_shape)
        map_ranges_indices = map_ranges[axis:axis + idx_len]
        map_ranges_data = map_ranges[:axis] + map_ranges[axis + idx_len:]

        # compute the indexing expressions
        fst = lambda x: x[0]
        output_idx_str = 'output[' + ', '.join(map(fst, map_ranges)) + ']'
        # the memlet string used to read data, which reads the whole axis
        data_memlet_elems = list(map(fst, map_ranges_data))
        data_memlet_elems.insert(axis, f'0:{data_shape[axis]}')

        data_memlet_str = 'data[' + ', '.join(data_memlet_elems) + ']'

        indices_idx_str = 'indices'
        if map_ranges_indices:
            indices_idx_str += '[' + ', '.join(map(fst,
                                                   map_ranges_indices)) + ']'
        else:
            indices_idx_str += '[0]'

        tasklet, me, mx = nstate.add_mapped_tasklet(
            node.label + "_tasklet",
            map_ranges=map_ranges,
            inputs={
                "__data": dace.Memlet(data_memlet_str),
                "idx": dace.Memlet(indices_idx_str),
            },
            code=f"__output = __data[idx]",
            outputs={"__output": dace.Memlet(output_idx_str)},
            external_edges=True)

        return nsdfg


def insert_at_indices(v, xs, idx):
    xs = list(copy.deepcopy(xs))
    for i in idx:
        xs.insert(i, v)
    return xs


@python_pure_op_implementation(
    shape=lambda node, data: insert_at_indices(1, data.shape, node.axes))
def Unsqueeze(data, expanded):
    expanded[:] = np.reshape(data, shape)
