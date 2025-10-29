import logging
from typing import Optional

import numpy as np

import dace
import torch
from dace.properties import make_properties
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil

import dace.libraries.onnx as donnx
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.nodes.onnx_op import ONNXOp
from dace.libraries.onnx import ONNXModel

log = logging.getLogger(__name__)

# blocklist of nondeterministic ops
# yapf: disable
NONDETERMINISTIC_OPS = {'ONNXDropout',
                        'ONNXGradient',
                        'ONNXGraphCall',
                        'ONNXIf',
                        'ONNXLoop',
                        'ONNXMomentum',
                        'ONNXMultinomial',
                        'ONNXRandomNormal',
                        'ONNXRandomNormalLike',
                        'ONNXRandomUniform',
                        'ONNXRandomUniformLike',
                        'ONNXSVMClassifier',
                        'ONNXSVMRegressor',
                        'ONNXScan',
                        'ONNXTreeEnsembleClassifier',
                        'ONNXTreeEnsembleRegressor'}
# Supported ops list
SUPPORTED_OPS = {'ONNXShape',
                 'ONNXConstantOfShape',
                 'ONNXRange',
                 'ONNXMul',
                 'ONNXAdd',
                 'ONNXSub',
                 'ONNXDiv',
                 'ONNXEqual',
                 'ONNXGreater',
                 'ONNXLess',
                 'ONNXGreaterOrEqual',
                 'ONNXLessOrEqual',
                 'ONNXWhere',
                 'ONNXUnsqueeze',
                 'ONNXConcat',
                 'ONNXReshape',
                 'ONNXTrilu',
                 'ONNXCast'}
# yapf: enable


@make_properties
class ConstantFolding(transformation.SingleStateTransformation):
    """ Remove nodes where all inputs are known and replace them with constant nodes by precomputing the output.
    """
    # pattern matching only checks that the type of the node matches,
    onnx_node = transformation.PatternNode(ONNXOp)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.onnx_node)]

    @staticmethod
    def is_constant(sdfg: dace.SDFG, state: dace.SDFGState, node) -> bool:
        if len(state.in_edges(node)) > 0:
            return False

        # the ONNX importer adds a _parent_onnx_model attribute to the sdfg
        if isinstance(node, nd.AccessNode) and node.data in sdfg._parent_onnx_model.clean_weights:
            return True

        return False

    def can_be_applied(self,
                       graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       expr_index: int,
                       sdfg,
                       permissive: bool = False):

        node = self.onnx_node

        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        # Check the supported list of operations
        if "ONNX" + node.schema.name not in SUPPORTED_OPS:
            return False

        # Check the blocklist of operations
        if 'ONNX' + node.schema.name in NONDETERMINISTIC_OPS:
            return False

        if node.schema.name == "Shape":
            assert len(graph.in_edges(node)) == 1
            shape_in_edge = graph.in_edges(node)[0]
            assert shape_in_edge.dst_conn == "data"
            shape_desc = sdfg.arrays[shape_in_edge.src.data]
            try:
                np.array(shape_desc.shape, np.int64)
            except Exception:
                # this happens if the shape is symbolic, for example
                return False

            return True

        if node.schema.name == "ConstantOfShape":
            # Check that the shape input is a constant
            assert len(graph.in_edges(node)) == 1
            shape_in_edge = graph.in_edges(node)[0]
            assert shape_in_edge.dst_conn == "input"

            # Check if the shape is in clean_weights
            if shape_in_edge.src.data not in sdfg._parent_onnx_model.clean_weights:
                return False

            return True

        # all inputs are constant
        for edge in graph.in_edges(node):
            if not ConstantFolding.is_constant(sdfg, graph, edge.src):
                return False

        return True

    @classmethod
    def match_to_str(cls, graph):
        node: ONNXOp = cls.onnx_node
        return "Precompute outputs of {}".format(node)

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG):
        parent: ONNXModel = sdfg._parent_onnx_model
        node = self.onnx_node
        log.debug(f"Applying constant folding: {node} in {state}")

        if node.schema.name == "Shape":
            # if we have a shape node, replace it with a constant
            assert len(state.in_edges(node)) == 1
            shape_in_edge = state.in_edges(node)[0]
            assert shape_in_edge.dst_conn == "data"
            shape_desc = sdfg.arrays[shape_in_edge.src.data]

            # Use the existing output name instead of creating a new temp name
            # This ensures downstream ops can find the constant
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update the array descriptor to have the correct shape
            sdfg.arrays[constant_name].shape = (len(shape_desc.shape), )
            sdfg.arrays[constant_name].dtype = dace.int64

            # Add the shape value to weights (use original name, not cleaned)
            # The clean_weights property will handle the name cleaning
            shape_value = np.array(shape_desc.shape, np.int64)
            parent.weights[constant_name] = torch.from_numpy(shape_value)

            # Mark as non-transient since it's now a constant
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "ConstantOfShape":
            # if we have a ConstantOfShape node with constant shape input, compute the output
            assert len(state.in_edges(node)) == 1
            shape_in_edge = state.in_edges(node)[0]
            assert shape_in_edge.dst_conn == "input"

            # Get the shape value from weights
            shape_data = parent.clean_weights[shape_in_edge.src.data]
            if hasattr(shape_data, 'cpu'):
                shape_value = shape_data.cpu().numpy()
            else:
                shape_value = np.array(shape_data)

            # Get the fill value (default is 0.0)
            fill_value = 0.0
            if hasattr(node, 'value') and node.value is not None:
                # value is a tensor proto
                fill_value = node.value
                if hasattr(fill_value, 'float_data') and len(fill_value.float_data) > 0:
                    fill_value = fill_value.float_data[0]
                elif hasattr(fill_value, 'int64_data') and len(fill_value.int64_data) > 0:
                    fill_value = fill_value.int64_data[0]
                elif hasattr(fill_value, 'int32_data') and len(fill_value.int32_data) > 0:
                    fill_value = fill_value.int32_data[0]

            # Get the output descriptor to determine dtype
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data
            output_dtype = sdfg.arrays[constant_name].dtype.as_numpy_dtype()

            # Create the filled tensor
            output_tensor = np.full(tuple(shape_value), fill_value, dtype=output_dtype)

            # Update the array descriptor
            sdfg.arrays[constant_name].shape = tuple(shape_value)

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(output_tensor)

            # Mark as non-transient since it's now a constant
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Range":
            # if we have a Range node with constant inputs, compute the range output
            # Get the constant inputs
            start_val = None
            limit_val = None
            delta_val = None

            for edge in state.in_edges(node):
                if edge.dst_conn == "start":
                    start_val = parent.clean_weights[edge.src.data].cpu().numpy()
                elif edge.dst_conn == "limit":
                    limit_val = parent.clean_weights[edge.src.data].cpu().numpy()
                elif edge.dst_conn == "delta":
                    delta_val = parent.clean_weights[edge.src.data].cpu().numpy()

            # Compute the range output
            # Handle both scalar and array inputs
            start_scalar = start_val.item() if hasattr(start_val, 'item') else start_val
            limit_scalar = limit_val.item() if hasattr(limit_val, 'item') else limit_val
            delta_scalar = delta_val.item() if hasattr(delta_val, 'item') else delta_val

            range_output = np.arange(start_scalar, limit_scalar, delta_scalar)

            # Determine the output data type based on the node's output connector
            output_edge = state.out_edges(node)[0]
            output_desc = sdfg.arrays[output_edge.src.data] if hasattr(output_edge.src,
                                                                       'data') else sdfg.arrays[output_edge.dst.data]
            output_dtype = output_desc.dtype.as_numpy_dtype()

            range_output = range_output.astype(output_dtype)

            # Create a constant for the range output
            constant_name = sdfg.temp_data_name()
            clean_constant_name = clean_onnx_name(constant_name)
            # Use add_scalar for 0-dimensional arrays (scalars)
            if len(range_output.shape) == 0:
                sdfg.add_scalar(clean_constant_name, dace.dtypes.typeclass(output_dtype.type))
            else:
                sdfg.add_array(clean_constant_name, range_output.shape, dace.dtypes.typeclass(output_dtype.type))

            assert constant_name not in parent.clean_weights
            parent.weights[constant_name] = torch.from_numpy(range_output)

            # Replace the Range node output with the constant
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            access_range = state.add_access(clean_constant_name)
            state.add_edge(access_range, None, output_edge.dst, output_edge.dst_conn,
                           sdfg.make_array_memlet(clean_constant_name))

        elif node.schema.name in ["Mul", "Add", "Sub", "Div"]:
            # Binary arithmetic operations with all constant inputs
            assert len(state.in_edges(node)) == 2

            # Get the two input values
            inputs = {}
            for edge in state.in_edges(node):
                input_name = edge.src.data
                input_data = parent.clean_weights[input_name]
                if hasattr(input_data, 'cpu'):
                    inputs[edge.dst_conn] = input_data.cpu().numpy()
                else:
                    inputs[edge.dst_conn] = np.array(input_data)

            # Get A and B (ONNX uses A and B connectors)
            # Can't use 'or' with numpy arrays, need explicit None check
            A = inputs.get('A') if 'A' in inputs else inputs.get('a') if 'a' in inputs else list(inputs.values())[0]
            B = inputs.get('B') if 'B' in inputs else inputs.get('b') if 'b' in inputs else list(inputs.values())[1]

            # Compute result based on operation
            if node.schema.name == "Mul":
                result = A * B
            elif node.schema.name == "Add":
                result = A + B
            elif node.schema.name == "Sub":
                result = A - B
            elif node.schema.name == "Div":
                result = A / B

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            result = np.asarray(result)

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name in ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"]:
            # Comparison operations with all constant inputs
            assert len(state.in_edges(node)) == 2

            # Get the two input values
            inputs = {}
            for edge in state.in_edges(node):
                input_name = edge.src.data
                input_data = parent.clean_weights[input_name]
                if hasattr(input_data, 'cpu'):
                    inputs[edge.dst_conn] = input_data.cpu().numpy()
                else:
                    inputs[edge.dst_conn] = np.array(input_data)

            # Get A and B
            # Can't use 'or' with numpy arrays, need explicit None check
            A = inputs.get('A') if 'A' in inputs else inputs.get('a') if 'a' in inputs else list(inputs.values())[0]
            B = inputs.get('B') if 'B' in inputs else inputs.get('b') if 'b' in inputs else list(inputs.values())[1]

            # Compute result based on operation
            if node.schema.name == "Equal":
                result = np.equal(A, B)
            elif node.schema.name == "Greater":
                result = np.greater(A, B)
            elif node.schema.name == "Less":
                result = np.less(A, B)
            elif node.schema.name == "GreaterOrEqual":
                result = np.greater_equal(A, B)
            elif node.schema.name == "LessOrEqual":
                result = np.less_equal(A, B)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            result = np.asarray(result)

            # If result is 0-dimensional (scalar), reshape to (1,) to avoid memlet issues
            if result.shape == ():
                result = result.reshape((1, ))

            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Where":
            # Where operation with all constant inputs
            assert len(state.in_edges(node)) == 3

            # Get the three input values (condition, X, Y)
            inputs = {}
            for edge in state.in_edges(node):
                input_name = edge.src.data
                input_data = parent.clean_weights[input_name]
                if hasattr(input_data, 'cpu'):
                    inputs[edge.dst_conn] = input_data.cpu().numpy()
                else:
                    inputs[edge.dst_conn] = np.array(input_data)

            # Where has inputs: condition, X, Y
            # Can't use 'or' with numpy arrays, need explicit None check
            condition = inputs.get('condition') if 'condition' in inputs else list(inputs.values())[0]
            X = inputs.get('X') if 'X' in inputs else list(inputs.values())[1]
            Y = inputs.get('Y') if 'Y' in inputs else list(inputs.values())[2]

            # Compute result
            result = np.where(condition, X, Y)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            result = np.asarray(result)

            # If result is 0-dimensional (scalar), reshape to (1,) to avoid memlet issues
            if result.shape == ():
                result = result.reshape((1, ))

            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Unsqueeze":
            # Unsqueeze operation with constant input
            # Get the data input
            data_input = None
            axes_input = None

            for edge in state.in_edges(node):
                if edge.dst_conn == "data":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        data_input = input_data.cpu().numpy()
                    else:
                        data_input = np.array(input_data)
                elif edge.dst_conn == "axes":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        axes_input = input_data.cpu().numpy()
                    else:
                        axes_input = np.array(input_data)

            # Get axes - could be from input connector or from attribute
            if axes_input is not None:
                axes = axes_input.tolist() if hasattr(axes_input, 'tolist') else axes_input
            elif hasattr(node, 'axes') and node.axes is not None:
                axes = node.axes
            else:
                raise ValueError("Unsqueeze requires axes to be specified")

            # Normalize axes to a list
            if isinstance(axes, (int, np.integer)):
                axes = [int(axes)]
            elif not isinstance(axes, list):
                axes = list(axes)

            # Apply unsqueeze by inserting dimensions at the specified axes
            result = data_input
            # Sort axes to insert from smallest to largest
            for axis in sorted(axes):
                result = np.expand_dims(result, axis=axis)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Concat":
            # Concat operation with all constant inputs
            # Get all inputs and the axis
            inputs = []
            axis = 0

            # Get axis from node attribute
            if hasattr(node, 'axis') and node.axis is not None:
                axis = node.axis

            # Collect all inputs in order
            # ONNX Concat has variable number of inputs named like __in_0, __in_1, etc.
            in_edges = state.in_edges(node)

            # Sort edges by connector name to maintain order
            def extract_index(edge):
                conn = edge.dst_conn
                if '__in_' in conn:
                    return int(conn.split('__in_')[1])
                return 0

            sorted_edges = sorted(in_edges, key=extract_index)

            for edge in sorted_edges:
                input_name = edge.src.data
                input_data = parent.clean_weights[input_name]
                if hasattr(input_data, 'cpu'):
                    inputs.append(input_data.cpu().numpy())
                else:
                    inputs.append(np.array(input_data))

            # Perform concatenation
            result = np.concatenate(inputs, axis=axis)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Reshape":
            # Reshape operation with constant inputs
            # Get data input and shape input
            data_input = None
            shape_input = None

            for edge in state.in_edges(node):
                if edge.dst_conn == "data":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        data_input = input_data.cpu().numpy()
                    else:
                        data_input = np.array(input_data)
                elif edge.dst_conn == "shape":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        shape_input = input_data.cpu().numpy()
                    else:
                        shape_input = np.array(input_data)

            # Convert shape to tuple
            new_shape = tuple(shape_input.tolist() if hasattr(shape_input, 'tolist') else shape_input)

            # Handle special case: -1 in shape means infer dimension
            if -1 in new_shape:
                # Calculate the inferred dimension
                total_elements = np.prod(data_input.shape)
                known_elements = 1
                infer_idx = -1
                for i, dim in enumerate(new_shape):
                    if dim == -1:
                        infer_idx = i
                    else:
                        known_elements *= dim

                if infer_idx >= 0:
                    new_shape = list(new_shape)
                    new_shape[infer_idx] = total_elements // known_elements
                    new_shape = tuple(new_shape)

            # Perform reshape
            result = data_input.reshape(new_shape)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Trilu":
            # Trilu operation with constant input(s)
            # Get data input and optional k input
            data_input = None
            k_input = None

            for edge in state.in_edges(node):
                if edge.dst_conn == "input":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        data_input = input_data.cpu().numpy()
                    else:
                        data_input = np.array(input_data)
                elif edge.dst_conn == "k":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        k_input = input_data.cpu().numpy()
                    else:
                        k_input = np.array(input_data)

            # Get upper attribute (default is 1 for upper triangular)
            upper = 1
            if hasattr(node, 'upper') and node.upper is not None:
                upper = node.upper

            # Get k value (diagonal offset, default is 0)
            k = 0
            if k_input is not None:
                k = int(k_input.item() if hasattr(k_input, 'item') else k_input)

            # Apply triangular operation
            if upper:
                # Upper triangular
                result = np.triu(data_input, k=k)
            else:
                # Lower triangular
                result = np.tril(data_input, k=k)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor
            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        elif node.schema.name == "Cast":
            # Cast operation with constant input
            # Get data input
            data_input = None

            for edge in state.in_edges(node):
                if edge.dst_conn == "input":
                    input_name = edge.src.data
                    input_data = parent.clean_weights[input_name]
                    if hasattr(input_data, 'cpu'):
                        data_input = input_data.cpu().numpy()
                    else:
                        data_input = np.array(input_data)

            # Get target dtype from node attribute
            if hasattr(node, 'to') and node.to is not None:
                # ONNX uses TensorProto data types (integers)
                # Map ONNX type to numpy dtype
                onnx_to_numpy_dtype = {
                    1: np.float32,  # FLOAT
                    2: np.uint8,  # UINT8
                    3: np.int8,  # INT8
                    5: np.int16,  # INT16
                    6: np.int32,  # INT32
                    7: np.int64,  # INT64
                    9: np.bool_,  # BOOL
                    10: np.float16,  # FLOAT16
                    11: np.float64,  # DOUBLE
                    12: np.uint32,  # UINT32
                    13: np.uint64,  # UINT64
                    16: np.float16,  # BFLOAT16 (map to float16)
                }

                target_dtype = onnx_to_numpy_dtype.get(node.to, np.float32)
            else:
                # If no target type specified, use the output descriptor's type
                output_edge = state.out_edges(node)[0]
                output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
                constant_name = output_access_node.data
                target_dtype = sdfg.arrays[constant_name].dtype.as_numpy_dtype()

            # Perform cast
            result = data_input.astype(target_dtype)

            # Get output name
            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            output_access_node = output_edge.dst if isinstance(output_edge.dst, nd.AccessNode) else output_edge.src
            constant_name = output_access_node.data

            # Update array descriptor (shape should be same, just dtype changes)
            sdfg.arrays[constant_name].shape = result.shape

            # Add to weights
            parent.weights[constant_name] = torch.from_numpy(result)

            # Mark as non-transient
            sdfg.arrays[constant_name].transient = False

        # remove all now useless nodes with a reverse BFS
        remove_node_and_computation(sdfg, state, node)


def remove_node_and_computation(sdfg: dace.SDFG, state: dace.SDFGState, node: nd.Node, connector: Optional[str] = None):
    """ Remove a node and the parent nodes that compute this node, if the outputs are not used elsewhere.

        :param sdfg: the sdfg containing the node.
        :param state: the state containing the node.
        :param node: the node to remove
        :param connector: if not None, the computation of the connector of
                          ``node`` will be removed, but not ``node`` itself.
    """
    if connector is not None:
        if connector not in node.in_connectors:
            return
        node.remove_in_connector(connector)
        edges = state.in_edges_by_connector(node, connector)
        for e in edges:
            state.remove_edge(e)
    else:
        # Remove all edges connected to the node
        edges = list(state.in_edges(node)) + list(state.out_edges(node))
        for e in edges:
            state.remove_edge(e)

        # Remove the node itself from the state
        state.remove_node(node)

    # remove dangling nodes, this can happen with non-transients
    for node, parent in sdfg.all_nodes_recursive():
        if (isinstance(node, nd.AccessNode) and parent.in_degree(node) + parent.out_degree(node) == 0):
            parent.remove_node(node)
