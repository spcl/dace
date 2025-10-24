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

            # If result is 0-dimensional (scalar), reshape to (1,) to avoid memlet issues
            if result.shape == ():
                result = result.reshape((1, ))

            sdfg.arrays[constant_name].shape = result.shape

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
