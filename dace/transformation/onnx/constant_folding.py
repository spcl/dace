import copy
import logging
from collections import deque
from typing import Dict, Optional

import numpy as np

import dace
import torch
from dace import data as dt, dtypes, memlet as mm
from dace import registry
from dace.properties import make_properties
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.transformation.passes import dead_dataflow_elimination
from dace.transformation import pass_pipeline

import dace.libraries.onnx as donnx
from dace.libraries.onnx.binary_utilities.python_onnx_node_evaluation import evaluate_node
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.nodes.onnx_op import ONNXOp
from dace.libraries.onnx import ONNXModel
from dace.libraries.onnx.environments import ONNXRuntime

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
        if not ONNXRuntime.is_installed():
            return False

        node = self.onnx_node

        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if not 'ONNX' + node.schema.name not in NONDETERMINISTIC_OPS:
            return False

        if isinstance(node, donnx.ONNXShape):
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

        if isinstance(node, donnx.ONNXShape):
            # if we have a shape node, replace it with a constant
            assert len(state.in_edges(node)) == 1
            shape_in_edge = state.in_edges(node)[0]
            assert shape_in_edge.dst_conn == "data"
            shape_desc = sdfg.arrays[shape_in_edge.src.data]

            constant_name = sdfg.temp_data_name()
            clean_constant_name = clean_onnx_name(constant_name)
            sdfg.add_array(clean_constant_name, (len(shape_desc.shape), ), dace.int64)

            assert constant_name not in parent.clean_weights
            parent.weights[constant_name] = torch.from_numpy(np.array(shape_desc.shape, np.int64))

            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            access_shape = state.add_access(clean_constant_name)
            state.add_edge(access_shape, None, output_edge.dst, output_edge.dst_conn,
                           sdfg.make_array_memlet(clean_constant_name))
        else:
            # otherwise compute the result of the op using the ORT API

            inputs = {}
            for edge in state.in_edges(node):
                # we know from can_be_applied that all in edges are from AccessNodes
                assert (isinstance(edge.src, nd.AccessNode) and hasattr(sdfg, "_parent_onnx_model")
                        and edge.src.data in sdfg._parent_onnx_model.clean_weights)

                input_value = sdfg._parent_onnx_model.clean_weights[edge.src.data]

                inputs[edge.dst_conn] = input_value.clone()

            outputs = evaluate_node(sdfg, state, node, inputs)

            for edge in state.out_edges(node):
                desc = copy.deepcopy(sdfg.arrays[edge.data.data])
                desc.transient = False
                output_value = outputs[edge.src_conn]

                constant_name = sdfg.temp_data_name()
                clean_constant_name = clean_onnx_name(constant_name)
                sdfg.add_datadesc(clean_constant_name, desc)

                assert constant_name not in parent.weights
                assert type(output_value) is torch.Tensor

                if not dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, desc.storage):
                    cpu_desc = copy.deepcopy(desc)
                    cpu_desc.storage = dtypes.StorageType.CPU_Heap
                    cpu_desc.transient = False
                    desc.transient = True
                    copy_in_name = sdfg.temp_data_name()
                    clean_copy_in_name = clean_onnx_name(copy_in_name)
                    sdfg.add_datadesc(clean_copy_in_name, cpu_desc)

                    access_constant = state.add_access(clean_constant_name)
                    state.add_edge(state.add_read(clean_copy_in_name), None, access_constant, None,
                                   sdfg.make_array_memlet(clean_copy_in_name))

                    name_to_add = copy_in_name
                else:
                    access_constant = state.add_read(clean_constant_name)
                    name_to_add = constant_name

                parent.weights[name_to_add] = output_value

                state.add_edge(access_constant, None, edge.dst, edge.dst_conn,
                               sdfg.make_array_memlet(clean_constant_name))

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
        edges = state.out_edges(node)
        for e in edges:
            state.remove_edge(e)

    # remove dangling nodes, this can happen with non-transients
    for node, parent in sdfg.all_nodes_recursive():
        if (isinstance(node, nd.AccessNode) and parent.in_degree(node) + parent.out_degree(node) == 0):
            parent.remove_node(node)
