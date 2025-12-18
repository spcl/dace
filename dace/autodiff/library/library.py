# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Dace library for autodiff

Includes the BackwardPass library node, and the replacements for the python frontend
"""
from typing import Dict, Set, Optional
import copy

import dace
import dace.library
from dace import data, properties
from dace.transformation import transformation as pm
from dace.sdfg import SDFG, SDFGState, graph, nodes

from dace.autodiff import backward_pass_generator as engine, analysis as autodiff_analysis
from dace.autodiff.utils import init_grad
from dace.sdfg.utils import in_edge_with_name
from dace.transformation.passes.analysis import AccessSets

# Import ParameterArray from the data package for backward compatibility
from dace.data.ml import ParameterArray


@dace.library.expansion
class ExpandBackwardPass(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'BackwardPass', state: SDFGState, sdfg: SDFG):

        node.validate(sdfg, state)

        in_array_name = lambda connector_name: in_edge_with_name(node, state, connector_name).data.data

        array_grad_map = {}

        access_sets = AccessSets().apply_pass(sdfg, {})

        nsdfg = SDFG("backward_" + sdfg.label)

        # Check for other BackwardPasses that also compute the same gradients as us
        node.propagate_conflicts(sdfg, state)

        # get the names of the output arrays in the forward pass
        given_gradients = node.outer_names_given_gradients(state)

        array_grad_map.update(node.required_gradients)
        array_grad_map.update((in_array_name(value_conn_name), grad_conn_name)
                              for grad_conn_name, value_conn_name in node.given_gradients.items())

        # remove the non-grad arrays as inputs from the forward pass;
        # they were also just added for control dependencies
        for forward_non_grad_conn_name in node.given_gradients.values():
            for edge in list(state.in_edges_by_connector(node, forward_non_grad_conn_name)):
                state.remove_edge(edge)
                if state.in_degree(edge.src) + state.out_degree(edge.src) == 0:
                    state.remove_node(edge.src)
            node.remove_in_connector(forward_non_grad_conn_name)

        gen = engine.BackwardPassGenerator(sdfg=sdfg,
                                           given_gradients=given_gradients,
                                           required_gradients=node.required_gradients.keys(),
                                           backward_sdfg=nsdfg,
                                           array_grad_map=array_grad_map,
                                           conflicted_gradient_buffers=node._conflicted_gradients)

        _, _, required_forwarded_values = gen.backward()

        # Add zero initialization for all gradients which we are the first to compute
        for outer_edge in state.out_edges(node):
            gradient_we_are_writing: str = outer_edge.data.data
            is_written_with_wcr = any(edge.data.wcr is not None and edge.data.data == outer_edge.src_conn
                                      for edge, _ in nsdfg.all_edges_recursive()
                                      if isinstance(edge, graph.MultiConnectorEdge))

            anyone_written_before_us = autodiff_analysis.is_previously_written(sdfg,
                                                                               state,
                                                                               node,
                                                                               gradient_we_are_writing,
                                                                               access_sets=access_sets)
            if not anyone_written_before_us and is_written_with_wcr:
                init_grad(gradient_we_are_writing, sdfg, state)

        for name in required_forwarded_values:
            # get the access to the forwarded_value
            # there should only be one since we don't allow inplace modification
            n = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == name]
            if len(n) > 1:
                raise ValueError(
                    "Expected only one access node for forwarded value, does the graph have in-place modification?")
            elif len(n) == 0:
                n = state.add_read(name)
            else:
                n = n[0]

            node.add_in_connector(name)
            state.add_edge(n, None, node, name, sdfg.make_array_memlet(name))

        nsdfg.validate()

        return nsdfg


@dace.library.node
class BackwardPass(nodes.LibraryNode):
    """
    The BackwardPass library node expands to an implementation of a
    BackwardPass that computes the requested gradients.

    These gradients are computed using the DaCe autograd engine.

    The gradient will be computed for each array in the output connectors.
    For this, the names of the output connectors must match the name of the
    array for which the gradient is to be computed.
    """

    # Global properties
    implementations = {
        "differentiate": ExpandBackwardPass,
    }
    default_implementation = "differentiate"

    given_gradients = properties.DictProperty(
        key_type=str,
        value_type=str,
        desc="Mapping between connector names of the given gradients and the names of the arrays they correspond to.")
    required_gradients = properties.DictProperty(
        key_type=str,
        value_type=str,
        desc=
        "Mapping from array name for which a gradient should be computed to the name of the connector that will receive the gradient."
    )

    _conflicted_gradients = properties.SetProperty(
        element_type=str,
        desc="Keys from required_gradients for which the gradients are also computed elsewhere, and thus writes to the "
        " buffer need to be with write-conflict-resolution. Note: this field is automatically populated upon expansion."
    )

    def __init__(self, name, given_gradients: Dict[str, str], *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.given_gradients = given_gradients
        self.required_gradients = {}

    def outer_names_given_gradients(self, state: SDFGState) -> Set[str]:
        """
        Returns the names of the arrays that are passed as given gradients.
        """
        in_array_name = lambda connector_name: in_edge_with_name(self, state, connector_name).data.data
        return set(map(in_array_name, self.given_gradients.values()))

    def propagate_conflicts(self, sdfg: SDFG, state: SDFGState):
        """
        Across this SDFG, check for other BackwardPasses that also compute the same gradients as us.

        If there are multiple BackwardPasses that compute the same gradients, update their list of conflicts.
        """

        ours = set(self.required_gradients)

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, BackwardPass):
                    if node is self:
                        continue
                    conflicts = ours.intersection(node.required_gradients)
                    if conflicts:
                        self._conflicted_gradients |= conflicts
                        node._conflicted_gradients |= conflicts

    def validate(self, sdfg, state):
        # Check that there is a correspondence between given gradients and inputs
        all_inputs = set(self.in_connectors)
        for given_grad, tensor_name in self.given_gradients.items():
            if given_grad not in all_inputs:
                raise ValueError("Given gradient '{}' is not an input of the node".format(given_grad))

            all_inputs.remove(given_grad)
            all_inputs.remove(tensor_name)

        if all_inputs:
            raise ValueError("The following in connectors were not included in given_gradients: {}".format(
                ', '.join(all_inputs)))

        # Check that we are computing at least one gradient
        if len(self.out_connectors) == 0:
            raise ValueError("BackwardPass node '{}' does not compute any gradients".format(self.name))
