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
from dace.util import in_edge_with_name
from dace.transformation.passes.analysis import AccessSets


@properties.make_properties
class ParameterArray(data.Array):
    """
    An array for which a gradient can be computed.
    """
    # since this can be None, this is not a DataProperty
    gradient = properties.Property(dtype=str, desc="The corresponding gradient buffer", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "Parameter" + data.Array.__repr__(self)

    def add_gradient_buffer(self, sdfg: SDFG, name: str) -> str:
        """
        Find or create a gradient buffer for the parameter in the given SDFG.

        :param sdfg: the SDFG containing the parameter
        :param name: the name of the parameter
        :return: the name of the gradient buffer
        """

        if self.gradient:
            return self.gradient

        # First, check if this array already has a gradient buffer in a nested
        # SDFG. This happens, for example when pytorch modules are used in the
        # frontend. In that case:
        #  1. the parser assembles the closure of the module, which adds
        #      descriptors for all the parameters and their gradients (if they
        #      are required).
        #  2. A nested sdfg is added for the module, with those array names.
        #  3. The DaceProgram will then pass these arrays in when the
        #     DaceProgram is called, using the names from the closure that
        #     match the names from the NestedSDFG
        #  4. When parsing the backward nodes, we want the gradient buffers in
        #     the closure to match the gradient buffers that we pass in. Thus,
        #     we need to make sure that we use the same name as the NestedSDFG
        #
        # Note that we do not currently do any nesting beyond this level,
        # because nested modules are converted to one SDFG.

        cands = set()
        for state in sdfg.nodes():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue

                nested_names = set()

                for edge in state.in_edges(node):
                    if edge.data.data == name:
                        nested_names.add(edge.dst_conn)
                for edge in state.out_edges(node):
                    if edge.data.data == name:
                        nested_names.add(edge.dst_conn)

                for name in nested_names:
                    nested_desc = node.sdfg.arrays[name]
                    if isinstance(nested_desc, ParameterArray) and nested_desc.gradient:
                        cands.add(nested_desc.gradient)

        if len(cands) > 1:
            raise ValueError("Multiple gradient buffers found for parameter " + name)
        elif len(cands) == 1:
            # we found a name of a gradient buffer in a nested SDFG:
            # reuse the same name in the outer sdfg if there is a matching descriptor
            grad_name = cands.pop()
            if grad_name in sdfg.arrays:
                self.gradient = grad_name
                return grad_name
        else:
            grad_name = sdfg._find_new_name('gradient_' + name)

        # Create a gradient buffer for the array
        grad_desc = copy.deepcopy(self)
        grad_desc.__class__ = data.Array
        grad_desc.transient = True
        grad_name = sdfg.add_datadesc(grad_name, grad_desc, find_new_name=True)
        self.gradient = grad_name
        return grad_name

    @staticmethod
    def make_parameter(sdfg: SDFG, name: str):
        """
        Converts an existing array into a parameter, without copying.

        :param sdfg: the SDFG containing the array.
        :param name: the name of the array.
        """
        desc = sdfg.arrays[name]
        if isinstance(desc, ParameterArray):
            return

        new_desc = ParameterArray(desc.dtype,
                                  desc.shape,
                                  storage=desc.storage,
                                  location=desc.location,
                                  allow_conflicts=desc.allow_conflicts,
                                  transient=desc.transient,
                                  strides=desc.strides,
                                  offset=desc.offset,
                                  lifetime=desc.lifetime,
                                  alignment=desc.alignment,
                                  debuginfo=desc.debuginfo,
                                  total_size=desc.total_size)
        sdfg.arrays[name] = new_desc


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
