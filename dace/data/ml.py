# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
ML-related data descriptors.

This module contains data descriptors that are specific to machine learning workflows,
such as ParameterArray for automatic differentiation.
"""
import copy

from dace import properties
from dace.data.core import Array
from dace.sdfg import SDFG, nodes


@properties.make_properties
class ParameterArray(Array):
    """
    An array for which a gradient can be computed.
    """
    # since this can be None, this is not a DataProperty
    gradient = properties.Property(dtype=str, desc="The corresponding gradient buffer", default=None, allow_none=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "Parameter" + Array.__repr__(self)

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
        grad_desc.__class__ = Array
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

        new_desc = copy.deepcopy(desc)
        new_desc.__class__ = ParameterArray
        new_desc.gradient = None
        sdfg.arrays[name] = new_desc
