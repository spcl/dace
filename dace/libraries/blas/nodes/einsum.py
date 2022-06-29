# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
General purpose Einstein sum (einsum) library node.

Specialization expansions of this node convert it to fast BLAS operations (e.g., matrix multiplications) if possible.
"""

from copy import deepcopy

from dace import SDFG, SDFGState, library, nodes, properties
from dace import transformation as xf
from dace.frontend.common import einsum


# Define the library node itself
@library.node
class Einsum(nodes.LibraryNode):
    # Set the default expansion of the node to 'specialize' (registered below)
    implementations = {}
    default_implementation = 'specialize'

    # Configurable properties of the einsum node
    einsum_str = properties.Property(dtype=str,
                                     default='',
                                     desc='The Einstein notation string that describes this einsum')


# Define the expansion, which specializes the einsum by lowering it to either a BLAS operation or a direct contraction
@library.register_expansion(Einsum, 'specialize')
class SpecializeEinsum(xf.ExpandTransformation):
    # Define environments necessary for this expansion (optional, can be an empty list)
    environments = []

    # The following method returns the SDFG that results from expanding the library node.
    # Upon expansion, DaCe will insert the returned SDFG into the graph as a nested SDFG node (which can be inlined).
    @staticmethod
    def expansion(node: Einsum, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        # Make an SDFG for the expansion
        sdfg = SDFG('einsum')
        state = sdfg.add_state()

        # Add the given arrays (as given by memlets) to the expansion SDFG
        inputs = []
        output = None
        for e in parent_state.in_edges(node):
            inputs.append(e.dst_conn)
            desc = parent_sdfg.arrays[e.data.data]
            insubset = deepcopy(e.data.src_subset)
            isqdim = insubset.squeeze()
            sdfg.add_array(e.dst_conn,
                           insubset.size(),
                           desc.dtype,
                           strides=[s for i, s in enumerate(desc.strides) if i in isqdim],
                           storage=desc.storage)

        for e in parent_state.out_edges(node):
            output = e.src_conn
            desc = parent_sdfg.arrays[e.data.data]
            outsubset = deepcopy(e.data.dst_subset)
            osqdim = outsubset.squeeze()
            sdfg.add_array(output,
                           outsubset.size(),
                           desc.dtype,
                           strides=[s for i, s in enumerate(desc.strides) if i in osqdim],
                           storage=desc.storage)
        #######################################

        # Fill SDFG with einsum contents
        einsum.create_einsum_sdfg(None, sdfg, state, node.einsum_str, *sorted(inputs), output=output)
        return sdfg
