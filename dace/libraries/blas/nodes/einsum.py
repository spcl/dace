# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
General purpose Einstein sum (einsum) library node.

Specialization expansions of this node convert it to fast BLAS operations (e.g., matrix multiplications) if possible.
"""

from copy import deepcopy

from dace import SDFG, SDFGState, library, nodes, properties, symbolic
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

    alpha = properties.SymbolicProperty(desc='The coefficient to multiply the inputs with', default=1.0)
    beta = properties.SymbolicProperty(desc='The coefficient to multiply the output with when added to the product',
                                       default=0.0)


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

        # The contraction coefficients alpha/beta may be supplied either as the
        # symbolic ``alpha``/``beta`` node properties (default) or as explicit
        # scalar input connectors ``_alpha``/``_beta`` (data-driven coefficients,
        # e.g. gemm's runtime ``alpha`` argument). For a connector, the scalar is
        # added as an input array and promoted to an SDFG-internal symbol bound
        # from its single element (BLAS/contraction consume a scalar value); the
        # effective coefficient is ``property * connector`` so the two compose.
        # The promoted symbol is local to this expansion and never leaks into the
        # parent SDFG, keeping the data read explicit at the node boundary.
        coeff_of = {'_alpha': 'alpha', '_beta': 'beta'}
        coeffs = {'alpha': node.alpha, 'beta': node.beta}
        coeff_assignments = {}

        # Add the given arrays (as given by memlets) to the expansion SDFG
        inputs = []
        output = None
        for e in parent_state.in_edges(node):
            desc = parent_sdfg.arrays[e.data.data]
            if e.dst_conn in coeff_of:
                which = coeff_of[e.dst_conn]
                sym = f'__einsum_{which}'
                sdfg.add_array(e.dst_conn, [1], desc.dtype, storage=desc.storage)
                sdfg.add_symbol(sym, desc.dtype)
                coeffs[which] = coeffs[which] * symbolic.symbol(sym)
                coeff_assignments[sym] = f'{e.dst_conn}[0]'
                continue
            inputs.append(e.dst_conn)
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

        # Bind each promoted coefficient symbol from its scalar array before the einsum.
        if coeff_assignments:
            sdfg.add_state_before(state, 'einsum_coeffs', assignments=coeff_assignments)

        # Fill SDFG with einsum contents
        einsum.create_einsum_sdfg(sdfg,
                                  state,
                                  node.einsum_str,
                                  *sorted(inputs),
                                  output=output,
                                  output_name=output,
                                  alpha=coeffs['alpha'],
                                  beta=coeffs['beta'])
        return sdfg
