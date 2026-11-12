# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
General purpose Einstein sum (einsum) library node.

Specialization expansions of this node convert it to fast BLAS operations (e.g., matrix multiplications) if possible.
"""

from copy import deepcopy

from dace import SDFG, Memlet, SDFGState, library, nodes, properties, symbolic
from dace import transformation as xf
from dace.frontend.common import einsum
from dace.libraries.blas import block
from dace.libraries.standard.environments.cuda import CUDA


def _is_dot_einsum(einsum_str: str, num_tensor_inputs: int) -> bool:
    """True for a 2-operand single-contracted-index dot ``i,i->`` (scalar output). Such a
    contraction is a DDOT, not a GEMM -- the degenerate 1x1 GEMM the contraction path would
    emit carries an illegal BLAS leading dimension for a strided operand."""
    lhs, sep, rhs = einsum_str.partition("->")
    terms = [t.strip() for t in lhs.split(",")]
    return (
        sep == "->"
        and rhs.strip() == ""
        and num_tensor_inputs == 2
        and len(terms) == 2
        and len(terms[0]) == 1
        and terms[0] == terms[1]
    )


# Define the library node itself
@library.node
class Einsum(nodes.LibraryNode):
    # Set the default expansion of the node to 'specialize' (registered below)
    implementations = {}
    default_implementation = "specialize"

    # Configurable properties of the einsum node
    einsum_str = properties.Property(
        dtype=str, default="", category="Semantics", desc="The Einstein notation string that describes this einsum"
    )

    alpha = properties.SymbolicProperty(
        category="Semantics", desc="The coefficient to multiply the inputs with", default=1.0
    )

    def folds_onto_output(self) -> bool:
        return self.folds_by_beta(self.beta)

    beta = properties.SymbolicProperty(
        category="Semantics", desc="The coefficient to multiply the output with when added to the product", default=0.0
    )

    def scalar_code(self, state) -> str | None:
        if "_beta" in self.in_connectors or not symbolic.equal_valued(0, self.beta):
            return None
        (out,) = self.out_connectors
        return f"{out} = ({self.alpha}) * " + " * ".join(sorted(self.in_connectors))


# Define the expansion, which specializes the einsum by lowering it to either a BLAS operation or a direct contraction
@library.register_expansion(Einsum, "specialize")
class SpecializeEinsum(xf.ExpandTransformation):
    # Define environments necessary for this expansion (optional, can be an empty list)
    environments = []

    # The following method returns the SDFG that results from expanding the library node.
    # Upon expansion, DaCe will insert the returned SDFG into the graph as a nested SDFG node (which can be inlined).
    @staticmethod
    def expansion(node: Einsum, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        # Make an SDFG for the expansion
        sdfg = SDFG("einsum")
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
        coeff_of = {"_alpha": "alpha", "_beta": "beta"}
        coeffs = {"alpha": node.alpha, "beta": node.beta}
        coeff_assignments = {}

        # Add the given arrays (as given by memlets) to the expansion SDFG
        inputs = []
        output = None
        for e in parent_state.in_edges(node):
            if e.data.is_empty():
                continue
            desc = parent_sdfg.arrays[e.data.data]
            if e.dst_conn in coeff_of:
                which = coeff_of[e.dst_conn]
                sym = f"__einsum_{which}"
                sdfg.add_array(e.dst_conn, [1], desc.dtype, storage=desc.storage)
                sdfg.add_symbol(sym, desc.dtype)
                coeffs[which] = coeffs[which] * symbolic.symbol(sym)
                coeff_assignments[sym] = f"{e.dst_conn}[0]"
                continue
            inputs.append(e.dst_conn)
            insubset = deepcopy(e.data.src_subset)
            isqdim = insubset.squeeze()
            sdfg.add_array(
                e.dst_conn,
                insubset.size(),
                desc.dtype,
                strides=[s for i, s in enumerate(desc.strides) if i in isqdim],
                storage=desc.storage,
            )

        for e in parent_state.out_edges(node):
            if e.data.is_empty():
                continue
            output = e.src_conn
            desc = parent_sdfg.arrays[e.data.data]
            outsubset = deepcopy(e.data.dst_subset)
            osqdim = outsubset.squeeze()
            sdfg.add_array(
                output,
                outsubset.size(),
                desc.dtype,
                strides=[s for i, s in enumerate(desc.strides) if i in osqdim],
                storage=desc.storage,
            )
        #######################################

        # Bind each promoted coefficient symbol from its scalar array before the einsum.
        if coeff_assignments:
            sdfg.add_state_before(state, "einsum_coeffs", assignments=coeff_assignments)

        # A scalar-output dot ``i,i->`` lowers to a stride-aware DDOT (``Dot`` node), NOT the
        # degenerate 1x1 GEMM the contraction path emits (whose leading dimension ``lda=1`` is
        # illegal for a strided operand -> DGEMM no-ops, silently dropping the contraction).
        # ``out = alpha * dot(x, y) + beta * out_prior``.
        if _is_dot_einsum(node.einsum_str, len(inputs)):
            from dace.libraries.blas.nodes.dot import Dot

            x_conn, y_conn = sorted(inputs)
            dtype = sdfg.arrays[output].dtype
            sdfg.add_scalar("__dot_res", dtype, transient=True, storage=sdfg.arrays[output].storage)
            dot = Dot("dot")
            state.add_node(dot)
            state.add_edge(state.add_read(x_conn), None, dot, "_x", Memlet.from_array(x_conn, sdfg.arrays[x_conn]))
            state.add_edge(state.add_read(y_conn), None, dot, "_y", Memlet.from_array(y_conn, sdfg.arrays[y_conn]))
            state.add_edge(dot, "_result", state.add_write("__dot_res"), None, Memlet("__dot_res[0]"))

            alpha, beta = coeffs["alpha"], coeffs["beta"]
            # ``symbolic.equal_valued`` (not ``beta == 0``): a ``sympy.Float(0.0)`` -- what the
            # SymbolicProperty stores for a ``0.0`` beta -- compares UNEQUAL to the Python/int
            # ``0`` under sympy's structural ``__eq__``, so ``beta == 0`` is ``False`` even for a
            # zero beta. That would emit the fold branch's ``__oin = output[0]`` read-back of an
            # uninitialized reduction temp (``0.0 * NaN = NaN`` -> nondeterministic garbage). This
            # is the value-aware zero test gemm/``create_einsum_sdfg`` use.
            if symbolic.equal_valued(0, beta):
                scale = state.add_tasklet("dot_scale", {"__d"}, {"__o"}, f"__o = ({alpha}) * __d")
            else:
                scale = state.add_tasklet(
                    "dot_scale", {"__d": None, "__oin": None}, {"__o"}, f"__o = ({alpha}) * __d + ({beta}) * __oin"
                )
                state.add_edge(state.add_read(output), None, scale, "__oin", Memlet(f"{output}[0]"))
            state.add_edge(state.add_read("__dot_res"), None, scale, "__d", Memlet("__dot_res[0]"))
            state.add_edge(scale, "__o", state.add_write(output), None, Memlet(f"{output}[0]"))
            return sdfg

        # Fill SDFG with einsum contents
        einsum.create_einsum_sdfg(
            sdfg,
            state,
            node.einsum_str,
            *sorted(inputs),
            output=output,
            output_name=output,
            alpha=coeffs["alpha"],
            beta=coeffs["beta"],
        )
        return sdfg


@library.register_expansion(Einsum, "CUDA (block strided)")
class ExpandEinsumCUDABlock(xf.ExpandTransformation):
    """In-kernel contraction by ONE thread block: each lane computes elements of the output, strided over the block,
    summing the product of the operands over every index the output does not name. Runtime coefficients
    (``_alpha`` / ``_beta``) and operands whose rank the subscripts do not match take ``specialize`` instead."""

    runs_inside_kernel = True
    environments = [CUDA]

    @staticmethod
    def expansion(node: Einsum, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        parsed = einsum.EinsumParser(node.einsum_str)
        sdfg = SDFG(node.label + "_block")
        inputs = sorted(c for c in node.in_connectors if c not in ("_alpha", "_beta"))
        outputs = list(node.out_connectors)
        if len(inputs) != len(node.in_connectors) or len(inputs) != len(parsed.inputs) or len(outputs) != 1:
            return SpecializeEinsum.expansion(node, parent_state, parent_sdfg)
        extents = {}
        for conn, term in zip(inputs + outputs, parsed.inputs + [parsed.output]):
            edge = next(e for e in parent_state.all_edges(node) if conn in (e.dst_conn, e.src_conn))
            desc = parent_sdfg.arrays[edge.data.data]
            subset = deepcopy(edge.data.subset)
            dims = subset.squeeze()
            shape, strides = subset.size(), [desc.strides[d] * edge.data.subset[d][2] for d in dims]
            if len(shape) != len(term) or desc.dtype.veclen > 1:
                return SpecializeEinsum.expansion(node, parent_state, parent_sdfg)
            sdfg.add_array(conn, shape, desc.dtype, strides=strides, storage=desc.storage)
            for letter, extent in zip(term, shape):
                extents.setdefault(letter, extent)
        index = {letter: f"__e{letter}" for letter in extents}
        reduced = [letter for letter in extents if letter not in parsed.output]
        term = " * ".join(block.at(sdfg, c, *(index[x] for x in t)) for c, t in zip(inputs, parsed.inputs))
        out = block.at(sdfg, outputs[0], *(index[x] for x in parsed.output))
        zero = symbolic.equal_valued(0, node.beta)
        values = {"alpha": node.alpha} if zero else {"alpha": node.alpha, "beta": node.beta}
        dtype = sdfg.arrays[outputs[0]].dtype
        code = block.coefficients(dtype, **values) + block.block_contraction_code(
            [(index[x], extents[x]) for x in parsed.output],
            [(index[x], extents[x]) for x in reduced],
            term,
            f"{out} = __alpha * __acc;" if zero else f"{out} = __alpha * __acc + __beta * {out};",
            dtype.ctype,
        )
        return block.block_sdfg(node, parent_state, sdfg, code, tuple(values.values()))
