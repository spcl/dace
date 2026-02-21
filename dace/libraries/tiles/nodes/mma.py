import copy

from dace.libraries.tiles.environments.native import NATIVE
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation

from dace.libraries.tiles.utils import _get_connector_info

# ---------------------------------------------------------------------------
# MMA  (Matrix Multiply-Accumulate:  C += A @ B)
# ---------------------------------------------------------------------------

@dace.library.expansion
class ExpandMMAPure(ExpandTransformation):
    """
    Pure (sequential C++) expansion of MMA.

    Implements::

        C[i, j] = Acc[i, j] + A[i, k] * B[k, j]   for all i, k, j

    The inner-product dimension ``k`` is inferred from A's column count
    (equivalently B's row count).
    """

    environments = [NATIVE]

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        a_edge, a_outer, a_shape, a_strides = _get_connector_info(
            node, parent_state, parent_sdfg, "_in_a", is_input=True
        )
        b_edge, b_outer, b_shape, b_strides = _get_connector_info(
            node, parent_state, parent_sdfg, "_in_b", is_input=True
        )
        acc_edge, acc_outer, acc_shape, acc_strides = _get_connector_info(
            node, parent_state, parent_sdfg, "_in_acc", is_input=True
        )
        c_edge, c_outer, c_shape, c_strides = _get_connector_info(
            node, parent_state, parent_sdfg, "_out_c", is_input=False
        )

        M, K_a = a_shape
        K_b, N = b_shape

        dtype = node.dtype
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        _, a = sdfg.add_array("_in_a", a_shape, dtype,
                       strides=a_strides, storage=a_outer.storage)
        _, b = sdfg.add_array("_in_b", b_shape, dtype,
                       strides=b_strides, storage=b_outer.storage)
        _, c = sdfg.add_array("_out_c", c_shape, dtype,
                       strides=c_strides, storage=c_outer.storage)
        _, acc = sdfg.add_array("_in_acc", c_shape, dtype,
                       strides=c_strides, storage=c_outer.storage)

        # C += A @ B  via a triply-nested map
        mma_tasklet = state.add_tasklet(
            name="mma_compute",
            inputs={"_mma_in_a", "_mma_in_b", "_mma_in_acc"},
            outputs={"_mma_out_c"},
            code=f"""dace_tile::mma_static<{M}, {K_a}, {N}>(
                _mma_in_a, _mma_in_b, _mma_in_acc, _mma_out_c,
                {K_a}, {N}, {N}, {N}, 1, 1, 1, 1
                );""",
            language=dace.dtypes.Language.CPP
        )

        for inp in {"_in_a", "_in_b", "_in_acc"}:
            state.add_edge(state.add_access(inp), None, mma_tasklet, f"_mma{inp}", dace.memlet.Memlet.from_array(inp, sdfg.arrays[inp]))
        state.add_edge(mma_tasklet, "_mma_out_c", state.add_access("_out_c"), None, dace.memlet.Memlet.from_array("_out_c", sdfg.arrays["_out_c"]))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        return ExpandMMAPure.make_sdfg(node, state, sdfg)


@dace.library.node
class MMA(dace.sdfg.nodes.LibraryNode):
    """
    Library node for Matrix Multiply-Accumulate: **C += A @ B**.

    Connectors
    ----------
    _a : in  – left-hand matrix tile  (M × K), must be exactly 2-D.
    _b : in  – right-hand matrix tile (K × N), must be exactly 2-D.
    _c : out – accumulator tile        (M × N), must be exactly 2-D.

    Additional inputs (e.g. alpha, beta scalars) are **not** supported and
    will cause ``validate`` to raise a ``ValueError``.  If you need a scaled
    GEMM, compose explicit Tasklets around this node.
    """

    implementations = {"pure": ExpandMMAPure}
    default_implementation = "pure"

    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name, location=location,
                         inputs={"_in_a", "_in_b", "_in_acc"}, outputs={"_out_c"})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)

        # Reject unexpected connectors (alpha, beta, etc.)
        in_connectors = {e.dst_conn for e in in_edges}
        out_connectors = {e.src_conn for e in out_edges}
        unexpected_in = in_connectors - {"_in_a", "_in_b", "_in_acc"}
        unexpected_out = out_connectors - {"_out_c"}
        if unexpected_in:
            raise ValueError(
                f"MMA does not support extra input connectors: {unexpected_in}. "
                "Scalar factors such as alpha/beta are not accepted; compose "
                "explicit Tasklets around the MMA node instead."
            )
        if unexpected_out:
            raise ValueError(
                f"MMA does not support extra output connectors: {unexpected_out}."
            )

        if in_connectors != {"_in_a", "_in_b", "_in_acc"}:
            raise ValueError(
                f"MMA requires exactly three inputs (_in_a, _in_b, _in_acc); got {in_connectors}."
            )
        if out_connectors != {"_out_c"}:
            raise ValueError(
                f"MMA requires exactly one output (_out_c); got {out_connectors}."
            )

        # Collect shapes
        shapes = {}
        for edge in in_edges:
            subset = copy.deepcopy(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            if len(size) != 2:
                raise ValueError(
                    f"MMA: connector '{edge.dst_conn}' must be 2-D "
                    f"(got {len(size)}-D after squeezing)."
                )
            shapes[edge.dst_conn] = size

        out_subset = copy.deepcopy(out_edges[0].data.subset)
        out_subset.squeeze()
        out_size = out_subset.size()
        if len(out_size) != 2:
            raise ValueError(
                f"MMA: output connector '_c' must be 2-D "
                f"(got {len(out_size)}-D after squeezing)."
            )
        shapes["_out_c"] = out_size

        M, K_a = shapes["_in_a"]
        K_b, N = shapes["_in_b"]
        M_c, N_c = shapes["_out_c"]

        if K_a != K_b:
            raise ValueError(
                f"MMA: inner dimensions of A and B must match "
                f"(A columns = {K_a}, B rows = {K_b})."
            )
        if M_c != M or N_c != N:
            raise ValueError(
                f"MMA: C shape ({M_c}x{N_c}) must equal MxN = {M}x{N}."
            )