import copy

from dace.libraries.tiles.environments.native import NATIVE
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation

from dace.libraries.tiles.utils import _get_connector_info

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dace.library.expansion
class ExpandStorePure(ExpandTransformation):
    """Pure (sequential C++) expansion of the Store node."""

    environments = [NATIVE]

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        in_edge, in_outer_array, in_shape, in_strides = _get_connector_info(
            node, parent_state, parent_sdfg, "_inp", is_input=True
        )
        out_edge, out_outer_array, out_shape, out_strides = _get_connector_info(
            node, parent_state, parent_sdfg, "_out", is_input=False
        )

        dtype = node.dtype
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        sdfg.add_array("_inp", in_shape, dtype,
                       strides=in_strides, storage=in_outer_array.storage)
        sdfg.add_array("_out", out_shape, dtype,
                       strides=out_strides, storage=out_outer_array.storage)

        state.add_mapped_tasklet(
            name="store",
            map_ranges={
                "__i0": f"0:{in_shape[0]}",
                "__i1": f"0:{in_shape[1]}",
            },
            inputs={
                "__inp": dace.memlet.Memlet.simple("_inp", "__i0, __i1"),
            },
            code="__out = __inp",
            outputs={
                "__out": dace.memlet.Memlet.simple("_out", "__i0, __i1"),
            },
            external_edges=True,
        )
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandStorePure.make_sdfg(node, state, sdfg)


@dace.library.node
class Store(dace.sdfg.nodes.LibraryNode):
    """
    Library node that stores a 2-D tile into a destination array.

    Connectors
    ----------
    _inp : in  – the 2-D source tile.
    _out : out – the 2-D destination region (must be exactly 2-D after squeezing).
    """

    implementations = {"pure": ExpandStorePure}
    default_implementation = "pure"

    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name, location=location,
                         inputs={"_inp"}, outputs={"_out"})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Store node expects exactly one input edge (_inp).")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Store node expects exactly one output edge (_out).")

        # Validate _inp dimensionality
        in_subset = copy.deepcopy(in_edges[0].data.subset)
        in_subset.squeeze()
        if len(in_subset.size()) != 2:
            raise ValueError(
                "Store: input memlet must describe a 2-D region "
                f"(got {len(in_subset.size())}-D after squeezing)."
            )

        # Validate _out dimensionality
        out_subset = copy.deepcopy(out_edges[0].data.subset)
        out_subset.squeeze()
        if len(out_subset.size()) != 2:
            raise ValueError(
                "Store: output memlet must describe a 2-D region "
                f"(got {len(out_subset.size())}-D after squeezing)."
            )

        # Shape consistency
        if list(in_subset.size()) != list(out_subset.size()):
            raise ValueError(
                f"Store: input shape {in_subset.size()} must match "
                f"output shape {out_subset.size()}."
            )
