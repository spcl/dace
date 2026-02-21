import copy

from dace.libraries.tiles.environments.native import NATIVE
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation

from dace.libraries.tiles.utils import _get_connector_info
# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

@dace.library.expansion
class ExpandLoadPure(ExpandTransformation):
    """Pure (sequential C++) expansion of the Load node."""
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
            name="load",
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
        return ExpandLoadPure.make_sdfg(node, state, sdfg)


@dace.library.node
class Load(dace.sdfg.nodes.LibraryNode):
    """
    Library node that loads a 2-D tile from a source array.

    Connectors
    ----------
    _inp : in  – the 2-D source region (must be exactly 2-D after squeezing).
    _out : out – the 2-D destination tile.
    """

    implementations = {"pure": ExpandLoadPure}
    default_implementation = "pure"

    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name, location=location,
                         inputs={"_inp"}, outputs={"_out"})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Load node expects exactly one input edge (_inp).")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Load node expects exactly one output edge (_out).")

        # Validate _inp dimensionality
        for _, _, _, dst_conn, memlet in in_edges:
            if dst_conn == "_inp":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                if len(subset.size()) != 2:
                    raise ValueError(
                        "Load: input memlet must describe a 2-D region "
                        f"(got {len(subset.size())}-D after squeezing)."
                    )

        # Validate _out dimensionality
        for _, _, _, dst_conn, memlet in [(e.src, e.src_conn, e.dst, e.dst_conn, e.data)
                                          for e in out_edges]:
            out_subset = copy.deepcopy(out_edges[0].data.subset)
            out_subset.squeeze()
            if len(out_subset.size()) != 2:
                raise ValueError(
                    "Load: output memlet must describe a 2-D region "
                    f"(got {len(out_subset.size())}-D after squeezing)."
                )
            break

        # Shape consistency
        in_subset = copy.deepcopy(in_edges[0].data.subset)
        in_subset.squeeze()
        out_subset = copy.deepcopy(out_edges[0].data.subset)
        out_subset.squeeze()
        if list(in_subset.size()) != list(out_subset.size()):
            raise ValueError(
                f"Load: input shape {in_subset.size()} must match "
                f"output shape {out_subset.size()}."
            )



