from dace import library, nodes, properties
from dace.transformation.transformation import ExpandTransformation
from numbers import Number
import dace.subsets


@library.expansion
class ExpandPure(ExpandTransformation):
    """Implements pure expansion of the Fill library node."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        output = None
        for e in parent_state.out_edges(node):
            if e.src_conn == "_output":
                output = parent_sdfg.arrays[e.data.data]
        sdfg = dace.SDFG(f"{node.label}_sdfg")
        _, out_arr = sdfg.add_array(
            "_output",
            output.shape,
            output.dtype,
            output.storage,
            strides=output.strides,
        )

        state = sdfg.add_state(f"{node.label}_state")
        map_params = [f"__i{i}" for i in range(len(out_arr.shape))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, out_arr.shape)}
        out_mem = dace.Memlet(expr=f"_output[{','.join(map_params)}]")
        inputs = {}
        outputs = {"_out": out_mem}
        code = f"_out = {node.value}"
        state.add_mapped_tasklet(
            f"{node.label}_tasklet", map_rng, inputs, code, outputs, external_edges=True
        )

        return sdfg


@library.node
class Fill(nodes.LibraryNode):
    """Implements filling data containers with a single value"""

    implementations = {"pure": ExpandPure}
    default_implementation = "pure"
    value = properties.SymbolicProperty(
        dtype=Number, default=0, desc="value to fill data container"
    )

    def __init__(self, name, value=0):
        super().__init__(name, outputs={"_output"})
        self.value = value
        self.name = name
