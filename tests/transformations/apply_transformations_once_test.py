# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace import transformation as dace_transformation, properties as dace_properties
from dace.sdfg import nodes as dace_nodes
from typing import Any, Union

N = dace.symbol('N')


@dace_properties.make_properties
class DummyTransformation(dace_transformation.SingleStateTransformation):
    tasklet = dace_transformation.PatternNode(dace_nodes.Tasklet)

    def __init__(self):
        super().__init__()

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.tasklet)]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        my_tasklet: dace_nodes.Tasklet = self.tasklet
        return "1.0" in my_tasklet.code.as_string

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        my_tasklet: dace_nodes.Tasklet = self.tasklet
        my_tasklet.code = dace_properties.CodeBlock(my_tasklet.code.as_string.replace('1.0', '10.0'),
                                                    language=my_tasklet.code.language)


def sdfg_with_two_simple_maps():
    sdfg = dace.SDFG("two_simple_maps")
    state = sdfg.add_state(is_start_block=True)
    A, _ = sdfg.add_array("A", [N], dace.float32)
    B, _ = sdfg.add_array("B", [N], dace.float32)
    C, _ = sdfg.add_array("C", [N], dace.float32)
    D, _ = sdfg.add_array("D", [N], dace.float32)
    E, _ = sdfg.add_array("E", [N], dace.float32)
    F, _ = sdfg.add_array("F", [N], dace.float32)
    A_node = state.add_access("A")
    B_node = state.add_access("B")
    C_node = state.add_access("C")
    D_node = state.add_access("D")
    E_node = state.add_access("E")
    F_node = state.add_access("F")
    state.add_mapped_tasklet(
        "plus1_A",
        {"i": "0:N"},
        code="_out = _inp + 1.0",
        inputs={"_inp": dace.Memlet(data=A, subset="i")},
        outputs={"_out": dace.Memlet(data=B, subset="i")},
        input_nodes={A_node},
        output_nodes={B_node},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "plus2_C",
        {"i": "0:N"},
        code="_out = _inp + 2.0",
        inputs={"_inp": dace.Memlet(data=C, subset="i")},
        outputs={"_out": dace.Memlet(data=D, subset="i")},
        input_nodes={C_node},
        output_nodes={D_node},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "plus1_E",
        {"i": "0:N"},
        code="_out = _inp + 1.0",
        inputs={"_inp": dace.Memlet(data=E, subset="i")},
        outputs={"_out": dace.Memlet(data=F, subset="i")},
        input_nodes={E_node},
        output_nodes={F_node},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_apply_transformations_once():
    sdfg = sdfg_with_two_simple_maps()
    count = sdfg.apply_transformations_once_everywhere(DummyTransformation, validate=True)
    assert count == 2
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace_nodes.Tasklet):
            assert "1.0" not in node.code.as_string


def test_apply_transformations_once_no_order_by_transformation():
    sdfg = sdfg_with_two_simple_maps()
    count = sdfg.apply_transformations_once_everywhere(DummyTransformation,
                                                       validate=True,
                                                       order_by_transformation=False)
    assert count == 2
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace_nodes.Tasklet):
            assert "1.0" not in node.code.as_string


if __name__ == "__main__":
    test_apply_transformations_once()
    test_apply_transformations_once_no_order_by_transformation()
