from typing import Any, List, Union
from dace import float32, Memlet
from dace.data import Tensor, Array, TensorIndex
from dace.libraries.sparse.nodes.tensor_index_notation import TensorIndexNotation, TensorIndexCompressed
from dace.properties import ShapeProperty, make_properties
from dace.transformation.transformation import SingleStateTransformation
from dace.sdfg import SDFG, SDFGState, nodes, state as st
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@make_properties
class SwitchTensorFormat(SingleStateTransformation):
    """
    Allows switching of tensor storage format when working with Tensor Index
    Notation (TIN) library nodes.
    """

    lib_node_a = transformation.PatternNode(nodes.LibraryNode)
    data_node = transformation.PatternNode(nodes.AccessNode)
    lib_node_b = transformation.PatternNode(nodes.LibraryNode)

    indices = ShapeProperty(dtype=tuple, default=None, desc="Indices in new storage format")

    @classmethod
    def expressions(cls) -> List[st.StateSubgraphView]:
        return [sdutil.node_path_graph(cls.lib_node_a, cls.data_node, cls.lib_node_b)]
    

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        if not isinstance(self.lib_node_a, TensorIndexNotation) or not isinstance(self.lib_node_b, TensorIndexNotation): 
            return False
        
        desc = self.data_node.desc(sdfg)
        if not isinstance(desc, Tensor) and not isinstance(desc, Array):
            return False
        
        return True
    

    def apply(self, graph: SDFG | SDFGState, sdfg: SDFG) -> Any | None:

        print(f"DEBUG {self.indices=}")
        print(f"DEBUG {self.data_node.data=}")
        print(f"DEBUG {sdfg.data(self.data_node.data)=}")

        csf_obj = Tensor(float32,
                    sdfg.data(self.data_node.data).shape,
                    [(TensorIndexCompressed(), 0), (TensorIndexCompressed(), 1)],
                    sdfg.data(self.data_node.data).shape[1],
                    "CSF_Tensor",
                    transient=True)
        
        del sdfg.arrays[self.data_node.data]
        sdfg.add_datadesc(self.data_node.data, csf_obj)

        # graph.add_edge(C, None, tin_node, 'tin_C', memlet=dace.Memlet(data='C'))
        # state.add_edge(tin_node, "tin_A", A, None, memlet=dace.Memlet(data='A'))

        print(f"DEBUG {sdfg.data(self.data_node.data)=}")

        print(f"DEBUG {graph.in_edges(self.data_node)=}")

        for e in graph.in_edges(self.data_node):
            # e.data = Memlet(data=self.data_node.data)
            graph.remove_edge(e)
            graph.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, memlet=Memlet(data=self.data_node.data))
        
        print(f"DEBUG {graph.in_edges(self.data_node)=}")

        print(f"DEBUG {graph.out_edges(self.data_node)=}")

        for e in graph.out_edges(self.data_node):
            # e.data = Memlet(data=self.data_node.data)
            graph.remove_edge(e)
            graph.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, memlet=Memlet(data=self.data_node.data))

        print(f"DEBUG {graph.out_edges(self.data_node)=}")

