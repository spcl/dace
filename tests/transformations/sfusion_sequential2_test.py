import dace
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np

N = dace.symbol('N')


def fusion(sdfg: dace.SDFG,
           graph: dace.SDFGState,
           subgraph: Union[SubgraphView, List[SubgraphView]] = None,
           **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    map_fusion = SubgraphFusion()
    for (property, val) in kwargs.items():
        setattr(map_fusion, property, val)

    for sg in subgraph:
        map_entries = get_lowest_scope_maps(sdfg, graph, sg)
        # remove map_entries and their corresponding exits from the subgraph
        # already before applying transformation
        if isinstance(sg, SubgraphView):
            for map_entry in map_entries:
                sg.nodes().remove(map_entry)
                if graph.exit_node(map_entry) in sg.nodes():
                    sg.nodes().remove(graph.exit_node(map_entry))
        print(f"Subgraph Fusion on map entries {map_entries}")
        map_fusion.fuse(sdfg, graph, map_entries)
        if isinstance(sg, SubgraphView):
            sg.nodes().append(map_fusion._global_map_entry)


@dace.program
def TEST(A: dace.float64[N], C: dace.float64[N]):
    B = np.ndarray(shape=[N], dtype=np.float64)
    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << A[i]
            out1 >> B[i]
            out1 = in1 + 1

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << B[i]
            out1 >> C[i]
            out1 = in1 + 1


if __name__ == "__main__":
    N.set(1000)

    sdfg = TEST.to_sdfg()
    state = sdfg.nodes()[0]

    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(N.get()).astype(np.float64)
    C1 = np.random.rand(N.get()).astype(np.float64)
    C2 = np.random.rand(N.get()).astype(np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N)

    fusion(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N)

    assert np.allclose(C1, C2)
