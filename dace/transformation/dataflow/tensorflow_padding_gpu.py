import copy
from dace import data as dt, types, subsets, symbolic
from dace.memlet import Memlet
from dace.graph import nodes, nxutil
from dace.sdfg import SDFGState
from dace.transformation import pattern_matching as pm
from dace.properties import ShapeProperty
from dace.config import Config


class TensorflowPaddingGPU(pm.Transformation):
    """ Implements the redundant array removal transformation. Removes array B
        in pattern A -> B -> A.
    """

    _gpu_array_source = nodes.AccessNode("_")
    _cpu_array_source = nodes.AccessNode("_")
    _cpu_array_dest = nodes.AccessNode("_")
    _gpu_array_dest = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(
                TensorflowPaddingGPU._gpu_array_source,
                TensorflowPaddingGPU._cpu_array_source,
                TensorflowPaddingGPU._cpu_array_dest,
                TensorflowPaddingGPU._gpu_array_dest,
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        gpu_source = graph.nodes()[candidate[TensorflowPaddingGPU._gpu_array_source]]
        cpu_source = graph.nodes()[candidate[TensorflowPaddingGPU._cpu_array_source]]
        cpu_dest = graph.nodes()[candidate[TensorflowPaddingGPU._cpu_array_dest]]
        gpu_dest = graph.nodes()[candidate[TensorflowPaddingGPU._gpu_array_dest]]

        # Ensure out degree is one (only one target, which is host array)
        if graph.out_degree(gpu_source) != 1:
            return False
        if graph.out_degree(cpu_source) != 1:
            return False
        if graph.out_degree(cpu_dest) != 1:
            return False

        cpu_types = [
            types.StorageType.CPU_Heap,
            types.StorageType.CPU_Pinned,
            types.StorageType.CPU_Stack,
            types.StorageType.Default,
        ]
        # gpu_types = [
        #    types.StorageType.GPU_Global,
        #    types.StorageType.GPU_Shared,
        #    types.StorageType.GPU_Stack,
        # ]
        print(gpu_source.desc(sdfg).storage)
        print(cpu_source.desc(sdfg).storage)
        print(cpu_dest.desc(sdfg).storage)
        print(gpu_dest.desc(sdfg).storage)
        # Make sure the pattern is GPU-CPU-CPU-GPU, for GPU, kernel local variables should be
        # skipped I guess.
        if gpu_source.desc(sdfg).storage != types.StorageType.GPU_Global:
            return False
        if cpu_source.desc(sdfg).storage not in cpu_types:
            return False
        if cpu_dest.desc(sdfg).storage not in cpu_types:
            return False
        if gpu_dest.desc(sdfg).storage != types.StorageType.GPU_Global:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        cpu_source = graph.nodes()[candidate[TensorflowPaddingGPU._cpu_array_source]]
        cpu_dest = graph.nodes()[candidate[TensorflowPaddingGPU._cpu_array_dest]]

        return "Remove " + str(cpu_source)
        return "Remove " + str(cpu_dest)

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        graph = sdfg.nodes()[self.state_id]

        gpu_source = gnode(TensorflowPaddingGPU._gpu_array_source)
        cpu_source = gnode(TensorflowPaddingGPU._cpu_array_source)
        cpu_dest = gnode(TensorflowPaddingGPU._cpu_array_dest)
        gpu_dest = gnode(TensorflowPaddingGPU._gpu_array_dest)

        cpu_edge = graph.out_edges(cpu_source)[0]

        new_memlet = copy.deepcopy(cpu_edge.data)
        new_memlet.data = gpu_source.data

        graph.add_edge(gpu_source, None, gpu_dest, None, new_memlet)

        graph.remove_edge(cpu_edge)
        graph.remove_node(cpu_source)
        graph.remove_node(cpu_dest)

        #for med_e in graph.out_edges(med_array):
        #    if (
        #        isinstance(med_e.dst, nodes.AccessNode)
        #        and med_e.dst.data == out_array.data
        #    ):
        #        # Modify all outcoming edges to point to in_array
        #        for out_e in graph.out_edges(med_e.dst):
        #            path = graph.memlet_path(out_e)
        #            for pe in path:
        #                if pe.data.data == out_array.data:
        #                    pe.data.data = in_array.data
        #            # Redirect edge to in_array
        #            graph.remove_edge(out_e)
        #            graph.add_edge(
        #                in_array, out_e.src_conn, out_e.dst, out_e.dst_conn, out_e.data
        #            )
        #        # Remove out_array
        #        for e in graph.edges_between(med_e, med_e.dst):
        #            graph.remove_edge(e)
        #        graph.remove_node(med_e.dst)
        #        med_out_edges += 1

        # Finally, med_array node
        #if med_array.desc(sdfg).transient and med_edges == med_out_edges:
        #    for e in graph.edges_between(in_array, med_array):
        #        graph.remove_edge(e)
        #    graph.remove_node(med_array)
        if Config.get_bool("debugprint"):
            TensorflowPaddingGPU._arrays_removed += 1

    def modifies_graph(self):
        return True

    @staticmethod
    def print_debuginfo():
        print(
            "Automatically transformed {} paddings using TensorflowPaddingGPU transform.".format(
                TensorflowPaddingGPU._arrays_removed
            )
        )


pm.Transformation.register_pattern(TensorflowPaddingGPU)
