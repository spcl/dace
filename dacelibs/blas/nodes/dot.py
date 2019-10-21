import dace.library
import dace.properties as properties
import dace.graph.nodes as nodes
from dace.memlet import Memlet
from dace.transformation.pattern_matching import ExpandTransformation
from dace.graph import nxutil


@dace.library.node
class Dot(nodes.LibraryNode):

    # Global properties
    implementations = {}  # Entries defined below to avoid cyclic dependency
    default_implementation = "pure"

    # Object fields
    dtype = properties.TypeClassProperty(allow_none=True)

    def __init__(self, dtype=None, *args, **kwargs):
        self.dtype = dtype
        super().__init__(*args, **kwargs)


@dace.library.transformation
class ExpandDotPure(ExpandTransformation):

    _node = Dot()

    @staticmethod
    def expand(node, state, sdfg):
        in_edges = state.in_edges(node)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to dot product")
        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(node)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from dot product")
        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()
        veclen = in_memlets[0].veclen
        if len(size) != 1:
            raise ValueError(
                "dot product only supported on 1-dimensional arrays")
        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to dot product must have equal size")
        if out_memlet.subset.num_elements() != 1 or out_memlet.veclen != 1:
            raise ValueError("Output of dot product must be a single element")
        if veclen != in_memlets[1].veclen:
            raise ValueError(
                "Vector lengths of inputs to dot product must be identical")
        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None
                or out_memlet.wcr is not None):
            raise ValueError("WCR on dot product memlets not supported")
        prefix = "_" + str(node) + "_"
        iterator = prefix + "i"
        in_memlets = [
            Memlet.simple(m.data, iterator, veclen=veclen, num_accesses=1)
            for m in in_memlets
        ]
        out_memlet = Memlet.simple(
            out_memlet.data, "0", wcr_str="lambda a, b: a + b")
        tasklet = nodes.Tasklet(
            str(node), {prefix + "x", prefix + "y"}, [prefix + "result"],
            "{prefix}result = {prefix}x + {prefix}y".format(prefix=prefix))
        entry, exit = state.add_map(prefix + "map",
                                    {iterator: "0:" + str(size[0])})
        state.add_memlet_path(
            in_edges[0].src,
            entry,
            tasklet,
            memlet=in_memlets[0],
            dst_conn=prefix + "x")
        state.add_memlet_path(
            in_edges[1].src,
            entry,
            tasklet,
            memlet=in_memlets[1],
            dst_conn=prefix + "y")
        state.add_memlet_path(
            tasklet,
            exit,
            out_edges[0].dst,
            memlet=out_memlet,
            src_conn=prefix + "result")
        state.remove_node(node)


# Register implementation
Dot.implementations["pure"] = (ExpandDotPure, [])


@dace.library.transformation
class ExpandDotOpenBLAS(ExpandTransformation):

    _node = Dot()

    @staticmethod
    def expand(node, state, sdfg):
        dtype = node.dtype
        if dtype == dace.float32:
            func = "sdot"
        elif dtype == dace.float64:
            func = "ddot"
        else:
            raise ValueError("Unsupported type for OpenBLAS dot product: " +
                             str(dtype))
        raise NotImplementedError("NYI")


# Register implementation
Dot.implementations["OpenBLAS"] = (ExpandDotOpenBLAS, ["OpenBLAS"])
