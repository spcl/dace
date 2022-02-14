import dace
import numpy as np
from dace import nodes as nd, data as dt
from dace.transformation.interstate import InlineSDFG


@dace.program
def nested_add1(A: dace.float64[3, 3], B: dace.float64[3, 3]):
    return A + B


@dace.program
def nested_add2(A: dace.float64[9], B: dace.float64[9]):
    return A + B


@dace.program
def reshape_node(A: dace.float64[9]):
    return A.reshape([3, 3])


@dace.program
def reshape_node_both_args(A: dace.float64[9], B: dace.float64[3, 3]):
    B[:] = np.reshape(A, [3, 3])


def test_inline_reshape_views_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[9], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = np.reshape(nested_add1(A, B), [9])
        return nested_add1(result, B)

    sdfg = test_inline_reshape_views_work.to_sdfg(simplify=True)

    arrays = 0
    views = 0
    sdfg_used_desc = set([n.desc(sdfg) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.AccessNode)])
    for desc in sdfg_used_desc:
        # View is subclass of Array, so we must do this check first
        if isinstance(desc, dace.data.View):
            views += 1
        elif isinstance(desc, dace.data.Array):
            arrays += 1

    assert arrays == 4
    assert views == 3


def _dml_disambiguate_direction_dependent_views(sdfg: dace.SDFG):
    """ Consider the following subgraph:
            (A) -- y --> (n) -- x --> (C)
            In dace, if B is a View node and A and C are access nodes, and y and x both have data set to A.data and
            B.data respectively, the semantics of the graph depend on the order in which it is executed, i.e. reversing
            the subgraph doesn't perform as expected anymore. To disambiguate this case, we set y.data to the View's
            data.
        """

    for n, state in sdfg.all_nodes_recursive():
        if isinstance(n, nd.AccessNode) and type(n.desc(sdfg)) is dt.View:
            in_edges = state.in_edges(n)
            out_edges = state.out_edges(n)

            if len(in_edges) == 1 and len(out_edges) == 1:
                A = in_edges[0].src
                y = in_edges[0].data
                C = out_edges[0].dst
                x = out_edges[0].data
                if (isinstance(A, nd.AccessNode) and isinstance(C, nd.AccessNode) and y.data == A.data
                        and x.data == C.data):

                    # flip the memlet
                    y.subset, y.other_subset = y.other_subset, y.subset
                    y.data = n.data
                    y.try_initialize(sdfg, state, in_edges[0])


def test_inline_flipped_reshape_works():
    # related to autodiff in daceml: to disambiguate views, daceml will
    # flip the memlet on reshape nodes

    reshape_sdfg = reshape_node_both_args.to_sdfg()
    _dml_disambiguate_direction_dependent_views(reshape_sdfg)

    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[9]):
        result = dace.define_local([3, 3], dace.float64)
        reshape_sdfg(A, result)
        return result

    sdfg = test_inline_reshape_views_work.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(InlineSDFG)


def test_views_between_maps_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[3, 3], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = nested_add2(A, B)
        result_reshaped = reshape_node(result)

        return np.transpose(result_reshaped)

    sdfg = test_inline_reshape_views_work.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()


if __name__ == "__main__":
    test_inline_reshape_views_work()
    test_views_between_maps_work()
    test_inline_flipped_reshape_works()
