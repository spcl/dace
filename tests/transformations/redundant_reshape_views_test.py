import dace
import numpy as np


@dace.program
def nested_add1(A: dace.float64[3, 3], B: dace.float64[3, 3]):
    return A + B


@dace.program
def nested_add2(A: dace.float64[9], B: dace.float64[9]):
    return A + B


@dace.program
def reshape_node(A: dace.float64[9]):
    return A.reshape([3, 3])


def test_inline_reshape_views_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[9], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = np.reshape(nested_add1(A, B), [9])
        return nested_add1(result, B)

    sdfg = test_inline_reshape_views_work.to_sdfg(coarsen=True)

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


def test_views_between_maps_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[3, 3], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = nested_add2(A, B)
        result_reshaped = reshape_node(result)

        return np.transpose(result_reshaped)

    sdfg = test_inline_reshape_views_work.to_sdfg(coarsen=False)
    sdfg.expand_library_nodes()
    sdfg.coarsen_dataflow()


if __name__ == "__main__":
    test_inline_reshape_views_work()
    test_views_between_maps_work()
