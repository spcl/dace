import dace


@dace.program
def nested_add1(A: dace.float64[3, 3], B: dace.float64[3, 3]):
    return A + B


def test_inline_reshape_views_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[9], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = nested_add1(A, B)
        return nested_add1(result, B)


    sdfg = test_inline_reshape_views_work.to_sdfg(strict=True)

    arrays = 0
    views = 0
    sdfg_used_desc = set([n.desc(sdfg) for n, _ in sdfg.all_nodes_recursive()
                          if isinstance(n, dace.nodes.AccessNode)])
    for desc in sdfg_used_desc:
        # View is subclas of Array, so we must do this check first
        if isinstance(desc, dace.data.View):
            views += 1
        elif isinstance(desc, dace.data.Array):
            arrays += 1
    
    assert(arrays == 4)
    assert(views == 3)


if __name__ == "__main__":
    test_inline_reshape_views_work()
