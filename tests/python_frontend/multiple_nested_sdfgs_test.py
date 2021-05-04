import copy

import numpy as np

import dace


def test_call_multiple_sdfgs():
    inparr = dace.float32[2, 2]
    axis = 1
    out_tmp_shape = inparr.shape
    out_tmp_dtype = inparr.dtype

    tmp_max_shape = list(copy.deepcopy(inparr.shape))
    tmp_max_shape.pop(axis)

    ##################
    # exp - max
    exp_minus_max = dace.SDFG("exp_minus_max")
    exp_minus_max.add_array("tmp_max", tmp_max_shape, inparr.dtype)
    exp_minus_max.add_array("original_input", inparr.shape, inparr.dtype)
    exp_minus_max.add_array("output", out_tmp_shape, out_tmp_dtype)
    exp_minus_max.add_state().add_mapped_tasklet(
        "_softmax_exp_",
        map_ranges={
            "__i" + str(i): "0:" + str(shape)
            for i, shape in enumerate(inparr.shape)
        },
        inputs={
            '__max':
            dace.Memlet.simple(
                "tmp_max", ','.join("__i" + str(i)
                                    for i in range(len(inparr.shape))
                                    if i != axis)),
            '__x':
            dace.Memlet.simple(
                "original_input",
                ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        code='__out = exp(__x - __max)',
        outputs={
            '__out':
            dace.Memlet.simple(
                "output",
                ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        external_edges=True)

    ##################
    # out_tmp / sum
    out_tmp_div_sum = dace.SDFG("out_tmp_div_sum")
    out_tmp_div_sum.add_array("out_tmp", inparr.shape, inparr.dtype)
    out_tmp_div_sum.add_array("tmp_sum", tmp_max_shape, inparr.dtype)
    out_tmp_div_sum.add_array("output", out_tmp_shape, out_tmp_dtype)

    out_tmp_div_sum.add_state().add_mapped_tasklet(
        "_softmax_div_",
        map_ranges={
            "__i" + str(i): "0:" + str(shape)
            for i, shape in enumerate(inparr.shape)
        },
        inputs={
            '__sum':
            dace.Memlet.simple(
                "tmp_sum", ','.join("__i" + str(i)
                                    for i in range(len(inparr.shape))
                                    if i != axis)),
            '__exp':
            dace.Memlet.simple(
                "out_tmp",
                ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        code='__out = __exp / __sum',
        outputs={
            '__out':
            dace.Memlet.simple(
                "output",
                ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        external_edges=True)

    ##################
    # put everything together as a program
    @dace.program
    def multiple_nested_sdfgs(input: dace.float32[2, 2],
                              output: dace.float32[2, 2]):
        tmp_max = np.max(input, axis=axis)

        out_tmp = dace.define_local(out_tmp_shape, out_tmp_dtype)
        exp_minus_max(tmp_max=tmp_max, original_input=input, output=out_tmp)

        tmp_sum = np.sum(out_tmp, axis=axis)

        out_tmp_div_sum(out_tmp=out_tmp, tmp_sum=tmp_sum, output=output)

    sdfg = multiple_nested_sdfgs.to_sdfg(strict=False)
    state = sdfg.nodes()[-1]
    for n in state.nodes():
        if isinstance(n, dace.sdfg.nodes.AccessNode):
            assert (n.data in {'out_tmp', 'tmp_sum', 'output'})
        elif isinstance(n, dace.sdfg.nodes.CodeNode):
            for src, _, _, _, _ in state.in_edges(n):
                assert (src.data in {'out_tmp', 'tmp_sum'})
            for _, _, dst, _, _ in state.out_edges(n):
                assert (dst.data in {'output'})


if __name__ == "__main__":
    test_call_multiple_sdfgs()
