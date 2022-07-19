# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
        map_ranges={"__i" + str(i): "0:" + str(shape)
                    for i, shape in enumerate(inparr.shape)},
        inputs={
            '__max': dace.Memlet.simple("tmp_max",
                                        ','.join("__i" + str(i) for i in range(len(inparr.shape)) if i != axis)),
            '__x': dace.Memlet.simple("original_input", ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        code='__out = exp(__x - __max)',
        outputs={'__out': dace.Memlet.simple("output", ','.join("__i" + str(i) for i in range(len(inparr.shape))))},
        external_edges=True)

    ##################
    # out_tmp / sum
    out_tmp_div_sum = dace.SDFG("out_tmp_div_sum")
    out_tmp_div_sum.add_array("out_tmp", inparr.shape, inparr.dtype)
    out_tmp_div_sum.add_array("tmp_sum", tmp_max_shape, inparr.dtype)
    out_tmp_div_sum.add_array("output", out_tmp_shape, out_tmp_dtype)

    out_tmp_div_sum.add_state().add_mapped_tasklet(
        "_softmax_div_",
        map_ranges={"__i" + str(i): "0:" + str(shape)
                    for i, shape in enumerate(inparr.shape)},
        inputs={
            '__sum': dace.Memlet.simple("tmp_sum",
                                        ','.join("__i" + str(i) for i in range(len(inparr.shape)) if i != axis)),
            '__exp': dace.Memlet.simple("out_tmp", ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        code='__out = __exp / __sum',
        outputs={'__out': dace.Memlet.simple("output", ','.join("__i" + str(i) for i in range(len(inparr.shape))))},
        external_edges=True)

    ##################
    # put everything together as a program
    @dace.program
    def multiple_nested_sdfgs(input: dace.float32[2, 2], output: dace.float32[2, 2]):
        tmp_max = np.max(input, axis=axis)

        out_tmp = dace.define_local(out_tmp_shape, out_tmp_dtype)
        exp_minus_max(tmp_max=tmp_max, original_input=input, output=out_tmp)

        tmp_sum = np.sum(out_tmp, axis=axis)

        out_tmp_div_sum(out_tmp=out_tmp, tmp_sum=tmp_sum, output=output)

    sdfg = multiple_nested_sdfgs.to_sdfg(simplify=False)
    state = sdfg.nodes()[-1]
    for n in state.nodes():
        if isinstance(n, dace.sdfg.nodes.AccessNode):
            assert (n.data in {'out_tmp', 'tmp_sum', 'output'})
        elif isinstance(n, dace.sdfg.nodes.CodeNode):
            for src, _, _, _, _ in state.in_edges(n):
                assert (src.data in {'out_tmp', 'tmp_sum'})
            for _, _, dst, _, _ in state.out_edges(n):
                assert (dst.data in {'output'})


def test_nested_sdfg_with_return_value():

    @dace.program
    def nested(A: dace.float64[20]):
        return A + 20

    sdfg = nested.to_sdfg()

    @dace.program
    def mainprog(A: dace.float64[30], B: dace.float64[20]):
        return sdfg(A[10:]) + B

    A = np.random.rand(30)
    B = np.random.rand(20)
    expected = A[10:] + 20 + B
    assert np.allclose(mainprog(A, B), expected)


def test_nested_sdfg_with_return_value_assignment():

    @dace.program
    def nested(A: dace.float64[20]):
        return A + 20

    sdfg = nested.to_sdfg()

    @dace.program
    def mainprog(A: dace.float64[30], B: dace.float64[20]):
        B[:] = sdfg(A[10:])

    A = np.random.rand(30)
    B = np.random.rand(20)
    expected = A[10:] + 20
    mainprog(A, B)
    assert np.allclose(B, expected)


def test_multiple_calls():

    @dace.program
    def nested(a: dace.float64[20], b: dace.float64[20]):
        return a + b + b

    @dace.program
    def tester(a: dace.float64[20], b: dace.float64[20]):
        if a[0] < 0.5:
            a += 0.5
            c = nested(a, b)
        else:
            c = nested(a, b)

        return c

    a = np.random.rand(20)
    b = np.random.rand(20)
    a[0] = 1.0

    # Regression: calling ``nested`` before ``tester`` affects validation
    nested(a, b)

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.validate()

    c = tester(a, b)
    assert np.allclose(c, a + b + b)


if __name__ == "__main__":
    test_call_multiple_sdfgs()
    test_nested_sdfg_with_return_value()
    test_nested_sdfg_with_return_value_assignment()
    test_multiple_calls()
