import re

import numpy as np
import pytest

import dace


def make_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("Repr")
    state = sdfg.add_state(is_start_block=True)
    N = dace.symbol(sdfg.add_symbol("N", dace.int32))
    for name in "BC":
        sdfg.add_array(
            name=name,
            dtype=dace.float64,
            shape=(N, N),
            strides=(N, 1),
            transient=False,
        )

    # `A` uses a stride that is not used by any of the other arrays.
    #  However, the stride is used if we want to index array `A`.
    second_stride_A = dace.symbol(sdfg.add_symbol("second_stride_A", dace.int32))
    sdfg.add_array(
        name="A",
        dtype=dace.float64,
        shape=(N, ),
        strides=(second_stride_A, ),
        transient=False,
    )

    # Also array `D` uses a stride that is not used by any other array.
    second_stride_D = dace.symbol(sdfg.add_symbol("second_stride_D", dace.int32))
    sdfg.add_array(
        name="D",
        dtype=dace.float64,
        shape=(N, N),
        strides=(second_stride_D, 1),
        transient=False,
    )

    # Simplest way to generate a mapped Tasklet, we will later modify it.
    state.add_mapped_tasklet(
        "computation",
        map_ranges={
            "__i0": "0:N",
            "__i1": "0:N"
        },
        inputs={
            "__in0": dace.Memlet("A[__i1]"),
            "__in1": dace.Memlet("B[__i0, __i1]"),
        },
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("C[__i0, __i1]")},
        external_edges=True,
    )

    # Instead of going from the MapEntry to the Tasklet we will go through
    #  an temporary AccessNode that is only used inside the map scope.
    #  Thus there is no direct reference to `A` inside the map scope, that would
    #  need `second_stride_A`.
    sdfg.add_scalar("tmp_in", transient=True, dtype=dace.float64)
    tmp_in = state.add_access("tmp_in")
    for e in state.edges():
        if e.dst_conn == "__in0":
            iedge = e
            break
    state.add_edge(
        iedge.src,
        iedge.src_conn,
        tmp_in,
        None,
        # The important thing is that the Memlet, that connects the MapEntry with the
        #  AccessNode, does not refers to the memory outside (its source) but to the transient
        #  inside (its destination)
        dace.Memlet(data="tmp_in", subset="0", other_subset="__i1"),  # This does not work!
        #dace.Memlet(data="A", subset="__i1", other_subset="0"),   # This would work!
    )
    state.add_edge(
        tmp_in,
        None,
        iedge.dst,
        iedge.dst_conn,
        dace.Memlet(f"{tmp_in.data}[0]"),
    )
    state.remove_edge(iedge)

    # Here we are doing something similar as for `A`, but this time for the output.
    #  The output of the Tasklet is stored inside a temporary scalar.
    #  From that scalar we then go to `C`, here the Memlet on the inside is still
    #  referring to `C`, thus it is referenced directly.
    #  We also add a second output that goes to `D` , but the inner Memlet does
    #  not refer to `D` but to the temporary. Thus there is no direct mention of
    #  `D` inside the map scope.
    sdfg.add_scalar("tmp_out", transient=True, dtype=dace.float64)
    tmp_out = state.add_access("tmp_out")
    for e in state.edges():
        if e.src_conn == "__out":
            oedge = e
            assert oedge.data.data == "C"
            break

    state.add_edge(
        oedge.src,
        oedge.src_conn,
        tmp_out,
        None,
        dace.Memlet(data="tmp_out", subset="0"),
    )
    state.add_edge(
        tmp_out,
        None,
        oedge.dst,
        oedge.dst_conn,
        dace.Memlet(data="C", subset="__i0, __i1"),
    )

    # Now we create a new output that uses `tmp_out` but goes into `D`.
    #  The memlet on the inside will not use `D` but `tmp_out`.
    state.add_edge(
        tmp_out,
        None,
        oedge.dst,
        "IN_D",
        dace.Memlet(data=tmp_out.data, subset="0", other_subset="__i1, __i0"),
    )
    state.add_edge(
        oedge.dst,
        "OUT_D",
        state.add_access("D"),
        None,
        dace.Memlet(data="D", subset="__i0, __i1", other_subset="0"),
    )
    oedge.dst.add_in_connector("IN_D", force=True)
    oedge.dst.add_out_connector("OUT_D", force=True)
    state.remove_edge(oedge)

    # Without this the test does not work properly
    #  It is related to [Issue#1703](https://github.com/spcl/dace/issues/1703)
    sdfg.validate()
    for edge in state.edges():
        edge.data.try_initialize(edge=edge, sdfg=sdfg, state=state)

    for array in sdfg.arrays.values():
        if isinstance(array, dace.data.Array):
            array.storage = dace.StorageType.GPU_Global
        else:
            array.storage = dace.StorageType.Register
    sdfg.apply_gpu_transformations(simplify=False)
    sdfg.validate()

    return sdfg


def test_argument_signature_test():
    """Tests if the argument signature is computed correctly.

    The test is focused on if data dependencies are picked up if they are only
    referenced indirectly. This effect is only directly visible for GPU.
    """
    sdfg = make_sdfg()

    map_entry = None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                map_entry = node
                break
        if map_entry is not None:
            break

    # Now get the argument list of the map.
    res_arglist = {k: v for k, v in state.scope_subgraph(map_entry).arglist().items()}

    ref_arglist = {
        'A': dace.data.Array,
        'B': dace.data.Array,
        'C': dace.data.Array,
        'D': dace.data.Array,
        'N': dace.data.Scalar,
        'second_stride_A': dace.data.Scalar,
        'second_stride_D': dace.data.Scalar,
    }

    assert len(ref_arglist) == len(res_arglist), f"Expected {len(ref_arglist)} but got {len(res_arglist)}"
    for aname in ref_arglist.keys():
        atype_ref = ref_arglist[aname]
        atype_res = res_arglist[aname]
        assert isinstance(atype_res,
                          atype_ref), f"Expected '{aname}' to have type {atype_ref}, but it had {type(atype_res)}."


@pytest.mark.gpu
def test_the_indirectly_referenced_arguments_reach_the_kernel():
    """The signature above is what the kernel is declared with, so a missing one does not compile.

    The GPU transformation tiles the map, and the write to ``D`` then leaves through two exits with
    the inner memlet still naming the transient -- which is how ``D`` and its stride went missing
    from the kernel while the arglist above was already right.
    """
    import cupy as cp

    sdfg = make_sdfg()
    csdfg = sdfg.compile()

    # Every array is GPU_Global, so the arguments are device arrays; the strides are the ones the
    # descriptors were declared with, which for a contiguous array is the row length.
    n = 8
    a = cp.asarray(np.random.rand(n))
    b = cp.asarray(np.random.rand(n, n))
    c = cp.zeros((n, n))
    d = cp.zeros((n, n))
    csdfg(A=a, B=b, C=c, D=d, N=n, second_stride_A=1, second_stride_D=n)

    # ``D`` is written through the transposing ``other_subset`` on the inner memlet, so it is C's
    # transpose -- an all-zero D is the symptom of the kernel never receiving it at all.
    expected = cp.asnumpy(a)[np.newaxis, :] + cp.asnumpy(b)
    assert np.allclose(cp.asnumpy(c), expected), 'the mapped tasklet did not compute A + B into C'
    assert np.allclose(cp.asnumpy(d), expected.T), 'the second output, reached only indirectly, is wrong'


def make_init_symbol_sdfg() -> dace.SDFG:
    """An SDFG whose init signature is a strict subset of its free symbols.

    ``GM`` appears only in a descriptor shape, so it is not needed as an argument to the generated
    code and ``__dace_init_`` does not take it; ``Px`` is read by a tasklet, so it is. The name
    ordering matters: both signatures are sorted, so ``GM`` sitting first is what makes a set
    mismatch land as a value shift rather than a harmless extra trailing argument.
    """
    sdfg = dace.SDFG('init_symbol_signature')
    GM = dace.symbol(sdfg.add_symbol('GM', dace.int32))
    sdfg.add_symbol('Px', dace.int32)
    sdfg.add_array('out', (GM, ), dace.float64, transient=False)
    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet('write_px', {}, {'o'}, 'o = Px')
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


def test_init_receives_the_symbols_its_signature_declares():
    """``__dace_init_`` takes only the symbols the code needs, so the caller must filter its init
    arguments by the same set. Filtering by the wider ``free_symbols`` passes one value too many and
    silently shifts every later parameter -- ctypes cannot catch it, and pgemv handed
    ``Cblacs_gridinit`` an array extent as the process-grid width."""
    sdfg = make_init_symbol_sdfg()
    # Non-vacuity: without this gap the two filters agree and the test proves nothing.
    assert sdfg.used_symbols(all_symbols=False) != sdfg.free_symbols, 'the two symbol sets must differ here'

    code = sdfg.generate_code()[0].clean_code
    params = re.search(r'__dace_init_' + sdfg.name + r'\(([^)]*)\)', code).group(1)
    assert [p.split()[-1] for p in params.split(',')] == ['Px'], f'unexpected init signature: {params}'

    # The init code runs with the init parameters in scope, so a wrong value for ``Px`` there is
    # observable: a non-zero ``__result`` makes the initializer return a null handle.
    sdfg.append_init_code('if (Px != 3) { __result = 1; }')
    out = np.zeros(4)
    sdfg(out=out, GM=4, Px=3)
    assert out[0] == 3, f'the tasklet wrote {out[0]}, so the program itself got the wrong Px'


if __name__ == "__main__":
    test_argument_signature_test()
    test_the_indirectly_referenced_arguments_reach_the_kernel()
    test_init_receives_the_symbols_its_signature_declares()
