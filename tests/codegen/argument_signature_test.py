import dace
import numpy as np
import pytest


def _make_indirect_reference_sdfg() -> dace.SDFG:
    """Build the ``Repr`` SDFG where arrays ``A`` and ``D`` are referenced only
    indirectly through scope-internal scalar transients.

    Each Map-scope inner Memlet references the internal transient
    (``tmp_in`` / ``tmp_out``); the outer ``A``/``D`` arrays are reachable
    only by walking the memlet path through the surrounding scope. This is
    the case ``DataflowGraphView.arglist`` must resolve to a correct kernel
    argument signature.
    """
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

    # ``A`` uses a stride that is not used by any of the other arrays.
    second_stride_A = dace.symbol(sdfg.add_symbol("second_stride_A", dace.int32))
    sdfg.add_array(
        name="A",
        dtype=dace.float64,
        shape=(N, ),
        strides=(second_stride_A, ),
        transient=False,
    )

    # ``D`` likewise uses a stride symbol not shared with any other array.
    second_stride_D = dace.symbol(sdfg.add_symbol("second_stride_D", dace.int32))
    sdfg.add_array(
        name="D",
        dtype=dace.float64,
        shape=(N, N),
        strides=(second_stride_D, 1),
        transient=False,
    )

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

    # Replace the direct ``A -> MapEntry -> tasklet`` chain with a scope-internal
    # scalar transient -- the inside-scope Memlet refers to the transient, so
    # ``A`` and ``second_stride_A`` are not directly visible inside the scope.
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
        dace.Memlet(data="tmp_in", subset="0", other_subset="__i1"),
    )
    state.add_edge(
        tmp_in,
        None,
        iedge.dst,
        iedge.dst_conn,
        dace.Memlet(f"{tmp_in.data}[0]"),
    )
    state.remove_edge(iedge)

    # Symmetric for the output: the scope-internal Memlet references a
    # ``tmp_out`` scalar transient; ``C`` flows out as before, and ``D`` is
    # added as a second sink whose internal Memlet also refers to ``tmp_out``.
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

    # Trigger Memlet initialisation; see https://github.com/spcl/dace/issues/1703.
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


def _map_entry(sdfg: dace.SDFG):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                return state, node
    raise AssertionError("No MapEntry found.")


def test_argument_signature_test():
    """``arglist`` resolves arrays referenced only via outer memlet paths.

    With the SDFG built by :func:`_make_indirect_reference_sdfg`, the scope
    subgraph's inner Memlets reference the scope-local transients
    ``tmp_in`` / ``tmp_out`` rather than ``A`` / ``D``. The outer arrays must
    still be reported as arguments by ``arglist`` so a downstream codegen
    can build a complete kernel signature.
    """
    sdfg = _make_indirect_reference_sdfg()
    state, map_entry = _map_entry(sdfg)

    res_arglist = dict(state.scope_subgraph(map_entry).arglist())
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
    for aname, atype_ref in ref_arglist.items():
        atype_res = res_arglist[aname]
        assert isinstance(atype_res,
                          atype_ref), f"Expected '{aname}' to have type {atype_ref}, but it had {type(atype_res)}."


@pytest.mark.gpu
def test_argument_signature_compiles_and_runs():
    """End-to-end CUDA compile + run: the kernel signature must include the
    indirect ``D`` / ``second_stride_D`` references emitted by the
    AccessNode->AccessNode lowering, otherwise ``nvcc`` rejects the kernel
    body with ``identifier "D" is undefined``.
    """
    cp = pytest.importorskip("cupy")

    sdfg = _make_indirect_reference_sdfg()
    csdfg = sdfg.compile()

    N_VAL = 8
    A = cp.arange(N_VAL, dtype=cp.float64)
    B = cp.arange(N_VAL * N_VAL, dtype=cp.float64).reshape(N_VAL, N_VAL)
    C = cp.zeros((N_VAL, N_VAL), dtype=cp.float64)
    D = cp.zeros((N_VAL, N_VAL), dtype=cp.float64)
    csdfg(A=A, B=B, C=C, D=D, N=N_VAL, second_stride_A=1, second_stride_D=N_VAL)

    expected = cp.asnumpy(A)[cp.newaxis, :] + cp.asnumpy(B)
    np.testing.assert_array_equal(cp.asnumpy(C), expected)


if __name__ == "__main__":
    pytest.main([__file__])
