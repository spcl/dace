import dace


def test_add_scope_connectors():
    sdfg = dace.SDFG("add_scope_connectors_sdfg")
    state = sdfg.add_state(is_start_block=True)
    me: dace.nodes.MapEntry
    mx: dace.nodes.MapExit
    me, mx = state.add_map("test_map", ndrange={"__i0": "0:10"})
    assert all(len(mn.in_connectors) == 0 and len(mn.out_connectors) == 0 for mn in [me, mx])
    me.add_in_connector("IN_T", dtype=dace.float64)
    assert len(me.in_connectors) == 1 and me.in_connectors["IN_T"] is dace.float64 and len(me.out_connectors) == 0
    assert len(mx.in_connectors) == 0 and len(mx.out_connectors) == 0

    # Because there is already an `IN_T` this call will fail.
    assert not me.add_scope_connectors("T")
    assert len(me.in_connectors) == 1 and me.in_connectors["IN_T"] is dace.float64 and len(me.out_connectors) == 0
    assert len(mx.in_connectors) == 0 and len(mx.out_connectors) == 0

    # Now it will work, because we specify force, however, the current type for `IN_T` will be overridden.
    assert me.add_scope_connectors("T", force=True)
    assert len(me.in_connectors) == 1 and me.in_connectors["IN_T"].type is None
    assert len(me.out_connectors) == 1 and me.out_connectors["OUT_T"].type is None
    assert len(mx.in_connectors) == 0 and len(mx.out_connectors) == 0

    # Now tries to the full adding.
    assert mx.add_scope_connectors("B", dtype=dace.int64)
    assert len(mx.in_connectors) == 1 and mx.in_connectors["IN_B"] is dace.int64
    assert len(mx.out_connectors) == 1 and mx.out_connectors["OUT_B"] is dace.int64


if __name__ == "__main__":
    test_add_scope_connectors()
