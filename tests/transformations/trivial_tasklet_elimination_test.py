# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination


N = 10


def test_trivial_tasklet():
    ty_ = dace.int32
    sdfg = dace.SDFG("trivial_tasklet")
    sdfg.add_symbol("s", ty_)
    sdfg.add_array("v", (N,), ty_)
    st = sdfg.add_state()
    
    tmp1_name, _ = sdfg.add_scalar(sdfg.temp_data_name(), ty_, transient=True)
    tmp1_node = st.add_access(tmp1_name)
    init_tasklet = st.add_tasklet("init", {}, {"out"}, "out = s")
    st.add_edge(init_tasklet, "out", tmp1_node, None, dace.Memlet(tmp1_node.data))

    tmp2_name, _ = sdfg.add_scalar(sdfg.temp_data_name(), ty_, transient=True)
    tmp2_node = st.add_access(tmp2_name)
    copy_tasklet = st.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    st.add_edge(tmp1_node, None, copy_tasklet, "inp", dace.Memlet(tmp1_node.data))
    st.add_edge(copy_tasklet, "out", tmp2_node, None, dace.Memlet(tmp2_node.data))
    
    bcast_tasklet, _, _ = st.add_mapped_tasklet(
        "bcast",
        dict(i=f"0:{N}"),
        inputs={"inp": dace.Memlet(f"{tmp2_node.data}[0]")},
        input_nodes={tmp2_node.data: tmp2_node},
        code="out = inp",
        outputs={"out": dace.Memlet("v[i]")},
        external_edges=True,
    )

    sdfg.validate()
    tasklet_nodes = {x for x in st.nodes() if isinstance(x, dace.nodes.Tasklet)}
    assert tasklet_nodes == {init_tasklet, copy_tasklet, bcast_tasklet}

    count = sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    assert count == 1

    assert len(st.out_edges(tmp1_node)) == 1
    assert st.out_edges(tmp1_node)[0].dst == tmp2_node

    tasklet_nodes = {x for x in st.nodes() if isinstance(x, dace.nodes.Tasklet)}
    assert tasklet_nodes == {init_tasklet, bcast_tasklet}


def test_trivial_tasklet_with_map():
    ty_ = dace.int32
    sdfg = dace.SDFG("trivial_tasklet_with_map")
    sdfg.add_symbol("s", ty_)
    sdfg.add_array("v", (N,), ty_)
    st = sdfg.add_state()
    
    tmp1_name, _ = sdfg.add_scalar(sdfg.temp_data_name(), ty_, transient=True)
    tmp1_node = st.add_access(tmp1_name)
    init_tasklet = st.add_tasklet("init", {}, {"out"}, "out = s")
    st.add_edge(init_tasklet, "out", tmp1_node, None, dace.Memlet(tmp1_node.data))

    me, mx = st.add_map("bcast", dict(i=f"0:{N}"))

    copy_tasklet = st.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    st.add_memlet_path(tmp1_node, me, copy_tasklet, dst_conn="inp", memlet=dace.Memlet(f"{tmp1_node.data}[0]"))
    tmp2_name, _ = sdfg.add_scalar(sdfg.temp_data_name(), ty_, transient=True)
    tmp2_node = st.add_access(tmp2_name)
    st.add_edge(copy_tasklet, "out", tmp2_node, None, dace.Memlet(tmp2_node.data))
    
    bcast_tasklet = st.add_tasklet("bcast", {"inp"}, {"out"}, "out = inp")
    st.add_edge(tmp2_node, None, bcast_tasklet, "inp", dace.Memlet(tmp2_node.data))
    st.add_memlet_path(bcast_tasklet, mx, st.add_access("v"), src_conn="out", memlet=dace.Memlet("v[i]"))

    sdfg.validate()
    tasklet_nodes = {x for x in st.nodes() if isinstance(x, dace.nodes.Tasklet)}
    assert tasklet_nodes == {init_tasklet, copy_tasklet, bcast_tasklet}

    count = sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    assert count == 2

    tasklet_nodes = {x for x in st.nodes() if isinstance(x, dace.nodes.Tasklet)}
    assert tasklet_nodes == {init_tasklet}

    assert len(st.in_edges(tmp2_node)) == 1
    assert st.in_edges(tmp2_node)[0].src == me

    assert len(st.out_edges(tmp2_node)) == 1
    assert st.out_edges(tmp2_node)[0].dst == mx


def test_trivial_tasklet_with_implicit_cast():
    ty32_ = dace.int32
    ty64_ = dace.int64
    sdfg = dace.SDFG("trivial_tasklet_with_implicit_cast")
    sdfg.add_symbol("s", ty32_)
    sdfg.add_array("v", (N,), ty32_)
    st = sdfg.add_state()
    
    tmp1_name, _ = sdfg.add_scalar(sdfg.temp_data_name(), ty32_, transient=True)
    tmp1_node = st.add_access(tmp1_name)
    init_tasklet = st.add_tasklet("init", {}, {"out"}, "out = s")
    st.add_edge(init_tasklet, "out", tmp1_node, None, dace.Memlet(tmp1_node.data))

    me, mx = st.add_map("bcast", dict(i=f"0:{N}"))

    copy_tasklet = st.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    st.add_memlet_path(tmp1_node, me, copy_tasklet, dst_conn="inp", memlet=dace.Memlet(f"{tmp1_node.data}[0]"))
    tmp2_name, _ = sdfg.add_scalar(sdfg.temp_data_name(), ty64_, transient=True)
    tmp2_node = st.add_access(tmp2_name)
    st.add_edge(copy_tasklet, "out", tmp2_node, None, dace.Memlet(tmp2_node.data))
    
    bcast_tasklet = st.add_tasklet("bcast", {"inp"}, {"out"}, "out = inp")
    st.add_edge(tmp2_node, None, bcast_tasklet, "inp", dace.Memlet(tmp2_node.data))
    st.add_memlet_path(bcast_tasklet, mx, st.add_access("v"), src_conn="out", memlet=dace.Memlet("v[i]"))

    sdfg.validate()
    tasklet_nodes = {x for x in st.nodes() if isinstance(x, dace.nodes.Tasklet)}
    assert tasklet_nodes == {init_tasklet, copy_tasklet, bcast_tasklet}

    # not applied because of data types mismatch on read/write nodes
    count = sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    assert count == 0


if __name__ == '__main__':
    test_trivial_tasklet()
    test_trivial_tasklet_with_map()
    test_trivial_tasklet_with_implicit_cast()
