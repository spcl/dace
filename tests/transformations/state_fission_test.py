# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify state fission.
# We use vector addition

import dace
import numpy as np

from dace.memlet import Memlet
from dace.transformation import helpers


def make_vecAdd_sdfg(symbol_name: str, sdfg_name: str, access_nodes_dict: dict, dtype=dace.float32):
    n = dace.symbol(symbol_name)
    vecAdd_sdfg = dace.SDFG(sdfg_name)
    vecAdd_state = vecAdd_sdfg.add_state()

    # ---------- ----------
    # ACCESS NODES
    # ---------- ----------

    x_name = access_nodes_dict["x"]
    y_name = access_nodes_dict["y"]
    z_name = access_nodes_dict["z"]

    vecAdd_sdfg.add_array(x_name, [n], dtype=dtype)
    vecAdd_sdfg.add_array(y_name, [n], dtype=dtype)
    vecAdd_sdfg.add_array(z_name, [n], dtype=dtype)

    x_in = vecAdd_state.add_read(x_name)
    y_in = vecAdd_state.add_read(y_name)
    z_out = vecAdd_state.add_write(z_name)

    # ---------- ----------
    # COMPUTE
    # ---------- ----------
    vecMap_entry, vecMap_exit = vecAdd_state.add_map('vecAdd_map', dict(i='0:{}'.format(n)))

    vecAdd_tasklet = vecAdd_state.add_tasklet('vecAdd_task', ['x_con', 'y_con'], ['z_con'], 'z_con = x_con + y_con')

    vecAdd_state.add_memlet_path(x_in,
                                 vecMap_entry,
                                 vecAdd_tasklet,
                                 dst_conn='x_con',
                                 memlet=dace.Memlet.simple(x_in.data, 'i'))

    vecAdd_state.add_memlet_path(y_in,
                                 vecMap_entry,
                                 vecAdd_tasklet,
                                 dst_conn='y_con',
                                 memlet=dace.Memlet.simple(y_in.data, 'i'))

    vecAdd_state.add_memlet_path(vecAdd_tasklet,
                                 vecMap_exit,
                                 z_out,
                                 src_conn='z_con',
                                 memlet=dace.Memlet.simple(z_out.data, 'i'))

    return vecAdd_sdfg


def make_nested_sdfg_cpu():
    '''
    Build an SDFG with two nested SDFGs
    '''

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("two_vecAdd")
    state = sdfg.add_state("state")

    # build the first axpy: works with x,y, and z of n-elements
    access_nodes_dict = {"x": "x", "y": "y", "z": "z"}

    # ATTENTION: this two nested SDFG must have the same name as they are equal
    to_nest = make_vecAdd_sdfg("n", "vecAdd1", access_nodes_dict)

    sdfg.add_array("x", [n], dace.float32)
    sdfg.add_array("y", [n], dace.float32)
    sdfg.add_array("z", [n], dace.float32)
    x = state.add_read("x")
    y = state.add_read("y")
    z = state.add_write("z")

    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"})

    state.add_memlet_path(x, nested_sdfg, dst_conn="x", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(y, nested_sdfg, dst_conn="y", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(nested_sdfg, z, src_conn="z", memlet=Memlet.simple(z, "0:n", num_accesses=n))

    # Build the second axpy: works with v,w and u of m elements
    access_nodes_dict = {"x": "v", "y": "w", "z": "u"}
    to_nest = make_vecAdd_sdfg("m", "vecAdd2", access_nodes_dict)

    sdfg.add_array("v", [m], dace.float32)
    sdfg.add_array("w", [m], dace.float32)
    sdfg.add_array("u", [m], dace.float32)
    v = state.add_read("v")
    w = state.add_read("w")
    u = state.add_write("u")

    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"v", "w"}, {"u"})

    state.add_memlet_path(v, nested_sdfg, dst_conn="v", memlet=Memlet.simple(v, "0:m", num_accesses=m))
    state.add_memlet_path(w, nested_sdfg, dst_conn="w", memlet=Memlet.simple(w, "0:m", num_accesses=m))
    state.add_memlet_path(nested_sdfg, u, src_conn="u", memlet=Memlet.simple(u, "0:m", num_accesses=m))

    return sdfg


def test_state_fission():
    '''
    Tests state fission. The starting point is a stae SDFG with two
    Nested SDFGs. The state is splitted into two
    :return:
    '''

    size_n = 16
    size_m = 32
    sdfg = make_nested_sdfg_cpu()

    # state fission
    state = sdfg.states()[0]
    node_x = state.nodes()[0]
    node_y = state.nodes()[1]
    node_z = state.nodes()[2]
    vec_add1 = state.nodes()[3]

    subg = dace.sdfg.graph.SubgraphView(state, [node_x, node_y, vec_add1, node_z])
    helpers.state_fission(sdfg, subg)
    sdfg.validate()

    assert (len(sdfg.states()) == 2)

    # run the program
    vec_add = sdfg.compile()

    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_m).astype(np.float32)
    w = np.random.rand(size_m).astype(np.float32)
    u = np.random.rand(size_m).astype(np.float32)

    vec_add(x=x, y=y, z=z, v=v, w=w, u=u, n=size_n, m=size_m)

    ref1 = np.add(x, y)
    ref2 = np.add(v, w)

    diff1 = np.linalg.norm(ref1 - z) / size_n
    diff2 = np.linalg.norm(ref2 - u) / size_m

    assert (diff1 <= 1e-5 and diff2 <= 1e-5)


if __name__ == "__main__":
    test_state_fission()
