# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify state fission.
# We use vector addition

import dace
import numpy as np

from dace.memlet import Memlet
from dace.transformation import helpers


def make_vecAdd_sdfg(symbol_name: str,
                     sdfg_name: str,
                     access_nodes_dict: dict,
                     dtype=dace.float32):
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
    vecMap_entry, vecMap_exit = vecAdd_state.add_map("vecAdd_map",
                                                     dict(i="0:{}".format(n)))

    vecAdd_tasklet = vecAdd_state.add_tasklet("vecAdd_task", ["x_con", "y_con"],
                                              ["z_con"],
                                              "z_con = x_con + y_con")

    vecAdd_state.add_memlet_path(x_in,
                                 vecMap_entry,
                                 vecAdd_tasklet,
                                 dst_conn='x_con',
                                 memlet=dace.Memlet(f"{x_name}[i]"))

    vecAdd_state.add_memlet_path(y_in,
                                 vecMap_entry,
                                 vecAdd_tasklet,
                                 dst_conn='y_con',
                                 memlet=dace.Memlet(f"{y_name}[i]"))

    vecAdd_state.add_memlet_path(vecAdd_tasklet,
                                 vecMap_exit,
                                 z_out,
                                 src_conn='z_con',
                                 memlet=dace.Memlet(f"{z_name}[i]"))

    return vecAdd_sdfg


def make_nested_sdfg_cpu():
    '''
    Build an SDFG with three nested SDFGs that comput
    r = (x+y) + (v+w)
    '''

    n = dace.symbol("n")

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
    z = state.add_access("z")

    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"})

    state.add_memlet_path(x,
                          nested_sdfg,
                          dst_conn="x",
                          memlet=Memlet(f"x[0:{n}]"))
    state.add_memlet_path(y,
                          nested_sdfg,
                          dst_conn="y",
                          memlet=Memlet(f"y[0:{n}]"))
    state.add_memlet_path(nested_sdfg,
                          z,
                          src_conn="z",
                          memlet=Memlet(f"z[0:{n}]"))

    # Build the second axpy: works with v,w and u of n elements
    access_nodes_dict = {"x": "v", "y": "w", "z": "u"}
    to_nest = make_vecAdd_sdfg("n", "vecAdd2", access_nodes_dict)

    sdfg.add_array("v", [n], dace.float32)
    sdfg.add_array("w", [n], dace.float32)
    sdfg.add_array("u", [n], dace.float32)
    v = state.add_read("v")
    w = state.add_read("w")
    u = state.add_access("u")

    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"v", "w"}, {"u"})

    state.add_memlet_path(v,
                          nested_sdfg,
                          dst_conn="v",
                          memlet=Memlet(f"v[0:{n}]"))
    state.add_memlet_path(w,
                          nested_sdfg,
                          dst_conn="w",
                          memlet=Memlet(f"w[0:{n}]"))
    state.add_memlet_path(nested_sdfg,
                          u,
                          src_conn="u",
                          memlet=Memlet(f"u[0:{n}]"))

    # Build the third axpy: works with z,u and u of n elements
    access_nodes_dict = {"x": "z", "y": "u", "z": "r"}
    to_nest = make_vecAdd_sdfg("n", "vecAdd3", access_nodes_dict)

    sdfg.add_array("r", [n], dace.float32)
    r = state.add_write("r")

    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"z", "u"}, {"r"})

    state.add_memlet_path(z,
                          nested_sdfg,
                          dst_conn="z",
                          memlet=Memlet(f"z[0:{n}]"))
    state.add_memlet_path(u,
                          nested_sdfg,
                          dst_conn="u",
                          memlet=Memlet(f"u[0:{n}]"))
    state.add_memlet_path(nested_sdfg,
                          r,
                          src_conn="r",
                          memlet=Memlet(f"r[0:{n}]"))
    return sdfg


def test_state_fission():
    '''
    Tests state fission. The starting point is a stae SDFG with two
    Nested SDFGs. The state is splitted into two
    :return:
    '''

    size_n = 16
    sdfg = make_nested_sdfg_cpu()

    # state fission
    state = sdfg.states()[0]
    node_x = state.nodes()[0]
    node_y = state.nodes()[1]
    node_z = state.nodes()[2]
    vec_add1 = state.nodes()[3]

    subg = dace.sdfg.graph.SubgraphView(state,
                                        [node_x, node_y, vec_add1, node_z])
    helpers.state_fission(sdfg, subg)
    sdfg.validate()
    assert (len(sdfg.states()) == 2)

    # run the program
    vec_add = sdfg.compile()

    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_n).astype(np.float32)
    w = np.random.rand(size_n).astype(np.float32)
    u = np.random.rand(size_n).astype(np.float32)

    r = np.random.rand(size_n).astype(np.float32)

    vec_add(x=x, y=y, z=z, v=v, w=w, u=u, n=size_n, r=r)

    ref = np.add(x, y) + np.add(v, w)

    diff = np.linalg.norm(ref - r) / size_n

    assert diff <= 1e-5


if __name__ == "__main__":
    test_state_fission()
