# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# The scope of the test is to verify that code nested SDFGs with a unique name is generated only once
# The nested SDFG compute vector addition

import dace
import numpy as np
import argparse
import subprocess

from dace.memlet import Memlet

size_n = 32
size_m = 64


def make_vecAdd_sdfg(sdfg_name: str, dtype=dace.float32):
    '''
    Builds an SDFG for vector addition
    :param sdfg_name: name to give to the sdfg
    :param dtype: used data type
    :return: an SDFG
    '''
    n = dace.symbol("size")
    vecAdd_sdfg = dace.SDFG(sdfg_name)
    vecAdd_state = vecAdd_sdfg.add_state("vecAdd_nested")

    # ---------- ----------
    # ACCESS NODES
    # ---------- ----------

    x_name = "x"
    y_name = "y"
    z_name = "z"

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


def make_nested_vecAdd_sdfg(sdfg_name: str, dtype=dace.float32):
    '''
    Builds an SDFG for vector addition. Internally has a nested SDFG in charge of actually
    performing the computation.
    :param sdfg_name: name to give to the sdfg
    :param dtype: used data type
    :return: an SDFG
    '''
    n = dace.symbol("size")
    vecAdd_parent_sdfg = dace.SDFG(sdfg_name)
    vecAdd_parent_state = vecAdd_parent_sdfg.add_state("vecAdd_parent")

    # ---------- ----------
    # ACCESS NODES
    # ---------- ----------

    x_name = "x"
    y_name = "y"
    z_name = "z"

    vecAdd_parent_sdfg.add_array(x_name, [n], dtype=dtype)
    vecAdd_parent_sdfg.add_array(y_name, [n], dtype=dtype)
    vecAdd_parent_sdfg.add_array(z_name, [n], dtype=dtype)

    x_in = vecAdd_parent_state.add_read(x_name)
    y_in = vecAdd_parent_state.add_read(y_name)
    z_out = vecAdd_parent_state.add_write(z_name)

    # ---------- ----------
    # COMPUTE
    # ---------- ----------

    # Create the nested SDFG for vector addition
    nested_sdfg_name = sdfg_name + "_nested"
    to_nest = make_vecAdd_sdfg(nested_sdfg_name, dtype)

    # Nest it and connect memlets
    nested_sdfg = vecAdd_parent_state.add_nested_sdfg(to_nest, vecAdd_parent_sdfg, {"x", "y"}, {"z"})
    vecAdd_parent_state.add_memlet_path(x_in,
                                        nested_sdfg,
                                        dst_conn="x",
                                        memlet=Memlet.simple(x_in, "0:size", num_accesses=n))
    vecAdd_parent_state.add_memlet_path(y_in,
                                        nested_sdfg,
                                        dst_conn="y",
                                        memlet=Memlet.simple(y_in, "0:size", num_accesses=n))
    vecAdd_parent_state.add_memlet_path(nested_sdfg,
                                        z_out,
                                        src_conn="z",
                                        memlet=Memlet.simple(z_out, "0:size", num_accesses=n))

    return vecAdd_parent_sdfg


def make_nested_sdfg_cpu_single_state():
    '''
    Builds an SDFG with two identical nested SDFGs
    '''

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("two_vecAdd")
    state = sdfg.add_state("state")

    # build the first axpy: works with x,y, and z of n-elements

    # ATTENTION: this two nested SDFG must have the same name as they are equal
    to_nest = make_vecAdd_sdfg("vecAdd")

    sdfg.add_array("x", [n], dace.float32)
    sdfg.add_array("y", [n], dace.float32)
    sdfg.add_array("z", [n], dace.float32)
    x = state.add_read("x")
    y = state.add_read("y")
    z = state.add_write("z")

    # add it as nested SDFG, with proper symbol mapping
    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"}, {"size": "n"})

    state.add_memlet_path(x, nested_sdfg, dst_conn="x", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(y, nested_sdfg, dst_conn="y", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(nested_sdfg, z, src_conn="z", memlet=Memlet.simple(z, "0:n", num_accesses=n))

    # Build the second axpy: works with v,w and u of m elements
    to_nest = make_vecAdd_sdfg("vecAdd")

    sdfg.add_array("v", [m], dace.float32)
    sdfg.add_array("w", [m], dace.float32)
    sdfg.add_array("u", [m], dace.float32)
    v = state.add_read("v")
    w = state.add_read("w")
    u = state.add_write("u")

    nested_sdfg = state.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"}, {"size": "m"})

    state.add_memlet_path(v, nested_sdfg, dst_conn="x", memlet=Memlet.simple(v, "0:m", num_accesses=m))
    state.add_memlet_path(w, nested_sdfg, dst_conn="y", memlet=Memlet.simple(w, "0:m", num_accesses=m))
    state.add_memlet_path(nested_sdfg, u, src_conn="z", memlet=Memlet.simple(u, "0:m", num_accesses=m))

    return sdfg


def make_nested_sdfg_cpu_two_states():
    '''
    Builds an SDFG with two nested SDFGs, one per state
    '''

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("two_vecAdd")
    state_0 = sdfg.add_state("state_0")

    # build the first axpy: works with x,y, and z of n-elements

    # ATTENTION: this two nested SDFG must have the same name as they are equal
    to_nest = make_vecAdd_sdfg("vecAdd")

    sdfg.add_array("x", [n], dace.float32)
    sdfg.add_array("y", [n], dace.float32)
    sdfg.add_array("z", [n], dace.float32)
    x = state_0.add_read("x")
    y = state_0.add_read("y")
    z = state_0.add_write("z")

    # add it as nested SDFG, with proper symbol mapping
    nested_sdfg = state_0.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"}, {"size": "n"})

    state_0.add_memlet_path(x, nested_sdfg, dst_conn="x", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state_0.add_memlet_path(y, nested_sdfg, dst_conn="y", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state_0.add_memlet_path(nested_sdfg, z, src_conn="z", memlet=Memlet.simple(z, "0:n", num_accesses=n))

    # Build the second axpy: add another state works with v,w and u of m elements
    state_1 = sdfg.add_state_after(state_0, "state_1")

    to_nest = make_vecAdd_sdfg("vecAdd")

    sdfg.add_array("v", [m], dace.float32)
    sdfg.add_array("w", [m], dace.float32)
    sdfg.add_array("u", [m], dace.float32)
    v = state_1.add_read("v")
    w = state_1.add_read("w")
    u = state_1.add_write("u")

    nested_sdfg = state_1.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"}, {"size": "m"})

    state_1.add_memlet_path(v, nested_sdfg, dst_conn="x", memlet=Memlet.simple(v, "0:m", num_accesses=m))
    state_1.add_memlet_path(w, nested_sdfg, dst_conn="y", memlet=Memlet.simple(w, "0:m", num_accesses=m))
    state_1.add_memlet_path(nested_sdfg, u, src_conn="z", memlet=Memlet.simple(u, "0:m", num_accesses=m))

    return sdfg


def make_nested_nested_sdfg_cpu():
    '''
    Builds an SDFG with two nested SDFGs, each of them has internally another Nested SDFG
    '''

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("nested_nested_vecAdd")
    state_0 = sdfg.add_state("state_0")

    # build the first axpy: works with x,y, and z of n-elements

    # ATTENTION: this two nested SDFG must have the same name as they are equal
    to_nest = make_nested_vecAdd_sdfg("vecAdd")
    sdfg.add_array("x", [n], dace.float32)
    sdfg.add_array("y", [n], dace.float32)
    sdfg.add_array("z", [n], dace.float32)
    x = state_0.add_read("x")
    y = state_0.add_read("y")
    z = state_0.add_write("z")

    # add it as nested SDFG, with proper symbol mapping
    nested_sdfg = state_0.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"}, {"size": "n"})

    state_0.add_memlet_path(x, nested_sdfg, dst_conn="x", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state_0.add_memlet_path(y, nested_sdfg, dst_conn="y", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state_0.add_memlet_path(nested_sdfg, z, src_conn="z", memlet=Memlet.simple(z, "0:n", num_accesses=n))

    # Build the second axpy: add another state works with v,w and u of m elements
    state_1 = sdfg.add_state_after(state_0, "state_1")

    to_nest = make_nested_vecAdd_sdfg("vecAdd")

    sdfg.add_array("v", [m], dace.float32)
    sdfg.add_array("w", [m], dace.float32)
    sdfg.add_array("u", [m], dace.float32)
    v = state_1.add_read("v")
    w = state_1.add_read("w")
    u = state_1.add_write("u")

    nested_sdfg = state_1.add_nested_sdfg(to_nest, sdfg, {"x", "y"}, {"z"}, {"size": "m"})

    state_1.add_memlet_path(v, nested_sdfg, dst_conn="x", memlet=Memlet.simple(v, "0:m", num_accesses=m))
    state_1.add_memlet_path(w, nested_sdfg, dst_conn="y", memlet=Memlet.simple(w, "0:m", num_accesses=m))
    state_1.add_memlet_path(nested_sdfg, u, src_conn="z", memlet=Memlet.simple(u, "0:m", num_accesses=m))

    return sdfg


def test_single_state():
    sdfg = make_nested_sdfg_cpu_single_state()

    two_axpy = sdfg.compile()

    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_m).astype(np.float32)
    w = np.random.rand(size_m).astype(np.float32)
    u = np.random.rand(size_m).astype(np.float32)

    two_axpy(x=x, y=y, z=z, v=v, w=w, u=u, n=size_n, m=size_m)

    ref1 = np.add(x, y)
    ref2 = np.add(v, w)

    diff1 = np.linalg.norm(ref1 - z) / size_n
    diff2 = np.linalg.norm(ref2 - u) / size_m

    # There is no need to check that the Nested SDFG has been generated only once. If this is not the case
    # the test will fail while compiling
    assert diff1 <= 1e-5 and diff2 <= 1e-5


def test_two_states():
    sdfg = make_nested_sdfg_cpu_two_states()

    two_axpy = sdfg.compile()
    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_m).astype(np.float32)
    w = np.random.rand(size_m).astype(np.float32)
    u = np.random.rand(size_m).astype(np.float32)

    two_axpy(x=x, y=y, z=z, v=v, w=w, u=u, n=size_n, m=size_m)

    ref1 = np.add(x, y)
    ref2 = np.add(v, w)

    diff1 = np.linalg.norm(ref1 - z) / size_n
    diff2 = np.linalg.norm(ref2 - u) / size_m
    assert diff1 <= 1e-5 and diff2 <= 1e-5


def test_nested_nested():
    sdfg = make_nested_nested_sdfg_cpu()
    two_axpy = sdfg.compile()
    x = np.random.rand(size_n).astype(np.float32)
    y = np.random.rand(size_n).astype(np.float32)
    z = np.random.rand(size_n).astype(np.float32)

    v = np.random.rand(size_m).astype(np.float32)
    w = np.random.rand(size_m).astype(np.float32)
    u = np.random.rand(size_m).astype(np.float32)

    two_axpy(x=x, y=y, z=z, v=v, w=w, u=u, n=size_n, m=size_m)

    ref1 = np.add(x, y)
    ref2 = np.add(v, w)

    diff1 = np.linalg.norm(ref1 - z) / size_n
    diff2 = np.linalg.norm(ref2 - u) / size_m
    assert diff1 <= 1e-5 and diff2 <= 1e-5


if __name__ == "__main__":
    test_single_state()
    test_two_states()
    test_nested_nested()
