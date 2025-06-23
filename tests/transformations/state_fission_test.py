# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Tuple

import dace
import numpy as np
import pytest

from dace.memlet import Memlet
from dace.sdfg import nodes, graph
from dace.transformation import helpers

from .utility import count_nodes, unique_name, make_sdfg_args, compile_and_run_sdfg


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
    """
    Build an SDFG with two nested SDFGs
    """

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


def _make_simple_split_with_access_nodes_sdfg(
) -> Tuple[dace.SDFG, dace.SDFGState, nodes.AccessNode, nodes.Tasklet, nodes.AccessNode, nodes.AccessNode]:
    sdfg = dace.SDFG(unique_name("simple_split_with_access_nodes_sdfg"))
    state = sdfg.add_state()

    for name in "abc":
        sdfg.add_scalar(
            name,
            dtype=dace.float64,
            transient=False,
        )
    a, b, c = (state.add_access(name) for name in "abc")

    tlet = state.add_tasklet(
        "computation",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in + 3.0",
    )

    state.add_edge(a, None, tlet, "__in", dace.Memlet("a[0]"))
    state.add_edge(tlet, "__out", b, None, dace.Memlet("b[0]"))
    state.add_nedge(b, c, dace.Memlet("b[0] -> [0]"))

    sdfg.validate()

    return sdfg, state, a, tlet, b, c


def test_state_fission():
    """
    Tests state fission. The starting point is a stae SDFG with two
    Nested SDFGs. The state is splitted into two
    """

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
    helpers.state_fission(subg)
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


@pytest.mark.parametrize("allow_isolated_nodes", [True, False])
def test_simple_split_with_access_nodes_1(allow_isolated_nodes: bool):
    """
    We only put `a` into the subgraph, thus it will be alone inside the new state.
    The original state will still have the same structure.
    """
    sdfg, state, a, tlet, b, c = _make_simple_split_with_access_nodes_sdfg()
    assert state.number_of_nodes() == 4
    assert count_nodes(state, nodes.AccessNode) == 3
    assert sdfg.number_of_nodes() == 1

    subgraph = graph.SubgraphView(state, [a])
    new_state = helpers.state_fission(subgraph, allow_isolated_nodes=allow_isolated_nodes)

    # The new state is before the original state.
    assert sdfg.number_of_nodes() == 2
    assert sdfg.out_degree(new_state) == 1
    assert sdfg.in_degree(new_state) == 0
    assert {state} == {oedge.dst for oedge in sdfg.out_edges(new_state)}
    assert sdfg.out_degree(state) == 0
    assert sdfg.in_degree(state) == 1

    # The original state has still the same structure. Because of how the function
    #  is implemented the original nodes will still be there.
    assert state.number_of_nodes() == 4
    assert set(state.nodes()) == {a, tlet, b, c}

    if allow_isolated_nodes:
        # The new state only contains `a`, however, since it is only a boundary node
        #  it has been copied; this is an implementation detail.
        assert new_state.number_of_nodes() == 1
        assert a not in new_state.nodes()
        assert {"a"} == {ac.data for ac in new_state.data_nodes()}
        assert all(new_state.degree(node) == 0 for node in new_state.nodes())

        # Because of the isolated node it is not valid.
        assert not sdfg.is_valid()

        # But if we remove the node, then it will be okay.
        new_state.remove_nodes_from(list(new_state.nodes()))
        sdfg.validate()

    else:
        # If we do not allow isolated nodes, then the new state is empty and the SDFG is valid.
        assert new_state.number_of_nodes() == 0
        sdfg.validate()


@pytest.mark.parametrize("relocation_set", [0, 1, 2])
def test_simple_split_with_access_nodes_2(relocation_set: int):
    """
    There are several modes for this test, indicated by `relocation_set`:
    - 0: `a` and `tlet` are in the subgraph, but that has to be extend by `b`.
    - 1: `a`, `tlet` and `b` are in the relocation set.
    - 2: Only `tlet` is added, but `a` and `b` have to be added automatically.

    Regardless of `relocation_set` the nodes `a`, `tlet` and `b` will be moved to the new state
    while `c` remains in the original state.
    """
    sdfg, state, a, tlet, b, c = _make_simple_split_with_access_nodes_sdfg()
    assert state.number_of_nodes() == 4
    assert count_nodes(state, nodes.AccessNode) == 3
    assert sdfg.number_of_nodes() == 1

    if relocation_set == 0:
        subgraph = graph.SubgraphView(state, [a, tlet])
    elif relocation_set == 1:
        subgraph = graph.SubgraphView(state, [a, tlet, b])
    elif relocation_set == 2:
        subgraph = graph.SubgraphView(state, [tlet])
    else:
        raise NotImplementedError(f'`relocation_set={relocation_set}` is unknown.')

    new_state = helpers.state_fission(subgraph)
    sdfg.validate()

    # The new state is before the original state.
    assert sdfg.number_of_nodes() == 2
    assert sdfg.out_degree(new_state) == 1
    assert sdfg.in_degree(new_state) == 0
    assert {state} == {oedge.dst for oedge in sdfg.out_edges(new_state)}
    assert sdfg.out_degree(state) == 0
    assert sdfg.in_degree(state) == 1

    # `a`, `tlet` and `b` have been moved into the new state. However, since `b` is a boundary node
    #  it was not moved but copied (this is an implementation detail).
    assert new_state.number_of_nodes() == 3
    new_state_ac = count_nodes(new_state, nodes.AccessNode, True)
    assert len(new_state_ac) == 2
    assert {a, tlet}.issubset(new_state.nodes())
    assert a in new_state_ac
    assert b not in new_state_ac
    assert {'a', 'b'} == {ac.data for ac in new_state_ac}

    # The second (original) state contains the `b` AccessNode that copies into the
    #  `c` AccessNode. Both are the originals, which is an implementation detail.
    assert state.number_of_nodes() == 2
    org_state_ac = count_nodes(state, nodes.AccessNode, True)
    assert set(org_state_ac) == {b, c}


if __name__ == "__main__":
    test_state_fission()
