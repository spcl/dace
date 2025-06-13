# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import numpy as np
import os
import copy
import pytest
from typing import Tuple

import dace
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.helpers import nest_state_subgraph
from dace.sdfg.state import StateSubgraphView


def make_sdfg():
    """ Creates three SDFG nested within each other, where two input arrays and
        two output arrays are fed throughout the hierarchy. One input and one
        output are not used for anything in the innermost SDFG, and can thus be
        removed in all nestings.
    """

    n = dace.symbol("N")

    sdfg_outer = dace.SDFG("prune_connectors_test")
    sdfg_outer.set_global_code("#include <fstream>\n#include <mutex>")
    state_outer = sdfg_outer.add_state("state_outer")
    sdfg_outer.add_symbol("N", dace.int32)

    sdfg_middle = dace.SDFG("middle")
    sdfg_middle.add_symbol("N", dace.int32)
    nsdfg_middle = state_outer.add_nested_sdfg(sdfg_middle,
                                               sdfg_outer, {"read_used_middle", "read_unused_middle"},
                                               {"write_used_middle", "write_unused_middle"},
                                               name="middle")
    state_middle = sdfg_middle.add_state("middle")

    entry_middle, exit_middle = state_middle.add_map("map_middle", {"i": "0:N"})

    sdfg_inner = dace.SDFG("inner")
    sdfg_inner.add_symbol("N", dace.int32)
    nsdfg_inner = state_middle.add_nested_sdfg(sdfg_inner,
                                               sdfg_middle, {"read_used_inner", "read_unused_inner"},
                                               {"write_used_inner", "write_unused_inner"},
                                               name="inner")
    state_inner = sdfg_inner.add_state("inner")

    entry_inner, exit_inner = state_inner.add_map("map_inner", {"j": "0:N"})
    tasklet = state_inner.add_tasklet("tasklet", {"read_tasklet"}, {"write_tasklet"},
                                      "write_tasklet = read_tasklet + 1")

    for s in ["unused", "used"]:

        # Read

        sdfg_outer.add_array(f"read_{s}", [n, n], dace.uint16)
        sdfg_outer.add_array(f"read_{s}_outer", [n, n], dace.uint16)
        sdfg_middle.add_array(f"read_{s}_middle", [n, n], dace.uint16)
        sdfg_inner.add_array(f"read_{s}_inner", [n], dace.uint16)

        read_outer = state_outer.add_read(f"read_{s}")
        read_middle = state_middle.add_read(f"read_{s}_middle")

        state_outer.add_memlet_path(read_outer,
                                    nsdfg_middle,
                                    dst_conn=f"read_{s}_middle",
                                    memlet=dace.Memlet(f"read_{s}[0:N, 0:N]"))
        state_middle.add_memlet_path(read_middle,
                                     entry_middle,
                                     nsdfg_inner,
                                     dst_conn=f"read_{s}_inner",
                                     memlet=dace.Memlet(f"read_{s}_middle[i, 0:N]"))

        # Write

        sdfg_outer.add_array(f"write_{s}", [n, n], dace.uint16)
        sdfg_outer.add_array(f"write_{s}_outer", [n, n], dace.uint16)
        sdfg_middle.add_array(f"write_{s}_middle", [n, n], dace.uint16)
        sdfg_inner.add_array(f"write_{s}_inner", [n], dace.uint16)

        write_outer = state_outer.add_write(f"write_{s}")
        write_middle = state_middle.add_write(f"write_{s}_middle")

        state_outer.add_memlet_path(nsdfg_middle,
                                    write_outer,
                                    src_conn=f"write_{s}_middle",
                                    memlet=dace.Memlet(f"write_{s}[0:N, 0:N]"))
        state_middle.add_memlet_path(nsdfg_inner,
                                     exit_middle,
                                     write_middle,
                                     src_conn=f"write_{s}_inner",
                                     memlet=dace.Memlet(f"write_{s}_middle[i, 0:N]"))

    read_inner = state_inner.add_read(f"read_used_inner")
    write_inner = state_inner.add_write(f"write_used_inner")

    state_inner.add_memlet_path(read_inner,
                                entry_inner,
                                tasklet,
                                dst_conn=f"read_tasklet",
                                memlet=dace.Memlet(f"read_{s}_inner[j]"))

    state_inner.add_memlet_path(tasklet,
                                exit_inner,
                                write_inner,
                                src_conn=f"write_tasklet",
                                memlet=dace.Memlet(f"write_{s}_inner[j]"))

    # Create mapped nested SDFG where the map entry and exit would be orphaned
    # by pruning the read and write, and must have nedges added to them

    isolated_read = state_outer.add_read("read_unused_outer")
    isolated_write = state_outer.add_write("write_unused_outer")
    isolated_sdfg = dace.SDFG("isolated_sdfg")
    isolated_nsdfg = state_outer.add_nested_sdfg(isolated_sdfg,
                                                 sdfg_outer, {"read_unused_isolated"}, {"write_unused_isolated"},
                                                 name="isolated")
    isolated_sdfg.add_array("read_unused_isolated", shape=(n, n), dtype=dace.uint16, transient=False)
    isolated_sdfg.add_array("write_unused_isolated", shape=(n, n), dtype=dace.uint16, transient=False)

    isolated_sdfg.add_symbol("i", dace.int32)
    isolated_nsdfg.symbol_mapping["i"] = "i"
    isolated_nsdfg.symbol_mapping["N"] = "N"
    isolated_entry, isolated_exit = state_outer.add_map("isolated", {"i": "0:N"})
    state_outer.add_memlet_path(isolated_read,
                                isolated_entry,
                                isolated_nsdfg,
                                dst_conn="read_unused_isolated",
                                memlet=dace.Memlet("read_unused_outer[0:N, 0:N]"))
    state_outer.add_memlet_path(isolated_nsdfg,
                                isolated_exit,
                                isolated_write,
                                src_conn="write_unused_isolated",
                                memlet=dace.Memlet("write_unused_outer[0:N, 0:N]"))
    isolated_state = isolated_sdfg.add_state("isolated")
    isolated_state.add_tasklet("isolated", {}, {},
                               """\
static std::mutex mutex;
std::unique_lock<std::mutex> lock(mutex);
std::ofstream of("prune_connectors_test.txt", std::ofstream::app);
of << i << "\\n";""",
                               language=dace.Language.CPP)

    sdfg_outer.validate()

    return sdfg_outer


def _make_read_write_sdfg(conforming_memlet: bool, ) -> Tuple[dace.SDFG, dace.nodes.NestedSDFG]:
    """Creates an SDFG for the `test_read_write_{1, 2}` tests.

    The SDFG is rather synthetic, it has an input `in_arg` and adds to every element
    10 and stores that in array `A`, through access node `A1`. From this access node
    the data flows into a nested SDFG. However, the data is not read but overwritten,
    through a map that writes through access node `inner_A`. That access node
    then writes into container `inner_B`. Both `inner_A` and `inner_B` are outputs
    of the nested SDFG and are written back into data container `A` and `B`.

    Depending on `conforming_memlet` the memlet that copies `inner_A` into `inner_B`
    will either be associated to `inner_A` (`True`) or `inner_B` (`False`).

    Notes:
        This is most likely a bug, see [issue#1643](https://github.com/spcl/dace/issues/1643),
        however, it is the historical behaviour.
    """

    # Creating the outer SDFG.
    osdfg = dace.SDFG("Outer_sdfg")
    ostate = osdfg.add_state(is_start_block=True)

    osdfg.add_array("in_arg", dtype=dace.float64, shape=(4, 4), transient=False)
    osdfg.add_array("A", dtype=dace.float64, shape=(4, 4), transient=False)
    osdfg.add_array("B", dtype=dace.float64, shape=(4, 4), transient=False)
    in_arg, A1, A2, B = (ostate.add_access(name) for name in ["in_arg", "A", "A", "B"])

    ostate.add_mapped_tasklet(
        "producer",
        map_ranges={
            "i": "0:4",
            "j": "0:4"
        },
        inputs={"__in": dace.Memlet("in_arg[i, j]")},
        code="__out = __in + 10.",
        outputs={"__out": dace.Memlet("A[i, j]")},
        input_nodes={in_arg},
        output_nodes={A1},
        external_edges=True,
    )

    # Creating the inner SDFG
    isdfg = dace.SDFG("Inner_sdfg")
    istate = isdfg.add_state(is_start_block=True)

    isdfg.add_array("inner_A", dtype=dace.float64, shape=(4, 4), transient=False)
    isdfg.add_array("inner_B", dtype=dace.float64, shape=(4, 4), transient=False)
    inner_A, inner_B = (istate.add_access(name) for name in ["inner_A", "inner_B"])

    istate.add_mapped_tasklet(
        "inner_consumer",
        map_ranges={
            "i": "0:4",
            "j": "0:4"
        },
        inputs={},
        code="__out = 10",
        outputs={"__out": dace.Memlet("inner_A[i, j]")},
        output_nodes={inner_A},
        external_edges=True,
    )

    # Depending on to which data container this memlet is associated,
    #  the transformation will apply or it will not apply.
    if conforming_memlet:
        # Because the `data` field of the inncoming and outgoing memlet are both
        #  set to `inner_A` the read to `inner_A` will be removed and the
        #  transformation can apply.
        istate.add_nedge(
            inner_A,
            inner_B,
            dace.Memlet("inner_A[0:4, 0:4] -> [0:4, 0:4]"),
        )
    else:
        # Because the `data` filed of the involved memlets differs the read to
        #  `inner_A` will not be removed thus the transformation can not remove
        #  the incoming `inner_A`.
        istate.add_nedge(
            inner_A,
            inner_B,
            dace.Memlet("inner_B[0:4, 0:4] -> [0:4, 0:4]"),
        )

    # Add the nested SDFG
    nsdfg = ostate.add_nested_sdfg(
        sdfg=isdfg,
        parent=osdfg,
        inputs={"inner_A"},
        outputs={"inner_A", "inner_B"},
    )

    # Connecting the nested SDFG
    ostate.add_edge(A1, None, nsdfg, "inner_A", dace.Memlet("A[0:4, 0:4]"))
    ostate.add_edge(nsdfg, "inner_A", A2, None, dace.Memlet("A[0:4, 0:4]"))
    ostate.add_edge(nsdfg, "inner_B", B, None, dace.Memlet("B[0:4, 0:4]"))

    return osdfg, nsdfg


def test_prune_connectors(n=None):
    if n is None:
        n = 64

    sdfg = make_sdfg()

    if sdfg.apply_transformations_repeated(PruneConnectors) != 3:
        raise RuntimeError("PruneConnectors was not applied.")

    arr_in = np.zeros((n, n), dtype=np.uint16)
    arr_out = np.empty((n, n), dtype=np.uint16)

    try:
        os.remove("prune_connectors_test.txt")
    except FileNotFoundError:
        pass

    # The pruned connectors are not removed so they have to be supplied.
    sdfg(read_used=arr_in,
         read_unused=arr_in,
         read_used_outer=arr_in,
         read_unused_outer=arr_in,
         write_used=arr_out,
         write_unused=arr_out,
         write_used_outer=arr_out,
         write_unused_outer=arr_out,
         N=n)

    assert np.allclose(arr_out, arr_in + 1)

    numbers_written = []
    with open("prune_connectors_test.txt", "r") as f:
        for line in f:
            numbers_written.append(int(line.strip()))
    assert all(sorted(numbers_written) == np.arange(n))

    os.remove("prune_connectors_test.txt")


def test_unused_retval():
    sdfg = dace.SDFG('tester')
    sdfg.add_transient('tmp', [1], dace.float64)
    sdfg.add_array('output', [1], dace.float64)
    state = sdfg.add_state()
    nsdfg = dace.SDFG('nester')
    nsdfg.add_array('used', [1], dace.float64)
    nsdfg.add_array('__return', [1], dace.float64)
    nstate = nsdfg.add_state()
    a = nstate.add_access('used')
    nstate.add_edge(nstate.add_tasklet('do', {}, {'out'}, 'out = 1'), 'out', a, None, dace.Memlet('used[0]'))
    nstate.add_nedge(a, nstate.add_write('__return'), dace.Memlet('__return[0]'))
    nsnode = state.add_nested_sdfg(nsdfg, None, {}, {'used', '__return'})
    state.add_edge(nsnode, 'used', state.add_write('output'), None, dace.Memlet('output[0]'))
    state.add_edge(nsnode, '__return', state.add_write('tmp'), None, dace.Memlet('tmp[0]'))

    # Mark nested SDFG to not be inlineable
    nsnode.no_inline = True

    sdfg.simplify()
    assert len(sdfg.arrays) == 1
    assert len(nsdfg.arrays) == 1

    a = np.random.rand(1)
    sdfg(output=a)
    assert np.allclose(a, 1)


def test_unused_retval_2():
    sdfg = dace.SDFG('tester')
    sdfg.add_transient('tmp', [2], dace.float64)
    sdfg.add_array('output', [2], dace.float64)
    state = sdfg.add_state()
    nsdfg = dace.SDFG('nester')
    nsdfg.add_array('used', [1], dace.float64)
    nsdfg.add_array('__return', [1], dace.float64)
    nstate = nsdfg.add_state()
    a = nstate.add_access('used')
    nstate.add_edge(nstate.add_tasklet('do', {}, {'out'}, 'out = 1'), 'out', a, None, dace.Memlet('used[0]'))
    nstate.add_nedge(a, nstate.add_write('__return'), dace.Memlet('__return[0]'))
    nsnode = state.add_nested_sdfg(nsdfg, None, {}, {'used', '__return'})
    me, mx = state.add_map('doit', dict(i='0:2'))
    state.add_nedge(me, nsnode, dace.Memlet())
    state.add_memlet_path(nsnode, mx, state.add_write('output'), memlet=dace.Memlet('output[i]'), src_conn='used')
    state.add_memlet_path(nsnode, mx, state.add_write('tmp'), memlet=dace.Memlet('tmp[i]'), src_conn='__return')

    # Mark nested SDFG to not be inlineable
    nsnode.no_inline = True

    sdfg.simplify()
    assert len(sdfg.arrays) == 1
    assert len(nsdfg.arrays) == 1

    a = np.random.rand(2)
    sdfg(output=a)
    assert np.allclose(a, 1)


def test_prune_connectors_with_dependencies():
    sdfg = dace.SDFG('tester')
    A, A_desc = sdfg.add_array('A', [4], dace.float64)
    B, B_desc = sdfg.add_array('B', [4], dace.float64)
    C, C_desc = sdfg.add_array('C', [4], dace.float64)
    D, D_desc = sdfg.add_array('D', [4], dace.float64)

    state = sdfg.add_state()
    a = state.add_access("A")
    b1 = state.add_access("B")
    b2 = state.add_access("B")
    c1 = state.add_access("C")
    c2 = state.add_access("C")
    d = state.add_access("D")

    _, map_entry_a, map_exit_a = state.add_mapped_tasklet("a",
                                                          map_ranges={"i": "0:4"},
                                                          inputs={"_in": dace.Memlet(data="A", subset='i')},
                                                          outputs={"_out": dace.Memlet(data="B", subset='i')},
                                                          code="_out = _in + 1")
    state.add_edge(a, None, map_entry_a, None, dace.Memlet(data="A", subset="0:4"))
    state.add_edge(map_exit_a, None, b1, None, dace.Memlet(data="B", subset="0:4"))

    tasklet_c, map_entry_c, map_exit_c = state.add_mapped_tasklet("c",
                                                                  map_ranges={"i": "0:4"},
                                                                  inputs={"_in": dace.Memlet(data="C", subset='i')},
                                                                  outputs={"_out": dace.Memlet(data="C", subset='i')},
                                                                  code="_out = _in + 1")
    state.add_edge(c1, None, map_entry_c, None, dace.Memlet(data="C", subset="0:4"))
    state.add_edge(map_exit_c, None, c2, None, dace.Memlet(data="C", subset="0:4"))

    _, map_entry_d, map_exit_d = state.add_mapped_tasklet("d",
                                                          map_ranges={"i": "0:4"},
                                                          inputs={"_in": dace.Memlet(data="B", subset='i')},
                                                          outputs={"_out": dace.Memlet(data="D", subset='i')},
                                                          code="_out = _in + 1")
    state.add_edge(b2, None, map_entry_d, None, dace.Memlet(data="B", subset="0:4"))
    state.add_edge(map_exit_d, None, d, None, dace.Memlet(data="D", subset="0:4"))

    sdfg.fill_scope_connectors()

    subgraph = StateSubgraphView(state, subgraph_nodes=[map_entry_c, map_exit_c, tasklet_c])
    nsdfg_node = nest_state_subgraph(sdfg, state, subgraph=subgraph)

    nsdfg_node.sdfg.add_datadesc("B1", datadesc=copy.deepcopy(B_desc))
    nsdfg_node.sdfg.arrays["B1"].transient = False
    nsdfg_node.sdfg.add_datadesc("B2", datadesc=copy.deepcopy(B_desc))
    nsdfg_node.sdfg.arrays["B2"].transient = False

    nsdfg_node.add_in_connector("B1")
    state.add_edge(b1, None, nsdfg_node, "B1", dace.Memlet.from_array(dataname="B", datadesc=B_desc))
    nsdfg_node.add_out_connector("B2")
    state.add_edge(nsdfg_node, "B2", b2, None, dace.Memlet.from_array(dataname="B", datadesc=B_desc))

    np_a = np.random.random(4)
    np_a_ = np.copy(np_a)
    np_b = np.random.random(4)
    np_b_ = np.copy(np_b)
    np_c = np.random.random(4)
    np_c_ = np.copy(np_c)
    np_d = np.random.random(4)
    np_d_ = np.copy(np_d)

    sdfg(A=np_a, B=np_b, C=np_c, D=np_d)

    applied = sdfg.apply_transformations_repeated(PruneConnectors)
    assert applied == 1
    assert len(sdfg.states()) == 2
    assert "B1" not in nsdfg_node.in_connectors
    assert "B2" not in nsdfg_node.out_connectors

    sdfg(A=np_a_, B=np_b_, C=np_c_, D=np_d_)
    assert np.allclose(np_a, np_a_)
    assert np.allclose(np_b, np_b_)
    assert np.allclose(np_c, np_c_)
    assert np.allclose(np_d, np_d_)


def test_read_write():
    sdfg, nsdfg = _make_read_write_sdfg(True)
    assert not PruneConnectors.can_be_applied_to(nsdfg=nsdfg, sdfg=sdfg, expr_index=0, permissive=False)

    sdfg, nsdfg = _make_read_write_sdfg(False)
    assert not PruneConnectors.can_be_applied_to(nsdfg=nsdfg, sdfg=sdfg, expr_index=0, permissive=False)


def test_prune_connectors_with_conditional_block():
    """
    Verifies that a connector to scalar data (here 'cond') in a NestedSDFG is not removed,
    when this data is only accessed by condition expressions in ControlFlowRegion nodes.
    """
    sdfg = dace.SDFG('tester')
    A, A_desc = sdfg.add_array('A', [4], dace.float64)
    B, B_desc = sdfg.add_array('B', [4], dace.float64)
    COND, COND_desc = sdfg.add_array('COND', [4], dace.bool_)
    OUT, OUT_desc = sdfg.add_array('OUT', [4], dace.float64)

    nsdfg = dace.SDFG('nested')
    a, _ = nsdfg.add_scalar('a', A_desc.dtype)
    b, _ = nsdfg.add_scalar('b', B_desc.dtype)
    cond, _ = nsdfg.add_scalar('cond', COND_desc.dtype)
    out, _ = nsdfg.add_scalar('out', OUT_desc.dtype)

    if_region = dace.sdfg.state.ConditionalBlock("if")
    nsdfg.add_node(if_region)
    entry_state = nsdfg.add_state("entry", is_start_block=True)
    nsdfg.add_edge(entry_state, if_region, dace.InterstateEdge())

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=nsdfg)
    a_state = then_body.add_state("true_branch", is_start_block=True)
    if_region.add_branch(dace.sdfg.state.CodeBlock(cond), then_body)
    a_state.add_nedge(a_state.add_access(a), a_state.add_access(out), dace.Memlet(out))

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=nsdfg)
    b_state = else_body.add_state("false_branch", is_start_block=True)
    if_region.add_branch(dace.sdfg.state.CodeBlock(f"not ({cond})"), else_body)
    b_state.add_nedge(b_state.add_access(b), b_state.add_access(out), dace.Memlet(out))

    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(nsdfg, sdfg, inputs={a, b, cond}, outputs={out})
    me, mx = state.add_map('map', dict(i="0:4"))
    state.add_memlet_path(state.add_access(A), me, nsdfg_node, dst_conn=a, memlet=dace.Memlet(f"{A}[i]"))
    state.add_memlet_path(state.add_access(B), me, nsdfg_node, dst_conn=b, memlet=dace.Memlet(f"{B}[i]"))
    state.add_memlet_path(state.add_access(COND), me, nsdfg_node, dst_conn=cond, memlet=dace.Memlet(f"{COND}[i]"))
    state.add_memlet_path(nsdfg_node, mx, state.add_access(OUT), src_conn=out, memlet=dace.Memlet(f"{OUT}[i]"))

    assert 0 == sdfg.apply_transformations_repeated(PruneConnectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=64)
    args = parser.parse_args()

    n = np.int32(args.N)

    test_prune_connectors(False, n=n)
    test_prune_connectors(True, n=n)
    test_unused_retval()
    test_unused_retval_2()
    test_prune_connectors_with_dependencies()
    test_read_write_1()
    test_read_write_2()
    test_prune_connectors_with_conditional_block()
