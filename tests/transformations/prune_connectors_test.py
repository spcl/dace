# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import numpy as np
import os
import pytest
import dace
from dace.transformation.dataflow import PruneConnectors


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
    isolated_sdfg.add_symbol("i", dace.int32)
    isolated_nsdfg.symbol_mapping["i"] = "i"
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

    return sdfg_outer


@pytest.mark.parametrize("remove_unused_containers", [False, True])
def test_prune_connectors(remove_unused_containers, n=None):
    if n is None:
        n = 64

    sdfg = make_sdfg()

    if sdfg.apply_transformations_repeated(PruneConnectors,
                                           options=[{
                                               'remove_unused_containers': remove_unused_containers
                                           }]) != 3:
        raise RuntimeError("PruneConnectors was not applied.")

    arr_in = np.zeros((n, n), dtype=np.uint16)
    arr_out = np.empty((n, n), dtype=np.uint16)

    try:
        os.remove("prune_connectors_test.txt")
    except FileNotFoundError:
        pass

    if remove_unused_containers:
        sdfg(read_used=arr_in, write_used=arr_out, N=n)
    else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=64)
    args = parser.parse_args()

    n = np.int32(args.N)

    test_prune_connectors(n=n)
    test_unused_retval()
    test_unused_retval_2()
