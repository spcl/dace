# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Various tests for dead code elimination passes. """

from typing import Tuple

import numpy as np
import pytest
import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination


def test_dse_simple():
    sdfg = dace.SDFG('dsetester')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_symbol('s', dace.uint64)
    init = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    s1.add_mapped_tasklet('doit', dict(i='0:20'), {}, 'out = 1', dict(out=dace.Memlet('a[i]')), external_edges=True)
    s2.add_mapped_tasklet('doit', dict(i='0:20'), {}, 'out = 2', dict(out=dace.Memlet('a[i]')), external_edges=True)

    sdfg.add_edge(init, s1, dace.InterstateEdge('s > s'))  # Always false
    sdfg.add_edge(init, s2, dace.InterstateEdge('s <= s'))

    DeadStateElimination().apply_pass(sdfg, {})
    assert set(sdfg.nodes()) == {init, s2}


def test_dse_unconditional():
    sdfg = dace.SDFG('dse_tester')
    sdfg.add_symbol('a', dace.int32)
    s = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    s3 = sdfg.add_state()
    e = sdfg.add_state()
    sdfg.add_edge(s, s1, dace.InterstateEdge('a > 0'))
    sdfg.add_edge(s, s2, dace.InterstateEdge('a >= a'))  # Always True
    sdfg.add_edge(s, s3, dace.InterstateEdge('a < 0'))
    sdfg.add_edge(s1, e, dace.InterstateEdge())
    sdfg.add_edge(s2, e, dace.InterstateEdge())
    sdfg.add_edge(s3, e, dace.InterstateEdge())

    DeadStateElimination().apply_pass(sdfg, {})
    assert set(sdfg.states()) == {s, s2, e}


def test_dse_edge_condition_with_integer_as_boolean_regression():
    """
    This is a regression test for issue #1129, which describes dead state elimination incorrectly eliminating interstate
    edges when integers are used as boolean values in interstate edge conditions. Code taken from issue #1129.
    """
    sdfg = dace.SDFG('dse_edge_condition_with_integer_as_boolean_regression')
    sdfg.add_scalar('N', dtype=dace.int32, transient=True)
    sdfg.add_scalar('result', dtype=dace.int32)
    state_init = sdfg.add_state()
    state_middle = sdfg.add_state()
    state_end = sdfg.add_state()
    sdfg.add_edge(state_init, state_end,
                  dace.InterstateEdge(condition='(not ((N > 20) != 0))', assignments={'result': 'N'}))
    sdfg.add_edge(state_init, state_middle, dace.InterstateEdge(condition='((N > 20) != 0)'))
    sdfg.add_edge(state_middle, state_end, dace.InterstateEdge(assignments={'result': '20'}))

    res = DeadStateElimination().apply_pass(sdfg, {})
    assert res is None


def test_dse_inside_loop():
    sdfg = dace.SDFG('dse_inside_loop')
    sdfg.add_symbol('a', dace.int32)
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    start = sdfg.add_state(is_start_block=True)
    sdfg.add_node(loop)
    end = sdfg.add_state()
    sdfg.add_edge(start, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    s = loop.add_state(is_start_block=True)
    s1 = loop.add_state()
    s2 = loop.add_state()
    s3 = loop.add_state()
    e = loop.add_state()
    loop.add_edge(s, s1, dace.InterstateEdge('a > 0'))
    loop.add_edge(s, s2, dace.InterstateEdge('a >= a'))  # Always True
    loop.add_edge(s, s3, dace.InterstateEdge('a < 0'))
    loop.add_edge(s1, e, dace.InterstateEdge())
    loop.add_edge(s2, e, dace.InterstateEdge())
    loop.add_edge(s3, e, dace.InterstateEdge())

    DeadStateElimination().apply_pass(sdfg, {})
    assert set(sdfg.states()) == {start, s, s2, e, end}


def test_dse_inside_loop_conditional():
    sdfg = dace.SDFG('dse_inside_loop')
    sdfg.add_symbol('a', dace.int32)
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    start = sdfg.add_state(is_start_block=True)
    sdfg.add_node(loop)
    end = sdfg.add_state()
    sdfg.add_edge(start, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    s = loop.add_state(is_start_block=True)
    cond_block = ConditionalBlock('cond', sdfg, loop)
    loop.add_node(cond_block)
    b1 = ControlFlowRegion('b1', sdfg)
    b1.add_state()
    cond_block.add_branch(CodeBlock('a > 0'), b1)
    b2 = ControlFlowRegion('b2', sdfg)
    s2 = b2.add_state()
    cond_block.add_branch(CodeBlock('a >= a'), b2)
    b3 = ControlFlowRegion('b3', sdfg)
    b3.add_state()
    cond_block.add_branch(CodeBlock('a < 0'), b3)
    e = loop.add_state()
    loop.add_edge(s, cond_block, dace.InterstateEdge())
    loop.add_edge(cond_block, e, dace.InterstateEdge())

    DeadStateElimination().apply_pass(sdfg, {})
    assert set(sdfg.states()) == {start, s, s2, e, end}


def test_dse_dead_branches_match_declared_type():
    """``_find_dead_branches`` returns what its annotation promises: ``(Optional[CodeBlock], region)``.

    It used to append two-element *lists* while claiming ``List[Tuple[...]]``. Today's only consumers
    unpack and ``len()`` the result, so nothing broke -- but the annotation was a lie, and the guard of
    an ``else`` arm is ``None``, which the element type did not admit either.
    """
    sdfg = dace.SDFG('dse_dead_branch_types')
    sdfg.add_symbol('a', dace.int32)
    cond_block = ConditionalBlock('cond', sdfg)
    sdfg.add_node(cond_block, is_start_block=True)
    taken = ControlFlowRegion('taken', sdfg)
    taken.add_state()
    cond_block.add_branch(CodeBlock('a >= a'), taken)
    shadowed = ControlFlowRegion('shadowed', sdfg)
    shadowed.add_state()
    cond_block.add_branch(CodeBlock('a > 0'), shadowed)
    otherwise = ControlFlowRegion('otherwise', sdfg)
    otherwise.add_state()
    cond_block.add_branch(None, otherwise)

    dead = DeadStateElimination()._find_dead_branches(cond_block)

    # The first arm is always true, so both later arms -- including the guardless ``else`` -- are dead.
    assert [branch for _, branch in dead] == [shadowed, otherwise]
    for entry in dead:
        assert isinstance(entry, tuple), type(entry)
        cond, branch = entry
        assert cond is None or isinstance(cond, CodeBlock), type(cond)
        assert isinstance(branch, ControlFlowRegion), type(branch)


def test_dse_malformed_conditional_block():
    sdfg = dace.SDFG("dse_malformed_conditional_block")
    sdfg.add_symbol("a", dace.int32)
    condition = ConditionalBlock("cond", sdfg)
    sdfg.add_node(condition)

    branch_else = ControlFlowRegion("else", sdfg)
    branch_else.add_state()
    condition.add_branch(None, branch_else)

    branch_if = ControlFlowRegion("if", sdfg)
    branch_if.add_state()
    condition.add_branch(CodeBlock("a > 0"), branch_if)

    merge_state = sdfg.add_state("merge_state")
    sdfg.add_edge(condition, merge_state, dace.InterstateEdge())

    with pytest.raises(
            InvalidSDFGNodeError,
            match="Conditional block detected, where else branch is not the last branch.",
    ):
        DeadStateElimination().apply_pass(sdfg, {})


def test_dse_malformed_conditional_block_in_region_names_its_own_region():
    """The malformed-conditional error resolves its block id against the region the block lives in.

    ``block_id`` indexes the *parent region*. Passing only the SDFG made the error resolve the id
    against the SDFG's own block list, so a conditional at index 1 of a loop was reported as
    whichever top-level block happened to sit at index 1 -- here, an unrelated state.
    """
    sdfg = dace.SDFG('dse_malformed_conditional_in_loop')
    sdfg.add_symbol('a', dace.int32)
    entry = sdfg.add_state('entry', is_start_block=True)
    decoy = sdfg.add_state('decoy_block_at_index_one')
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(entry, decoy, dace.InterstateEdge())
    sdfg.add_edge(decoy, loop, dace.InterstateEdge())

    loop.add_state('preamble', is_start_block=True)
    condition = ConditionalBlock('cond', sdfg, loop)
    loop.add_node(condition)
    loop.add_edge(loop.node(0), condition, dace.InterstateEdge())
    branch_else = ControlFlowRegion('else', sdfg)
    branch_else.add_state()
    condition.add_branch(None, branch_else)
    branch_if = ControlFlowRegion('if', sdfg)
    branch_if.add_state()
    condition.add_branch(CodeBlock('a > 0'), branch_if)
    # ``add_node`` does not register the region, so the cfg ids are stale until this runs.
    sdfg.reset_cfg_list()

    # The conditional is block 1 of the loop, and an unrelated state is block 1 of the SDFG.
    assert condition.block_id == 1
    assert sdfg.node(1) is decoy

    with pytest.raises(InvalidSDFGNodeError) as excinfo:
        DeadStateElimination().apply_pass(sdfg, {})

    err = excinfo.value
    assert err.cfg is loop
    assert err.resolve_block() is condition
    assert err.to_json()['cfg_id'] == loop.cfg_id
    # Formatting must not raise, and must not blame the unrelated top-level state.
    assert 'decoy_block_at_index_one' not in str(err)


def test_dde_simple():

    @dace.program
    def dde_tester(a: dace.float64[20], b: dace.float64[20]):
        c = a + b
        b[:] = a

    sdfg = dde_tester.to_sdfg()
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    # Access node with c should not exist
    assert all(n.data != 'c' for n in sdfg.node(0).data_nodes())


def test_dde_libnode():

    @dace.program
    def dde_tester(a: dace.float64[20], b: dace.float64[20]):
        c = a @ b
        b[:] = a

    sdfg = dde_tester.to_sdfg()
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    # Access node with c should not exist
    assert all(n.data != 'c' for n in sdfg.node(0).data_nodes())


@pytest.mark.parametrize('second_tasklet', (False, True))
def test_dde_access_node_in_scope(second_tasklet):
    sdfg = dace.SDFG('dde_anis')
    sdfg.add_transient('a', [20], dace.float64)
    state = sdfg.add_state()

    # map->map->tasklet[,tasklet]->mapexit->(accessnode)->mapexit
    ome, omx = state.add_map('outer', dict(i='0:10'))
    ime, imx = state.add_map('inner', dict(j='0:2'))
    t = state.add_tasklet('doit', {}, {'out'}, 'out = 1')
    w = state.add_write('a')
    state.add_memlet_path(ome, ime, t, memlet=dace.Memlet())
    state.add_memlet_path(t, imx, w, memlet=dace.Memlet('a[i*2+j]'), src_conn='out')
    state.add_nedge(w, omx, dace.Memlet())

    if second_tasklet:
        t2 = state.add_tasklet('stateful', {}, {}, '', side_effects=True)
        state.add_nedge(ime, t2, dace.Memlet())
        state.add_nedge(t2, imx, dace.Memlet())

    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    if second_tasklet:
        assert set(state.nodes()) == {ome, ime, t2, imx, omx}
    else:
        assert state.number_of_nodes() == 0
    sdfg.validate()


def test_dde_connectors():
    """ Tests removal of connectors on tasklets and nested SDFGs. """

    @dace.program
    def dde_conntest(a: dace.float64[20], b: dace.float64[20]):
        c = dace.ndarray([20], dace.float64)
        for i in dace.map[0:20]:
            with dace.tasklet:
                inp << a[i]

                out1 = inp * 2
                out2 = inp + 3

                out1 >> c[i]
                out2 >> b[i]

    sdfg = dde_conntest.to_sdfg()  # Nested SDFG or tasklet depends on simplify configuration
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert all('c' not in [n.data for n in state.data_nodes()] for state in sdfg.nodes())
    sdfg.validate()


def test_dde_scope_reconnect():
    """
    Corner case:
    map {
        tasklet(callback()) -> tasklet(do nothing)
    }
    expected map to stay connected
    """
    sdfg = dace.SDFG('dde_scope_tester')
    sdfg.add_symbol('cb', dace.callback(dace.float64))
    sdfg.add_scalar('s', dace.float64, transient=True)

    state = sdfg.add_state()
    me, mx = state.add_map('doit', dict(i='0:2'))
    # Tasklet has a callback and cannot be removed
    t1 = state.add_tasklet('callback', {}, {'o'}, 'o = cb()', side_effects=True)
    # Tasklet has no output and thus can be removed
    t2 = state.add_tasklet('nothing', {'inp'}, {}, '')
    state.add_nedge(me, t1, dace.Memlet())
    state.add_edge(t1, 'o', t2, 'inp', dace.Memlet('s'))
    state.add_nedge(t2, mx, dace.Memlet())

    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert set(state.nodes()) == {me, t1, mx}
    sdfg.validate()


@pytest.mark.parametrize('libnode', (False, True))
def test_dde_inout(libnode):
    """ Tests nested SDFG with the same array as input and output. """
    sdfg = dace.SDFG('dde_inout')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_transient('b', [20], dace.float64)
    state = sdfg.add_state()

    if not libnode:

        @dace.program
        def nested(a: dace.float64[20], b: dace.float64[20]):
            for i in range(1, 20):
                b[i] = a[i - 1] + 1
            for i in range(19):
                a[i] = b[i + 1] + 1

        nsdfg = nested.to_sdfg(simplify=False)
        node = state.add_nested_sdfg(nsdfg, {'b'}, {'a', 'b'})
        outconn = 'b'
    else:
        node = dace.nodes.LibraryNode('tester')  # Library node without side effects
        node.add_in_connector('b')
        node.add_out_connector('a')
        node.add_out_connector('bout')
        state.add_node(node)
        outconn = 'bout'  # Library node has different output connector name

    state.add_edge(node, 'a', state.add_write('a'), None, dace.Memlet('a'))
    state.add_edge(state.add_read('b'), None, node, 'b', dace.Memlet('b'))
    state.add_edge(node, outconn, state.add_write('b'), None, dace.Memlet('b'))

    assert sorted([n.data for n in state.data_nodes()]) == ['a', 'b', 'b']
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert sorted([n.data for n in state.data_nodes()]) == ['a', 'b', 'b']
    sdfg.validate()


def test_dde_inout_two_states():
    """Test two states with read/write in second state."""

    sdfg = dace.SDFG("dde_inout_two_states")
    sdfg.add_scalar("tmp", dace.float32)
    sdfg.add_scalar("computed", dace.float32, transient=True)

    start_state = sdfg.add_state("start_state", is_start_block=True)
    s1_read_tmp = start_state.add_read("tmp")
    s1_write_computed = start_state.add_write("computed")

    # upstream tasklet that writes a transient (to be read in a separate state)
    first_tasklet = start_state.add_tasklet("write", {"read_tmp"}, {"write_computed"},
                                            "write_computed = read_tmp * 2 + 1")
    start_state.add_memlet_path(s1_read_tmp, first_tasklet, dst_conn="read_tmp", memlet=dace.Memlet(data="tmp"))
    start_state.add_memlet_path(first_tasklet,
                                s1_write_computed,
                                src_conn="write_computed",
                                memlet=dace.Memlet(data="computed"))

    next_state = sdfg.add_state_after(start_state, "next_state")
    s2_write_computed = next_state.add_write("computed")
    s2_read_computed = next_state.add_read("computed")
    s2_write_tmp = next_state.add_write("tmp")

    # downstream tasklet that reads _and_ writes a transient
    second_tasklet = next_state.add_tasklet(
        "read_write", {"read_computed"}, {"write_tmp", "write_computed"},
        "write_computed = 2 * read_computed\nwrite_tmp = write_computed + read_computed")
    next_state.add_memlet_path(s2_read_computed,
                               second_tasklet,
                               dst_conn="read_computed",
                               memlet=dace.Memlet(data="computed"))
    next_state.add_memlet_path(second_tasklet, s2_write_tmp, src_conn="write_tmp", memlet=dace.Memlet(data="tmp"))
    next_state.add_memlet_path(second_tasklet,
                               s2_write_computed,
                               src_conn="write_computed",
                               memlet=dace.Memlet(data="computed"))

    results = {}
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, results)

    dde_results = results["DeadDataflowElimination"][0]
    assert dde_results.get(start_state) is None, "No changes to `start_state` expected."
    expected_cleanup = dde_results.get(next_state)
    assert expected_cleanup == {s2_write_computed}, "Expected to clean up write to `computed` from `next_state`."


def test_dce():
    """ End-to-end test evaluating both dataflow and state elimination. """
    # Code should end up as b[:] = a + 2; b += 1
    @dace.program
    def dce_tester(a: dace.float64[20], b: dace.float64[20]):
        c = a + 1
        d = 4
        e = 5
        f = c + e
        # Unused branch
        if a[0] > 1:
            b[:] = 5
        b[:] = a + 1  # Ends up overwritten
        b[:] = a + 2
        b += 1

    sdfg = dce_tester.to_sdfg(simplify=False)
    result = Pipeline([DeadDataflowElimination(), DeadStateElimination()]).apply_pass(sdfg, {})
    sdfg.simplify()
    assert sdfg.number_of_nodes() <= 4

    # Check that arrays were removed
    assert all('c' not in [n.data for n in state.data_nodes()] for state in sdfg.nodes())
    assert any('f' in [n.data for n in rstate if isinstance(n, dace.nodes.AccessNode)]
               for rstate in result[DeadDataflowElimination.__name__][0].values())


def test_dce_callback():

    def dace_inhibitor(f):
        return f

    @dace_inhibitor
    def callback(arr):
        # Do something with the array
        print(arr)

    @dace.program
    def dce_cb(a: dace.float64[20]):
        callback(a)

    with pytest.warns(match="Automatically creating callback"):
        sdfg = dce_cb.to_sdfg()
    num_tasklets_before = len([n for n, p in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)])
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    num_tasklets_after = len([n for n, p in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)])
    assert num_tasklets_after > 0 and num_tasklets_after == num_tasklets_before


def test_dce_callback_manual():
    sdfg = dace.SDFG('dce_cbman')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_symbol('cb', dace.callback(None, dace.float64))
    state = sdfg.add_state()

    r = state.add_read('a')
    t = state.add_tasklet('callback', {'inp'}, {}, 'cb(inp)')
    state.add_edge(r, None, t, 'inp', dace.Memlet('a[0:20]'))

    sdfg.validate()
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert set(state.nodes()) == {r, t}
    sdfg.validate()


@pytest.mark.parametrize('dtype', (dace.float64, dace.bool, np.float64))
def test_dce_add_type_hint_of_variable(dtype):
    """
    The code of this test comes from this issue: https://github.com/spcl/dace/issues/1150#issue-1445418361
    and this issue: https://github.com/spcl/dace/issues/1710
    and this PR: https://github.com/spcl/dace/pull/1721
    """
    if dtype is dace.bool:
        true_value = True
        false_value = False
    else:
        true_value = 3.0
        false_value = 7.0

    sdfg = dace.SDFG("test")
    state = sdfg.add_state()
    sdfg.add_array("out", dtype=dtype, shape=(10, ))
    sdfg.add_array("cond", dtype=dace.bool, shape=(10, ))
    sdfg.add_array("tmp", dtype=dtype, shape=(10, ), transient=True)
    tasklet, *_ = state.add_mapped_tasklet(
        code=f"""
if _cond:
    _tmp = {true_value}
else:
    _tmp = {false_value}
_out = _tmp
        """,
        inputs={"_cond": dace.Memlet(subset="k", data="cond")},
        outputs={
            "_out": dace.Memlet(subset="k", data="out"),
            "_tmp": dace.Memlet(subset="k", data="tmp"),
        },
        map_ranges={"k": "0:10"},
        name="test_tasklet",
        external_edges=True,
    )
    sdfg.simplify()
    assert tasklet.code.as_string.startswith("_tmp:")

    compiledsdfg = sdfg.compile()
    cond = np.random.choice(a=[True, False], size=(10, ))
    if isinstance(dtype, dace.typeclass):
        out = np.zeros((10, ), dtype=dtype.as_numpy_dtype())
    else:
        out = np.zeros((10, ), dtype=dtype)

    compiledsdfg(cond=cond, out=out)
    assert np.all(out == np.where(cond, true_value, false_value))


def test_prune_single_branch_conditional_block():
    sdfg = dace.SDFG("conditional_sdfg")

    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["b"].transient = True

    first_state = sdfg.add_state("first_state")
    first_state.add_mapped_tasklet(
        "first_comp",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i0]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("b[__i0]")},
        external_edges=True,
    )

    # create states inside the nested SDFG for the if-branches
    if_region = dace.sdfg.state.ConditionalBlock("if")
    sdfg.add_node(if_region)
    sdfg.add_edge(first_state, if_region, dace.InterstateEdge())

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=sdfg)
    then_state = then_body.add_state("true_branch", is_start_block=True)
    if_region.add_branch(dace.sdfg.state.CodeBlock("True"), then_body)
    then_state.add_mapped_tasklet(
        "second_comp",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("b[__i0]")},
        code="__out = __in1 + 1.0",
        outputs={"__out": dace.Memlet("c[__i0]")},
        external_edges=True,
    )
    sdfg.validate()
    res = DeadStateElimination().apply_pass(sdfg, {})
    assert res and len(res) == 1
    assert sdfg.out_edges(first_state)[0].dst == then_body


def test_dde_loop_condition():

    @dace.program
    def loop_condition(A: dace.float64[20]):
        for i in dace.map[0:20]:
            f = 0.0
            while f < 10.0:
                with dace.tasklet:
                    a << A[i]
                    f_out = a + 1.0
                    b = f_out
                    f_out >> f
                    b >> A[i]

    sdfg = loop_condition.to_sdfg(simplify=False)
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    count_f_nodes = sum(1 for a, _ in sdfg.all_nodes_recursive()
                        if isinstance(a, dace.nodes.AccessNode) and a.data == 'f')
    assert count_f_nodes == 2


def _two_writes_to_one_transient(second_write_range: str) -> Tuple[dace.SDFG, dace.SDFGState, nodes.AccessNode]:
    """``B[:] = 0`` then ``B[:, <range>] = A``, read back through a second access node.

    The two ``B`` access nodes are joined by no path -- the initializer's only out-edge is an
    ordering edge to ``A`` -- so deadness cannot be read off its out-edges.
    """
    sdfg = dace.SDFG('dde_two_writes')
    sdfg.add_array('A', [6, 6], dace.float64)
    sdfg.add_array('out', [6, 6], dace.float64)
    sdfg.add_transient('B', [6, 6], dace.float64)
    state = sdfg.add_state()

    init = state.add_tasklet('init', {}, {'o'}, 'o = 0.0')
    ient, iex = state.add_map('init_map', dict(i='0:6', j='0:6'))
    b_init = state.add_access('B')
    state.add_nedge(ient, init, dace.Memlet())
    state.add_memlet_path(init, iex, b_init, src_conn='o', memlet=dace.Memlet('B[i, j]'))

    a_read = state.add_access('A')
    state.add_nedge(b_init, a_read, dace.Memlet())

    copy = state.add_tasklet('copy', {'inp'}, {'o'}, 'o = inp')
    went, wex = state.add_map('write_map', dict(i='0:6', j=second_write_range))
    b_part = state.add_access('B')
    state.add_memlet_path(a_read, went, copy, dst_conn='inp', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(copy, wex, b_part, src_conn='o', memlet=dace.Memlet('B[i, j]'))

    read = state.add_tasklet('read', {'inp'}, {'o'}, 'o = inp')
    rent, rex = state.add_map('read_map', dict(i='0:6', j='0:6'))
    out = state.add_access('out')
    state.add_memlet_path(b_part, rent, read, dst_conn='inp', memlet=dace.Memlet('B[i, j]'))
    state.add_memlet_path(read, rex, out, src_conn='o', memlet=dace.Memlet('out[i, j]'))

    sdfg.validate()
    return sdfg, state, b_init


def test_dde_keeps_an_initializer_the_second_write_does_not_cover():
    sdfg, state, b_init = _two_writes_to_one_transient('1:6')
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})

    assert b_init in state.nodes(), 'the zero-init was removed although column 0 is still read'
    assert any(isinstance(n, nodes.Tasklet) and n.label == 'init' for n in state.nodes())

    A = np.random.rand(6, 6)
    out = np.full((6, 6), 12345.0)
    expected = A.copy()
    expected[:, 0] = 0.0
    sdfg(A=A.copy(), out=out)
    assert np.allclose(out, expected)


def test_dde_removes_an_initializer_the_second_write_fully_covers():
    """Control: the same shape, but the second write covers all of ``B``."""
    sdfg, state, b_init = _two_writes_to_one_transient('0:6')
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})

    assert b_init not in state.nodes()
    assert not any(isinstance(n, nodes.Tasklet) and n.label == 'init' for n in state.nodes())


def _init_then_reduce(identity) -> Tuple[dace.SDFG, dace.SDFGState, nodes.AccessNode]:
    """``s = 0`` then ``Reduce(s <- sum(A))``, read back through a second ``s`` access node.

    The two ``s`` nodes are joined only by an ordering edge, so deadness of the initializer is
    decided by whether the reduction's write is a full overwrite of ``s``.
    """
    sdfg = dace.SDFG('dde_init_then_reduce')
    sdfg.add_array('A', [6], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_transient('s', [1], dace.float64)
    state = sdfg.add_state()

    init = state.add_tasklet('init', {}, {'o'}, 'o = 0.0')
    s_init = state.add_access('s')
    state.add_edge(init, 'o', s_init, None, dace.Memlet('s[0]'))

    a_read = state.add_access('A')
    state.add_nedge(s_init, a_read, dace.Memlet())

    red = state.add_reduce('lambda a, b: a + b', None, identity)
    s_red = state.add_access('s')
    state.add_edge(a_read, None, red, '_in', dace.Memlet('A[0:6]'))
    state.add_edge(red, '_out', s_red, None, dace.Memlet('s[0]'))

    copy = state.add_tasklet('copy', {'inp'}, {'o'}, 'o = inp')
    out = state.add_access('out')
    state.add_edge(s_red, None, copy, 'inp', dace.Memlet('s[0]'))
    state.add_edge(copy, 'o', out, None, dace.Memlet('out[0]'))

    sdfg.validate()
    return sdfg, state, s_init


def test_dde_keeps_an_initializer_a_reduction_without_identity_folds_onto():
    """A reduction with no identity has nothing to start the fold from, so it accumulates onto
    whatever ``s`` already holds. Removing the zero-init leaves it summing into garbage."""
    sdfg, state, s_init = _init_then_reduce(None)
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})

    assert s_init in state.nodes(), 'the zero-init was removed although the reduction folds onto it'
    assert any(isinstance(n, nodes.Tasklet) and n.label == 'init' for n in state.nodes())

    A = np.random.rand(6)
    out = np.full((1, ), 12345.0)
    sdfg(A=A.copy(), out=out)
    assert np.allclose(out, A.sum())


def test_dde_removes_an_initializer_a_reduction_with_identity_overwrites():
    """Control: an identity makes the expansion emit its own init, so the write covers all of
    ``s`` and the upstream zero-init is genuinely dead."""
    sdfg, state, s_init = _init_then_reduce(0.0)
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})

    assert s_init not in state.nodes()
    assert not any(isinstance(n, nodes.Tasklet) and n.label == 'init' for n in state.nodes())


def test_dde_keeps_an_ordering_edge_whose_producer_is_live():
    """A dead transient's empty out-edge orders a LIVE producer before a reader."""
    sdfg = dace.SDFG('dde_live_ordering')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    sdfg.add_transient('t', [1], dace.float64)
    state = sdfg.add_state()

    src = state.add_access('A')
    me, mx = state.add_map('reader', dict(i='0:4'))
    tasklet = state.add_tasklet('r', {'x': None}, {'z': None}, 'z = x')
    state.add_memlet_path(src, me, tasklet, dst_conn='x', memlet=dace.Memlet('A[0]'))
    state.add_memlet_path(tasklet, mx, state.add_access('B'), src_conn='z', memlet=dace.Memlet('B[i]'))

    # 'w' survives on its A[0] output; 't' is never read, so DDE removes it.
    w = state.add_tasklet('w', {}, {'o': None, 'p': None}, 'o = 100.0\np = 999.0')
    state.add_edge(w, 'o', state.add_access('A'), None, dace.Memlet('A[0]'))
    tnode = state.add_access('t')
    state.add_edge(w, 'p', tnode, None, dace.Memlet('t[0]'))
    state.add_edge(tnode, None, src, None, dace.Memlet())
    sdfg.validate()

    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    sdfg.validate()
    assert state.in_degree(src) == 1, 'the happens-before edge into the reader source was dropped'

    A = np.zeros(4)
    B = np.zeros(4)
    sdfg(A=A, B=B)
    assert np.allclose(B, 100.0)


if __name__ == '__main__':
    test_dse_simple()
    test_dse_unconditional()
    test_dse_edge_condition_with_integer_as_boolean_regression()
    test_dse_inside_loop()
    test_dse_inside_loop_conditional()
    test_dse_dead_branches_match_declared_type()
    test_dse_malformed_conditional_block()
    test_dse_malformed_conditional_block_in_region_names_its_own_region()
    test_dde_simple()
    test_dde_libnode()
    test_dde_access_node_in_scope(False)
    test_dde_access_node_in_scope(True)
    test_dde_connectors()
    test_dde_scope_reconnect()
    test_dde_inout(False)
    test_dde_inout(True)
    test_dde_inout_two_states()
    test_dce()
    test_dce_callback()
    test_dce_callback_manual()
    test_dce_add_type_hint_of_variable(dace.float64)
    test_dce_add_type_hint_of_variable(dace.bool)
    test_dce_add_type_hint_of_variable(np.float64)
    test_dde_keeps_an_initializer_a_reduction_without_identity_folds_onto()
    test_dde_removes_an_initializer_a_reduction_with_identity_overwrites()
    test_prune_single_branch_conditional_block()
    test_dde_loop_condition()
    test_dde_keeps_an_initializer_the_second_write_does_not_cover()
    test_dde_removes_an_initializer_the_second_write_fully_covers()
    test_dde_keeps_an_ordering_edge_whose_producer_is_live()
