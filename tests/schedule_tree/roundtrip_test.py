# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests conversion of schedule trees to SDFGs.
"""
import dace
import numpy as np
import pytest

from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s, treenodes as tn
from dace.properties import CodeBlock
from dace.sdfg.state import BreakBlock, ConditionalBlock, LoopRegion, NamedRegion, ReturnBlock
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising


def _roundtrip(sdfg: dace.SDFG,
               expected_node_type: type,
               state_boundary_behavior: t2s.StateBoundaryBehavior = t2s.StateBoundaryBehavior.STATE_TRANSITION,
               simplify: bool | None = None) -> dace.SDFG:
    """
    Converts an SDFG to a schedule tree and back, ensuring the tree contains a node of the given type.

    Unless ``simplify`` is given, simplification of the resulting SDFG follows the configuration, so that both the
    simplified and the unsimplified variants are covered by the CI matrix rather than by an extra test parameter. Tests
    that check the structure of the unsimplified SDFG must pass ``simplify=False``, since the ``DACE_*`` environment
    variables set by CI take precedence over ``dace.config.set_temporary``.
    """
    stree = sdfg.as_schedule_tree()
    assert any(type(node) is expected_node_type for node in stree.preorder_traversal())
    if simplify is None:
        simplify = dace.config.Config.get_bool('optimizer', 'automatic_simplification')
    return stree.as_sdfg(simplify=simplify, state_boundary_behavior=state_boundary_behavior)


def _roundtrip_and_compare(
        sdfg: dace.SDFG,
        expected_node_type: type,
        *arguments: dict,
        state_boundary_behavior: t2s.StateBoundaryBehavior = t2s.StateBoundaryBehavior.STATE_TRANSITION,
        simplify: bool | None = None) -> dace.SDFG:
    """
    Converts an SDFG to a schedule tree and back, ensuring the tree contains a node of the given type, and that both
    SDFGs compute the same outputs and return values for each of the given argument sets (on copies of the arrays).
    """
    new_sdfg = _roundtrip(sdfg, expected_node_type, state_boundary_behavior, simplify)
    new_sdfg.name = f'{sdfg.name}_roundtrip'  # Avoid overwriting the compiled original SDFG
    compiled = sdfg.compile()
    new_compiled = new_sdfg.compile()

    for args in arguments:
        expected_args = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in args.items()}
        actual_args = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in args.items()}
        expected_result = compiled(**expected_args)
        actual_result = new_compiled(**actual_args)

        for name, expected in expected_args.items():
            if isinstance(expected, np.ndarray):
                assert np.allclose(actual_args[name], expected), f'Argument "{name}" differs for {args}'
        if expected_result is None:
            assert actual_result is None
        else:
            expected_result = expected_result if isinstance(expected_result, tuple) else (expected_result, )
            actual_result = actual_result if isinstance(actual_result, tuple) else (actual_result, )
            assert len(actual_result) == len(expected_result)
            for expected, actual in zip(expected_result, actual_result):
                assert np.allclose(actual, expected), f'Return value differs for {args}'

    return new_sdfg


def test_implicit_inline_and_constants():
    """
    Tests implicit inlining upon roundtrip conversion, as well as constants with conflicting names.
    """

    @dace
    def nester(A: dace.float64[20]):
        A[:] = 12

    @dace.program
    def tester(A: dace.float64[20, 20]):
        for i in dace.map[0:20]:
            nester(A[:, i])

    sdfg = tester.to_sdfg(simplify=False)

    # Inject constant into nested SDFG
    assert len(list(sdfg.all_sdfgs_recursive())) > 1
    sdfg.add_constant('cst', 13)  # Add an unused constant
    sdfg.cfg_list[-1].add_constant('cst', 1, dace.data.Scalar(dace.float64))
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code.as_string = tasklet.code.as_string.replace('12', 'cst')

    # Perform a roundtrip conversion
    stree = sdfg.as_schedule_tree()
    new_sdfg = stree.as_sdfg()

    assert len(list(new_sdfg.all_sdfgs_recursive())) == 1
    assert new_sdfg.constants['cst_0'].dtype == np.float64

    # Test SDFG
    a = np.random.rand(20, 20)
    new_sdfg(A=a)  # Tests arg_names
    assert np.allclose(a, 1)


def test_name_propagation():
    name = "my_complicated_sdfg_test_name"
    sdfg = dace.SDFG(name)
    sdfg.add_state("empty", is_start_block=True)

    stree = sdfg.as_schedule_tree()
    assert stree.name == name

    sdfg = stree.as_sdfg()
    assert sdfg.name == name


def test_view_of_slice():

    @dace.program
    def tester(a: dace.float64[30]):
        b = a[1:21]
        b[:] = 5

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode)

    a = np.random.rand(30)
    expected = a.copy()
    expected[1:21] = 5
    new_sdfg(a=a)
    assert np.allclose(a, expected)


def test_view_read_after_write():

    @dace.program
    def tester(a: dace.float64[30]):
        b = a[5:25]
        b[3] = b[2] + a[7]
        a[9] = b[3] * 2

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode)

    a = np.random.rand(30)
    expected = a.copy()
    expected[8] = expected[7] + expected[7]
    expected[9] = expected[8] * 2
    new_sdfg(a=a)
    assert np.allclose(a, expected)


def test_view_of_view():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20, 20], dace.float64)
    sdfg.add_view('Av', [400], dace.float64)
    sdfg.add_view('Avv', [10, 40], dace.float64)
    sdfg.add_view('Bv', [400], dace.float64)
    sdfg.add_view('Bvv', [10, 40], dace.float64)
    state = sdfg.add_state()
    av = state.add_access('Av')
    bv = state.add_access('Bv')
    bvv = state.add_access('Bvv')
    avv = state.add_access('Avv')
    state.add_edge(state.add_read('A'), None, av, None, dace.Memlet('A[0:20, 0:20]'))
    state.add_edge(av, None, avv, 'views', dace.Memlet('Av[0:400]'))
    state.add_edge(avv, None, bvv, None, dace.Memlet('Avv[0:10, 0:40]'))
    state.add_edge(bvv, 'views', bv, None, dace.Memlet('Bv[0:400]'))
    state.add_edge(bv, 'views', state.add_write('B'), None, dace.Memlet('Bv[0:400]'))

    new_sdfg = _roundtrip(sdfg, tn.ViewNode)

    a = np.random.rand(20, 20)
    b = np.random.rand(20, 20)
    new_sdfg(A=a, B=b)
    assert np.allclose(a, b)


def test_view_in_map_scope():

    @dace.program
    def tester(a: dace.float64[10, 30], b: dace.float64[10]):
        for i in dace.map[0:10]:
            r = a[i]
            b[i] = r[4] + r[5]

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode)

    a = np.random.rand(10, 30)
    b = np.random.rand(10)
    new_sdfg(a=a, b=b)
    assert np.allclose(b, a[:, 4] + a[:, 5])


def test_view_write_in_map_scope():

    @dace.program
    def tester(a: dace.float64[10, 30]):
        for i in dace.map[0:10]:
            r = a[i]
            r[3] = i

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode)

    a = np.random.rand(10, 30)
    expected = a.copy()
    expected[:, 3] = np.arange(10)
    new_sdfg(a=a)
    assert np.allclose(a, expected)


def test_view_passed_to_nested_sdfg():

    @dace.program
    def nested(x: dace.float64[20]):
        x[:] = x + 1

    @dace.program
    def tester(a: dace.float64[30]):
        nested(a[1:21])

    new_sdfg = _roundtrip(tester.to_sdfg(simplify=False), tn.ViewNode)

    a = np.random.rand(30)
    expected = a.copy()
    expected[1:21] += 1
    new_sdfg(a=a)
    assert np.allclose(a, expected)


def test_dynamic_map_range():
    H = dace.symbol('H')
    nnz = dace.symbol('nnz')

    @dace.program
    def tester(A_row: dace.uint32[H + 1], A_val: dace.float32[nnz], b: dace.float32[H]):
        for i in dace.map[0:H]:
            for j in dace.map[A_row[i]:A_row[i + 1]]:
                b[i] += A_val[j]

    new_sdfg = _roundtrip(tester.to_sdfg(), tn.DynScopeCopyNode)

    A_row = np.array([0, 2, 3, 5], dtype=np.uint32)
    A_val = np.random.rand(5).astype(np.float32)
    b = np.zeros(3, dtype=np.float32)
    new_sdfg(A_row=A_row, A_val=A_val, b=b, H=3, nnz=5)
    assert np.allclose(b, [A_val[0] + A_val[1], A_val[2], A_val[3] + A_val[4]])


def test_reference_set():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('n', dace.int32)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_array('C', [20], dace.float64)
    sdfg.add_reference('ref', [20], dace.float64)

    init = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    end = sdfg.add_state()
    sdfg.add_edge(init, s1, dace.InterstateEdge('n > 0'))
    sdfg.add_edge(init, s2, dace.InterstateEdge('n <= 0'))
    sdfg.add_edge(s1, end, dace.InterstateEdge())
    sdfg.add_edge(s2, end, dace.InterstateEdge())

    s1.add_edge(s1.add_access('A'), None, s1.add_access('ref'), 'set', dace.Memlet('A[0:20]'))
    s2.add_edge(s2.add_access('B'), None, s2.add_access('ref'), 'set', dace.Memlet('B[0:20]'))
    end.add_nedge(end.add_access('ref'), end.add_access('C'), dace.Memlet('ref[0:20]'))

    FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})
    new_sdfg = _roundtrip(sdfg, tn.RefSetNode)

    a = np.random.rand(20)
    b = np.random.rand(20)
    c = np.random.rand(20)
    new_sdfg(A=a, B=b, C=c, n=1)
    assert np.allclose(c, a)
    new_sdfg(A=a, B=b, C=c, n=0)
    assert np.allclose(c, b)


def test_library_call():

    @dace.program
    def tester(a: dace.float64[5, 4], b: dace.float64[4, 3]):
        return a @ b

    new_sdfg = _roundtrip(tester.to_sdfg(), tn.LibraryCall)

    a = np.random.rand(5, 4)
    b = np.random.rand(4, 3)
    assert np.allclose(new_sdfg(a=a, b=b), a @ b)


def _inverted_loop_sdfg(name: str, loop: LoopRegion) -> dace.SDFG:
    """
    Creates an SDFG that increments ``A[i]`` in the body of the given (inverted) loop.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={} if loop.loop_variable else {'i': '0'}))
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('increment', {'inp'}, {'out'}, 'out = inp + 1')
    body.add_edge(body.add_read('A'), None, tasklet, 'inp', dace.Memlet('A[i]'))
    body.add_edge(tasklet, 'out', body.add_write('A'), None, dace.Memlet('A[i]'))
    if not loop.loop_variable:
        loop.add_state_after(body, 'increment_i', assignments={'i': 'i + 1'})
    return sdfg


def test_do_while_loop():
    # The condition never holds, the body is executed once
    sdfg = _inverted_loop_sdfg('tester', LoopRegion('loop', 'i < 0', inverted=True))
    new_sdfg = _roundtrip(sdfg, tn.DoWhileScope)

    a = np.zeros(10)
    new_sdfg(A=a)
    assert np.allclose(a, [1] + [0] * 9)


@pytest.mark.parametrize('update_before_condition', (False, True))
def test_do_for_loop(update_before_condition: bool):
    loop = LoopRegion('loop',
                      'i < 3',
                      'i',
                      'i = 0',
                      'i = i + 1',
                      inverted=True,
                      update_before_condition=update_before_condition)
    sdfg = _inverted_loop_sdfg('tester', loop)
    _roundtrip_and_compare(sdfg, tn.LoopScope, dict(A=np.zeros(10)))


def test_transients_and_nested_sdfg() -> None:

    def nestedSDFG() -> dace.SDFG:

        def get_start_state(sdfg: dace.SDFG) -> dace.SDFGState:
            state = sdfg.add_state("my_state", is_start_block=True)
            read = state.add_read("B")
            write = state.add_write("tmp_condition")
            tasklet = state.add_tasklet("masklet", inputs={"B0"}, outputs={"out"}, code="out = B0 < 0")
            state.add_edge(read, None, tasklet, "B0", dace.Memlet("B[0]"))
            state.add_edge(tasklet, "out", write, None, dace.Memlet("tmp_condition[0]"))
            return state

        def get_if_block(sdfg: dace.SDFG) -> dace.sdfg.ControlFlowRegion:
            if_block = ConditionalBlock('if_region', sdfg=sdfg)
            then_body = dace.sdfg.ControlFlowRegion('then_body', sdfg=sdfg, parent=if_block)
            then_state = then_body.add_state('then_state', is_start_block=True)
            then_write = then_state.add_write('B')
            then_tasklet = then_state.add_tasklet('write_zero', {}, {'out'}, 'out = 0')
            then_state.add_edge(then_tasklet, 'out', then_write, None, dace.Memlet('B[0]'))
            if_block.add_branch("tmp_condition", then_body)
            return if_block

        def get_map_state(sdfg: dace.SDFG) -> dace.SDFGState:
            state = sdfg.add_state("map_state")
            access_A = state.add_access("A")
            write_B = state.add_write("B")
            state.add_mapped_tasklet("write_one", {"j": dace.subsets.Range.from_string("0:10")}, {},
                                     "out = 1.0", {"out": dace.Memlet("A[15*i + 3*j]")},
                                     external_edges=True,
                                     output_nodes={"A": access_A})
            state.add_mapped_tasklet("copy", {"k": dace.subsets.Range.from_string("10:20")},
                                     {"read": dace.Memlet("A[k]")},
                                     "write = read", {"write": dace.Memlet("B[k]")},
                                     external_edges=True,
                                     input_nodes={"A": access_A},
                                     output_nodes={"B": write_B})
            return state

        sdfg = dace.SDFG(name="nested")
        sdfg.add_scalar("tmp_condition", dace.bool, transient=True)
        sdfg.add_array("A", [60], dace.float32)
        sdfg.add_array("B", [60], dace.float32)

        start_state = get_start_state(sdfg)

        if_block = get_if_block(sdfg)
        sdfg.add_node(if_block)
        sdfg.add_edge(start_state, if_block, dace.InterstateEdge())

        map_state = get_map_state(sdfg)
        sdfg.add_edge(if_block, map_state, dace.InterstateEdge())

        return sdfg

    sdfg = dace.SDFG(name="tester")
    _, A_desc = sdfg.add_array("A", [60], dace.float32, transient=True)
    _, B_desc = sdfg.add_array("B", [60], dace.float32)
    state = sdfg.add_state("state")
    access_A = state.add_access("A")
    state.add_mapped_tasklet("fill", {"i": dace.subsets.Range.from_string("0:60")},
                             inputs={},
                             code="out = 42.42",
                             outputs={"out": dace.Memlet("A[i]")},
                             external_edges=True,
                             output_nodes={"A": access_A})

    read_B = state.add_read("B")
    map_entry, map_exit = state.add_map("second_map", {"i": dace.subsets.Range.from_string("0:2")})

    # map_entry
    map_entry.add_in_connector("IN_A")
    map_entry.add_out_connector("OUT_A")
    map_entry.add_in_connector("IN_B")
    map_entry.add_out_connector("OUT_B")
    state.add_edge(access_A, None, map_entry, "IN_A", dace.Memlet.from_array("A", A_desc))
    state.add_edge(read_B, None, map_entry, "IN_B", dace.Memlet.from_array("B", B_desc))

    # nested SDFG
    nsdfg = nestedSDFG()
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        inputs={
            "A": None,
            "B": None
        },
        outputs={
            "A": None,
            "B": None
        },
        name="nested_sdfg",
    )
    state.add_edge(map_entry, "OUT_A", nsdfg_node, "A", dace.Memlet.from_array("A", A_desc))
    state.add_edge(map_entry, "OUT_B", nsdfg_node, "B", dace.Memlet.from_array("B", B_desc))

    # map_exit
    map_exit.add_in_connector("IN_A")
    map_exit.add_out_connector("OUT_A")
    state.add_edge(nsdfg_node, "A", map_exit, "IN_A", dace.Memlet.from_array("A", A_desc))
    write_A = state.add_write("A")
    state.add_edge(map_exit, "OUT_A", write_A, None, dace.Memlet.from_array("A", A_desc))

    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")
    state.add_edge(nsdfg_node, "B", map_exit, "IN_B", dace.Memlet.from_array("B", B_desc))
    write_B = state.add_write("B")
    state.add_edge(map_exit, "OUT_B", write_B, None, dace.Memlet.from_array("B", B_desc))

    sdfg.validate()
    dace.sdfg.propagation.propagate_memlets_sdfg(sdfg)
    stree = sdfg.as_schedule_tree()
    roundtrip_sdfg = stree.as_sdfg(validate=True)

    assert roundtrip_sdfg.arrays["A"].transient
    assert not roundtrip_sdfg.arrays["B"].transient

    roundtrip_nested: dace.SDFG = list(filter(lambda node: node.label == "nested_sdfg", roundtrip_sdfg.cfg_list))[0]
    assert not roundtrip_nested.arrays["A"].transient

    tmp_condition = roundtrip_nested.symbols.get("tmp_condition", None)
    assert tmp_condition == dace.bool


def _write_tasklet(state: dace.SDFGState, code: str, inputs: dict[str, str], output: str) -> None:
    """
    Adds a tasklet with the given code, which reads the input memlets (by connector) and writes ``out`` to the output.
    """
    tasklet = state.add_tasklet('compute', set(inputs.keys()), {'out'}, code)
    for connector, memlet in inputs.items():
        state.add_edge(state.add_read(memlet.split('[')[0]), None, tasklet, connector, dace.Memlet(memlet))
    state.add_edge(tasklet, 'out', state.add_write(output.split('[')[0]), None, dace.Memlet(output))


def _add_conditional_return(region: dace.sdfg.state.ControlFlowRegion, condition: str,
                            after: dace.SDFGState) -> ConditionalBlock:
    """
    Adds a conditional block that returns if the condition holds to the start of a region, followed by ``after``.
    """
    block = ConditionalBlock('maybe_return')
    region.add_node(block, is_start_block=True)
    branch = dace.sdfg.ControlFlowRegion('return_branch', sdfg=region.sdfg if region.sdfg else region, parent=block)
    branch.add_node(ReturnBlock('return'), is_start_block=True)
    block.add_branch(CodeBlock(condition), branch)
    region.add_edge(block, after, dace.InterstateEdge())
    return block


def test_return_block():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [2], dace.float64)
    after = sdfg.add_state('after')
    _write_tasklet(after, 'out = 1', {}, 'A[1]')
    _add_conditional_return(sdfg, 'N > 5', after)

    _roundtrip_and_compare(sdfg, tn.GotoNode, dict(A=np.zeros(2), N=10), dict(A=np.zeros(2), N=0))


def test_conditional_edge_exit():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [2], dace.float64)
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    _write_tasklet(first, 'out = 1', {}, 'A[0]')
    _write_tasklet(second, 'out = 2', {}, 'A[1]')
    sdfg.add_edge(first, second, dace.InterstateEdge('N > 0'))

    _roundtrip_and_compare(sdfg, tn.StateIfScope, dict(A=np.zeros(2), N=1), dict(A=np.zeros(2), N=0))


def _nested_return_sdfg(in_map: bool, in_loop: bool) -> dace.SDFG:
    """
    Creates an SDFG that calls a nested SDFG that returns early, followed by a computation on its output.

    :param in_map: If True, the nested SDFG and the subsequent computation are in a map scope.
    :param in_loop: If True, the nested SDFG returns from within a loop.
    """
    inner = dace.SDFG('inner')
    inner.add_symbol('N', dace.int64)
    inner.add_array('X', [1], dace.float64)
    increment = dace.SDFGState('increment')
    if in_loop:
        # for j in range(3): if j >= N: return; X[0] += 1
        inner.add_symbol('j', dace.int64)
        loop = LoopRegion('loop', 'j < 3', 'j', 'j = 0', 'j = j + 1')
        inner.add_node(loop, is_start_block=True)
        loop.add_node(increment)
        _add_conditional_return(loop, 'j >= N', increment)
    else:
        # if N > 5: return; X[0] += 1
        inner.add_node(increment)
        _add_conditional_return(inner, 'N > 5', increment)
    _write_tasklet(increment, 'out = inp + 1', {'inp': 'X[0]'}, 'X[0]')

    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    nsdfg = state.add_nested_sdfg(inner, {'X'}, {'X'}, symbol_mapping={'N': 'N'})
    tasklet = state.add_tasklet('after_call', {'inp'}, {'out'}, 'out = inp + 10')
    index = 'i' if in_map else '0'
    written = state.add_access('A')
    state.add_edge(nsdfg, 'X', written, None, dace.Memlet(f'A[{index}]'))
    state.add_edge(written, None, tasklet, 'inp', dace.Memlet(f'A[{index}]'))
    if in_map:
        entry, exit_node = state.add_map('map', dict(i='0:10'))
        state.add_memlet_path(state.add_read('A'), entry, nsdfg, dst_conn='X', memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(tasklet, exit_node, state.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
        state.add_memlet_path(written, exit_node, state.add_write('A'), memlet=dace.Memlet('A[i]'))
    else:
        state.add_edge(state.add_read('A'), None, nsdfg, 'X', dace.Memlet('A[0]'))
        state.add_edge(tasklet, 'out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg


@pytest.mark.parametrize('in_loop', (False, True))
@pytest.mark.parametrize('in_map', (False, True))
def test_nested_sdfg_return(in_map: bool, in_loop: bool):
    sdfg = _nested_return_sdfg(in_map, in_loop)
    sdfg.validate()
    arguments = [dict(A=np.random.rand(10), B=np.zeros(10), N=n) for n in (0, 2, 10)]
    new_sdfg = _roundtrip_and_compare(sdfg, tn.StateLabel, *arguments)
    new_sdfg.validate()


def test_break_in_conditional():
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg = _inverted_loop_sdfg('tester', loop)
    sdfg.add_symbol('N', dace.int64)
    body = loop.start_block
    block = ConditionalBlock('maybe_break')
    loop.add_node(block, is_start_block=True)
    branch = dace.sdfg.ControlFlowRegion('break_branch', sdfg=sdfg, parent=block)
    branch.add_node(BreakBlock('break'), is_start_block=True)
    block.add_branch(CodeBlock('i == N'), branch)
    loop.add_edge(block, body, dace.InterstateEdge())

    _roundtrip_and_compare(sdfg, tn.BreakNode, dict(A=np.zeros(10), N=4), dict(A=np.zeros(10), N=20))


def _state_machine_if_else() -> dace.SDFG:
    """
    Creates an SDFG with an if/else that is expressed through conditional inter-state edges.
    """
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [2], dace.float64)
    guard = sdfg.add_state('guard', is_start_block=True)
    then_state = sdfg.add_state('then_state')
    else_state = sdfg.add_state('else_state')
    merge = sdfg.add_state('merge')
    _write_tasklet(then_state, 'out = 1', {}, 'A[0]')
    _write_tasklet(else_state, 'out = 2', {}, 'A[0]')
    _write_tasklet(merge, 'out = inp + 10', {'inp': 'A[0]'}, 'A[1]')
    sdfg.add_edge(guard, then_state, dace.InterstateEdge('N > 0'))
    sdfg.add_edge(guard, else_state, dace.InterstateEdge('N <= 0'))
    sdfg.add_edge(then_state, merge, dace.InterstateEdge())
    sdfg.add_edge(else_state, merge, dace.InterstateEdge())
    return sdfg


def test_state_machine_if_else():
    sdfg = _state_machine_if_else()
    _roundtrip_and_compare(sdfg, tn.GBlock, dict(A=np.zeros(2), N=1), dict(A=np.zeros(2), N=-1))


def test_state_machine_loop():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    guard = sdfg.add_state('guard')
    body = sdfg.add_state('body')
    after = sdfg.add_state('after')
    _write_tasklet(body, 'out = i', {}, 'A[i]')
    _write_tasklet(after, 'out = i + 100', {}, 'A[9]')
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': '0'}))
    sdfg.add_edge(guard, body, dace.InterstateEdge('i < N'))
    sdfg.add_edge(guard, after, dace.InterstateEdge('i >= N'))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))

    new_sdfg = _roundtrip_and_compare(sdfg, tn.GBlock, dict(A=np.zeros(10), N=5), dict(A=np.zeros(10), N=0))

    # The loop can be raised again after the conversion
    if not dace.config.Config.get_bool('optimizer', 'automatic_simplification'):
        dace.sdfg.utils.inline_control_flow_regions(new_sdfg)
        FixedPointPipeline([ControlFlowRaising()]).apply_pass(new_sdfg, {})
        assert any(isinstance(block, LoopRegion) for block in new_sdfg.all_control_flow_blocks(recursive=True))


def test_state_machine_in_loop():
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg = _inverted_loop_sdfg('tester', loop)
    sdfg.add_symbol('N', dace.int64)
    body = loop.start_block
    guard = loop.add_state('guard', is_start_block=True)
    skip = loop.add_state('skip')
    _write_tasklet(skip, 'out = 100', {}, 'A[9]')
    loop.add_edge(guard, body, dace.InterstateEdge('i != N'))
    loop.add_edge(guard, skip, dace.InterstateEdge('i == N'))

    _roundtrip_and_compare(sdfg, tn.GBlock, dict(A=np.zeros(10), N=2), dict(A=np.zeros(10), N=7))


def _fibonacci_consume_sdfg(chunked: bool) -> dace.SDFG:
    """
    Creates an SDFG that computes Fibonacci numbers by consuming a stream and pushing smaller values back into it.

    :param chunked: If True, consumes two elements at a time and stops once the result reaches 44.
    """
    sdfg = dace.SDFG('tester')
    sdfg.add_array('iv', [1], dace.int32)
    sdfg.add_stream('S', dace.int32, transient=True, buffer_size=256)
    sdfg.add_array('res', [1], dace.float32)
    state = sdfg.add_state('state')

    if chunked:
        entry, exit_node = state.add_consume('cons', ('p', '1'), 'res[0] >= 44', chunksize=2)
        code = """
for i in range(__dace_cons_numelems):
    if s[i] == 1:
        val = 1
    elif s[i] > 1:
        sout = s[i] - 1
        sout = s[i] - 2
"""
        element_memlet = dace.Memlet('S[0:2]')
        element_memlet.allow_oob = True
    else:
        entry, exit_node = state.add_consume('cons', ('p', '1'))
        code = """
if s == 1:
    val = 1
elif s > 1:
    sout = s - 1
    sout = s - 2
"""
        element_memlet = dace.Memlet('S[0]')
    tasklet = state.add_tasklet('fibonacci', {'s'}, {'sout', 'val'}, code)

    state.add_nedge(state.add_read('iv'), state.add_write('S'), dace.Memlet('S[0]'))
    stream_edge = state.add_edge(state.add_read('S'), None, entry, 'IN_stream', dace.Memlet('S[0]'))
    stream_edge.data.allow_oob = chunked
    state.add_edge(entry, 'OUT_stream', tasklet, 's', element_memlet)
    state.add_memlet_path(tasklet,
                          exit_node,
                          state.add_write('S'),
                          src_conn='sout',
                          memlet=dace.Memlet('S[0]', volume=-1))
    state.add_memlet_path(tasklet,
                          exit_node,
                          state.add_write('res'),
                          src_conn='val',
                          memlet=dace.Memlet('res[0]', wcr='lambda a, b: a + b', volume=-1))
    return sdfg


@pytest.mark.parametrize('chunked', (False, True))
def test_consume_fibonacci(chunked: bool):
    _roundtrip_and_compare(_fibonacci_consume_sdfg(chunked), tn.ConsumeScope,
                           dict(iv=np.array([10], np.int32), res=np.zeros(1, np.float32)))


def _consume_body_sdfg(in_map: bool, push: bool, multistate: bool) -> dace.SDFG:
    """
    Creates an SDFG that fills a stream from an array and consumes it with a nested SDFG, whose result is summed up.

    :param in_map: If True, the consume scope is in a map scope.
    :param push: If True, the consume body also pushes smaller values back into the stream.
    :param multistate: If True, the consume body writes twice to a temporary, i.e., requires multiple states.
    """
    sdfg = dace.SDFG('tester')
    sdfg.add_array('V', [4], dace.int32)
    sdfg.add_stream('S', dace.int32, transient=True)
    sdfg.add_array('R', [1], dace.int32)
    fill = sdfg.add_state('fill', is_start_block=True)
    _, fill_entry, _ = fill.add_mapped_tasklet('fill',
                                               dict(k='0:4'), {'inp': dace.Memlet('V[k]')},
                                               'out = inp', {'out': dace.Memlet('S[0]')},
                                               external_edges=True)
    fill_entry.map.schedule = dace.ScheduleType.Sequential

    inner = dace.SDFG('inner')
    inner.add_scalar('x', dace.int32)
    inner.add_scalar('y', dace.int32)
    inner.add_array('T', [1], dace.int32, transient=True)
    first = inner.add_state('first', is_start_block=True)
    _write_tasklet(first, 'out = inp * 2', {'inp': 'x'}, 'T[0]')
    last = first
    if multistate:
        last = inner.add_state_after(first, 'overwrite')
        _write_tasklet(last, 'out = inp + 1', {'inp': 'x'}, 'T[0]')
    result_state = inner.add_state_after(last, 'result')
    outputs = {'y'}
    if push:
        inner.add_stream('Q', dace.int32)
        tasklet = result_state.add_tasklet('compute', {'inp', 'elem'}, {'out', 'q'},
                                           'out = inp\nif elem > 1:\n    q = elem - 1')
        result_state.add_edge(result_state.add_read('T'), None, tasklet, 'inp', dace.Memlet('T[0]'))
        result_state.add_edge(result_state.add_read('x'), None, tasklet, 'elem', dace.Memlet('x'))
        result_state.add_edge(tasklet, 'out', result_state.add_write('y'), None, dace.Memlet('y'))
        result_state.add_edge(tasklet, 'q', result_state.add_write('Q'), None, dace.Memlet('Q[0]'))
        outputs.add('Q')
    else:
        _write_tasklet(result_state, 'out = inp', {'inp': 'T[0]'}, 'y')

    state = sdfg.add_state_after(fill, 'consume')
    entry, exit_node = state.add_consume('cons', ('p', '1'))
    body = state.add_nested_sdfg(inner, {'x'}, outputs)
    stream = state.add_read('S')
    state.add_edge(entry, 'OUT_stream', body, 'x', dace.Memlet('S[0]'))
    path = []
    if in_map:
        map_entry, map_exit = state.add_map('map', dict(i='0:1'))
        state.add_memlet_path(stream, map_entry, entry, dst_conn='IN_stream', memlet=dace.Memlet('S[0]'))
        path = [map_exit]
    else:
        state.add_edge(stream, None, entry, 'IN_stream', dace.Memlet('S[0]'))
    state.add_memlet_path(body,
                          exit_node,
                          *path,
                          state.add_write('R'),
                          src_conn='y',
                          memlet=dace.Memlet('R[0]', wcr='lambda a, b: a + b'))
    if push:
        state.add_memlet_path(body, exit_node, *path, state.add_write('S'), src_conn='Q', memlet=dace.Memlet('S[0]'))
    return sdfg


@pytest.mark.parametrize('multistate', (False, True))
@pytest.mark.parametrize('push', (False, True))
@pytest.mark.parametrize('in_map', (False, True))
def test_consume_body(in_map: bool, push: bool, multistate: bool):
    sdfg = _consume_body_sdfg(in_map, push, multistate)
    # Simplification would inline the nested SDFGs counted below
    new_sdfg = _roundtrip_and_compare(sdfg,
                                      tn.ConsumeScope,
                                      dict(V=np.arange(4, dtype=np.int32), R=np.zeros(1, np.int32)),
                                      simplify=False)
    # A body that requires multiple states is nested (within the nested SDFG of the map, if any)
    assert len(list(new_sdfg.all_sdfgs_recursive())) == 1 + int(multistate) * (1 + int(in_map))


def _nview_sdfg(in_loop: bool) -> dace.SDFG:
    """
    Creates an SDFG that passes a slice of an array to a nested SDFG with a shape that cannot be mapped to the slice.

    :param in_loop: If True, the nested SDFG is called in a loop over the sliced dimension.
    """
    inner = dace.SDFG('inner')
    inner.add_array('X', [40], dace.float64)
    inner_state = inner.add_state()
    _write_tasklet(inner_state, 'out = inp + 1', {'inp': 'X[3]'}, 'X[3]')

    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [4, 5, 10], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    if in_loop:
        sdfg.add_symbol('i', dace.int64)
        loop = LoopRegion('loop', 'i < 5', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(loop)
        sdfg.add_edge(init, loop, dace.InterstateEdge())
        state = loop.add_state('call', is_start_block=True)
        index = 'i'
    else:
        state = sdfg.add_state_after(init, 'call')
        index = '1'
    nsdfg = state.add_nested_sdfg(inner, {'X'}, {'X'})
    state.add_edge(state.add_read('A'), None, nsdfg, 'X', dace.Memlet(f'A[0:4, {index}, 0:10]'))
    state.add_edge(nsdfg, 'X', state.add_write('A'), None, dace.Memlet(f'A[0:4, {index}, 0:10]'))
    return sdfg


@pytest.mark.parametrize('in_loop', (False, True))
def test_nview_outside_map(in_loop: bool):
    _roundtrip_and_compare(_nview_sdfg(in_loop), tn.NView, dict(A=np.random.rand(4, 5, 10)))


def test_reference_set_from_tasklet():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_reference('ref', [19], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('ptrset', {'a': dace.pointer(dace.float64)}, {'o'}, 'o = a + 1')
    state.add_edge(state.add_read('A'), None, tasklet, 'a', dace.Memlet('A'))
    ref = state.add_access('ref')
    state.add_edge(tasklet, 'o', ref, 'set', dace.Memlet('ref'))
    _write_tasklet(state, 'out = inp + 1', {'inp': 'ref[0]'}, 'B[0]')
    # Reuse the reference access node for the read
    read_edge = next(e for e in state.edges() if e.dst_conn == 'inp')
    state.remove_node(read_edge.src)
    state.add_edge(ref, None, read_edge.dst, 'inp', dace.Memlet('ref[0]'))

    new_sdfg = _roundtrip_and_compare(sdfg, tn.RefSetNode, dict(A=np.random.rand(20), B=np.zeros(1)))
    assert any(e.dst_conn == 'set' for state in new_sdfg.states() for e in state.edges())


@pytest.mark.parametrize('side_effect', (False, True))
def test_empty_map(side_effect: bool):
    sdfg = dace.SDFG('tester')
    state = sdfg.add_state()
    entry, exit_node = state.add_map('map', dict(i='0:10'))
    if side_effect:
        tasklet = state.add_tasklet('print', {}, {}, 'printf("%d\\n", i);', language=dace.Language.CPP)
        state.add_nedge(entry, tasklet, dace.Memlet())
        state.add_nedge(tasklet, exit_node, dace.Memlet())
    else:
        state.add_nedge(entry, exit_node, dace.Memlet())

    # Simplification would remove the empty map checked below
    new_sdfg = _roundtrip(sdfg, tn.MapScope, simplify=False)
    new_sdfg.validate()
    assert any(isinstance(n, dace.nodes.MapExit) for n, _ in new_sdfg.all_nodes_recursive())


def test_named_region():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [2], dace.float64)
    region = NamedRegion('my_region')
    sdfg.add_node(region, is_start_block=True)
    _write_tasklet(region.add_state('write', is_start_block=True), 'out = 1', {}, 'A[0]')
    _write_tasklet(sdfg.add_state_after(region, 'after'), 'out = inp + 1', {'inp': 'A[0]'}, 'A[1]')

    # Simplification would inline the named region checked below
    new_sdfg = _roundtrip_and_compare(sdfg, tn.NamedRegionScope, dict(A=np.zeros(2)), simplify=False)
    assert [block.label for block in new_sdfg.all_control_flow_blocks()
            if isinstance(block, NamedRegion)] == ['my_region']


def test_sdfg_metadata():
    """
    Code blocks (including those of nested SDFGs), callback mappings, and argument names are preserved.
    """
    inner = dace.SDFG('inner')
    inner.add_array('X', [1], dace.float64)
    inner.set_global_code('static double nested_helper(double x) { return x * 3; }')
    inner.set_init_code('/* nested init */')
    state = inner.add_state()
    tasklet = state.add_tasklet('call_helper', {}, {'out'}, 'out = nested_helper(2);', language=dace.Language.CPP)
    state.add_edge(tasklet, 'out', state.add_write('X'), None, dace.Memlet('X[0]'))

    sdfg = dace.SDFG('tester')
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.arg_names = ['B', 'A']
    sdfg.set_global_code('static double helper(double x) { return x + 1; }')
    sdfg.set_exit_code('/* exit */')
    sdfg.callback_mapping = {'__dace_callback_0': 'my_callback'}
    state = sdfg.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {'X'})
    written = state.add_access('A')
    state.add_edge(nsdfg, 'X', written, None, dace.Memlet('A[0]'))
    tasklet = state.add_tasklet('call_helper', {'inp'}, {'out'}, 'out = helper(inp);', language=dace.Language.CPP)
    state.add_edge(written, None, tasklet, 'inp', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'out', state.add_write('B'), None, dace.Memlet('B[0]'))

    stree = sdfg.as_schedule_tree()
    assert 'nested_helper' in stree.global_code['frame'].as_string
    assert 'helper(double x) { return x + 1; }' in stree.global_code['frame'].as_string
    new_sdfg = stree.as_sdfg(simplify=False)

    assert 'nested_helper' in new_sdfg.global_code['frame'].as_string
    assert 'return x + 1' in new_sdfg.global_code['frame'].as_string
    assert 'nested init' in new_sdfg.init_code['frame'].as_string
    assert 'exit' in new_sdfg.exit_code['frame'].as_string
    assert new_sdfg.callback_mapping == {'__dace_callback_0': 'my_callback'}
    assert new_sdfg.arg_names == ['B', 'A']

    a = np.zeros(1)
    b = np.zeros(1)
    new_sdfg(b, a)
    assert a[0] == 6 and b[0] == 7


def _waw_program_sdfg() -> dace.SDFG:

    @dace.program
    def tester(A: dace.float64[10], B: dace.float64[10]):
        A[0] = 1
        B[1] = A[0] + A[2]
        A[0] = B[1] * 2
        A[2] = A[0] + 1
        for i in dace.map[0:10]:
            B[i] = A[i] + 1
        A[:] = B

    return tester.to_sdfg(simplify=False)


def _state_machine_nested_waw() -> dace.SDFG:
    """
    Creates an SDFG with an if/else expressed through conditional inter-state edges, where one branch calls a nested
    SDFG that writes to the same element twice in separate states.
    """
    sdfg = _state_machine_if_else()
    inner = dace.SDFG('inner')
    inner.add_array('X', [2], dace.float64)
    first = inner.add_state('first', is_start_block=True)
    _write_tasklet(first, 'out = inp + 5', {'inp': 'X[1]'}, 'X[0]')
    second = inner.add_state_after(first, 'second')
    _write_tasklet(second, 'out = inp * 3', {'inp': 'X[1]'}, 'X[0]')
    then_state = next(state for state in sdfg.states() if state.label == 'then_state')
    nsdfg = then_state.add_nested_sdfg(inner, {'X'}, {'X'})
    then_state.add_edge(then_state.add_read('A'), None, nsdfg, 'X', dace.Memlet('A[0:2]'))
    then_state.add_edge(nsdfg, 'X', then_state.add_write('A'), None, dace.Memlet('A[0:2]'))
    # The original write of the branch happens before the nested SDFG
    for edge in list(then_state.edges()):
        if edge.dst_conn is None and edge.src_conn == 'out':
            then_state.remove_edge(edge)
            then_state.add_edge(edge.src, 'out', edge.dst, None, edge.data)
            then_state.add_nedge(edge.dst, nsdfg, dace.Memlet())
    return sdfg


#: SDFGs (with a node type their schedule tree contains) and arguments, for converting state boundaries into empty
#: memlets. Some cases contain no boundaries within states, but test that control flow is unaffected by the behavior.
_EMPTY_MEMLET_ROUNDTRIPS = {
    'waw_program': (_waw_program_sdfg, tn.MapScope, [dict(A=np.random.rand(10), B=np.random.rand(10))]),
    'nested_return_in_map_loop': (lambda: _nested_return_sdfg(True, True), tn.StateLabel,
                                  [dict(A=np.random.rand(10), B=np.zeros(10), N=n) for n in (0, 2, 10)]),
    'consume_body': (lambda: _consume_body_sdfg(True, True, True), tn.ConsumeScope,
                     [dict(V=np.arange(4, dtype=np.int32), R=np.zeros(1, np.int32))]),
    'state_machine': (_state_machine_if_else, tn.GBlock, [dict(A=np.zeros(2), N=n) for n in (1, -1)]),
    'state_machine_nested_waw':
    (_state_machine_nested_waw, tn.GBlock, [dict(A=np.random.rand(2), N=n) for n in (1, -1)]),
    'nview_in_loop': (lambda: _nview_sdfg(True), tn.NView, [dict(A=np.random.rand(4, 5, 10))]),
}


@pytest.mark.parametrize('name', _EMPTY_MEMLET_ROUNDTRIPS.keys())
def test_roundtrip_empty_memlet_boundaries(name: str):
    factory, node_type, arguments = _EMPTY_MEMLET_ROUNDTRIPS[name]
    _roundtrip_and_compare(factory(),
                           node_type,
                           *arguments,
                           state_boundary_behavior=t2s.StateBoundaryBehavior.EMPTY_MEMLET,
                           simplify=False)


if __name__ == '__main__':
    test_implicit_inline_and_constants()
    test_name_propagation()
    test_view_of_slice()
    test_view_read_after_write()
    test_view_of_view()
    test_view_in_map_scope()
    test_view_write_in_map_scope()
    test_view_passed_to_nested_sdfg()
    test_dynamic_map_range()
    test_reference_set()
    test_library_call()
    test_do_while_loop()
    test_do_for_loop(False)
    test_do_for_loop(True)
    test_return_block()
    test_conditional_edge_exit()
    for in_map in (False, True):
        for in_loop in (False, True):
            test_nested_sdfg_return(in_map, in_loop)
    test_break_in_conditional()
    test_state_machine_if_else()
    test_state_machine_loop()
    test_state_machine_in_loop()
    test_consume_fibonacci(False)
    test_consume_fibonacci(True)
    test_nview_outside_map(False)
    test_nview_outside_map(True)
    test_reference_set_from_tasklet()
    for in_map in (False, True):
        for push in (False, True):
            for multistate in (False, True):
                test_consume_body(in_map, push, multistate)
    test_empty_map(False)
    test_empty_map(True)
    test_transients_and_nested_sdfg()
    test_named_region()
    test_sdfg_metadata()
    for name in _EMPTY_MEMLET_ROUNDTRIPS:
        test_roundtrip_empty_memlet_boundaries(name)
