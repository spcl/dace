from typing import Tuple
import pytest
import sympy
import dace
import numpy as np
import dace.sdfg.utils as sdutil

N = dace.symbol("N")

input_sets = [
    (0, "_if_cond_44",
     "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 1)] >= rcldtopcf))",
     3),
    (1, "_if_cond_44",
     "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] >= rcldtopcf))",
     2),
    (2, "_if_cond_44",
     "((za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 2)] < rcldtopcf) and (za[((_for_it_47 + 1) - 1), ((_for_it_23 + 1) - 1)] >= rcldtopcf) and (za[((_for_it_47 + 1) - 2), ((_for_it_23 + 1) - 0)] >= rcldtopcf))",
     4),
    (3, "_if_cond_44", "0", 0),
    (4, "_if_cond_44", "rcldtopcf", 1),
    (5, "_if_cond_44", "sym_rcldtopcf", 1),
]


def make_type2_sdfg():
    assignment_key = "_if_cond_1"
    assignment_val = "(((zqx[((_for_it_12 + 1) - 1), ((_for_it_11 + 1) - 1), (1 - 1)] + zqx[((_for_it_12 + 1) - 1), ((_for_it_11 + 1) - 1), (2 - 1)]) < rlmin) or (za[((_for_it_12 + 1) - 1), ((_for_it_11 + 1) - 1)] < ramin))"
    sdfg = dace.SDFG(f"single_complex_expression_sdfg_type_2")
    state1 = sdfg.add_state("complex_tasklet_state")

    sdfg.add_symbol("_for_it_11", dace.int64)
    sdfg.add_symbol("_for_it_12", dace.int64)

    for inm in ["ramin", "rlmin", "os"]:
        sdfg.add_scalar(inm, dace.float64, dace.dtypes.StorageType.Default, transient=False)

    sdfg.add_array("za", (
        5,
        5,
    ), dace.float64, dace.dtypes.StorageType.Default, transient=False)
    sdfg.add_array("zqx", (5, 5, 2), dace.float64, dace.dtypes.StorageType.Default, transient=False)

    state2 = sdfg.add_state("complex_tasklet_state2")

    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments={assignment_key: assignment_val}))

    sdfg.validate()
    return sdfg


# Create SDFG
def make_sdfg(assignment_key, assignment_val):
    sdfg = dace.SDFG('cond_edge_sdfg')

    # Symbols
    sdfg.add_symbol('_for_it_47', dace.int64)
    sdfg.add_symbol('_for_it_23', dace.int64)
    sdfg.add_scalar('rcldtopcf', dace.float64)
    sdfg.add_scalar('sym_rcldtopcf', dace.float64)

    # Arrays
    sdfg.add_array('za', [N, N], dace.float64)  # sizes arbitrary

    # States
    s0 = sdfg.add_state('start')
    s1 = sdfg.add_state('end')

    # Add defined _if_cond_44 as a symbol
    sdfg.add_symbol('_if_cond_44', dace.float64)

    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={assignment_key: assignment_val}))

    return sdfg


# Pytest-style runner
def run_two_sdfgs(sdfg1, sdfg2, inputs: dict):
    sdfg1(**inputs)
    sdfg2(**inputs)


# Pytest test function
@pytest.mark.parametrize("input", input_sets)
def test_single_edge(input: Tuple[int, str, str, int]):
    sdfgA = make_sdfg(input[1], input[2])
    sdfgB = make_sdfg(input[1], input[2])
    sdfgA.name = f"pattern_{input[0]}_original"
    sdutil.demote_symbol_to_scalar(sdfgB, "_if_cond_44", dace.float64)
    sdfgB.name = f"pattern_{input[0]}_demoted"

    tasklets = set()
    for state in sdfgB.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                tasklets.add((node, state))
    assert len(tasklets) == 1

    tasklet, state = tasklets.pop()
    assert len(state.in_edges(tasklet)) == input[3]

    _N = 10

    inputs = {
        'za': np.random.rand(_N, _N),
        '_for_it_47': np.int64(5),
        '_for_it_23': np.int64(7),
        'rcldtopcf': np.float64(0.3),
        'sym_rcldtopcf': np.float64(0.3),
        'N': _N,
    }

    run_two_sdfgs(sdfgA, sdfgB, inputs)


# Pytest test function
def test_complex_expr_and_connector_names():
    sdfgA = make_type2_sdfg()
    sdfgB = make_type2_sdfg()
    sdfgA.name = f"pattern_type2_original"
    sdutil.demote_symbol_to_scalar(sdfgB, "_if_cond_1", dace.float64)
    sdfgB.name = f"pattern_type2_demoted"

    tasklets = set()
    for state in sdfgB.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                tasklets.add((node, state))
    assert len(tasklets) == 1

    tasklet, state = tasklets.pop()

    sym_expr = dace.symbolic.SymExpr(tasklet.code.as_string.split(" = ")[1].strip())
    func_and_sym_names = {str(s)
                          for s in sym_expr.free_symbols}.union({str(f.func)
                                                                 for f in sym_expr.atoms(sympy.Function)})
    for arr_name in ["rlmin", "rlmax", "za", "zqx"]:
        assert arr_name not in func_and_sym_names


if __name__ == "__main__":
    for iset in input_sets:
        test_single_edge(iset)
    test_complex_expr_and_connector_names()


def test_a_cpp_guard_tasklet_reading_the_symbol_is_rewritten():
    """A tasklet body is not necessarily one Python assignment.

    ``demote_symbol_to_scalar`` used to sanity-check "no tasklet assigns the symbol" by splitting
    the body on ``" = "``, and to rewrite the body ``py_only``. A C++ guard tasklet --
    ``if (s > 0) { std::abort(); }``, which canonicalization emits for a scatter-guard assumption
    -- has no assignment at all: the split crashed with ``not enough values to unpack``, and the
    rewrite behind it asserted. Both are the same mistake, that every body is ``lhs = rhs``.

    The tasklet must come out reading the new scalar through a connector, with the symbol gone
    from the SDFG.
    """
    sdfg = dace.SDFG('cpp_guard_demotion')
    sdfg.add_array('data', [4], dace.int64)
    sdfg.add_symbol('guard_count', dace.int64)
    start = sdfg.add_state('start', is_start_block=True)
    body = sdfg.add_state('body')
    sdfg.add_edge(start, body, dace.InterstateEdge(assignments={'guard_count': 'data[0]'}))
    guard = body.add_tasklet('check_assumption', {}, {},
                             'if (guard_count > 0) { std::abort(); }',
                             language=dace.dtypes.Language.CPP)

    sdutil.demote_symbol_to_scalar(sdfg, 'guard_count', dace.int64, None)

    assert 'guard_count' not in sdfg.symbols, 'the symbol is gone from the SDFG'
    assert 'guard_count' in sdfg.arrays, 'and is now a scalar'
    assert '_in_guard_count' in guard.in_connectors, 'the guard reads the scalar through a connector'
    code = guard.code.as_string
    assert '_in_guard_count' in code and 'std::abort()' in code, f'the C++ body must survive intact: {code!r}'
    assert 'if (guard_count' not in code, f'the bare symbol must be gone from the body: {code!r}'


def test_tasklet_assigns_name_reads_the_statements_not_the_source_text():
    """The sanity check behind the demotion: it must answer for a body with no assignment, with
    several, and for a non-Python body, without assuming any shape."""
    from dace.sdfg import tasklet_utils as tutil

    sdfg = dace.SDFG('assigns_name')
    state = sdfg.add_state()

    def tasklet(code, language=dace.dtypes.Language.Python):
        return state.add_tasklet(f't{len(state.nodes())}', {'a': None}, {'b': None}, code, language=language)

    assert tutil.tasklet_assigns_name(tasklet('b = a'), 'b')
    assert not tutil.tasklet_assigns_name(tasklet('b = a'), 'a')
    assert tutil.tasklet_assigns_name(tasklet('tmp = a\nb = tmp * 2'), 'tmp'), 'a later statement counts'
    assert tutil.tasklet_assigns_name(tasklet('b = a\nb += 1'), 'b'), 'an augmented assignment counts'
    assert not tutil.tasklet_assigns_name(tasklet('b = a == 2'), 'a'), 'a comparison is not an assignment'
    cpp = tasklet('if (a > 0) { std::abort(); }', dace.dtypes.Language.CPP)
    assert not tutil.tasklet_assigns_name(cpp, 'a'), 'a body with no assignment assigns nothing'
    assert tutil.tasklet_assigns_name(tasklet('b = a;', dace.dtypes.Language.CPP), 'b'), 'C++ assignment counts'
    assert tutil.tasklet_assigns_name(tasklet('b += a;', dace.dtypes.Language.CPP), 'b'), 'so does C++ +='
    assert not tutil.tasklet_assigns_name(tasklet('if (b >= a) { }', dace.dtypes.Language.CPP), 'b'), '>= is not ='


def writes_scalar(state: dace.SDFGState, name: str) -> bool:
    return any(
        isinstance(node, dace.nodes.AccessNode) and node.data == name and state.in_degree(node) > 0
        for node in state.nodes())


def writes_per_path(sdfg: dace.SDFG, src: dace.SDFGState, dst: dace.SDFGState, name: str) -> list[int]:
    return [sum(writes_scalar(block, name) for block in path) for path in sdfg.all_simple_paths(src, dst)]


def stacked_writes(sdfg: dace.SDFG, name: str) -> list[str]:
    return [
        block.label for block in sdfg.nodes()
        if writes_scalar(block, name) and any(writes_scalar(pred, name) for pred in sdfg.predecessors(block))
    ]


def branches_joining_at_use(label: str, start_to_right: dict[str, str],
                            right_to_use: dict[str, str]) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """``flag`` picks ``left`` or ``right``; ``left -> use`` assigns ``s = X[0]``; ``use`` copies ``s`` to ``out``."""
    sdfg = dace.SDFG(label)
    for name in ('X', 'Y', 'out'):
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_symbol('s', dace.float64)
    sdfg.add_symbol('flag', dace.int64)
    start = sdfg.add_state('start', is_start_block=True)
    left = sdfg.add_state('left')
    right = sdfg.add_state('right')
    use = sdfg.add_state('use')
    sdfg.add_edge(start, left, dace.InterstateEdge(condition='flag > 0'))
    sdfg.add_edge(start, right, dace.InterstateEdge(condition='flag <= 0', assignments=start_to_right))
    sdfg.add_edge(left, use, dace.InterstateEdge(assignments={'s': 'X[0]'}))
    sdfg.add_edge(right, use, dace.InterstateEdge(assignments=right_to_use))
    tasklet = use.add_tasklet('copy', {}, {'res'}, 'res = s')
    use.add_edge(tasklet, 'res', use.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg, start, use


def run_branch(sdfg: dace.SDFG, flag: int) -> float:
    out = np.zeros(1)
    sdfg(X=np.array([2.0]), Y=np.array([5.0]), out=out, flag=np.int64(flag))
    return float(out[0])


def test_two_branches_assigning_a_symbol_into_one_state_each_keep_their_own_value():
    sdfg, start, use = branches_joining_at_use('two_assigning_branches', {}, {'s': 'Y[0]'})

    sdutil.demote_symbol_to_scalar(sdfg, 's', dace.float64)

    assert writes_per_path(sdfg, start, use, 's') == [1, 1], 'each branch must run exactly its own write'
    assert stacked_writes(sdfg, 's') == [], 'no write may run right after another write of the same scalar'
    assert (run_branch(sdfg, 1), run_branch(sdfg, 0)) == (2.0, 5.0)


def test_a_non_assigning_branch_into_the_same_state_does_not_run_the_other_branch_write():
    sdfg, start, use = branches_joining_at_use('one_assigning_branch', {'s': 'Y[0]'}, {})

    sdutil.demote_symbol_to_scalar(sdfg, 's', dace.float64)

    assert writes_per_path(sdfg, start, use, 's') == [1, 1], 'each branch must run exactly its own write'
    assert stacked_writes(sdfg, 's') == []
    assert (run_branch(sdfg, 1), run_branch(sdfg, 0)) == (2.0, 5.0)


def test_a_write_on_a_back_edge_into_the_start_block_does_not_run_on_entry():
    """Structural only: ``s`` is undefined on entry, so an entry write is observable only as undefined behavior."""
    sdfg = dace.SDFG('back_edge_into_start_block')
    sdfg.add_array('X', [4], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_symbol('s', dace.float64)
    sdfg.add_symbol('i', dace.int64)
    head = sdfg.add_state('head', is_start_block=True)
    body = sdfg.add_state('body')
    done = sdfg.add_state('done')
    sdfg.add_edge(head, body, dace.InterstateEdge(condition='i < 4', assignments={'i': 'i + 1'}))
    sdfg.add_edge(body, head, dace.InterstateEdge(assignments={'s': 'X[i - 1]'}))
    sdfg.add_edge(head, done, dace.InterstateEdge(condition='i >= 4'))
    sink = done.add_tasklet('read', {}, {'res'}, 'res = s')
    done.add_edge(sink, 'res', done.add_access('out'), None, dace.Memlet('out[0]'))

    sdutil.demote_symbol_to_scalar(sdfg, 's', dace.float64)

    assert sdfg.start_block is head, 'entering the SDFG must not run the back edge write (X[i - 1] at i = 0)'
    assert writes_per_path(sdfg, body, head, 's') == [1], 'the back edge runs its write exactly once'
    sdfg.validate()
