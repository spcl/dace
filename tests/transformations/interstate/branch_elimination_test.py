import copy
import functools
import numpy as np
import dace
import pytest
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes import ConstantPropagation, EliminateBranches
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.symbol_propagation import SymbolPropagation

N = 32
S = 32
S1 = 0
S2 = 32


def temporarily_disable_autoopt_and_serialization(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Save original values
        orig_autoopt = dace.config.Config.get("optimizer", "autooptimize")
        orig_serialization = dace.config.Config.get("testing", "serialization")
        try:
            # Set both to False
            dace.config.Config.set("optimizer", "autooptimize", value=False)
            dace.config.Config.set("testing", "serialization", value=False)
            return func(*args, **kwargs)
        finally:
            # Restore original values
            dace.config.Config.set("optimizer", "autooptimize", value=orig_autoopt)
            dace.config.Config.set("testing", "serialization", value=orig_serialization)

    return wrapper


@dace.program
def branch_dependent_value_write(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] > 0.5:
            c[i, j] = a[i, j] * b[i, j]
            d[i, j] = 1 - c[i, j]
        else:
            c[i, j] = 0.0
            d[i, j] = 0.0


@dace.program
def branch_dependent_value_write_with_transient_reuse(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] > 0.5:
            c[i, j] = a[i, j] * b[i, j]
            c_scl = 1 - c[i, j]
        else:
            c[i, j] = 0.1
            c_scl = 1.0

        if b[i, j] > 0.5:
            c[i, j] = c_scl * c[i, j]
        else:
            c[i, j] = -1.5 * c_scl * c[i, j]


@dace.program
def branch_dependent_value_write_two(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] > 0.3:
            b[i, j] = 1.1
            d[i, j] = 0.8
        else:
            b[i, j] = -1.1
            d[i, j] = 2.2
        c[i, j] = max(0, b[i, j])
        d[i, j] = max(0, d[i, j])


@dace.program
def top_level_if(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64[N, N],
):
    if a[0, 0] > 0.3:
        b[0, 0] = 1.1
        d[0, 0] = 0.8
    else:
        b[1, 1] = -1.1
        d[1, 1] = 2.2
    c[1, 1] = max(0, b[1, 1])


@dace.program
def weird_condition(a: dace.float64[N, N], b: dace.float64[N, N], ncldtop: dace.int64):
    for i, j in dace.map[0:N:1, 0:N:1]:
        cond = i + 1 > ncldtop
        if cond == 1:
            a[i, j] = 1.1
            b[i, j] = 0.8


@dace.program
def multi_state_branch_body(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N], d: dace.float64[N, N],
                            s: dace.int64):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] > 0.3:
            b[i + s, j + s] = 1.1
            d[i + s, j + s] = 0.8
        else:
            b[i + s, j + s] = -1.1
            d[i + s, j + s] = 2.2
        c[i + s, j + s] = max(0, b[i + s, j + s])
        d[i + s, j + s] = max(0, d[i + s, j + s])


@dace.program
def nested_if(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N], d: dace.float64[N, N],
              s: dace.int64):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] > 0.3:
            if s == 0:
                b[i, j] = 1.1
                d[i, j] = 0.8
            else:
                b[i, j] = 1.2
                d[i, j] = 0.9
        else:
            b[i, j] = -1.1
            d[i, j] = 2.2
        c[i + s, j + s] = max(0, b[i + s, j + s])
        d[i + s, j + s] = max(0, d[i + s, j + s])


SN = dace.symbol("SN")


@dace.program
def condition_on_bounds(a: dace.float64[SN, SN], b: dace.float64[SN, SN], c: dace.float64[SN, SN],
                        d: dace.float64[SN, SN], s: dace.int64):
    for i in dace.map[0:2]:
        j = 1
        c1 = (i < s)
        if c1:
            b[i, j] = 1.1 * c[i + 1, j] * a[i + 1, j]
            d[i, j] = 0.8 * c[i + 1, j] * a[i + 1, j]


@dace.program
def nested_if_two(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N], d: dace.float64[N, N]):
    for i, j in dace.map[0:N:1, 0:N:1]:
        c[i, j] = b[i, j] * c[i, j]
        d[i, j] = b[i, j] * d[i, j]
        if a[i, j] > 0.3:
            if b[i, j] > 0.1:
                b[i, j] = 1.1
                d[i, j] = 0.8
            else:
                b[i, j] = 1.2
                d[i, j] = 0.9
        else:
            b[i, j] = -1.1
            d[i, j] = 2.2
        c[i, j] = max(0, b[i, j])
        d[i, j] = max(0, d[i, j])


@dace.program
def branch_dependent_value_write_single_branch(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] < 0.65:
            b[i, j] = 0.0
        d[i, j] = b[i, j] + a[i, j]  # ensure d is always written for comparison


@dace.program
def branch_dependent_value_write_single_branch_nonzero_write(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if a[i, j] < 0.65:
            b[i, j] = 1.2 * a[i, j]
        d[i, j] = b[i, j] + a[i, j]  # ensure d is always written for comparison


@dace.program
def complicated_if(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if (a[i, j] + d[i, j] * 3.0) < 0.65:
            b[i, j] = 0.0
        d[i, j] = b[i, j] + a[i, j]  # ensure d is always written for comparison


@dace.program
def tasklets_in_if(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    for i in dace.map[S1:S2:1]:
        for j in dace.map[S1:S2:1]:
            if a[i, j] > c:
                b[i, j] = b[i, j] + d[i, j]
            else:
                b[i, j] = b[i, j] - d[i, j]
            b[i, j] = (1 - a[i, j]) * c


@dace.program
def single_branch_connectors(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    for i in dace.map[S1:S2:1]:
        for j in dace.map[S1:S2:1]:
            if a[i, j] > c:
                b[i, j] = d[i, j]


@dace.program
def if_over_map(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    if c == 2.0:
        for i in dace.map[S1:S2:1]:
            for j in dace.map[S1:S2:1]:
                cond = (a[0, 0] + b[0, 0] + d[0, 0]) > 2.0
                if cond:
                    b[i, j] = d[i, j]


@dace.program
def if_over_map_with_top_level_tasklets(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    if c == 2.0:
        for i in dace.map[S1:S2:1]:
            for j in dace.map[S1:S2:1]:
                cond = (b[0, 0] + d[0, 0]) > 2.0
                if cond:
                    b[i, j] = d[i, j]
        a[2, 2] = 3.0
        a[1, 1] = 3.0
        a[4, 4] = 3.0


@dace.program
def disjoint_subsets(
    if_cond_58: dace.int32,
    A: dace.float64[N],
    B: dace.float64[N, 3, 3],
    C: dace.float64[N, 3, 3],
    E: dace.float64[N],
):
    i = 4
    if if_cond_58 == 1:
        B[i, 2, 0] = A[i] + B[i, 2, 0]
        B[i, 2, 0] = E[i] + B[i, 2, 0]
        B[i, 0, 2] = B[i, 2, 0] + C[i, 0, 2]
        B[i, 0, 2] = A[i] + B[i, 0, 2]
    else:
        B[i, 1, 0] = A[i] + B[i, 1, 0]
        B[i, 1, 0] = E[i] + B[i, 1, 0]
        B[i, 0, 1] = B[i, 2, 0] + C[i, 0, 1]
        B[i, 0, 1] = A[i] + B[i, 0, 1]


@dace.program
def disjoint_subsets_two(
    if_cond_58: dace.int32,
    A: dace.float64[N],
    B: dace.float64[N, 3, 3],
    C: dace.float64[N, 3, 3],
    D: dace.float64[N, 3, 3],
    E: dace.float64[N],
    F: dace.float64[N, 3, 3],
):
    for i in dace.map[0:N:1]:
        if if_cond_58 == 1:
            B[i, 2, 0] = A[i] + B[i, 2, 0]
            C[i, 2, 0] = E[i] + B[i, 2, 0]
            D[i, 0, 2] = A[i] + C[i, 0, 2]
            F[i, 0, 2] = A[i] + D[i, 0, 2]
        else:
            B[i, 1, 0] = A[i] + B[i, 1, 0]
            C[i, 1, 0] = E[i] + B[i, 1, 0]
            D[i, 0, 1] = A[i] + C[i, 0, 1]
            F[i, 0, 1] = A[i] + D[i, 0, 1]


def _get_parent_state(sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG):
    for n, g in sdfg.all_nodes_recursive():
        if n == nsdfg_node:
            return g
    return None


def apply_branch_elimination(sdfg, nestedness: int = 1):
    """Apply BranchElimination transformation to all eligible conditionals."""
    # Pattern matching with conditional branches to not work (9.10.25), avoid it
    for i in range(nestedness):
        for node, graph in sdfg.all_nodes_recursive():
            parent_nsdfg_node = graph.sdfg.parent_nsdfg_node
            parent_state = None
            if parent_nsdfg_node is not None:
                parent_state = _get_parent_state(sdfg, parent_nsdfg_node)
            if isinstance(node, ConditionalBlock):
                t = branch_elimination.BranchElimination()
                if t.can_be_applied_to(graph.sdfg, conditional=node, options={"parent_nsdfg_state": parent_state}):
                    t.apply_to(graph.sdfg, conditional=node, options={"parent_nsdfg_state": parent_state})


def run_and_compare(
    program,
    num_expected_branches,
    use_pass,
    sdfg_name,
    **arrays,
):
    # Run SDFG version (no transformation)
    sdfg = program.to_sdfg()
    sdfg.validate()
    sdfg.name = sdfg_name

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg_name + "_branch_eliminated"

    # Run SDFG version (with transformation)
    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    out_fused = {k: v.copy() for k, v in arrays.items()}

    # Apply transformation
    if use_pass:
        fb = EliminateBranches()
        fb.try_clean = True
        fb.apply_to_top_level_ifs = True
        fb.apply_pass(copy_sdfg, {})
    else:
        apply_branch_elimination(copy_sdfg, 2)

    copy_sdfg.save(f"{copy_sdfg.name}.sdfgz", compress=True)
    c_sdfg(**out_no_fuse)
    c_copy_sdfg(**out_fused)

    branch_code = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(
        branch_code) == num_expected_branches, f"(actual) len({branch_code}) != (desired) {num_expected_branches}"

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


def run_and_compare_sdfg(
    sdfg,
    permissive,
    sdfg_name,
    **arrays,
):
    # Run SDFG version (no transformation)
    sdfg.validate()
    sdfg.name = sdfg_name
    # Run SDFG version (with transformation)
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg_name + "_branch_eliminated"

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    out_fused = {k: v.copy() for k, v in arrays.items()}

    fb = EliminateBranches()
    fb.try_clean = True
    fb.apply_to_top_level_ifs = True
    fb.permissive = permissive
    fb.apply_pass(copy_sdfg, {})

    c_sdfg(**out_no_fuse)
    c_copy_sdfg(**out_fused)

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)

    return copy_sdfg


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_branch_dependent_value_write(use_pass_flag):
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write,
                    0,
                    use_pass_flag,
                    f"branch_dependent_value_write_use_pass_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    c=c,
                    d=d)


@temporarily_disable_autoopt_and_serialization
def test_weird_condition():
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    ncldtop = np.array([N // 2], dtype=np.int64)
    run_and_compare(weird_condition, 1, False, f"weird_condition", a=a, b=b, ncldtop=ncldtop[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_branch_dependent_value_write_two(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.zeros((N, N))
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_two,
                    0,
                    use_pass_flag,
                    f"branch_dependent_value_write_two_use_pass_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    c=c,
                    d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_branch_dependent_value_write_single_branch(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_single_branch,
                    0,
                    use_pass_flag,
                    f"branch_dependent_value_write_single_branch_use_pass_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_complicated_if(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.zeros((N, N))
    run_and_compare(complicated_if,
                    0,
                    use_pass_flag,
                    f"complicated_if_use_pass_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_multi_state_branch_body(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    c = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(multi_state_branch_body,
                    0 if use_pass_flag else 1,
                    use_pass_flag,
                    f"multistate_branch_body_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    s=s[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_nested_if(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    c = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.random.choice([0.001, 5.0], size=(N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(nested_if,
                    0,
                    use_pass_flag,
                    f"nested_if_use_pass_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    s=s[0])


@temporarily_disable_autoopt_and_serialization
def test_condition_on_bounds():
    a = np.random.choice([0.001, 3.0], size=(2, 2))
    b = np.random.choice([0.001, 5.0], size=(2, 2))
    c = np.random.choice([0.001, 5.0], size=(2, 2))
    d = np.random.choice([0.001, 5.0], size=(2, 2))

    sdfg = condition_on_bounds.to_sdfg()
    sdfg.validate()
    sdfg.name = "condition_on_bounds"
    arrays = {"a": a, "b": b, "c": c, "d": d}
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(a=out_no_fuse["a"], b=out_no_fuse["b"], c=out_no_fuse["c"], d=out_no_fuse["d"], s=1, SN=2)
    # Apply transformation
    eb = EliminateBranches()
    eb.apply_to_top_level_ifs = True
    eb.apply_pass(sdfg, {})
    sdfg.validate()

    nsdfgs = {(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}
    assert len(nsdfgs) == 1  # Can be applied should return false


@temporarily_disable_autoopt_and_serialization
def test_nested_if_two():
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    c = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.random.choice([0.001, 5.0], size=(N, N))
    run_and_compare(nested_if_two, 0, True, f"nested_if_two", a=a, b=b, c=c, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_tasklets_in_if(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    c = np.zeros((1, ))
    d = np.zeros((N, N))
    run_and_compare(tasklets_in_if,
                    0,
                    use_pass_flag,
                    f"tasklets_in_if_use_pass{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    d=d,
                    c=c[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_branch_dependent_value_write_single_branch_nonzero_write(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.random.choice([0.001, 5.0], size=(N, N))
    run_and_compare(branch_dependent_value_write_single_branch_nonzero_write,
                    0,
                    use_pass_flag,
                    f"branch_dependent_value_write_single_branch_nonzero_write_use_pass_{str(use_pass_flag).lower()}",
                    a=a,
                    b=b,
                    d=d)


@temporarily_disable_autoopt_and_serialization
def test_branch_dependent_value_write_with_transient_reuse():
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 3.0], size=(N, N))
    c = np.random.choice([0.001, 3.0], size=(N, N))
    run_and_compare(branch_dependent_value_write_with_transient_reuse,
                    0,
                    True,
                    f"branch_dependent_value_write_with_transient_reuse",
                    a=a,
                    b=b,
                    c=c)


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_single_branch_connectors(use_pass_flag):
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 5.0], size=(N, N))
    d = np.random.choice([0.001, 5.0], size=(N, N))
    c = np.random.randn(1, )

    sdfg = single_branch_connectors.to_sdfg()
    sdfg.validate()
    sdfg.name = f"test_single_branch_connectors_use_pass_{str(use_pass_flag).lower()}"
    arrays = {"a": a, "b": b, "c": c, "d": d}
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(a=out_no_fuse["a"], b=out_no_fuse["b"], c=out_no_fuse["c"][0], d=out_no_fuse["d"])
    # Apply transformation
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"test_single_branch_connectors_use_pass_{str(use_pass_flag).lower()}_branch_eliminated"
    if use_pass_flag:
        eb = EliminateBranches()
        eb.apply_to_top_level_ifs = True
        eb.apply_pass(copy_sdfg, {})
    else:
        apply_branch_elimination(copy_sdfg, 2)

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg(a=out_fused["a"], b=out_fused["b"], c=out_fused["c"][0], d=out_fused["d"])

    branch_code = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branch_code) == 0, f"(actual) len({branch_code}) != (desired) 0"

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)

    nsdfgs = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}
    assert len(nsdfgs) == 1
    nsdfg, parent_state = nsdfgs.pop()
    assert len(nsdfg.in_connectors) == 4, f"{nsdfg.in_connectors}, length is not 4 but {len(nsdfg.in_connectors)}"
    assert len(nsdfg.out_connectors) == 1, f"{nsdfg.out_connectors}, length is not 1 but {len(nsdfg.out_connectors)}"


@pytest.mark.parametrize("use_pass_flag", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_disjoint_subsets(use_pass_flag):
    if_cond_58 = np.array([1], dtype=np.int32)
    A = np.random.choice([0.001, 3.0], size=(N, ))
    B = np.random.randn(N, 3, 3)
    C = np.random.randn(N, 3, 3)
    E = np.random.choice([0.001, 3.0], size=(N, 3, 3))
    run_and_compare(disjoint_subsets,
                    0,
                    use_pass_flag,
                    f"disjoint_subsets_use_pass_{str(use_pass_flag).lower()}",
                    A=A,
                    B=B,
                    C=C,
                    E=E,
                    if_cond_58=if_cond_58[0])


@dace.program
def _multi_state_nested_if(
    A: dace.float64[
        N,
    ],
    B: dace.float64[N, 3, 3],
    C: dace.float64[
        N,
    ],
    if_cond_1: dace.float64,
    offset: dace.int64,
):
    if if_cond_1 > 1.0:
        _if_cond_2 = C[offset]
        if _if_cond_2 > 1.0:
            B[6, 1, 0] = A[6] + B[6, 1, 0]
            B[6, 0, 1] = A[6] + B[6, 0, 1]
        else:
            B[6, 2, 0] = A[6] + B[6, 2, 0]
            B[6, 0, 2] = A[6] + B[6, 0, 2]


@temporarily_disable_autoopt_and_serialization
def test_try_clean():
    sdfg1 = _multi_state_nested_if.to_sdfg()
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 2

    for cblock in cblocks:
        parent_sdfg = cblock.parent_graph.sdfg
        parent_graph = cblock.parent_graph
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        xform.try_clean(graph=parent_graph, sdfg=parent_sdfg)

    # Should have move states before
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 2
    # A state must have been moved before)
    assert isinstance(sdfg1.start_block, dace.SDFGState)
    sdfg1.validate()

    fbpass = EliminateBranches()
    fbpass.try_clean = False
    fbpass.apply_to_top_level_ifs = True
    fbpass.apply_pass(sdfg1, {})
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    # 1 left because now the if branch has 2 states -> not anymore thanks to force fuse and empty state elim
    assert len(cblocks) == 0, f"{cblocks}"
    sdfg1.validate()

    for cblock in cblocks:
        parent_sdfg = cblock.parent_graph.sdfg
        parent_graph = cblock.parent_graph
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        applied = xform.try_clean(graph=parent_graph, sdfg=parent_sdfg, lift_multi_state=False)
        assert applied is False
        applied = xform.try_clean(graph=parent_graph, sdfg=parent_sdfg, lift_multi_state=True)
        assert applied is True
    sdfg1.validate()

    fbpass = EliminateBranches()
    fbpass.try_clean = False
    fbpass.apply_to_top_level_ifs = True
    fbpass.apply_pass(sdfg1, {})
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0, f"{cblocks}"
    sdfg1.validate()

    if_cond_1 = np.array([1.2], dtype=np.float64)
    offset = np.array([0], dtype=np.int64)
    A = np.random.choice([0.001, 3.0], size=(N, ))
    B = np.random.randn(N, 3, 3)
    C = np.random.choice([0.001, 3.0], size=(N, ))
    run_and_compare_sdfg(sdfg1,
                         permissive=False,
                         sdfg_name=f"multi_state_nested_if_sdfg",
                         A=A,
                         B=B,
                         C=C,
                         if_cond_1=if_cond_1[0],
                         offset=offset[0])


@temporarily_disable_autoopt_and_serialization
def test_try_clean_as_pass():
    # This is a test to check the different configurations of try clean, applicability depends on the SDFG and the pass
    sdfg = _multi_state_nested_if.to_sdfg()
    fbpass = EliminateBranches()
    fbpass.clean_only = True
    fbpass.try_clean = False
    fbpass.apply_to_top_level_ifs = True
    fbpass.apply_pass(sdfg, {})
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0, f"{cblocks}"
    sdfg.validate()

    if_cond_1 = np.array([1.2], dtype=np.float64)
    offset = np.array([0], dtype=np.int64)
    A = np.random.choice([0.001, 3.0], size=(N, ))
    B = np.random.randn(N, 3, 3)
    C = np.random.choice([0.001, 3.0], size=(N, ))
    run_and_compare_sdfg(sdfg,
                         permissive=False,
                         sdfg_name=f"multi_state_nested_if_sdfg_try_clean_variant",
                         A=A,
                         B=B,
                         C=C,
                         if_cond_1=if_cond_1[0],
                         offset=offset[0])


def _get_sdfg_with_interstate_array_condition():
    sdfg = dace.SDFG("sd1")
    sdfg.add_array("llindex", (4, 4, 4), dtype=dace.int64)
    sdfg.add_array("zratio", (4, 4, 4), dtype=dace.float64)
    sdfg.add_array("zsolqa", (4, 4, 4), dtype=dace.float64)
    sdfg.add_scalar(
        "zzratio",
        dtype=dace.float64,
        transient=True,
        storage=dace.dtypes.StorageType.Register,
    )
    sdfg.add_symbol("_if_cond_1", dace.int64)
    s1 = sdfg.add_state("s1", is_start_block=True)
    cb1 = ConditionalBlock("cb1", sdfg, sdfg)
    cfg = ControlFlowRegion("cfg1", sdfg, cb1)
    cb1.add_branch(condition=CodeBlock("_if_cond_1 == 1"), branch=cfg)
    s2 = cfg.add_state("s2", is_start_block=True)

    sdfg.add_edge(s1, cb1, dace.InterstateEdge(assignments={"_if_cond_1": "llindex[2,2,2]"}))

    z1 = s1.add_access("zratio")
    zz1 = s1.add_access("zzratio")
    zz2 = s2.add_access("zzratio")
    zs1 = s2.add_access("zsolqa")
    zs2 = s2.add_access("zsolqa")

    t1 = s1.add_tasklet("T_1", {"_in_zratio"}, {"_out_zzratio"}, "_out_zzratio = _in_zratio")
    t2 = s2.add_tasklet("T_2", {"_in_zzratio", "_in_zsolqa"}, {"_out_zsolqa"}, "_out_zsolqa = _in_zzratio * _in_zsolqa")

    s1.add_edge(z1, None, t1, "_in_zratio", dace.memlet.Memlet("zratio[3,3,3]"))
    s1.add_edge(t1, "_out_zzratio", zz1, None, dace.memlet.Memlet("zzratio[0]"))
    s2.add_edge(zz2, None, t2, "_in_zzratio", dace.memlet.Memlet("zzratio[0]"))
    s2.add_edge(zs1, None, t2, "_in_zsolqa", dace.memlet.Memlet("zsolqa[2,2,2]"))
    s2.add_edge(t2, "_out_zsolqa", zs2, None, dace.memlet.Memlet("zsolqa[2,2,2]"))

    sdfg.validate()
    return sdfg


@temporarily_disable_autoopt_and_serialization
def test_sdfg_with_interstate_array_condition():
    sdfg = _get_sdfg_with_interstate_array_condition()
    llindex = np.ones(shape=(4, 4, 4), dtype=np.int64)
    zsolqa = np.random.choice([0.001, 3.0], size=(4, 4, 4))
    zratio = np.random.choice([0.001, 3.0], size=(4, 4, 4))
    run_and_compare_sdfg(
        sdfg,
        permissive=False,
        sdfg_name=f"sdfg_with_interstate_array_condition",
        llindex=llindex,
        zsolqa=zsolqa,
        zratio=zratio,
    )

    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.Tasklet):
            assert "[" not in n.code.as_string, f"Tasklet {n} has code: {n.code.as_string}"
            assert "]" not in n.code.as_string, f"Tasklet {n} has code: {n.code.as_string}"


@dace.program
def repeated_condition_variables(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    conds: dace.float64[4, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        cond_1 = conds[0, j]
        if cond_1 > 0.5:
            c[i, j] = a[i, j] * b[i, j]
        cond_1 = conds[1, j]
        if cond_1 > 0.5:
            c[i, j] = a[i, j] * b[i, j]
        cond_1 = conds[2, j]
        if cond_1 > 0.5:
            c[i, j] = a[i, j] * b[i, j]
        cond_1 = conds[3, j]
        if cond_1 > 0.5:
            c[i, j] = a[i, j] * b[i, j]


@temporarily_disable_autoopt_and_serialization
def test_repeated_condition_variables():
    a = np.random.choice([0.001, 3.0], size=(N, N))
    b = np.random.choice([0.001, 3.0], size=(N, N))
    c = np.random.choice([0.001, 3.0], size=(N, N))
    conds = np.random.choice([1.0, 3.0], size=(4, N))
    run_and_compare(repeated_condition_variables, 0, True, f"repeated_condition_variables", a=a, b=b, c=c, conds=conds)


def _find_state(root_sdfg: dace.SDFG, node):
    for n, g in root_sdfg.all_nodes_recursive():
        if n == node:
            return g
    return None


@temporarily_disable_autoopt_and_serialization
def test_if_over_map():
    sdfg = if_over_map.to_sdfg()
    cblocks = {n for n in sdfg.all_control_flow_regions() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1
    inner_cblocks = {
        n
        for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, ConditionalBlock) and g is not None and g.sdfg.parent_nsdfg_node is not None
    }
    assert len(inner_cblocks) == 1

    xform = branch_elimination.BranchElimination()
    xform.conditional = cblocks.pop()
    xform.parent_nsdfg_state = None
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform = branch_elimination.BranchElimination()
    xform.conditional = inner_cblocks.pop()
    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is True


@temporarily_disable_autoopt_and_serialization
def test_if_over_map_with_top_level_tasklets():
    sdfg = if_over_map.to_sdfg()
    cblocks = {n for n in sdfg.all_control_flow_regions() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1
    inner_cblocks = {
        n
        for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, ConditionalBlock) and g is not None and g.sdfg.parent_nsdfg_node is not None
    }
    assert len(inner_cblocks) == 1

    xform = branch_elimination.BranchElimination()
    xform.conditional = cblocks.pop()
    xform.parent_nsdfg_state = None
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform = branch_elimination.BranchElimination()
    xform.conditional = inner_cblocks.pop()
    xform.parent_nsdfg_state = None
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is True


@temporarily_disable_autoopt_and_serialization
def test_can_be_applied_parameters_on_nested_sdfg():
    sdfg = nested_if.to_sdfg()
    cblocks = {n for n in sdfg.all_control_flow_regions() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0
    inner_cblocks = {
        n
        for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, ConditionalBlock) and g is not None and g.sdfg.parent_nsdfg_node is not None
    }
    assert len(inner_cblocks) == 2

    full_inner_cblocks = {
        n
        for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock) and g is not None and g != g.sdfg
    }
    assert len(full_inner_cblocks) == 1

    upper_inner_cblocks = {
        n
        for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock) and g is not None and g == g.sdfg
    }
    assert len(upper_inner_cblocks) == 1

    xform = branch_elimination.BranchElimination()
    xform.conditional = full_inner_cblocks.pop()

    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)

    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is True


@dace.program
def non_trivial_subset_after_combine_tasklet(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64[N, N],
    e: dace.float64[N, N],
    f: dace.float64,
    g: dace.float64[N, N],
):
    _if_cond_1 = a[1, 2] > (1 - f)
    if _if_cond_1:
        tc1 = b[3, 4] + c[3, 4] + d[3, 4]
        tc2 = tc1 * 2.3
        tc3 = max(0.0, tc2)
        g[6, 6] = tc3
    else:
        tc1 = b[3, 4] + c[3, 4] + e[5, 5]
        tc2 = tc1 * 2.3
        tc3 = max(0.767, tc2)
        tc4 = e[4, 4] * tc3
        g[6, 6] = tc4


@temporarily_disable_autoopt_and_serialization
def test_non_trivial_subset_after_combine_tasklet():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.random.choice([0.001, 5.0], size=(N, N))
    E = np.random.choice([0.001, 5.0], size=(N, N))
    F = np.random.randn(1, )
    G = np.random.choice([0.001, 5.0], size=(N, N))
    run_and_compare(
        non_trivial_subset_after_combine_tasklet,
        0,
        True,
        f"non_trivial_subset_after_combine_tasklet",
        a=A,
        b=B,
        c=C,
        d=D,
        e=E,
        f=F[0],
        g=G,
    )


@dace.program
def split_on_disjoint_subsets(
    a: dace.float64[N, N, 2],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64,
):
    _if_cond_1 = b[1, 2] > (1.0 - d)
    if _if_cond_1:
        tc1 = b[6, 6] * c[6, 6]
        a[6, 6, 0] = tc1
    else:
        tc1 = b[6, 6] * c[6, 6]
        a[6, 6, 1] = tc1
    c[3, 3] = 0.0
    b[4, 4] = 0.0


@dace.program
def split_on_disjoint_subsets_nested(
    a: dace.float64[N, N, 2],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64,
):
    for i in dace.map[0:N]:
        _if_cond_1 = b[i, 2] > (1.0 - d)
        if _if_cond_1:
            tc1 = b[i, 6] * c[i, 6]
            a[i, 6, 0] = tc1
        else:
            tc1 = b[i, 6] * c[i, 6]
            a[i, 6, 1] = tc1
        c[i, 3] = 0.0
        b[i, 4] = 0.0


@temporarily_disable_autoopt_and_serialization
def test_split_on_disjoint_subsets():
    A = np.random.choice([0.001, 5.0], size=(N, N, 2))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = split_on_disjoint_subsets.to_sdfg()

    # Is disjoit subset needs to return true
    cblocks = {n for n in sdfg.all_control_flow_regions() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1
    cblock = cblocks.pop()

    (cond0, body0), (cond1, body1) = cblock.branches[0:2]
    assert len(body0.nodes()) == 1
    assert len(body1.nodes()) == 1
    state0 = body0.nodes()[0]
    state1 = body1.nodes()[0]
    assert isinstance(state0, dace.SDFGState)
    assert isinstance(state1, dace.SDFGState)

    xform = branch_elimination.BranchElimination()
    xform.conditional = cblock
    assert xform._is_disjoint_subset(state0, state1) is True

    # If we split we will make not applicable anymore
    xform._split_branches(cblock.parent_graph, cblock)

    sdfg.validate()

    run_and_compare(
        split_on_disjoint_subsets,
        0,
        True,
        f"split_on_disjoint_subsets",
        a=A,
        b=B,
        c=C,
        d=D[0],
    )


@temporarily_disable_autoopt_and_serialization
def test_split_on_disjoint_subsets_nested():
    A = np.random.choice([0.001, 5.0], size=(N, N, 2))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = split_on_disjoint_subsets_nested.to_sdfg()

    # Is disjoit subset needs to return true
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1
    cblock = cblocks.pop()

    (cond0, body0), (cond1, body1) = cblock.branches[0:2]
    assert len(body0.nodes()) == 1
    assert len(body1.nodes()) == 1
    state0 = body0.nodes()[0]
    state1 = body1.nodes()[0]
    assert isinstance(state0, dace.SDFGState)
    assert isinstance(state1, dace.SDFGState)

    xform = branch_elimination.BranchElimination()
    xform.conditional = cblock
    assert xform._is_disjoint_subset(state0, state1) is True

    # If we split we will make not applicable anymore
    xform._split_branches(cblock.parent_graph, cblock)

    sdfg.validate()

    run_and_compare(
        split_on_disjoint_subsets_nested,
        0,
        True,
        f"split_on_disjoint_subsets_nested",
        a=A,
        b=B,
        c=C,
        d=D[0],
    )


@dace.program
def write_to_transient(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.int64,
):
    for i in dace.map[0:N]:
        zmdn = 0.0
        _if_cond_1 = d < 5.0
        if _if_cond_1:
            tc1 = b[i, 6] + a[i, 6]
            zmdn = tc1
        b[i, 3] = zmdn
        b[i, 3] = zmdn


@dace.program
def write_to_transient_two(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.int64,
):
    for i in dace.map[0:N]:
        zmdn = 0.0
        _if_cond_1 = d < 5.0
        if _if_cond_1:
            tc1 = b[i, 6] + a[i, 6]
            zmdn = tc1
        else:
            zmdn = 1.0
        b[i, 3] = zmdn
        b[i, 3] = zmdn


@temporarily_disable_autoopt_and_serialization
def test_write_to_transient():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    run_and_compare(
        write_to_transient,
        0,
        True,
        f"write_to_transient",
        a=A,
        b=B,
        d=D[0],
    )


@temporarily_disable_autoopt_and_serialization
def test_write_to_transient_two():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    run_and_compare(
        write_to_transient_two,
        0,
        True,
        f"write_to_transient_two",
        a=A,
        b=B,
        d=D[0],
    )


@temporarily_disable_autoopt_and_serialization
def test_double_empty_state():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = write_to_transient_two.to_sdfg()

    nested_sdfgs = {(n, g) for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}

    sdfg.add_state_before(sdfg.start_block, label=f"empty_prepadding_{sdfg.label}", is_start_block=True)

    for nsdfg, parent_state in nested_sdfgs:
        nsdfg.sdfg.add_state_before(nsdfg.sdfg.start_block,
                                    label=f"empty_prepardding_{nsdfg.sdfg.label}",
                                    is_start_block=True)

    run_and_compare_sdfg(
        sdfg,
        permissive=False,
        sdfg_name=f"double_empty_state",
        a=A,
        b=B,
        d=D[0],
    )


@dace.program
def complicated_pattern_for_manual_clean_up_one(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64,
):
    for i in dace.map[0:N]:
        _if_cond_1 = i < d
        if _if_cond_1:
            zalfaw = a[i, 0]
            tc1 = b[i, 6] + zalfaw
            tc2 = b[i, 3] * zalfaw + tc1
            c[i, 0] = tc2
        else:
            c[i, 0] = 0.0


@temporarily_disable_autoopt_and_serialization
def test_complicated_pattern_for_manual_clean_up_one():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = complicated_pattern_for_manual_clean_up_one.to_sdfg()

    nested_sdfgs = {(n, g) for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}

    # Force scalar promotion like in CloudSC
    ssp = ScalarToSymbolPromotion()
    ssp.integers_only = False
    ssp.transients_only = True
    scalar_names = {
        arr_name
        for arr_name, arr in sdfg.arrays.items()
        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and (arr.shape == [
            1,
        ] or arr.shape == (1, )))
    }.difference({"zalfaw"})
    ssp.ignore = scalar_names
    ssp.apply_pass(sdfg, {})

    for nsdfg, parent_state in nested_sdfgs:
        ssp = ScalarToSymbolPromotion()
        ssp.integers_only = False
        ssp.transients_only = True
        scalar_names = {
            arr_name
            for arr_name, arr in nsdfg.sdfg.arrays.items()
            if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and (arr.shape == [
                1,
            ] or arr.shape == (1, )))
        }.difference({"zalfaw"})
        ssp.ignore = scalar_names
        ssp.apply_pass(nsdfg.sdfg, {})

    for nsdfg, parent_state in nested_sdfgs:
        for cb in nsdfg.sdfg.all_control_flow_regions():
            if isinstance(cb, ConditionalBlock):
                xform = branch_elimination.BranchElimination()
                xform.parent_nsdfg_state = parent_state
                xform.conditional = cb
                assert xform.can_be_applied(graph=cb.parent_graph, expr_index=0, sdfg=cb.sdfg) is False
                assert xform.symbol_reused_outside_conditional(sym_name="zalfaw") is False
                assert len(cb.branches) == 2
                # Clean-up should be able to catch this pattern
                xform.demote_branch_only_symbols_appearing_only_a_single_branch_to_scalars_and_try_fuse(
                    graph=cb.parent_graph, sdfg=cb.sdfg)
                (cond0, body0), (cond1, body1) = cb.branches[0:2]
                assert len(body0.nodes()) == 1
                assert len(body1.nodes()) == 1
                assert all({isinstance(n, dace.SDFGState) for n in body0.nodes()})
                assert all({isinstance(n, dace.SDFGState) for n in body1.nodes()})


@temporarily_disable_autoopt_and_serialization
def test_try_clean_on_complicated_pattern_for_manual_clean_up_one():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = complicated_pattern_for_manual_clean_up_one.to_sdfg()

    nested_sdfgs = {(n, g) for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}

    # Force scalar promotion like in CloudSC
    ssp = ScalarToSymbolPromotion()
    ssp.integers_only = False
    ssp.transients_only = True
    scalar_names = {
        arr_name
        for arr_name, arr in sdfg.arrays.items()
        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and (arr.shape == [
            1,
        ] or arr.shape == (1, )))
    }.difference({"zalfaw"})
    ssp.ignore = scalar_names
    ssp.apply_pass(sdfg, {})

    for nsdfg, parent_state in nested_sdfgs:
        ssp = ScalarToSymbolPromotion()
        ssp.integers_only = False
        ssp.transients_only = True
        scalar_names = {
            arr_name
            for arr_name, arr in nsdfg.sdfg.arrays.items()
            if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and (arr.shape == [
                1,
            ] or arr.shape == (1, )))
        }.difference({"zalfaw"})
        ssp.ignore = scalar_names
        ssp.apply_pass(nsdfg.sdfg, {})

    transformed_sdfg = run_and_compare_sdfg(sdfg,
                                            permissive=True,
                                            sdfg_name="try_clean_on_complicated_pattern_for_manual_cleanup_one",
                                            a=A,
                                            b=B,
                                            c=C,
                                            d=D[0])

    branch_code = {n for n, g in transformed_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branch_code) == 0, f"(actual) len({branch_code}) != (desired) {0}"


@dace.program
def complicated_pattern_for_manual_clean_up_two(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    c: dace.float64[N, N],
    d: dace.float64,
    e: dace.float64,
):
    for i in dace.map[0:N]:
        _if_cond_1 = i < d
        if _if_cond_1:
            zlcrit = d
            zalfaw = a[i, 0]
            tc1 = b[i, 6] + zalfaw
            tc2 = b[i, 3] * zalfaw + tc1
            c[i, 0] = tc2
        else:
            zlcrit = e
            c[i, 0] = 0.0
        a[i, 3] = zlcrit * 2.0


@temporarily_disable_autoopt_and_serialization
def test_try_clean_on_complicated_pattern_for_manual_clean_up_two():
    A = np.random.choice([0.001, 5.0], size=(N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    E = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = complicated_pattern_for_manual_clean_up_two.to_sdfg()

    nested_sdfgs = {(n, g) for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}

    # Force scalar promotion like in CloudSC
    ssp = ScalarToSymbolPromotion()
    ssp.integers_only = False
    ssp.transients_only = True
    scalar_names = {
        arr_name
        for arr_name, arr in sdfg.arrays.items()
        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and (arr.shape == [
            1,
        ] or arr.shape == (1, )))
    }.difference({"zlcrit"})
    ssp.ignore = scalar_names
    ssp.apply_pass(sdfg, {})

    for nsdfg, parent_state in nested_sdfgs:
        ssp = ScalarToSymbolPromotion()
        ssp.integers_only = False
        ssp.transients_only = True
        scalar_names = {
            arr_name
            for arr_name, arr in nsdfg.sdfg.arrays.items()
            if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and (arr.shape == [
                1,
            ] or arr.shape == (1, )))
        }.difference({"zlcrit"})
        ssp.ignore = scalar_names
        ssp.apply_pass(nsdfg.sdfg, {})

    transformed_sdfg = run_and_compare_sdfg(sdfg,
                                            permissive=True,
                                            sdfg_name="try_clean_on_complicated_pattern_for_manual_cleanup_two",
                                            a=A,
                                            b=B,
                                            c=C,
                                            d=D[0],
                                            e=E[0])

    branch_code = {n for n, g in transformed_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branch_code) == 0, f"(actual) len({branch_code}) != (desired) {0}"


@dace.program
def single_assignment(a: dace.float64[
    N,
], _if_cond_1: dace.float64):
    for i in dace.map[0:N]:
        if _if_cond_1 > 0.0:
            a[i] = 0.0


@dace.program
def single_assignment_cond_from_scalar(a: dace.float64[512]):
    for i in dace.map[0:256]:
        _if_cond_1 = a[256 + i]
        if _if_cond_1 > 0.0:
            a[i] = 0.0


@temporarily_disable_autoopt_and_serialization
def test_single_assignment():
    if_cond_1 = np.array([1], dtype=np.float64)
    A = np.ones(shape=(N, ), dtype=np.float64)
    run_and_compare(single_assignment, 0, True, "single_assignment", a=A, _if_cond_1=if_cond_1[0])


@temporarily_disable_autoopt_and_serialization
def test_single_assignment_cond_from_scalar():
    A = np.ones(shape=(512, ), dtype=np.float64)
    before = single_assignment_cond_from_scalar.to_sdfg()
    before.compile()
    run_and_compare(single_assignment_cond_from_scalar, 0, True, "single_assignment_cond_from_scalar", a=A)


def _get_sdfg_with_condition_from_transient_scalar() -> dace.SDFG:
    sdfg = dace.SDFG("sd1")

    sdfg.add_scalar("zacond_0", transient=True, dtype=dace.float64)
    sdfg.add_scalar("_if_cond_41", transient=True, dtype=dace.float64)
    sdfg.add_array("zsolac", (N, ), dace.float64)
    sdfg.add_array("zlcond2", (N, ), dace.float64)
    sdfg.add_array("za", (N, ), dace.float64)
    sdfg.add_symbol("_if_cond_42", dace.float64)

    s1 = sdfg.add_state("s1", is_start_block=True)
    cb1 = ConditionalBlock(label="cb1", sdfg=sdfg, parent=sdfg)
    cfg1 = ControlFlowRegion(label="cfg1", sdfg=sdfg, parent=cb1)
    cb1.add_branch(condition=CodeBlock("_if_cond_41 == 1"), branch=cfg1)
    cb2 = ConditionalBlock(label="cb2", sdfg=sdfg, parent=sdfg)
    cfg2 = ControlFlowRegion(label="cfg2", sdfg=sdfg, parent=cb2)
    cb2.add_branch(condition=CodeBlock("_if_cond_42 == 1"), branch=cfg2)
    s2 = cfg1.add_state("s2", is_start_block=True)
    s3 = cfg2.add_state("s3", is_start_block=True)

    sdfg.add_edge(s1, cb1, InterstateEdge())
    sdfg.add_edge(cb1, cb2, InterstateEdge())
    s4 = sdfg.add_state_after(state=cb2, label="s4")

    # _if_cond_42 is a symbol (free symbol)
    # Calculate _if_cond_41, zacond_0 in s1
    # Calculate zacond_1 in s2
    # Calculate zsolac using zacond_0

    t1 = s1.add_tasklet(name="t1",
                        inputs={"_in1", "_in2"},
                        outputs={"_out"},
                        code="_out = ((_in1 < 0.3) and (_in2 < 0.5))")
    s1.add_edge(s1.add_access("za"), None, t1, "_in1", dace.memlet.Memlet("za[4]"))
    s1.add_edge(s1.add_access("zlcond2"), None, t1, "_in2", dace.memlet.Memlet("zlcond2[4]"))
    s1.add_edge(t1, "_out", s1.add_access("_if_cond_41"), None, dace.memlet.Memlet("_if_cond_41[0]"))

    t2 = s2.add_tasklet(name="t2_1", inputs=set(), outputs={"_out"}, code="_out = 0.0")
    s2.add_edge(t2, "_out", s2.add_access("zacond_0"), None, dace.memlet.Memlet("zacond_0[0]"))

    t2_2 = s2.add_tasklet(name="t2_2", inputs=set(), outputs={"_out"}, code="_out = 0.0")
    s2.add_edge(t2_2, "_out", s2.add_access("zlcond2"), None, dace.memlet.Memlet("zlcond2[4]"))

    t3 = s3.add_tasklet(name="t3", inputs=set(), outputs={"_out"}, code="_out = 0.5")
    s3.add_edge(t3, "_out", s3.add_access("zacond_0"), None, dace.memlet.Memlet("zacond_0[0]"))

    t4 = s4.add_tasklet(name="t4", inputs={"_in1", "_in2"}, outputs={"_out"}, code="_out = _in1 + _in2")
    s4.add_edge(t4, "_out", s4.add_access("zsolac"), None, dace.memlet.Memlet("zsolac[4]"))
    s4.add_edge(s4.add_access("zsolac"), None, t4, "_in1", dace.memlet.Memlet("zsolac[4]"))
    s4.add_edge(s4.add_access("zacond_0"), None, t4, "_in2", dace.memlet.Memlet("zacond_0[0]"))

    sdfg.validate()
    return sdfg


@temporarily_disable_autoopt_and_serialization
def test_condition_from_transient_scalar():
    zsolac = np.random.choice([8.0, 11.0], size=(N, ))
    zlcond2 = np.random.choice([8.0, 11.0], size=(N, ))
    za = np.random.choice([8.0, 11.0], size=(N, ))
    _if_cond_42 = np.random.choice([8.0, 11.0], size=(1, ))
    sdfg = _get_sdfg_with_condition_from_transient_scalar()

    transformed_sdfg = run_and_compare_sdfg(sdfg,
                                            permissive=False,
                                            sdfg_name="condition_from_transient_scalar",
                                            zsolac=zsolac,
                                            zlcond2=zlcond2,
                                            za=za,
                                            _if_cond_42=_if_cond_42[0])

    branch_code = {n for n, g in transformed_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branch_code) == 0, f"(actual) len({branch_code}) != (desired) {0}"


def _get_disjoint_chain_sdfg() -> dace.SDFG:
    sd1 = dace.SDFG("disjoint_chain")
    cb1 = ConditionalBlock("cond_if_cond_58", sdfg=sd1, parent=sd1)
    ss1 = sd1.add_state(label="pre", is_start_block=True)
    sd1.add_node(cb1, is_start_block=False)

    cfg1 = ControlFlowRegion(label="cond_58_true", sdfg=sd1, parent=cb1)
    s1 = cfg1.add_state("main_1", is_start_block=True)
    cfg2 = ControlFlowRegion(label="cond_58_false", sdfg=sd1, parent=cb1)
    s2 = cfg2.add_state("main_2", is_start_block=True)

    cb1.add_branch(
        condition=CodeBlock("_if_cond_58 == 1"),
        branch=cfg1,
    )
    cb1.add_branch(
        condition=None,
        branch=cfg2,
    )
    for arr_name, shape in [
        ("zsolqa", (N, 5, 5)),
        ("zrainaut", (N, )),
        ("zrainacc", (N, )),
        ("ztp1", (N, )),
    ]:
        sd1.add_array(arr_name, shape, dace.float64)
    sd1.add_scalar("rtt", dace.float64)
    sd1.add_symbol("_if_cond_58", dace.float64)
    sd1.add_symbol("_for_it_52", dace.int64)
    sd1.add_edge(src=ss1, dst=cb1, data=InterstateEdge(assignments={
        "_if_cond_58": "ztp1[_for_it_52] <= rtt",
    }, ))

    for state, d1_access_str, zsolqa_access_str, zsolqa_access_str_rev in [
        (s1, "_for_it_52", "_for_it_52,3,0", "_for_it_52,0,3"), (s2, "_for_it_52", "_for_it_52,2,0", "_for_it_52,0,2")
    ]:
        zrainaut = state.add_access("zrainaut")
        zrainacc = state.add_access("zrainacc")
        zsolqa1 = state.add_access("zsolqa")
        zsolqa2 = state.add_access("zsolqa")
        zsolqa3 = state.add_access("zsolqa")
        zsolqa4 = state.add_access("zsolqa")
        zsolqa5 = state.add_access("zsolqa")
        for i, (tasklet_code, in1, instr1, in2, instr2, out, outstr) in enumerate([
            ("_out = _in1 + _in2", zrainaut, d1_access_str, zsolqa1, zsolqa_access_str, zsolqa2, zsolqa_access_str),
            ("_out = _in1 + _in2", zrainacc, d1_access_str, zsolqa2, zsolqa_access_str, zsolqa3, zsolqa_access_str),
            ("_out = (-_in1) + _in2", zrainaut, d1_access_str, zsolqa3, zsolqa_access_str_rev, zsolqa4,
             zsolqa_access_str_rev),
            ("_out = (-_in1) + _in2", zrainacc, d1_access_str, zsolqa4, zsolqa_access_str_rev, zsolqa5,
             zsolqa_access_str_rev),
        ]):
            t1 = state.add_tasklet("t1", {"_in1", "_in2"}, {"_out"}, tasklet_code)
            state.add_edge(in1, None, t1, "_in1", dace.memlet.Memlet(f"{in1.data}[{instr1}]"))
            state.add_edge(in2, None, t1, "_in2", dace.memlet.Memlet(f"{in2.data}[{instr2}]"))
            state.add_edge(t1, "_out", out, None, dace.memlet.Memlet(f"{out.data}[{outstr}]"))

    sd1.validate()

    sd2 = dace.SDFG("sd2")
    p_s1 = sd2.add_state("p_s1", is_start_block=True)

    map_entry, map_exit = p_s1.add_map(name="map1", ndrange={"_for_it_52": dace.subsets.Range([(0, N - 1, 1)])})
    nsdfg = p_s1.add_nested_sdfg(sdfg=sd1,
                                 inputs={"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"},
                                 outputs={"zsolqa"},
                                 symbol_mapping={"_for_it_52": "_for_it_52"})
    for arr_name, shape in [("zsolqa", (N, 5, 5)), ("zrainaut", (N, )), ("zrainacc", (N, )), ("ztp1", (N, ))]:
        sd2.add_array(arr_name, shape, dace.float64)
    sd2.add_scalar("rtt", dace.float64)
    for input_name in {"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"}:
        a = p_s1.add_access(input_name)
        p_s1.add_edge(a, None, map_entry, f"IN_{input_name}",
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        p_s1.add_edge(map_entry, f"OUT_{input_name}", nsdfg, input_name,
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        map_entry.add_in_connector(f"IN_{input_name}")
        map_entry.add_out_connector(f"OUT_{input_name}")
    for output_name in {"zsolqa"}:
        a = p_s1.add_access(output_name)
        p_s1.add_edge(map_exit, f"OUT_{output_name}", a, None,
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        p_s1.add_edge(nsdfg, output_name, map_exit, f"IN_{output_name}",
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        map_exit.add_in_connector(f"IN_{output_name}")
        map_exit.add_out_connector(f"OUT_{output_name}")

    nsdfg.sdfg.parent_nsdfg_node = nsdfg

    sd1.validate()
    sd2.validate()
    return sd2, p_s1


@pytest.mark.parametrize("rtt_val", [0.0, 4.0, 6.0])
@temporarily_disable_autoopt_and_serialization
def test_disjoint_chain_split_branch_only(rtt_val):
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg()
    sdfg.name = f"disjoint_chain_split_branch_only_rtt_val_{str(rtt_val).replace('.','_')}"
    zsolqa = np.random.choice([0.001, 5.0], size=(N, 5, 5))
    zrainacc = np.random.choice([0.001, 5.0], size=(N, ))
    zrainaut = np.random.choice([0.001, 5.0], size=(N, ))
    ztp1 = np.random.choice([3.5, 5.0], size=(N, ))
    rtt = np.random.choice([rtt_val], size=(1, ))

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_branch_eliminated"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)

    # Run SDFG version (with transformation)
    xform = branch_elimination.BranchElimination()
    cblocks = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1
    cblock = cblocks.pop()

    xform.conditional = cblock
    xform.parent_nsdfg_state = nsdfg_parent_state
    xform.sequentialize_if_else_branch_if_disjoint_subsets(cblock.parent_graph)

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg(**out_fused)

    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


@pytest.mark.parametrize("rtt_val", [0.0, 4.0, 6.0])
@temporarily_disable_autoopt_and_serialization
def test_disjoint_chain(rtt_val):
    sdfg, _ = _get_disjoint_chain_sdfg()
    zsolqa = np.random.choice([0.001, 5.0], size=(N, 5, 5))
    zrainacc = np.random.choice([0.001, 5.0], size=(N, ))
    zrainaut = np.random.choice([0.001, 5.0], size=(N, ))
    ztp1 = np.random.choice([3.5, 5.0], size=(N, ))
    rtt = np.random.choice([rtt_val], size=(1, ))

    run_and_compare_sdfg(sdfg,
                         permissive=False,
                         sdfg_name=f"disjoint_chain_rtt_val_{str(rtt_val).replace('.', '_')}",
                         zsolqa=zsolqa,
                         zrainacc=zrainacc,
                         zrainaut=zrainaut,
                         ztp1=ztp1,
                         rtt=rtt[0])


@dace.program
def pattern_from_cloudsc_one(
    A: dace.float64[2, N, N],
    B: dace.float64[N, N],
    c: dace.float64,
    D: dace.float64[N, N],
    E: dace.float64[N, N],
):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            B[i, j] = A[1, i, j] + A[0, i, j]
            _if_cond_5 = B[i, j] > c
            if _if_cond_5:
                D[i, j] = B[i, j] / A[0, i, j]
                E[i, j] = 1.0 - D[i, j]
            else:
                D[i, j] = 0.0
                E[i, j] = 0.0


@pytest.mark.parametrize("c_val", [0.0, 1.0, 6.0])
@temporarily_disable_autoopt_and_serialization
def test_pattern_from_cloudsc_one(c_val):
    A = np.random.choice([0.001, 5.0], size=(
        2,
        N,
        N,
    ))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.array([c_val], )
    D = np.random.choice([0.001, 5.0], size=(N, N))
    E = np.random.choice([0.001, 5.0], size=(N, N))

    run_and_compare(pattern_from_cloudsc_one,
                    0,
                    True,
                    f"pattern_from_cloudsc_one_c_val_{str(c_val).replace('.', '_')}",
                    A=A,
                    B=B,
                    c=C[0],
                    D=D,
                    E=E)


@dace.program
def map_param_usage(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i in dace.map[0:N]:
        _if_cond_1 = d[i, i] > 0.0
        if _if_cond_1:
            tc1 = d[i, i] + a[i, i]
            zmdn = tc1
        else:
            zmdn = 0.0
        b[i, i] = zmdn
        b[i, i] = zmdn


@temporarily_disable_autoopt_and_serialization
def test_can_be_applied_on_map_param_usage():
    A = np.random.choice([0.001, 5.0], size=(
        N,
        N,
    ))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    D = np.random.choice([0.001, 5.0], size=(N, N))

    sdfg = map_param_usage.to_sdfg()

    xform = branch_elimination.BranchElimination()
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1
    xform.conditional = cblocks.pop()
    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)

    assert xform.can_be_applied(xform.conditional.parent_graph, 0, xform.conditional.sdfg, False)

    run_and_compare(map_param_usage, 0, True, "can_be_applied_on_map_param_usage_tester", a=A, b=B, d=D)


def _get_safe_map_param_use_in_nested_sdfg() -> dace.SDFG:
    inner_sdfg = dace.SDFG("inner")
    outer_sdfg = dace.SDFG("outer")

    inner_symbol_mapping = {
        "_for_it_37": "_for_it_37",
    }
    for arr_name in ["zsolac", "zacust", "zfinalsum"]:
        inner_sdfg.add_array(arr_name, (N, ), dace.float64)
        outer_sdfg.add_array(arr_name, (N, ), dace.float64)
    inner_inputs = {"zsolac", "zacust", "zfinalsum"}
    inner_outputs = {"zacust", "zsolac"}

    i_s1 = inner_sdfg.add_state("i_s1", is_start_block=True)
    i_cb1 = ConditionalBlock("i_cb1", sdfg=inner_sdfg, parent=inner_sdfg)
    inner_sdfg.add_node(i_cb1)
    inner_sdfg.add_edge(i_s1, i_cb1, InterstateEdge(assignments={"_if_cond_22": "zfinalsum[_for_it_37] < 1e-14"}))
    inner_sdfg.add_symbol("_if_cond_22", dace.int32)
    inner_sdfg.add_symbol("_for_it_37", dace.int32)

    i_cfg1 = ControlFlowRegion("i_cfg1", sdfg=inner_sdfg, parent=i_cb1)
    i_cfg1_s1 = i_cfg1.add_state("i_cfg1_s1", is_start_block=True)

    t1 = i_cfg1_s1.add_tasklet("t1", inputs={}, outputs={"_out"}, code="_out = 0.0")
    i_cfg1_s1.add_edge(t1, "_out", i_cfg1_s1.add_access("zacust"), None, dace.memlet.Memlet("zacust[_for_it_37]"))

    i_cb1.add_branch(CodeBlock("_if_cond_22 == 1"), i_cfg1)

    i_s2 = inner_sdfg.add_state_after(i_cb1, label="i_s2")
    t2 = i_s2.add_tasklet("t2", inputs={"_in1", "_in2"}, outputs={"_out"}, code="_out = _in1 + _in2")
    for in_name, conn_name in [("zacust", "_in1"), ("zsolac", "_in2")]:
        i_s2.add_edge(i_s2.add_access(in_name), None, t2, conn_name, dace.memlet.Memlet(f"{in_name}[_for_it_37]"))
    i_s2.add_edge(t2, "_out", i_s2.add_access("zsolac"), None, dace.memlet.Memlet(f"zsolac[_for_it_37]"))

    o_s1 = outer_sdfg.add_state("o_s1", is_start_block=True)
    nsdfg = o_s1.add_nested_sdfg(sdfg=inner_sdfg,
                                 inputs=inner_inputs,
                                 outputs=inner_outputs,
                                 symbol_mapping=inner_symbol_mapping)

    map_entry, map_exit = o_s1.add_map(name="m1", ndrange={
        "_for_it_37": dace.subsets.Range([(0, N - 1, 1)]),
    })
    for in_name in inner_inputs:
        o_s1.add_edge(o_s1.add_access(in_name), None, map_entry, f"IN_{in_name}",
                      dace.memlet.Memlet.from_array(in_name, o_s1.sdfg.arrays[in_name]))
        map_entry.add_in_connector(f"IN_{in_name}")
        map_entry.add_out_connector(f"OUT_{in_name}")
        o_s1.add_edge(map_entry, f"OUT_{in_name}", nsdfg, in_name,
                      dace.memlet.Memlet.from_array(in_name, o_s1.sdfg.arrays[in_name]))
    for out_name in inner_outputs:
        o_s1.add_edge(nsdfg, out_name, map_exit, f"IN_{out_name}",
                      dace.memlet.Memlet.from_array(out_name, o_s1.sdfg.arrays[out_name]))
        map_exit.add_in_connector(f"IN_{out_name}")
        map_exit.add_out_connector(f"OUT_{out_name}")
        o_s1.add_edge(map_exit, f"OUT_{out_name}", o_s1.add_access(out_name), None,
                      dace.memlet.Memlet.from_array(out_name, o_s1.sdfg.arrays[out_name]))

    outer_sdfg.validate()
    return outer_sdfg


@temporarily_disable_autoopt_and_serialization
def test_safe_map_param_use_in_nested_sdfg():
    sdfg = _get_safe_map_param_use_in_nested_sdfg()
    sdfg.validate()

    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, ConditionalBlock):
            xform = branch_elimination.BranchElimination()
            xform.conditional = n
            xform.parent_nsdfg_state = _find_state(sdfg, g.sdfg.parent_nsdfg_node)
            assert xform.can_be_applied(graph=g, expr_index=0, sdfg=g.sdfg, permissive=False)
            assert xform.can_be_applied(graph=g, expr_index=0, sdfg=g.sdfg, permissive=True)

    # "zsolac", "zacust", "zlfinalsum"
    zsolac = np.random.choice([0.001, 5.0], size=(N, ))
    zfinalsum = np.random.choice([0.001, 5.0], size=(N, ))
    zacust = np.random.choice([0.001, 5.0], size=(N, ))
    run_and_compare_sdfg(sdfg,
                         False,
                         f"safe_map_param_use_in_nested_sdfg",
                         zsolac=zsolac,
                         zfinalsum=zfinalsum,
                         zacust=zacust)


def _get_nsdfg_with_return(return_arr: bool) -> dace.SDFG:
    inner_sdfg = dace.SDFG("inner")
    outer_sdfg = dace.SDFG("outer")

    inner_symbol_mapping = {}
    for outer_arr_name in ["ztp"]:
        outer_sdfg.add_array(outer_arr_name, (N, N), dace.float64)
    for outer_scalar_name in ["rtt"]:
        outer_sdfg.add_scalar(outer_scalar_name, dace.float64)
    outer_sdfg.add_array("zalfa_1", (1, ), dace.float64)
    if return_arr:
        inner_sdfg.add_array("foedelta__ret", (1, ), dace.float64)
    else:
        inner_sdfg.add_scalar("foedelta__ret", dace.float64)
    for inner_scalar_name in ["ptare_var_0", "rtt_var_1"]:
        inner_sdfg.add_scalar(inner_scalar_name, dace.float64)
    for inner_tmp_name in ["tmp_call_103", "tmp_call_1"]:
        inner_sdfg.add_scalar(inner_tmp_name, dace.float64, transient=True)

    inner_inputs = {"ptare_var_0", "rtt_var_1"}
    inner_outputs = {"foedelta__ret"}
    inner_to_outer_name_mapping_in = {
        "ptare_var_0": ("rtt", "[0]"),
        "rtt_var_1": ("ztp", "[4,4]"),
    }
    inner_to_outer_name_mapping_out = {"foedelta__ret": ("zalfa_1", "[0]")}

    i_s1 = inner_sdfg.add_state("i_s1", is_start_block=True)
    i_cb1 = ConditionalBlock("i_cb1", sdfg=inner_sdfg, parent=inner_sdfg)
    inner_sdfg.add_node(i_cb1)
    inner_sdfg.add_edge(i_s1, i_cb1, InterstateEdge())

    i_cfg1 = ControlFlowRegion("i_cfg1", sdfg=inner_sdfg, parent=i_cb1)
    i_cfg1_s1 = i_cfg1.add_state("i_cfg1_s1", is_start_block=True)
    i_cfg2 = ControlFlowRegion("i_cfg2", sdfg=inner_sdfg, parent=i_cb1)
    i_cfg2_s1 = i_cfg2.add_state("i_cfg2_s1", is_start_block=True)

    t1 = i_cfg1_s1.add_tasklet("t1", inputs={}, outputs={"_out"}, code="_out = 0.0")
    i_cfg1_s1.add_edge(t1, "_out", i_cfg1_s1.add_access("tmp_call_103"), None, dace.memlet.Memlet("tmp_call_103[0]"))
    t2 = i_cfg2_s1.add_tasklet("t2", inputs={}, outputs={"_out"}, code="_out = 1.0")
    taccess_1 = i_cfg2_s1.add_access("tmp_call_1")
    i_cfg2_s1.add_edge(t2, "_out", taccess_1, None, dace.memlet.Memlet("tmp_call_1[0]"))
    t3 = i_cfg2_s1.add_tasklet("t3", inputs={"_in1"}, outputs={"_out"}, code="_out = (- _in1)")
    i_cfg2_s1.add_edge(taccess_1, None, t3, "_in1", dace.memlet.Memlet("tmp_call_1[0]"))
    i_cfg2_s1.add_edge(t3, "_out", i_cfg2_s1.add_access("tmp_call_103"), None, dace.memlet.Memlet("tmp_call_103[0]"))

    i_cb1.add_branch(CodeBlock("(ptare_var_0 - rtt_var_1) >= 0.0"), i_cfg1)
    i_cb1.add_branch(None, i_cfg2)

    i_s2 = inner_sdfg.add_state_after(i_cb1, label="i_s2")
    t4 = i_s2.add_tasklet("t4", inputs={"_in1"}, outputs={"_out"}, code="_out = max(0.0, _in1)")
    i_s2.add_edge(i_s2.add_access("tmp_call_103"), None, t4, "_in1", dace.memlet.Memlet("tmp_call_103[0]"))
    i_s2.add_edge(t4, "_out", i_s2.add_access("foedelta__ret"), None, dace.memlet.Memlet("foedelta__ret[0]"))

    o_s1 = outer_sdfg.add_state("o_s1", is_start_block=True)
    nsdfg = o_s1.add_nested_sdfg(sdfg=inner_sdfg,
                                 inputs=inner_inputs,
                                 outputs=inner_outputs,
                                 symbol_mapping=inner_symbol_mapping)

    for inner_name, (outer_name, access_str) in inner_to_outer_name_mapping_in.items():
        o_s1.add_edge(o_s1.add_access(outer_name), None, nsdfg, inner_name,
                      dace.memlet.Memlet(f"{outer_name}{access_str}"))
    for inner_name, (outer_name, access_str) in inner_to_outer_name_mapping_out.items():
        o_s1.add_edge(nsdfg, inner_name, o_s1.add_access(outer_name), None,
                      dace.memlet.Memlet(f"{outer_name}{access_str}"))

    outer_sdfg.validate()
    return outer_sdfg


@pytest.mark.parametrize("ret_arr", [True, False])
@temporarily_disable_autoopt_and_serialization
def test_nested_sdfg_with_return(ret_arr):
    sdfg = _get_nsdfg_with_return(ret_arr)
    sdfg.validate()
    sdfg.name = f"nested_sdfg_with_return_ret_arr_{str(ret_arr).lower()}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_branch_eliminated"

    for n, g in copy_sdfg.all_nodes_recursive():
        if isinstance(n, ConditionalBlock):
            xform = branch_elimination.BranchElimination()
            xform.conditional = n
            xform.parent_nsdfg_state = _find_state(copy_sdfg, g.sdfg.parent_nsdfg_node)
            assert xform.can_be_applied(graph=g, expr_index=0, sdfg=g.sdfg, permissive=False)
            assert xform.can_be_applied(graph=g, expr_index=0, sdfg=g.sdfg, permissive=True)

    ztp = np.random.choice([0.001, 5.0], size=(N, N))
    rtt = np.random.choice([10.0, 15.0], size=(1, ))
    zalfa_1 = np.array([999.9])
    arrays = {"ztp": ztp, "rtt": rtt[0], "zalfa_1": zalfa_1}

    # Run SDFG version (no transformation)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    assert out_no_fuse["zalfa_1"][0] != 999.9

    # Run SDFG version (with transformation)
    fb = EliminateBranches()
    fb.try_clean = True
    fb.permissive = False
    fb.apply_to_top_level_ifs = True
    fb.apply_pass(copy_sdfg, {})
    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg(**out_fused)
    assert out_fused["zalfa_1"][0] != 999.9

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


@dace.program
def mid_sdfg(pap: dace.float64[N], ptsphy: dace.float64, r2es: dace.float64, r3ies: dace.float64, r4ies: dace.float64,
             rcldtopcf: dace.float64, rd: dace.float64, rdepliqrefdepth: dace.float64, rdepliqrefrate: dace.float64,
             rg: dace.float64, riceinit: dace.float64, rlmin: dace.float64, rlstt: dace.float64, rtt: dace.float64,
             rv: dace.float64, za: dace.float64[N], zdp: dace.float64[N], zfokoop: dace.float64[N],
             zicecld: dace.float64[N], zrho: dace.float64[N], ztp1: dace.float64[N], zcldtopdist: dace.float64[N],
             zicenuclei: dace.float64[N], zqxfg: dace.float64[N], zsolqa: dace.float64[N]):
    for it_47 in dace.map[
            0:N:1,
    ]:
        # Ice nucleation and deposition
        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            # Calculate ice saturation vapor pressure
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            zicenuclei[it_47] = 2.0 * np.exp(tmp_arg_72)
            # Deposition calculation parameters
            zadd = (1.6666666666667 * rlstt * (rlstt / ztp1[it_47]))
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47])
            # Update mixing ratios
            zqxfg[it_47] = zqxfg[it_47] + zadd
            zsolqa[it_47] = zqxfg[it_47] + zbdd


@dace.program
def huge_sdfg(pap: dace.float64[N], ptsphy: dace.float64, r2es: dace.float64, r3ies: dace.float64, r4ies: dace.float64,
              rcldtopcf: dace.float64, rd: dace.float64, rdepliqrefdepth: dace.float64, rdepliqrefrate: dace.float64,
              rg: dace.float64, riceinit: dace.float64, rlmin: dace.float64, rlstt: dace.float64, rtt: dace.float64,
              rv: dace.float64, za: dace.float64[N], zdp: dace.float64[N], zfokoop: dace.float64[N],
              zicecld: dace.float64[N], zrho: dace.float64[N], ztp1: dace.float64[N], zcldtopdist: dace.float64[N],
              zicenuclei: dace.float64[N], zqxfg: dace.float64[N], zsolqa: dace.float64[N]):
    for it_47 in dace.map[
            0:N:1,
    ]:
        # Check if crossing cloud top threshold
        if za[it_47] < rcldtopcf and za[it_47] >= rcldtopcf:
            zcldtopdist[it_47] = 0.0
        else:
            zcldtopdist[it_47] = zcldtopdist[it_47] + (zdp[it_47] / (rg * zrho[it_47]))

        # Ice nucleation and deposition
        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            # Calculate ice saturation vapor pressure
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            tmp_call_47 = r2es * np.exp(tmp_arg_72)
            zvpice = (rv * tmp_call_47) / rd

            # Calculate liquid vapor pressure
            zvpliq = zfokoop[it_47] * np.log(zvpice)

            # Ice nuclei concentration
            tmp_arg_27 = -0.639 + ((-1.96 * zvpice + 1.96 * zvpliq) / zvpliq)
            zicenuclei[it_47] = 1000.0 * np.exp(tmp_arg_27)

            # Nucleation factor
            zinfactor = min(1.0, 6.66666666666667e-05 * zicenuclei[it_47])

            # Deposition calculation parameters
            zadd = (1.6666666666667 * rlstt * (rlstt / (rv * ztp1[it_47]) - 1.0)) / ztp1[it_47]
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47]) / zvpice

            tmp_call_49 = (zicenuclei[it_47] / zrho[it_47])
            zcvds = (7.8 * tmp_call_49 * (zvpliq - zvpice)) / (zvpice * (zadd + zbdd))

            # Initial ice content
            zice0 = max(riceinit * zicenuclei[it_47] / zrho[it_47], zicecld[it_47])

            # New ice after deposition
            tmp_arg_30 = 0.666 * ptsphy * zcvds + zice0
            zinew = tmp_arg_30**1.5

            # Deposition amount
            zdepos1 = max(0.0, za[it_47] * (zinew - zice0))
            zdepos2 = min(zdepos1, 1.1)

            # Apply nucleation factor and cloud top distance factor
            tmp_arg_33 = zinfactor + (1.0 - zinfactor) * (rdepliqrefrate + zcldtopdist[it_47] / rdepliqrefdepth)
            zdepos3 = zdepos2 * min(1.0, tmp_arg_33)

            # Update mixing ratios
            zqxfg[it_47] = zqxfg[it_47] + zdepos3
            zsolqa[it_47] = zsolqa[it_47] + zdepos3


@pytest.mark.parametrize("eps_operator_type_for_log_and_div", ["max", "add"])
@temporarily_disable_autoopt_and_serialization
def test_huge_sdfg_with_log_exp_div(eps_operator_type_for_log_and_div: str):
    """Generate test data for the loop body function"""

    data = {
        'ptsphy': np.float64(36.0),  # timestep (s)
        'r2es': np.float64(6.11),  # saturation vapor pressure constant (hPa)
        'r3ies': np.float64(12.0),  # ice saturation constant
        'r4ies': np.float64(15.5),  # ice saturation constant
        'rcldtopcf': np.float64(16.8),  # cloud top threshold
        'rd': np.float64(287.0),  # gas constant for dry air (J/kg/K)
        'rdepliqrefdepth': np.float64(20.0),  # reference depth
        'rdepliqrefrate': np.float64(17.3),  # reference rate
        'rg': np.float64(9.81),  # gravity (m/s²)
        'riceinit': np.float64(5.3),  # initial ice content (kg/m³)
        'rlmin': np.float64(3.9),  # minimum liquid water (kg/m³)
        'rlstt': np.float64(2.5e6),  # latent heat (J/kg)
        'rtt': np.float64(273.15),  # triple point temperature (K)
        'rv': np.float64(461.5),  # gas constant for water vapor (J/kg/K)
    }

    # 1D arrays with safe ranges
    rng = np.random.default_rng(0)

    def safe_uniform(low, high, size):
        """Avoid near-zero or extreme values that could cause NaN in log/div."""
        return rng.uniform(low, high, size).astype(np.float64)

    # State variables (N = grid size)
    data['pap'] = safe_uniform(1.0, 2.0, (N, ))  # pressure-like
    data['za'] = safe_uniform(0.9, 1.5, (N, ))  # altitude/cloud-top
    data['ztp1'] = safe_uniform(260.0, 280.0, (N, ))  # temperature near freezing
    data['zqxfg'] = safe_uniform(5.0, 11.0, (N, ))  # mixing ratios
    data['zsolqa'] = safe_uniform(5.0, 11.0, (N, ))  # ice tendencies

    data['zdp'] = safe_uniform(0.5, 2.0, (N, ))  # layer depth
    data['zfokoop'] = safe_uniform(0.95, 1.05, (N, ))  # correction factor
    data['zicecld'] = safe_uniform(10.0, 11.0, (N, ))  # cloud ice
    data['zrho'] = safe_uniform(0.9, 1.2, (N, ))  # density
    data['zcldtopdist'] = safe_uniform(0.1, 1.0, (N, ))  # distance to cloud top
    data['zicenuclei'] = safe_uniform(1e2, 1e4, (N, ))  # ice nuclei concentration

    sdfg = huge_sdfg.to_sdfg()
    sdfg.name = f"huge_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}"
    sdfg.validate()
    #it_23: dace.int64, it_47: dace.int64
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    sdfg.validate()
    ConstantPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.auto_optimize(dace.dtypes.DeviceType.CPU, True, True)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in data.items()}
    sdfg(**out_no_fuse)

    # Apply transformation
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"huge_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}_branch_eliminated"
    fb = EliminateBranches()
    fb.try_clean = True
    fb.apply_to_top_level_ifs = True
    fb.eps_operator_type_for_log_and_div = eps_operator_type_for_log_and_div
    fb.apply_pass(copy_sdfg, {})

    cblocks = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in data.items()}

    copy_sdfg(**out_fused)

    # Compare all arrays
    for name in data.keys():
        print(name)
        print(out_fused[name] - out_no_fuse[name])
        np.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


@pytest.mark.parametrize("eps_operator_type_for_log_and_div", ["max", "add"])
@temporarily_disable_autoopt_and_serialization
def test_mid_sdfg_with_log_exp_div(eps_operator_type_for_log_and_div: str):
    """Generate test data for the loop body function"""

    data = {
        'ptsphy': np.float64(36.0),  # timestep (s)
        'r2es': np.float64(6.11),  # saturation vapor pressure constant (hPa)
        'r3ies': np.float64(12.0),  # ice saturation constant
        'r4ies': np.float64(15.5),  # ice saturation constant
        'rcldtopcf': np.float64(16.8),  # cloud top threshold
        'rd': np.float64(287.0),  # gas constant for dry air (J/kg/K)
        'rdepliqrefdepth': np.float64(20.0),  # reference depth
        'rdepliqrefrate': np.float64(17.3),  # reference rate
        'rg': np.float64(9.81),  # gravity (m/s²)
        'riceinit': np.float64(5.3),  # initial ice content (kg/m³)
        'rlmin': np.float64(3.9),  # minimum liquid water (kg/m³)
        'rlstt': np.float64(2.5e6),  # latent heat (J/kg)
        'rtt': np.float64(273.15),  # triple point temperature (K)
        'rv': np.float64(461.5),  # gas constant for water vapor (J/kg/K)
    }

    # 1D arrays with safe ranges
    rng = np.random.default_rng(0)

    def safe_uniform(low, high, size):
        """Avoid near-zero or extreme values that could cause NaN in log/div."""
        return rng.uniform(low, high, size).astype(np.float64)

    # State variables (N = grid size)
    data['pap'] = safe_uniform(1.0, 2.0, (N, ))  # pressure-like
    data['za'] = safe_uniform(0.9, 1.5, (N, ))  # altitude/cloud-top
    data['ztp1'] = safe_uniform(260.0, 280.0, (N, ))  # temperature near freezing
    data['zqxfg'] = safe_uniform(5.0, 11.0, (N, ))  # mixing ratios
    data['zsolqa'] = safe_uniform(5.0, 11.0, (N, ))  # ice tendencies

    data['zdp'] = safe_uniform(0.5, 2.0, (N, ))  # layer depth
    data['zfokoop'] = safe_uniform(0.95, 1.05, (N, ))  # correction factor
    data['zicecld'] = safe_uniform(10.0, 11.0, (N, ))  # cloud ice
    data['zrho'] = safe_uniform(0.9, 1.2, (N, ))  # density
    data['zcldtopdist'] = safe_uniform(0.1, 1.0, (N, ))  # distance to cloud top
    data['zicenuclei'] = safe_uniform(1e2, 1e4, (N, ))  # ice nuclei concentration
    sdfg = mid_sdfg.to_sdfg()
    sdfg.name = f"mid_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"mid_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}_branch_eliminated"

    sdfg.validate()
    copy_sdfg.validate()

    #it_23: dace.int64, it_47: dace.int64
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    sdfg.validate()
    ConstantPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.auto_optimize(dace.dtypes.DeviceType.CPU, True, True)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in data.items()}
    sdfg(**out_no_fuse)

    # Apply transformation

    fb = EliminateBranches()
    fb.try_clean = True
    fb.apply_to_top_level_ifs = True
    fb.eps_operator_type_for_log_and_div = eps_operator_type_for_log_and_div
    fb.apply_pass(copy_sdfg, {})

    cblocks = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in data.items()}

    copy_sdfg(**out_fused)

    # Compare all arrays
    for name in data.keys():
        print(name)
        print(out_fused[name] - out_no_fuse[name])
        np.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


@dace.program
def wcr_edge(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        cond = A[i, j]
        if cond > 0.00000000001:
            A[i, j] += 2.0


@dace.program
def loop_param_usage(A: dace.float64[6, N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i in range(6):
        for j in dace.map[0:N]:
            for k in dace.map[0:N]:
                if A[i, j, k] > 2.0:
                    C[i, j] = 2.0 + C[i, j]


@temporarily_disable_autoopt_and_serialization
def test_loop_param_usage():
    A = np.random.choice([0.001, 5.0], size=(6, N, N))
    B = np.random.choice([0.001, 5.0], size=(N, N))
    C = np.random.choice([0.001, 5.0], size=(N, N))

    sdfg = loop_param_usage.to_sdfg()
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1

    for cblock in cblocks:
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        xform.parent_nsdfg_state = _find_state(
            sdfg, cblock.sdfg.parent_nsdfg_node) if cblock.sdfg.parent_nsdfg_node is not None else None
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, False) is True
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, True) is True

    run_and_compare_sdfg(sdfg, False, "loop_param_usage", A=A, B=B, C=C)


@temporarily_disable_autoopt_and_serialization
def test_can_be_applied_on_wcr_edge():
    sdfg = wcr_edge.to_sdfg()

    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1

    for cblock in cblocks:
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        xform.parent_nsdfg_state = _find_state(
            sdfg, cblock.sdfg.parent_nsdfg_node) if cblock.sdfg.parent_nsdfg_node is not None else None
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, False) is False
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, True) is False

    from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
    sdfg.apply_transformations_repeated(WCRToAugAssign)
    sdfg.validate()
    sdfg.save("x0.sdfg")
    sdfg.compile()

    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    for cblock in cblocks:
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        xform.parent_nsdfg_state = _find_state(
            sdfg, cblock.sdfg.parent_nsdfg_node) if cblock.sdfg.parent_nsdfg_node is not None else None
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, False) is True
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, True) is True

    A = np.random.choice([0.001, 5.0], size=(N, N))
    sdfg.save("x1.sdfg")

    run_and_compare_sdfg(sdfg, False, "can_be_applied_wcr", A=A)


@dace.program
def interstate_boolean_op(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        if c3:
            A[i, j] = A[i, j] + B[i, j]


def test_interstate_boolean():
    sdfg = interstate_boolean_op.to_sdfg()
    sdfg.name = "interstate_boolean_op"
    nsdfg = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}.pop()
    inner_sdfg: dace.SDFG = nsdfg.sdfg
    syms = inner_sdfg.symbols
    last_state = {s for s in inner_sdfg.nodes() if inner_sdfg.out_degree(s) == 0}.pop()
    last_last_state = inner_sdfg.add_state_after(last_state, "ssss", assignments={"symsym": "__tmp0 or __tmp1"})
    inner_sdfg.add_symbol("symsym", dace.int64)
    sdfg.validate()
    sdfg.save("interstate_boolean_op.sdfg")
    eb = EliminateBranches()
    eb.apply_to_top_level_ifs = True
    eb.apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.save("interstate_boolean_op_transformed.sdfg")
    sdfg.compile()


@temporarily_disable_autoopt_and_serialization
def test_top_level_if():
    sdfg = top_level_if.to_sdfg()

    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1

    # Can be applied should be true for this top level if
    for cblock in cblocks:
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        xform.parent_nsdfg_state = _find_state(
            sdfg, cblock.sdfg.parent_nsdfg_node) if cblock.sdfg.parent_nsdfg_node is not None else None
        assert xform.can_be_applied(cblock.parent_graph, 0, cblock.sdfg, False) is True

    eb = EliminateBranches()
    eb.apply_to_top_level_ifs = False
    eb.apply_pass(sdfg, {})
    # Pass should not convert top level ifs
    assert len({n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}) == 1


LEN_1D = 64


@dace.program
def dace_s441(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(50):  # or iterations parameter
        for i in range(LEN_1D):
            if d[i] < 0.0:
                a[i] = a[i] + b[i] * c[i]
            elif d[i] == 0.0:
                a[i] = a[i] + b[i] * b[i]
            else:
                a[i] = a[i] + c[i] * c[i]


def test_s441():
    LEN_1D_val = LEN_1D

    # Allocate random inputs
    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.randn(LEN_1D_val).astype(np.float64)  # includes negative, zero, positive

    sdfg = dace_s441.to_sdfg()
    EliminateBranches().apply_pass(sdfg, {})
    branches = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    if len(branches) > 0:
        sdfg.save("branch_elimination_failed_s441.sdfg")
        assert False

    run_and_compare_sdfg(sdfg, False, "branch_elimination_failed_s441", a=a, b=b, c=c, d=d)

    return a


@dace.program
def interstate_boolean_op_two(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        c4 = c3 or (A[i, j] > B[i, j])
        if not c4:
            A[i, j] = A[i, j] + B[i, j]


def test_interstate_boolean_op_two():
    # Allocate random inputs
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    c0 = np.int64(0)

    sdfg = interstate_boolean_op_two.to_sdfg()
    sdfg.save("interstate_boolean_op_two.sdfg")

    EliminateBranches().apply_pass(sdfg, {})
    branches = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    if len(branches) > 0:
        sdfg.save("interstate_boolean_op_two.sdfg")
        assert False
    sdfg.save("interstate_boolean_op_two_transformed.sdfg")
    run_and_compare_sdfg(sdfg, False, "interstate_boolean_op_two", A=A, B=B, c0=c0)


LEN_1D = dace.symbol("LEN_1D")
ITERATIONS = dace.symbol("ITERATIONS")


@dace.program
def dace_s1161(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if c[i] < 0.0:
                b[i] = a[i] + d[i] * d[i]
            else:
                a[i] = c[i] + d[i] * e[i]


def test_s1161():
    sdfg = dace_s1161.to_sdfg()
    sdfg.save("s1161.sdfg")
    from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
    CleanDataToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    sdfg.save("s1161_v2.sdfg")
    be = branch_elimination.BranchElimination()
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    for cblock in cblocks:
        xform = branch_elimination.BranchElimination()
        xform.conditional = cblock
        xform._split_branches(parent_graph=cblock.parent_graph, if_block=cblock)
    EliminateBranches().apply_pass(sdfg, {})
    branches = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branches) == 0


if __name__ == "__main__":
    test_s1161()
    test_top_level_if()
    test_interstate_boolean()
    test_interstate_boolean_op_two()
    test_huge_sdfg_with_log_exp_div("max")
    test_huge_sdfg_with_log_exp_div("add")
    test_mid_sdfg_with_log_exp_div("max")
    test_mid_sdfg_with_log_exp_div("add")
    test_nested_sdfg_with_return(True)
    test_nested_sdfg_with_return(False)
    test_safe_map_param_use_in_nested_sdfg()
    test_can_be_applied_on_map_param_usage()
    test_pattern_from_cloudsc_one(0.0)
    test_pattern_from_cloudsc_one(1.0)
    test_pattern_from_cloudsc_one(6.0)
    test_condition_on_bounds()
    test_nested_if_two()
    test_disjoint_chain_split_branch_only(0.0)
    test_disjoint_chain_split_branch_only(4.0)
    test_disjoint_chain_split_branch_only(6.0)
    test_disjoint_chain(0.0)
    test_disjoint_chain(4.0)
    test_disjoint_chain(6.0)
    test_condition_from_transient_scalar()
    test_single_assignment()
    test_single_assignment_cond_from_scalar()
    test_sdfg_with_interstate_array_condition()
    test_branch_dependent_value_write_with_transient_reuse()
    test_try_clean()
    test_try_clean_as_pass()
    test_repeated_condition_variables()
    test_weird_condition()
    test_if_over_map()
    test_if_over_map_with_top_level_tasklets()
    test_can_be_applied_parameters_on_nested_sdfg()
    test_non_trivial_subset_after_combine_tasklet()
    test_split_on_disjoint_subsets()
    test_split_on_disjoint_subsets_nested()
    test_write_to_transient()
    test_write_to_transient_two()
    test_double_empty_state()
    test_complicated_pattern_for_manual_clean_up_one()
    test_try_clean_on_complicated_pattern_for_manual_clean_up_one()
    test_try_clean_on_complicated_pattern_for_manual_clean_up_two()
    for use_pass_flag in [True, False]:
        test_branch_dependent_value_write(use_pass_flag)
        test_branch_dependent_value_write_two(use_pass_flag)
        test_branch_dependent_value_write_single_branch(use_pass_flag)
        test_branch_dependent_value_write_single_branch_nonzero_write(use_pass_flag)
        test_single_branch_connectors(use_pass_flag)
        test_complicated_if(use_pass_flag)
        test_multi_state_branch_body(use_pass_flag)
        test_nested_if(use_pass_flag)
        test_tasklets_in_if(use_pass_flag)
        test_disjoint_subsets(use_pass_flag)
