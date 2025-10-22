import numpy as np
import dace
import pytest
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.interstate import fuse_branches
from dace.transformation.passes import fuse_branches_pass
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion

N = 32
S = 32
S1 = 0
S2 = 32


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


def apply_fuse_branches(sdfg, nestedness: int = 1):
    """Apply FuseBranches transformation to all eligible conditionals."""
    # Pattern matching with conditional branches to not work (9.10.25), avoid it
    for i in range(nestedness):
        for node, graph in sdfg.all_nodes_recursive():
            parent_nsdfg_node = graph.sdfg.parent_nsdfg_node
            parent_state = None
            if parent_nsdfg_node is not None:
                parent_state = _get_parent_state(sdfg, parent_nsdfg_node)
            if isinstance(node, ConditionalBlock):
                t = fuse_branches.FuseBranches()
                if t.can_be_applied_to(graph.sdfg, conditional=node, options={"parent_nsdfg_state": parent_state}):
                    t.apply_to(graph.sdfg, conditional=node, options={"parent_nsdfg_state": parent_state})


def run_and_compare(
    program,
    num_expected_branches,
    use_pass,
    **arrays,
):
    # Run SDFG version (no transformation)
    sdfg = program.to_sdfg()
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    sdfg.save(sdfg.label + "_before.sdfg")
    # Apply transformation
    if use_pass:
        fuse_branches_pass.FuseBranchesPass().apply_pass(sdfg, {})
    else:
        apply_fuse_branches(sdfg, 2)

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in arrays.items()}
    sdfg.save("b.sdfg")
    sdfg.save(sdfg.label + "_after3.sdfg")

    sdfg(**out_fused)

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(
        branch_code) == num_expected_branches, f"(actual) len({branch_code}) != (desired) {num_expected_branches}"

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


def run_and_compare_sdfg(
    sdfg,
    **arrays,
):
    # Run SDFG version (no transformation)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    sdfg.save(sdfg.label + "_before.sdfg")

    # Run SDFG version (with transformation)
    fb = fuse_branches_pass.FuseBranchesPass()
    fb.try_clean = True
    fb.apply_pass(sdfg, {})
    out_fused = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_fused)
    sdfg.save(sdfg.label + "_after3.sdfg")

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_branch_dependent_value_write(use_pass_flag):
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write, 0, use_pass_flag, a=a, b=b, c=c, d=d)


def test_weird_condition():
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    ncldtop = np.array([N // 2], dtype=np.int64)
    run_and_compare(weird_condition, 0, False, a=a, b=b, ncldtop=ncldtop[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_branch_dependent_value_write_two(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.zeros((N, N))
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_two, 0, use_pass_flag, a=a, b=b, c=c, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_branch_dependent_value_write_single_branch(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_single_branch, 0, use_pass_flag, a=a, b=b, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_complicated_if(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    d = np.zeros((N, N))
    run_and_compare(complicated_if, 0, use_pass_flag, a=a, b=b, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_multi_state_branch_body(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    c = np.random.choice([0.0, 5.0], size=(N, N))
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(multi_state_branch_body, 1, use_pass_flag, a=a, b=b, c=c, d=d, s=s[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_nested_if(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(nested_if, 0, use_pass_flag, a=a, b=b, c=c, d=d, s=s[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_tasklets_in_if(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    c = np.zeros((1, ))
    d = np.zeros((N, N))
    run_and_compare(tasklets_in_if, 0, use_pass_flag, a=a, b=b, d=d, c=c[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_branch_dependent_value_write_single_branch_nonzero_write(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    d = np.random.choice([0.0, 5.0], size=(N, N))
    run_and_compare(branch_dependent_value_write_single_branch_nonzero_write, 0, use_pass_flag, a=a, b=b, d=d)


def test_branch_dependent_value_write_with_transient_reuse():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 3.0], size=(N, N))
    c = np.random.choice([0.0, 3.0], size=(N, N))
    branch_dependent_value_write_with_transient_reuse.to_sdfg().save("t.sdfg")
    run_and_compare(branch_dependent_value_write_with_transient_reuse, 0, True, a=a, b=b, c=c)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_single_branch_connectors(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 5.0], size=(N, N))
    d = np.random.choice([0.0, 5.0], size=(N, N))
    c = np.random.randn(1, )

    sdfg = single_branch_connectors.to_sdfg()
    sdfg.validate()
    arrays = {"a": a, "b": b, "c": c, "d": d}
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(a=out_no_fuse["a"], b=out_no_fuse["b"], c=out_no_fuse["c"][0], d=out_no_fuse["d"])
    sdfg.save(sdfg.label + "_before.sdfg")
    # Apply transformation
    if use_pass_flag:
        fuse_branches_pass.FuseBranchesPass().apply_pass(sdfg, {})
    else:
        apply_fuse_branches(sdfg, 2)

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in arrays.items()}
    sdfg.save(sdfg.label + "_after.sdfg")
    sdfg(a=out_fused["a"], b=out_fused["b"], c=out_fused["c"][0], d=out_fused["d"])

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branch_code) == 0, f"(actual) len({branch_code}) != (desired) 0"

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)

    nsdfgs = {(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}
    assert len(nsdfgs) == 1
    nsdfg, parent_state = nsdfgs.pop()
    assert len(nsdfg.in_connectors) == 4, f"{nsdfg.in_connectors}, length is not 4 but {len(nsdfg.in_connectors)}"
    assert len(nsdfg.out_connectors) == 1, f"{nsdfg.out_connectors}, length is not 1 but {len(nsdfg.out_connectors)}"


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_disjoint_subsets(use_pass_flag):
    if_cond_58 = np.array([1], dtype=np.int32)
    A = np.random.choice([0.0, 3.0], size=(N, ))
    B = np.random.randn(N, 3, 3)
    C = np.random.randn(N, 3, 3)
    E = np.random.choice([0.0, 3.0], size=(N, 3, 3))
    run_and_compare(disjoint_subsets, 0, use_pass_flag, A=A, B=B, C=C, E=E, if_cond_58=if_cond_58[0])


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


def test_try_clean():
    sdfg1 = _multi_state_nested_if.to_sdfg()
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 2

    for cblock in cblocks:
        parent_sdfg = cblock.parent_graph.sdfg
        parent_graph = cblock.parent_graph
        xform = fuse_branches.FuseBranches()
        xform.conditional = cblock
        xform.try_clean(graph=parent_graph, sdfg=parent_sdfg)

    # Should have moe states before
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 2
    # A state must have been moved before)
    assert isinstance(sdfg1.start_block, dace.SDFGState)
    sdfg1.validate()

    fbpass = fuse_branches_pass.FuseBranchesPass()
    fbpass.try_clean = False
    fbpass.apply_pass(sdfg1, {})
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    # 1 left because now the if branch has 2 states
    assert len(cblocks) == 1, f"{cblocks}"
    sdfg1.validate()

    for cblock in cblocks:
        parent_sdfg = cblock.parent_graph.sdfg
        parent_graph = cblock.parent_graph
        xform = fuse_branches.FuseBranches()
        xform.conditional = cblock
        xform.try_clean(graph=parent_graph, sdfg=parent_sdfg)
    sdfg1.save("x5.sdfg")
    sdfg1.validate()

    fbpass = fuse_branches_pass.FuseBranchesPass()
    fbpass.try_clean = False
    fbpass.apply_pass(sdfg1, {})
    cblocks = {n for n, g in sdfg1.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    sdfg1.save("x6.sdfg")
    assert len(cblocks) == 0, f"{cblocks}"
    sdfg1.validate()

    if_cond_1 = np.array([1.2], dtype=np.float64)
    offset = np.array([0], dtype=np.int64)
    A = np.random.choice([0.0, 3.0], size=(N, ))
    B = np.random.randn(N, 3, 3)
    C = np.random.choice([0.0, 3.0], size=(N, ))
    run_and_compare_sdfg(sdfg1, A=A, B=B, C=C, if_cond_1=if_cond_1[0], offset=offset[0])


def test_try_clean_as_pass():
    # This is a test to check the different configurations of try clean, applicability depends on the SDFG and the pass
    sdfg = _multi_state_nested_if.to_sdfg()
    fbpass = fuse_branches_pass.FuseBranchesPass()
    fbpass.clean_only = True
    fbpass.try_clean = False
    fbpass.apply_pass(sdfg, {})
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 2, f"{cblocks}"
    fbpass.clean_only = False
    fbpass.try_clean = False
    fbpass.apply_pass(sdfg, {})
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1, f"{cblocks}"
    fbpass.clean_only = False
    fbpass.try_clean = False
    fbpass.apply_pass(sdfg, {})
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 1, f"{cblocks}"
    fbpass.clean_only = False
    fbpass.try_clean = True
    fbpass.apply_pass(sdfg, {})
    cblocks = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0, f"{cblocks}"
    sdfg.validate()

    if_cond_1 = np.array([1.2], dtype=np.float64)
    offset = np.array([0], dtype=np.int64)
    A = np.random.choice([0.0, 3.0], size=(N, ))
    B = np.random.randn(N, 3, 3)
    C = np.random.choice([0.0, 3.0], size=(N, ))
    run_and_compare_sdfg(sdfg, A=A, B=B, C=C, if_cond_1=if_cond_1[0], offset=offset[0])


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


def test_sdfg_with_interstate_array_condition():
    sdfg = _get_sdfg_with_interstate_array_condition()
    llindex = np.ones(shape=(4, 4, 4), dtype=np.int64)
    zsolqa = np.random.choice([0.0, 3.0], size=(4, 4, 4))
    zratio = np.random.choice([0.0, 3.0], size=(4, 4, 4))
    run_and_compare_sdfg(
        sdfg,
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


def test_repeated_condition_variables():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.choice([0.0, 3.0], size=(N, N))
    c = np.random.choice([0.0, 3.0], size=(N, N))
    conds = np.random.choice([1.0, 3.0], size=(4, N))
    run_and_compare(repeated_condition_variables, 0, True, a=a, b=b, c=c, conds=conds)


def _find_state(root_sdfg: dace.SDFG, node):
    for n, g in root_sdfg.all_nodes_recursive():
        if n == node:
            return g
    return None


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

    xform = fuse_branches.FuseBranches()
    xform.conditional = cblocks.pop()
    xform.parent_nsdfg_state = None
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform = fuse_branches.FuseBranches()
    xform.conditional = inner_cblocks.pop()
    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is True


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

    xform = fuse_branches.FuseBranches()
    xform.conditional = cblocks.pop()
    xform.parent_nsdfg_state = None
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform = fuse_branches.FuseBranches()
    xform.conditional = inner_cblocks.pop()
    xform.parent_nsdfg_state = None
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)
    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is True


def test_can_be_applied_parameters_on_nested_sdfg():
    sdfg = weird_condition.to_sdfg()
    cblocks = {n for n in sdfg.all_control_flow_regions() if isinstance(n, ConditionalBlock)}
    assert len(cblocks) == 0
    inner_cblocks = {
        n
        for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, ConditionalBlock) and g is not None and g.sdfg.parent_nsdfg_node is not None
    }
    assert len(inner_cblocks) == 1

    xform = fuse_branches.FuseBranches()
    xform.conditional = inner_cblocks.pop()

    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is False

    xform.parent_nsdfg_state = _find_state(sdfg, xform.conditional.sdfg.parent_nsdfg_node)

    assert xform.can_be_applied(graph=xform.conditional.parent_graph,
                                expr_index=0,
                                sdfg=xform.conditional.parent_graph.sdfg) is True

    xform.parent_nsdfg_state


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


def test_non_trivial_subset_after_combine_tasklet():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    C = np.random.choice([0.0, 5.0], size=(N, N))
    D = np.random.choice([0.0, 5.0], size=(N, N))
    E = np.random.choice([0.0, 5.0], size=(N, N))
    F = np.random.randn(1, )
    G = np.random.choice([0.0, 5.0], size=(N, N))
    run_and_compare(
        non_trivial_subset_after_combine_tasklet,
        0,
        True,
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


def test_split_on_disjoint_subsets():
    A = np.random.choice([0.0, 5.0], size=(N, N, 2))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    C = np.random.choice([0.0, 5.0], size=(N, N))
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

    xform = fuse_branches.FuseBranches()
    xform.conditional = cblock
    assert xform._is_disjoint_subset(state0, state1) is True

    # If we split we will make not applicable anymore
    xform._split_branches(cblock.parent_graph, cblock)

    sdfg.validate()

    run_and_compare(
        split_on_disjoint_subsets,
        0,
        True,
        a=A,
        b=B,
        c=C,
        d=D[0],
    )


def test_split_on_disjoint_subsets_nested():
    A = np.random.choice([0.0, 5.0], size=(N, N, 2))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    C = np.random.choice([0.0, 5.0], size=(N, N))
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

    xform = fuse_branches.FuseBranches()
    xform.conditional = cblock
    assert xform._is_disjoint_subset(state0, state1) is True

    # If we split we will make not applicable anymore
    xform._split_branches(cblock.parent_graph, cblock)

    sdfg.validate()

    run_and_compare(
        split_on_disjoint_subsets_nested,
        0,
        True,
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
        _if_cond_1 = i < d
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
        _if_cond_1 = i < d
        if _if_cond_1:
            tc1 = b[i, 6] + a[i, 6]
            zmdn = tc1
        else:
            zmdn = 1.0
        b[i, 3] = zmdn
        b[i, 3] = zmdn


def test_write_to_transient():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    run_and_compare(
        write_to_transient,
        0,
        True,
        a=A,
        b=B,
        d=D[0],
    )


def test_write_to_transient_two():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    run_and_compare(
        write_to_transient_two,
        0,
        True,
        a=A,
        b=B,
        d=D[0],
    )


def test_double_empty_state():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
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


def test_complicated_pattern_for_manual_clean_up_one():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    C = np.random.choice([0.0, 5.0], size=(N, N))
    D = np.ones([
        1,
    ], dtype=np.float64)
    sdfg = complicated_pattern_for_manual_clean_up_one.to_sdfg()
    sdfg.save("uwu.sdfg")

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
                xform = fuse_branches.FuseBranches()
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


def test_try_clean_on_complicated_pattern_for_manual_clean_up_one():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    C = np.random.choice([0.0, 5.0], size=(N, N))
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

    run_and_compare_sdfg(sdfg, a=A, b=B, c=C, d=D[0])

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
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


def test_try_clean_on_complicated_pattern_for_manual_clean_up_two():
    A = np.random.choice([0.0, 5.0], size=(N, N))
    B = np.random.choice([0.0, 5.0], size=(N, N))
    C = np.random.choice([0.0, 5.0], size=(N, N))
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

    run_and_compare_sdfg(sdfg, a=A, b=B, c=C, d=D[0], e=E[0])

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
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


def test_single_assignment():
    if_cond_1 = np.array([1], dtype=np.float64)
    A = np.ones(shape=(N, ), dtype=np.float64)
    run_and_compare(single_assignment, 0, True, a=A, _if_cond_1=if_cond_1[0])


def test_single_assignment_cond_from_scalar():
    A = np.ones(shape=(512, ), dtype=np.float64)
    before = single_assignment_cond_from_scalar.to_sdfg()
    before.name = "non_fusion_single_assignment_cond_from_scalar"
    before.compile()
    run_and_compare(single_assignment_cond_from_scalar, 0, True, a=A)


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


def test_condition_from_transient_scalar():
    zsolac = np.random.choice([8.0, 11.0], size=(N, ))
    zlcond2 = np.random.choice([8.0, 11.0], size=(N, ))
    za = np.random.choice([8.0, 11.0], size=(N, ))
    _if_cond_42 = np.random.choice([8.0, 11.0], size=(1, ))
    sdfg = _get_sdfg_with_condition_from_transient_scalar()

    sdfg.save("condition_from_transient_scalar_before.sdfg")

    run_and_compare_sdfg(sdfg, zsolac=zsolac, zlcond2=zlcond2, za=za, _if_cond_42=_if_cond_42[0])
    sdfg.save("condition_from_transient_scalar_after.sdfg")

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branch_code) == 0, f"(actual) len({branch_code}) != (desired) {0}"


if __name__ == "__main__":
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
