import numpy as np
import dace
import pytest
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import fuse_branches
from dace.transformation.passes import fuse_branches_pass

N = 8


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


S = 8
S1 = 0
S2 = 8


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
    sdfg(**out_fused)
    sdfg.save(sdfg.label + "_after.sdfg")

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(
        branch_code) == num_expected_branches, f"(actual) len({branch_code}) != (desired) {num_expected_branches}"

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
    b = np.random.randn(N, N)
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_single_branch, 0, use_pass_flag, a=a, b=b, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_complicated_if(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    d = np.zeros((N, N))
    run_and_compare(complicated_if, 0, use_pass_flag, a=a, b=b, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_multi_state_branch_body(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    c = np.random.randn(N, N)
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(multi_state_branch_body, 1, use_pass_flag, a=a, b=b, c=c, d=d, s=s[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_nested_if(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(nested_if, 0, use_pass_flag, a=a, b=b, c=c, d=d, s=s[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_tasklets_in_if(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    c = np.zeros((1, ))
    d = np.zeros((N, N))
    run_and_compare(tasklets_in_if, 0, use_pass_flag, a=a, b=b, d=d, c=c[0])


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_branch_dependent_value_write_single_branch_nonzero_write(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    d = np.random.randn(N, N)
    run_and_compare(branch_dependent_value_write_single_branch_nonzero_write, 0, use_pass_flag, a=a, b=b, d=d)


@pytest.mark.parametrize("use_pass_flag", [True, False])
def test_single_branch_connectors(use_pass_flag):
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    d = np.random.randn(N, N)
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
    sdfg(a=out_fused["a"], b=out_fused["b"], c=out_fused["c"][0], d=out_fused["d"])
    sdfg.save(sdfg.label + "_after.sdfg")

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


if __name__ == "__main__":
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
