import numpy as np
import dace
from dace.sdfg.state import ConditionalBlock
import dace.transformation.interstate.fuse_branches as fuse_branches

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
def complicated_if(
    a: dace.float64[N, N],
    b: dace.float64[N, N],
    d: dace.float64[N, N],
):
    for i, j in dace.map[0:N:1, 0:N:1]:
        if (a[i, j] + d[i, j] * 3.0) < 0.65:
            b[i, j] = 0.0
        d[i, j] = b[i, j] + a[i, j]  # ensure d is always written for comparison


def apply_fuse_branches(sdfg, nestedness: int = 1):
    """Apply FuseBranches transformation to all eligible conditionals."""
    # Pattern matching with conditional branches to not work (9.10.25), avoid it
    for i in range(nestedness):
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, ConditionalBlock):
                t = fuse_branches.FuseBranches()
                if t.can_be_applied_to(graph.sdfg, conditional=node):
                    t.apply_to(graph.sdfg, conditional=node)


def run_and_compare(
    program,
    num_expected_branches,
    **arrays,
):
    # Run SDFG version (no transformation)
    sdfg = program.to_sdfg()
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)

    # Apply transformation
    apply_fuse_branches(sdfg, 2)

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_fused)

    branch_code = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(
        branch_code) == num_expected_branches, f"(actual) len({branch_code}) != (desired) {num_expected_branches}"

    # Compare all arrays
    for name in arrays.keys():
        np.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


def test_branch_dependent_value_write():
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write, 0, a=a, b=b, c=c, d=d)


def test_branch_dependent_value_write_two():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.zeros((N, N))
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_two, 0, a=a, b=b, c=c, d=d)


def test_branch_dependent_value_write_single_branch():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    d = np.zeros((N, N))
    run_and_compare(branch_dependent_value_write_single_branch, 0, a=a, b=b, d=d)


def test_complicated_if():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    d = np.zeros((N, N))
    run_and_compare(complicated_if, 0, a=a, b=b, d=d)


def test_multi_state_branch_body():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    c = np.random.randn(N, N)
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(multi_state_branch_body, 1, a=a, b=b, c=c, d=d, s=s[0])


def test_nested_if():
    a = np.random.choice([0.0, 3.0], size=(N, N))
    b = np.random.randn(N, N)
    c = np.zeros((N, N))
    d = np.zeros((N, N))
    s = np.zeros((1, )).astype(np.int64)
    run_and_compare(nested_if, 0, a=a, b=b, c=c, d=d, s=s[0])


if __name__ == "__main__":
    test_branch_dependent_value_write()
    test_branch_dependent_value_write_two()
    test_branch_dependent_value_write_single_branch()
    test_complicated_if()
    test_multi_state_branch_body()
    test_nested_if()
