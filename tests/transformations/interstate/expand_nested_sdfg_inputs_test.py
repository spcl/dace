import numpy as np
import dace
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs

N = dace.symbol("N")


@dace.program
def scatter_gatter(
    A: dace.float64[N, N, N],
    B: dace.float64[N, N, N],
    C: dace.float64[N, N, N],
    idx0: dace.int64[N],
    idx1: dace.int64[N],
):
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        a = A[idx0[i], j, idx1[k]]
        b = B[i, j, k]
        c = C[idx0[i], j, idx1[k]]
        C[idx0[i], j, idx1[k]] = a * b * c * 2.0


def test_expand_nested_sdfg_inputs():
    sdfg = scatter_gatter.to_sdfg()
    sdfg.validate()
    print("there")
    nsdfgs = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(nsdfgs) == 1, (f"Expected exactly one NestedSDFG before transformation, "
                              f"found {len(nsdfgs)}.")
    nsdfg, parent_state = nsdfgs[0]

    sdfg.save("before.sdfg")
    ExpandNestedSDFGInputs().apply_to(sdfg=parent_state.sdfg, nested_sdfg=nsdfg)
    sdfg.save("after.sdfg")
    print("here")

    sdfg.validate()

    # Execute once to ensure the transformed SDFG is runnable.
    n = 16
    rng = np.random.default_rng(42)

    A = rng.random((n, n, n))
    B = rng.random((n, n, n))
    C = rng.random((n, n, n))

    idx0 = rng.permutation(n).astype(np.int64)
    idx1 = rng.permutation(n).astype(np.int64)

    sdfg(A=A, B=B, C=C, idx0=idx0, idx1=idx1, N=n)

    nsdfgs = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    assert len(nsdfgs) == 1, (f"Expected exactly one NestedSDFG after transformation, "
                              f"found {len(nsdfgs)}.")

    nsdfg, parent_state = nsdfgs[0]

    for edge in parent_state.in_edges(nsdfg):
        expected = dace.subsets.Range.from_array(parent_state.sdfg.arrays[edge.data.data])

        assert edge.data.subset == expected, (f"Input edge '{edge.data.data}' was not expanded to the full array.\n"
                                              f"Expected: {expected}\n"
                                              f"Actual:   {edge.data.subset}")

    for edge in parent_state.out_edges(nsdfg):
        expected = dace.subsets.Range.from_array(parent_state.sdfg.arrays[edge.data.data])

        assert edge.data.subset == expected, (f"Output edge '{edge.data.data}' was not expanded to the full array.\n"
                                              f"Expected: {expected}\n"
                                              f"Actual:   {edge.data.subset}")


import numpy as np
import dace

from dace.transformation.interstate.expand_nested_sdfg_inputs import (
    ExpandNestedSDFGInputs, )

N = dace.symbol("N")


# -------------------------
# Reference implementation
# -------------------------
def numpy_reference(A, B, idx):
    C = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[idx[i], j] = A[idx[i], j] + B[i, j] * 2.0
    return C


# -------------------------
# DaCe program (forces Subscript + scalar mix)
# -------------------------
@dace.program
def column_gather_scatter(
    A: dace.float64[N, N],
    B: dace.float64[N, N],
    C: dace.float64[N, N],
    idx: dace.int64[N],
):
    for i, j in dace.map[0:N, 0:N]:

        # key pattern:
        # scalar index + subscript reuse
        a = A[idx[i], j]
        b = B[i, j]

        C[idx[i], j] = a + b * 2.0


# -------------------------
# Test
# -------------------------
def test_expand_nested_sdfg_inputs_column_scalar_uncollapse_e2e():
    sdfg = column_gather_scatter.to_sdfg()
    sdfg.validate()

    # locate NestedSDFG
    nsdfgs = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    assert len(nsdfgs) == 1, (f"Expected exactly one NestedSDFG before transformation, "
                              f"found {len(nsdfgs)}")

    nsdfg, parent_state = nsdfgs[0]

    # snapshot input edges (pre-transform sanity)
    pre_in_edges = list(parent_state.in_edges(nsdfg))
    pre_out_edges = list(parent_state.out_edges(nsdfg))

    # -------------------------
    # Apply transformation
    # -------------------------
    sdfg.save("before_column_scalar_uncollapse.sdfg")
    ExpandNestedSDFGInputs().apply_to(
        sdfg=parent_state.sdfg,
        nested_sdfg=nsdfg,
    )
    sdfg.save("after_column_scalar_uncollapse.sdfg")

    sdfg.validate()

    # -------------------------
    # Run E2E execution
    # -------------------------
    n = 16
    rng = np.random.default_rng(0)

    A = rng.random((n, n))
    B = rng.random((n, n))
    C = np.zeros((n, n))
    idx = rng.permutation(n)

    sdfg(A=A, B=B, C=C, idx=idx, N=n)

    # -------------------------
    # Reference result
    # -------------------------
    C_ref = numpy_reference(A, B, idx)

    assert np.allclose(C, C_ref), ("Numerical mismatch after ExpandNestedSDFGInputs\n"
                                   f"max error = {np.max(np.abs(C - C_ref))}")

    # -------------------------
    # Structural validation
    # -------------------------
    nsdfgs_after = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    assert len(nsdfgs_after) == 1, (f"Expected exactly one NestedSDFG after transformation, "
                                    f"found {len(nsdfgs_after)}")

    nsdfg_after, parent_state_after = nsdfgs_after[0]

    # ensure memlets fully expanded
    for edge in parent_state_after.in_edges(nsdfg_after):
        expected = dace.subsets.Range.from_array(parent_state_after.sdfg.arrays[edge.data.data])
        assert edge.data.subset == expected, (f"Input edge not fully expanded for {edge.data.data}\n"
                                              f"Expected: {expected}\n"
                                              f"Actual:   {edge.data.subset}")

    for edge in parent_state_after.out_edges(nsdfg_after):
        expected = dace.subsets.Range.from_array(parent_state_after.sdfg.arrays[edge.data.data])
        assert edge.data.subset == expected, (f"Output edge not fully expanded for {edge.data.data}\n"
                                              f"Expected: {expected}\n"
                                              f"Actual:   {edge.data.subset}")

    # -------------------------
    # Regression guard: ensure transformation actually changed structure
    # -------------------------
    post_in_edges = list(parent_state_after.in_edges(nsdfg_after))

    assert len(pre_in_edges) == len(post_in_edges), ("Unexpected change in number of input edges after transformation")


if __name__ == "__main__":
    test_expand_nested_sdfg_inputs_column_scalar_uncollapse_e2e()
    test_expand_nested_sdfg_inputs()
