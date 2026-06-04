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
    nsdfgs = [
        (n, s)
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.NestedSDFG)
    ]
    assert len(nsdfgs) == 1, (
        f"Expected exactly one NestedSDFG before transformation, "
        f"found {len(nsdfgs)}."
    )
    nsdfg, parent_state = nsdfgs[0]

    ExpandNestedSDFGInputs().apply_to(sdfg=parent_state.sdfg, nested_sdfg=nsdfg)

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

    nsdfgs = [
        (n, s)
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.NestedSDFG)
    ]

    assert len(nsdfgs) == 1, (
        f"Expected exactly one NestedSDFG after transformation, "
        f"found {len(nsdfgs)}."
    )

    nsdfg, parent_state = nsdfgs[0]

    for edge in parent_state.in_edges(nsdfg):
        expected = dace.subsets.Range.from_array(
            parent_state.sdfg.arrays[edge.data.data]
        )

        assert edge.data.subset == expected, (
            f"Input edge '{edge.data.data}' was not expanded to the full array.\n"
            f"Expected: {expected}\n"
            f"Actual:   {edge.data.subset}"
        )

    for edge in parent_state.out_edges(nsdfg):
        expected = dace.subsets.Range.from_array(
            parent_state.sdfg.arrays[edge.data.data]
        )

        assert edge.data.subset == expected, (
            f"Output edge '{edge.data.data}' was not expanded to the full array.\n"
            f"Expected: {expected}\n"
            f"Actual:   {edge.data.subset}"
        )


if __name__ == "__main__":
    test_expand_nested_sdfg_inputs()