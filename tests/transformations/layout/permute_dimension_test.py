import copy
import numpy as np
import dace
from dace.transformation.layout.permute_dimensions import PermuteArrayDimensions


def test_standalone_execution():
    """Standalone test function that can be run without pytest."""
    print("Running standalone permute transformations test...")

    # Setup
    dace.Config.set('cache', value='unique')
    N_val = 8
    TSTEPS_val = 2

    # Create kernel
    N = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def kernel(
        TSTEPS: dace.int64,
        vals_A: dace.float64[N, N, N],
        vals_B: dace.float64[N, N, N],
        neighbors: dace.int64[N, N, 8],
    ):
        for _ in range(1, TSTEPS):
            for i, j, k in dace.map[0:N - 2, 0:N - 2, 0:N - 2]:
                vals_B[i + 1, j + 1, k +
                       1] = 0.2 * (vals_A[i + 1, j + 1, k + 1] + vals_A[i + 1, j, k + 1] + vals_A[i + 1, j + 2, k + 1] +
                                   vals_A[neighbors[i + 1, k + 1, 0], j + 1, neighbors[i + 1, k + 1, 4]] +
                                   vals_A[neighbors[i + 1, k + 1, 1], j + 1, neighbors[i + 1, k + 1, 5]] +
                                   vals_A[neighbors[i + 1, k + 1, 2], j + 1, neighbors[i + 1, k + 1, 6]] +
                                   vals_A[neighbors[i + 1, k + 1, 3], j + 1, neighbors[i + 1, k + 1, 7]])
            for i, j, k in dace.map[0:N - 2, 0:N - 2, 0:N - 2]:
                vals_A[i + 1, j + 1, k +
                       1] = 0.2 * (vals_B[i + 1, j + 1, k + 1] + vals_B[i + 1, j, k + 1] + vals_B[i + 1, j + 2, k + 1] +
                                   vals_B[neighbors[i + 1, k + 1, 0], j + 1, neighbors[i + 1, k + 1, 4]] +
                                   vals_B[neighbors[i + 1, k + 1, 1], j + 1, neighbors[i + 1, k + 1, 5]] +
                                   vals_B[neighbors[i + 1, k + 1, 2], j + 1, neighbors[i + 1, k + 1, 6]] +
                                   vals_B[neighbors[i + 1, k + 1, 3], j + 1, neighbors[i + 1, k + 1, 7]])

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify(skip=["ArrayElimination", "DeadDataflowElimination"])

    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = original_sdfg.name + "_transposed"

    # Apply transformations
    PermuteArrayDimensions(
        permute_map={
            "vals_A": [0, 2, 1],
            "vals_B": [0, 2, 1]
        },
        add_permute_maps=True,
    ).apply_pass(sdfg=transformed_sdfg, pipeline_results={})

    # Find and apply map transformations
    map_labels = {}
    for state in transformed_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                map_labels[node.label] = [0, 2, 1]

    # Validate SDFGs
    original_sdfg.validate()
    transformed_sdfg.validate()

    # Initialize data
    np.random.seed(42)
    vals_A_orig = np.fromfunction(lambda i, j, k: i * k * (j + 2) / N_val, (N_val, N_val, N_val), dtype=np.float64)
    vals_B_orig = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N_val, (N_val, N_val, N_val), dtype=np.float64)
    neighbors = np.random.randint(1, N_val - 1, size=(N_val, N_val, 8), dtype=np.int64)

    vals_A_trans = vals_A_orig.copy()
    vals_B_trans = vals_B_orig.copy()

    # Execute SDFGs
    original_sdfg(vals_A=vals_A_orig, vals_B=vals_B_orig, neighbors=neighbors, N=N_val, TSTEPS=TSTEPS_val)
    transformed_sdfg(vals_A=vals_A_trans, vals_B=vals_B_trans, neighbors=neighbors, N=N_val, TSTEPS=TSTEPS_val)

    # Check results
    vals_A_close = np.allclose(vals_A_orig, vals_A_trans, rtol=1e-10, atol=1e-12)
    vals_B_close = np.allclose(vals_B_orig, vals_B_trans, rtol=1e-10, atol=1e-12)

    print(f"vals_A results match: {vals_A_close}")
    print(f"vals_B results match: {vals_B_close}")

    if vals_A_close and vals_B_close:
        print("✅ All tests passed! Permute transformations preserve correctness.")
    else:
        print("❌ Test failed! Results differ between original and transformed SDFGs.")
        if not vals_A_close:
            print(f"vals_A max difference: {np.max(np.abs(vals_A_orig - vals_A_trans))}")
            print(f"vals_A difference: {np.abs(vals_A_orig - vals_A_trans)}")
        if not vals_B_close:
            print(f"vals_B max difference: {np.max(np.abs(vals_B_orig - vals_B_trans))}")
            print(f"vals_B difference: {np.abs(vals_B_orig - vals_B_trans)}")

    return vals_A_close and vals_B_close


if __name__ == "__main__":
    success = test_standalone_execution()
    exit(0 if success else 1)
