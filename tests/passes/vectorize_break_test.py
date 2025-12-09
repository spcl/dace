import dace
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.vectorization.vectorize_break import VectorizeBreak
import numpy as np

LEN_1D = dace.symbol("LEN_1D")
ITERATIONS = dace.symbol("ITERATIONS")
LEN_1D_VAL = 32
ITERATIONS_VAL = 1

@dace.program
def dace_s482(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            a[i] = a[i] + b[i] * c[i]
            if c[i] > b[i]:
                break

@dace.program
def dace_s481(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):
        for i in range(LEN_1D):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]


def run_sdfg(sdfg, a, b=None, c=None, d=None):
    """Helper to execute a compiled SDFG with provided arrays."""
    inputs = {}
    if a is not None: inputs['a'] = a
    if b is not None: inputs['b'] = b
    if c is not None: inputs['c'] = c
    if d is not None: inputs['d'] = d
    inputs['LEN_1D'] = LEN_1D_VAL
    inputs['ITERATIONS'] = ITERATIONS_VAL
    sdfg(**inputs)
    return a  # mutated array



def test_s482_vectorized_matches_baseline():
    # Random input
    LEN = 64
    ITERS = 1

    # Make 50%
    a0 = np.random.rand(LEN).astype(np.float64)
    b0 = np.random.rand(LEN).astype(np.float64)
    half = LEN // 2
    b0[:half] = np.random.uniform(0.0, 0.4, half)
    c0 = np.random.rand(LEN).astype(np.float64)
    c0[:half] = np.random.uniform(0.6, 1.0, half)

    # Make copies for the two runs
    a_ref = a0.copy()
    a_vec = a0.copy()

    # --- baseline SDFG ---
    sdfg_ref = dace_s482.to_sdfg()
    exe_ref = sdfg_ref.compile()
    run_sdfg(exe_ref, a_ref, b=b0.copy(), c=c0.copy())
    sdfg_ref.save("s482_v2.sdfg")


    # --- vectorized SDFG ---
    sdfg_vec = dace_s482.to_sdfg()
    sdfg_vec.apply_transformations_repeated(LoopToMap)
    VectorizeBreak(vector_width=8).apply_pass(sdfg_vec, {})
    sdfg_vec.save("s482_vectorized_v2.sdfg")
    exe_vec = sdfg_vec.compile()
    run_sdfg(exe_vec, a_vec, b=b0.copy(), c=c0.copy())
    try:
        np.testing.assert_allclose(a_ref, a_vec, equal_nan=True)
    except AssertionError as e:
        diff = a_ref - a_vec
        mism = np.where(~np.isclose(a_ref, a_vec, equal_nan=True))[0]

        print("\n=== VECTORIZATION MISMATCH DETECTED ===")
        print("First mismatching indices:", mism[:10], "..." if len(mism) > 10 else "")
        print("Mismatching indices:", mism)
        print("\nReference values at mismatches:", a_ref[mism][:10])
        print("Vectorized values at mismatches:", a_vec[mism][:10])
        print("Diff:", diff[mism][:10])
        print("\nFull AssertionError:")
        print(e)
        raise

# ----------------------------------------------------------------------
# Test s481
# ----------------------------------------------------------------------

def test_s481_vectorized_matches_baseline():
    # Random input
    a0 = np.random.rand(LEN_1D_VAL).astype(np.float64)
    b0 = np.random.rand(LEN_1D_VAL).astype(np.float64)
    c0 = np.random.rand(LEN_1D_VAL).astype(np.float64)
    d0 = np.random.uniform(-1.0, 1.0, LEN_1D_VAL).astype(np.float64)  # sign matters

    # Make copies
    a_ref = a0.copy()
    a_vec = a0.copy()

    # --- baseline ---
    sdfg_ref = dace_s481.to_sdfg()
    exe_ref = sdfg_ref.compile()
    run_sdfg(exe_ref, a_ref, b=b0.copy(), c=c0.copy(), d=d0.copy())
    sdfg_ref.save("s481.sdfg")

    # --- vectorized ---
    sdfg_vec = dace_s481.to_sdfg()

    sdfg_vec.apply_transformations_repeated(LoopToMap)
    VectorizeBreak(vector_width=8).apply_pass(sdfg_vec, {})
    exe_vec = sdfg_vec.compile()
    sdfg_vec.save("s481_veec.sdfg")

    run_sdfg(exe_vec, a_vec, b=b0.copy(), c=c0.copy(), d=d0.copy())

    try:
        np.testing.assert_allclose(a_ref, a_vec, equal_nan=True)
    except AssertionError as e:
        diff = a_ref - a_vec
        mism = np.where(~np.isclose(a_ref, a_vec, equal_nan=True))[0]

        print("\n=== VECTORIZATION MISMATCH DETECTED ===")
        print("First mismatching indices:", mism[:10], "..." if len(mism) > 10 else "")
        print("\nReference values at mismatches:", a_ref[mism][:10])
        print("Vectorized values at mismatches:", a_vec[mism][:10])
        print("Diff:", diff[mism][:10])
        print("\nFull AssertionError:")
        print(e)
        raise

if __name__ == "__main__":
    test_s482_vectorized_matches_baseline()
    test_s481_vectorized_matches_baseline()