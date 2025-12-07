# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os
from typing import Dict, List
import dace
import pytest
import copy
import numpy as np
from dace.frontend.python.parser import DaceProgram
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.passes.vectorization.detect_gather import DetectGather
from math import log, exp

###############################################################################
# Utility: Detect CPU Vector Capabilities
###############################################################################


def detect_cpu_vector_features():
    """
    Returns a dictionary describing available SIMD features.

    Example:
        {
            "avx2": True,
            "avx512": False,
            "neon": False,
            "sve": False,
            "sve2": False,
        }
    """
    import platform, subprocess, sys

    system = platform.machine().lower()

    if system == "x86_64":

        try:
            out = subprocess.check_output(["lscpu"], text=True).lower()
        except Exception:
            out = ""

        return {
            "avx2": ("avx2" in out),
            "avx512": ("avx512" in out),
            "neon": False,
            "sve": False,
            "sve2": False,
        }

    elif system in ("arm64", "aarch64"):
        try:
            out = subprocess.check_output(["lscpu"], text=True).lower()
        except Exception:
            out = ""

        return {
            "neon": ("asimd" in out or "neon" in out),
            "sve": ("sve" in out),
            "sve2": ("sve2" in out),
            "avx2": False,
            "avx512": False,
        }

    else:
        return dict(avx2=False, avx512=False, neon=False, sve=False, sve2=False)


###############################################################################
# Core Vectorization Test Harness
###############################################################################


def compile_and_run(sdfg, arrays, params):
    """Compile SDFG and execute it."""
    compiled = sdfg.compile()
    compiled(**arrays, **params)
    return compiled


def apply_vectorization_pass(
    sdfg,
    vector_width=8,
    fuse_overlapping_loads=False,
    insert_copies=True,
    no_inline=False,
    filter_map=None,
):
    VectorizeCPU(
        vector_width=vector_width,
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies,
        apply_on_maps=filter_map,
        no_inline=no_inline,
        fail_on_unvectorizable=True,
    ).apply_pass(sdfg, {})
    sdfg.validate()


def simplify_sdfg_if_needed(sdfg, simplify, skip_simplify):
    if simplify:
        sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())


###############################################################################
# Main Test Entry
###############################################################################


def run_vectorization_test(
    dace_func,
    arrays,
    params,
    vector_width=8,
    simplify=True,
    skip_simplify=None,
    save_sdfgs=False,
    sdfg_name=None,
    apply_loop_to_map=True,
    exact=None,
):
    """Runs baseline + vectorized SDFGs and compares correctness."""

    # deepcopy input arrays
    arr_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arr_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # === Build original SDFG ===
    sdfg = dace_func.to_sdfg(simplify=False)
    if sdfg_name:
        sdfg.name = sdfg_name

    if simplify:
        sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify)

    if apply_loop_to_map:
        sdfg.apply_transformations_repeated(LoopToMap())
        sdfg.simplify()

    if save_sdfgs and sdfg_name:
        sdfg.save(f"{sdfg_name}.sdfg")

    # === Build vectorized copy ===
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = (sdfg_name or sdfg.name) + "_vec"

    apply_vectorization_pass(vsdfg, vector_width=vector_width)

    if save_sdfgs and sdfg_name:
        vsdfg.save(f"{sdfg_name}_vec.sdfg")

    # === Execute both ===
    compile_and_run(sdfg, arr_orig, params)
    compile_and_run(vsdfg, arr_vec, params)

    # === Compare ===
    for name in arrays:
        assert np.allclose(arr_orig[name], arr_vec[name], rtol=1e-32), \
            f"{name} mismatch\n{arr_orig[name] - arr_vec[name]}"

        if exact is not None:
            diff = arr_vec[name] - exact
            assert np.allclose(arr_vec[name], exact, rtol=0, atol=1e-300), \
                f"{name}: max abs diff = {np.max(np.abs(diff))}"

    return vsdfg


def select_env_flags():
    """
    Returns a list of environment-variable dictionaries.
    Each dictionary corresponds to one configuration that should be tested.
    """

    feats = detect_cpu_vector_features()
    print("Detected SIMD:", feats)

    configs = []

    # ------------------------
    # 1. Scalar reference case
    # ------------------------
    configs.append({"__DACE_USE_INTRINSICS": "0"})

    # ------------------------
    # 2. AVX-512 available
    # ------------------------
    if feats.get("avx512", False):
        # AVX-512 intrinsics ON
        configs.append({"__DACE_USE_INTRINSICS": "1", "__DACE_USE_AVX512": "1"})

        # Also test AVX2 mode explicitly
        configs.append({
            "__DACE_USE_INTRINSICS": "1",
            "__DACE_USE_AVX512": "0"  # forces AVX2 fallback in headers
        })

    # ------------------------
    # 3. AVX2 but no AVX-512
    # ------------------------
    elif feats.get("avx2", False):
        configs.append({"__DACE_USE_INTRINSICS": "1", "__DACE_USE_AVX512": "0"})

    # ------------------------
    # 4. SVE2 / SVE available
    # ------------------------
    if feats.get("sve2", False) or feats.get("sve", False):
        configs.append({"__DACE_USE_INTRINSICS": "1", "__DACE_USE_SVE": "1"})

        # Test SVE disabled with intrinsics ON → should fall back to NEON/scalar
        configs.append({"__DACE_USE_INTRINSICS": "1", "__DACE_USE_SVE": "0"})

    # ------------------------
    # 5. NEON-only platform
    # ------------------------
    elif feats.get("neon", False):
        configs.append({"__DACE_USE_INTRINSICS": "1", "__DACE_USE_SVE": "0"})

    print("Generated test configurations:")
    for cfg in configs:
        print("  ", cfg)

    return configs


N = 64


@dace.program
def dace_add(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N:1]:
        C[i] = A[i] + B[i]


@dace.program
def dace_vector_mult(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * B[i]


@dace.program
def dace_vector_mult_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * 0.5


@dace.program
def dace_vector_add(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


@dace.program
def dace_vector_add_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + 0.5


@dace.program
def dace_vector_sub(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] - B[i]


@dace.program
def dace_vector_sub_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] - 0.5


@dace.program
def dace_vector_sub_w_scalar_c(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5 - A[i]


@dace.program
def dace_vector_div(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] / B[i]


@dace.program
def dace_vector_div_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] / 0.5


@dace.program
def dace_vector_div_w_scalar_c(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5 / A[i]


@dace.program
def dace_vector_copy(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i]


@dace.program
def dace_vector_copy_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5


@dace.program
def dace_vector_exp(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = exp(A[i])


@dace.program
def dace_vector_log(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = log(A[i])


@dace.program
def dace_vector_min(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = min(A[i], B[i])


@dace.program
def dace_vector_min_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = min(A[i], 0.5)


@dace.program
def dace_vector_max(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = min(A[i], B[i])


@dace.program
def dace_vector_max_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = min(A[i], 0.5)


@dace.program
def dace_vector_gt(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] > B[i]


@dace.program
def dace_vector_gt_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] > 0.5


@dace.program
def dace_vector_gt_w_scalar_c(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5 > A[i]


@dace.program
def dace_vector_lt(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] < B[i]


@dace.program
def dace_vector_lt_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] < 0.5


@dace.program
def dace_vector_lt_w_scalar_c(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5 < A[i]


@dace.program
def dace_vector_ge(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] >= B[i]


@dace.program
def dace_vector_ge_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] >= 0.5


@dace.program
def dace_vector_ge_w_scalar_c(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5 >= A[i]


@dace.program
def dace_vector_le(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] <= B[i]


@dace.program
def dace_vector_le_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] <= 0.5


@dace.program
def dace_vector_le_w_scalar_c(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = 0.5 <= A[i]


@dace.program
def dace_vector_eq(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] == B[i]


@dace.program
def dace_vector_eq_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] == 0.5


@dace.program
def dace_vector_ne(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] != B[i]


@dace.program
def dace_vector_ne_w_scalar(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] != 0.5


###############################################################################
# Collect all intrinsic DaCe programs
###############################################################################

INTRINSIC_FUNCS = [
    dace_vector_mult,
    dace_vector_mult_w_scalar,
    dace_vector_add,
    dace_vector_add_w_scalar,
    dace_vector_sub,
    dace_vector_sub_w_scalar,
    dace_vector_sub_w_scalar_c,
    dace_vector_div,
    dace_vector_div_w_scalar,
    dace_vector_div_w_scalar_c,
    dace_vector_copy,
    dace_vector_copy_w_scalar,
    dace_vector_exp,
    dace_vector_log,
    dace_vector_min,
    dace_vector_min_w_scalar,
    dace_vector_max,
    dace_vector_max_w_scalar,
    dace_vector_gt,
    dace_vector_gt_w_scalar,
    dace_vector_gt_w_scalar_c,
    dace_vector_lt,
    dace_vector_lt_w_scalar,
    dace_vector_lt_w_scalar_c,
    dace_vector_ge,
    dace_vector_ge_w_scalar,
    dace_vector_ge_w_scalar_c,
    dace_vector_le,
    dace_vector_le_w_scalar,
    dace_vector_le_w_scalar_c,
    dace_vector_eq,
    dace_vector_eq_w_scalar,
    dace_vector_ne,
    dace_vector_ne_w_scalar,
]


@pytest.mark.parametrize("config", select_env_flags())
@pytest.mark.parametrize("func", INTRINSIC_FUNCS)
def test_intrinsic(func: DaceProgram, config: List[Dict[str, str]]):
    for k, v in config.items():
        os.environ[k] = v

    A = np.random.rand(N)
    B = np.random.rand(N)
    C = np.zeros_like(A)

    arrays = {"A": A, "B": B, "C": C}

    cfg_label = "_".join(f"{k}{v}" for k, v in config.items())

    run_vectorization_test(
        dace_func=func,
        arrays=arrays,
        params={},
        save_sdfgs=True,
        sdfg_name=f"{func.name}_{cfg_label}",
        apply_loop_to_map=True,
        simplify=True,
        skip_simplify=["ScalarToSymbolPromotion"],
    )


@dace.program
def dace_strided_gather(A: dace.float64[N], B: dace.int64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[B[i]] * D[i]


@pytest.mark.parametrize("config", select_env_flags())
def test_strided_gather(config: List[Dict[str, str]]):
    for k, v in config.items():
        os.environ[k] = v
    # -------------------------
    # Prepare test arrays
    # -------------------------
    A = np.random.rand(N)
    D = np.random.rand(N)
    B = np.random.randint(0, N, size=N, dtype=np.int64)  # ensures B[i]*2 < N
    C = np.zeros_like(A)

    arrays = dict(A=A, B=B, C=C, D=D)

    # Deepcopy for baseline and vectorized execution
    arr_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arr_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # -------------------------
    # Build baseline SDFG
    # -------------------------
    sdfg = dace_strided_gather.to_sdfg(simplify=False)
    sdfg.name = "strided_gather"

    simplify_sdfg_if_needed(sdfg, simplify=True, skip_simplify=["ScalarToSymbolPromotion"])
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    # -------------------------
    # Build vectorized SDFG
    # -------------------------
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = "strided_gather_vec"

    apply_vectorization_pass(vsdfg, vector_width=8)
    DetectGather().apply_pass(vsdfg, {})
    vsdfg.simplify()

    # -------------------------
    # Execute baseline + vectorized
    # -------------------------
    compile_and_run(sdfg, arr_orig, params=dict())
    compile_and_run(vsdfg, arr_vec, params=dict())

    # -------------------------
    # Compare results
    # -------------------------
    for name in arrays:
        assert np.allclose(arr_orig[name], arr_vec[name], rtol=1e-32), \
            f"{name} mismatch:\n{arr_orig[name] - arr_vec[name]}"

    return vsdfg
