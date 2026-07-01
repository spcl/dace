import pytest
import numpy as np
import dace
import importlib
import matplotlib.pyplot as plt
from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA
from copy import deepcopy
from pathlib import Path
from time import time

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"No `gpu_block_size` property specified on map.*",
    category=UserWarning,
    module=r"dace\.transformation\.dataflow\.add_threadblock_map",
)

#########################################################
###                      Globals                      ###
#########################################################
pytest.mark.polybench = pytest.mark.polybench
pytest.mark.current = pytest.mark.current
pytest.mark.no_offload = pytest.mark.no_offload
pytest.mark.polybench_small = pytest.mark.polybench_small
pytest.mark.polybench_medium = pytest.mark.polybench_medium
pytest.mark.polybench_large = pytest.mark.polybench_large
pytest.mark.intrastate_copies = pytest.mark.intrastate_copies
pytest.mark.non_polybench = pytest.mark.non_polybench

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "polybench: mark test as NPBench/Polybench offload suite")
    config.addinivalue_line("markers", "current: tests of current interest")
    config.addinivalue_line("markers", "no_offload: mark graphs which do not require/enable offloading")
    config.addinivalue_line("markers", "polybench_small: mark test as small Polybench SDFG")
    config.addinivalue_line("markers", "polybench_medium: mark test as medium Polybench SDFG")
    config.addinivalue_line("markers", "polybench_large: mark test as large Polybench SDFG")
    config.addinivalue_line("markers", "intrastate_copies: mark graphs which require intrastate copies")
    config.addinivalue_line("markers", "non_polybench: mark test as non-Polybench NPBench SDFG")

VIEW_ORG = False
VIEW_MOD = False

# Some NPBench initializers (e.g., scipy CSR `.data`) can return NumPy views.
# Enable passing them to DaCe during this test suite.
dace.Config.set('compiler', 'allow_view_arguments', value=True)


#########################################################
##          Automatic Testing Infrastructure          ###
#########################################################

"""
def _discover_polybench_kernel_modules() -> list[str]:
    polybench_root = Path(__file__).resolve().parents[5] / "npbench" / "npbench" / "benchmarks" / "polybench"
    modules: list[str] = []
    for file_path in sorted(polybench_root.glob("**/*_dace.py")):
        rel_path = file_path.relative_to(polybench_root).with_suffix("")
        modules.append("npbench.benchmarks.polybench." + ".".join(rel_path.parts))
    return modules

ALL_POLYBENCH_KERNEL_MODULES = _discover_polybench_kernel_modules()
print("ALL_POLYBENCH_KERNEL_MODULES:\n" + "\n".join([name.split(".")[-1][:-5] for name in ALL_POLYBENCH_KERNEL_MODULES]))
"""

def _make_scalar(dtype):
    np_dtype = np.dtype(dtype.as_numpy_dtype())
    scalar_type = np_dtype.type
    if np.issubdtype(np_dtype, np.floating):
        return scalar_type(1.5)
    if np.issubdtype(np_dtype, np.integer):
        return scalar_type(8)
    if np.issubdtype(np_dtype, np.bool_):
        return scalar_type(True)
    return scalar_type(1)


def _evaluate_dim(dim, symbols: dict[str, int]) -> int:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    try:
        return int(dim.subs(symbols))
    except Exception:
        try:
            return int(dace.symbolic.evaluate(dim, symbols))
        except Exception:
            return 8


def _make_array(desc: dace.data.Array, symbols: dict[str, int]) -> np.ndarray:
    shape = tuple(max(2, _evaluate_dim(dim, symbols)) for dim in desc.shape)
    np_dtype = desc.dtype.as_numpy_dtype()
    arr = np.arange(np.prod(shape), dtype=np_dtype).reshape(shape)
    if np.issubdtype(np_dtype, np.floating):
        return arr / (arr.size + 1.0) + 1.0
    if np.issubdtype(np_dtype, np.integer):
        return (arr % 17) + 1
    if np.issubdtype(np_dtype, np.bool_):
        return (arr % 2) == 0
    return arr


def _generate_inputs_for_sdfg(sdfg: dace.SDFG) -> dict[str, object]:
    symbols: dict[str, int] = {str(s): 8 for s in sdfg.free_symbols}

    for argname, desc in sdfg.arglist().items():
        if argname.startswith("__return"):
            continue
        if isinstance(desc, dace.data.Scalar):
            np_dtype = np.dtype(desc.dtype.as_numpy_dtype())
            if np.issubdtype(np_dtype, np.integer):
                symbols.setdefault(argname, 8)

    inputs: dict[str, object] = {}

    for argname, desc in sdfg.arglist().items():
        if argname.startswith("__return"):
            continue

        if argname in symbols:
            inputs[argname] = np.int64(symbols[argname])
            continue

        if desc is None:
            inputs[argname] = np.int64(8)
            continue

        if isinstance(desc, dace.data.Array):
            inputs[argname] = _make_array(desc, symbols)
        elif isinstance(desc, dace.data.Scalar):
            inputs[argname] = _make_scalar(desc.dtype)
        else:
            inputs[argname] = np.int64(8)

    for symname, symval in symbols.items():
        if symname not in inputs:
            inputs[symname] = np.int64(symval)

    return inputs


def _run_generic_offloading_test(sdfg: dace.SDFG):
    sdfg.validate()
    if VIEW_ORG: sdfg.view()

    baseline_inputs = _generate_inputs_for_sdfg(sdfg)
    offload_inputs = deepcopy(baseline_inputs)

    print("run orig")
    start_time = time()
    sdfg(**baseline_inputs)
    seq_time = time() - start_time

    print("apply pass")
    OtA().apply_pass(sdfg, {})
    if VIEW_MOD: sdfg.view()
    sdfg.validate()
    
    print("run new")
    sdfg._recompile = True
    start_time = time()
    sdfg(**offload_inputs)
    offl_time = time() - start_time

    for name, baseline_value in baseline_inputs.items():
        if isinstance(baseline_value, np.ndarray):
            assert np.allclose(baseline_value, offload_inputs[name], equal_nan=True), f"Mismatch in '{name}'"

    return seq_time, offl_time


########################################################
##                  Polybench Tests                  ###
########################################################

# helpers
def test_offload(short_name, submodule="", test_name="", program_name=""):
    if not test_name: test_name = short_name

    if not submodule:
        module_name = f"npbench.benchmarks.{short_name}.{test_name}_dace"
    else:
        module_name = f"npbench.benchmarks.{submodule}.{short_name}.{test_name}_dace"
    module = importlib.import_module(module_name)
    
    sdfg = None
    for sdfg_name in [program_name, short_name, "kernel"]:
        if sdfg_name and hasattr(module, sdfg_name):
            sdfg = getattr(module, sdfg_name).to_sdfg()
    if not sdfg:
        assert False, "sdfg not found"

    _run_generic_offloading_test(sdfg)

def test_polybench_offload(short_name): test_offload(short_name, "polybench")

def benchmark_polybench_offload(short_name):
    module_name = f"npbench.benchmarks.polybench.{short_name}.{short_name}_dace"
    module = importlib.import_module(module_name)
    assert hasattr(module, "kernel"), f"{module_name} has no kernel!"
            
    sdfg = module.kernel.to_sdfg()
    seq_time, offl_time = _run_generic_offloading_test(sdfg)
    return seq_time, offl_time

def gather_polybench_runtime_data(shortnames):
    results = {}
    failed = {}

    for name in shortnames:
        try:
            seq_time, offl_time = benchmark_polybench_offload(name)
        except Exception as ex:
            failed[name] = f"{type(ex).__name__}: {ex}"
            continue

        results[name] = [seq_time, offl_time]

    results["doitgen"] = test_polybench_doitgen()

    if failed:
        num = len(failed)
        print(f"\n\n{num} FAILURE{"S" if num > 1 else ""}:")
        for name, exception in failed.items():
            print(name, ":\t", exception)
        print("\n\n")
    
    return results

def create_graph(results):
    successful_names = list(results.keys())
    runtimes = np.array(list(results.values()))
    unoptimized_times = runtimes[:, 0]
    optimized_times = runtimes[:, 1]

    x = np.arange(len(successful_names), dtype=np.float64)
    bar_width = 0.38

    plt.figure(figsize=(max(8.0, 0.8 * len(successful_names)), 5.5))
    plt.bar(x - bar_width / 2, unoptimized_times, width=bar_width, color="red", label="unoptimized")
    plt.bar(x + bar_width / 2, optimized_times, width=bar_width, color="blue", label="optimized")

    plt.xlabel("polybench graphs")
    plt.ylabel("runtime (s)")
    plt.title("Polybench Runtime Comparison")
    plt.xticks(x, successful_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = Path(__file__).resolve().with_name("polybench_runtime_comparison.png")
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()
    else:
        plt.show()

######################################

POLYBENCH_SMALL_NAMES = [
    "atax",
    "bicg",
    #"doitgen",
    "floyd_warshall",
    "gemm",
    "k3mm",
    "lu",
    "mvt",
    "trisolv",
    "trmm",
]

POLYBENCH_MEDIUM_NAMES = [
    "cholesky",
    "cholesky2",
    "covariance",
    "gesummv",
    "jacobi_1d",
    "jacobi_2d",
    "k2mm",
    "syrk",
    "syr2k",
    "fdtd_2d",
    "gemver",
]

POLYBENCH_LARGE_NAMES = [
    "nussinov",
    "adi",
    "correlation",
    "deriche",
    "durbin",
    "gramschmidt",
    "heat_3d",
    "ludcmp",
    "seidel_2d",
    "symm",
]

POLYBENCH_NAMES = POLYBENCH_SMALL_NAMES + POLYBENCH_MEDIUM_NAMES + POLYBENCH_LARGE_NAMES

TOPLEVEL_BENCHMARKS = [
    "azimint_hist", 
    "azimint_naive", # works
    "cavity_flow", 
    "channel_flow", 
    "compute", # works
    "contour_integral",
    "crc16", # works
    "go_fast", # works
    "mandelbrot1", 
    "mandelbrot2", 
    "nbody", 
    "scattering_self_energies", # works
    "spmv", 
    "stockham_fft"
] 
     
DEEPLEARNING_BENCHMARKS = [
    "conv2d_bias", 
    "lenet", 
    "mlp", 
    "resnet", 
    "softmax", 
]

WEATHER_BENCHMARKS = [
    "hdiff", # works
    "vadv",
]

PYTRAN_BENCHMARKS = [
    "arc_distance", # works
]

#####################################

# Small SDFGs
@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_atax(): test_polybench_offload("atax") # fine

@pytest.mark.polybench
def test_polybench_bicg(): test_polybench_offload("bicg") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
# extra test case because the generic tester can't handle matrix inputs
def test_polybench_doitgen():
    @dace.program
    def doitgen_explicit(A: dace.float64[8, 10, 12], C4: dace.float64[12, 12]):
        for r in range(8):
            A[r, :, :] = np.reshape(np.reshape(A[r], (10, 12)) @ C4, (10, 12))

    nr, nq, np_size = 8, 10, 12
    a_init = np.fromfunction(lambda i, j, k: ((i * j + k) % np_size) / np_size, (nr, nq, np_size), dtype=np.float64)
    c4_init = np.fromfunction(lambda i, j: (i * j % np_size) / np_size, (np_size, np_size), dtype=np.float64)

    a_cpu = a_init.copy()
    a_offload = a_init.copy()
    c4_cpu = c4_init.copy()
    c4_offload = c4_init.copy()

    sdfg = doitgen_explicit.to_sdfg()
    sdfg.validate()
    if VIEW_ORG:
        sdfg.view()

    start_time = time()
    sdfg(A=a_cpu, C4=c4_cpu)
    seq_time = time() - start_time

    OtA().apply_pass(sdfg, {})
    sdfg.validate()
    if VIEW_MOD:
        sdfg.view()

    sdfg._recompile = True
    start_time = time()
    sdfg(A=a_offload, C4=c4_offload)
    offl_time = time() - start_time

    assert np.allclose(a_cpu, a_offload)

    return seq_time, offl_time

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_floyd_warshall(): test_polybench_offload("floyd_warshall") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_gemm(): test_polybench_offload("gemm") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_k3mm(): test_polybench_offload("k3mm") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
@pytest.mark.no_offload
def test_polybench_lu(): test_polybench_offload("lu") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_mvt(): test_polybench_offload("mvt") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
@pytest.mark.no_offload
def test_polybench_trisolv(): test_polybench_offload("trisolv") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_trmm(): test_polybench_offload("trmm") # fine


# Medium SDFGs
@pytest.mark.polybench
@pytest.mark.polybench_medium
@pytest.mark.no_offload
def test_polybench_cholesky(): test_polybench_offload("cholesky") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_cholesky2(): test_polybench_offload("cholesky2") # fails due to missing OpenBlas

@pytest.mark.polybench
@pytest.mark.polybench_medium
@pytest.mark.intrastate_copies
def test_polybench_covariance(): test_polybench_offload("covariance") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_gesummv(): test_polybench_offload("gesummv") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_jacobi_1d(): test_polybench_offload("jacobi_1d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_jacobi_2d(): test_polybench_offload("jacobi_2d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_k2mm(): test_polybench_offload("k2mm") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
@pytest.mark.intrastate_copies
def test_polybench_syrk(): test_polybench_offload("syrk") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_syr2k(): test_polybench_offload("syr2k") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_fdtd_2d(): test_polybench_offload("fdtd_2d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_gemver(): test_polybench_offload("gemver") # fine


# Large SDFGs
@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_nussinov(): test_polybench_offload("nussinov") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_adi(): test_polybench_offload("adi") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_correlation(): test_polybench_offload("correlation") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_deriche(): test_polybench_offload("deriche") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.no_offload
@pytest.mark.intrastate_copies
def test_polybench_durbin(): test_polybench_offload("durbin") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_gramschmidt(): test_polybench_offload("gramschmidt") #fine

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_heat_3d(): test_polybench_offload("heat_3d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_ludcmp(): test_polybench_offload("ludcmp") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_seidel_2d(): test_polybench_offload("seidel_2d") # fine (nested loop with copies! -> there's a slight inefficiency I can possibly fix at some point)

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.intrastate_copies
def test_polybench_symm(): test_polybench_offload("symm") # fine


# Non-Polybench SDFGs
@pytest.mark.non_polybench
def test_azimint_hist(): test_offload("azimint_hist") # fine (no more naming issue)

@pytest.mark.non_polybench
@pytest.mark.intrastate_copies
def test_azimint_naive(): test_offload("azimint_naive") # fine

@pytest.mark.non_polybench
@pytest.mark.intrastate_copies
def test_cavity_flow(): test_offload("cavity_flow") # fine

@pytest.mark.non_polybench
def test_channel_flow(): test_offload("channel_flow") # fine

@pytest.mark.non_polybench
def test_compute(): test_offload("compute") # fine

@pytest.mark.non_polybench
def test_contour_integral(): test_offload("contour_integral") # FAILS because lapacke.h is missing -> OPENBLAS ISSUE

@pytest.mark.non_polybench
@pytest.mark.no_offload
def test_crc16(): test_offload("crc16")

@pytest.mark.non_polybench
def test_conv2d_bias(): test_offload("conv2d_bias", "deep_learning", "conv2d") # fine

@pytest.mark.non_polybench
def test_lenet(): test_offload("lenet", "deep_learning", program_name="lenet5") # FAILS with dimensionality mismatch in SDFG.validate(); sth about subsets?

@pytest.mark.non_polybench
def test_mlp(): test_offload("mlp", "deep_learning") # fine

@pytest.mark.non_polybench
def test_resnet(): test_offload("resnet", "deep_learning", program_name="resnet_basicblock_gpu") # fine

@pytest.mark.non_polybench
def test_softmax(): test_offload("softmax", "deep_learning", program_name="softmax_gpu") # fine
@pytest.mark.non_polybench
def test_go_fast(): test_offload("go_fast") # fine

@pytest.mark.non_polybench
def test_mandelbrot1(): test_offload("mandelbrot1", program_name="mandelbrot") # fine

@pytest.mark.non_polybench
def test_mandelbrot2(): test_offload("mandelbrot2", program_name="mandelbrot") # fine

@pytest.mark.non_polybench
def test_nbody(): test_offload("nbody") # fine

#@pytest.mark.non_polybench
#def test_scattering_self_energies(): test_offload("scattering_self_energies")  # FAILS
# HANGS in run(orig) for 8+ minutes -> just needs to run longer, its 10 or so nested loops; offload looks correct
# FAILS in run(offl) with wcr-related compiler failure

@pytest.mark.non_polybench
def test_arc_distance(): test_offload("arc_distance", "pythran") # fine

@pytest.mark.no_offload
@pytest.mark.non_polybench
@pytest.mark.current
def test_spmv(): test_offload("spmv") # fine

@pytest.mark.non_polybench
def test_stockham_fft(): test_offload("stockham_fft") # FAILS huuge compiler error in orig and this error in offl:
"""
E   dace.sdfg.validation.InvalidSDFGEdgeError: Dimensionality mismatch between src/dst subsets 
(at state BinOp_39_0, edge expr_3_141592653589793_times_i_coord_gpu[0:R, 0:R] (_Mult__map[__i0=0:R, __i1=0:R]:OUT_expr_3_141592653589793_times_i_coord -> expr_3_141592653589793_times_i_coord_gpu:None))
"""
@pytest.mark.non_polybench
def test_hdiff(): test_offload("hdiff", "weather_stencils") # fine

@pytest.mark.non_polybench
def test_vadv(): test_offload("vadv", "weather_stencils") # FAILS with data mismatch!

# 17 of 25 run out of the box!

#######################################################
##                       Main                       ###
#######################################################

VIEW_ORG = False
VIEW_MOD = False

if __name__ == "__main__":
    # Run with: python npbench_testsuite.py
    #pytest.main([__file__, "-s", "-v", "--tb=short", "-m", "current"])
    pytest.main([__file__, "-v", "--tb=short", "-m", "polybench"])
    pytest.main([__file__, "-v", "--tb=short", "-m", "non_polybench"])

    #create_graph(gather_polybench_runtime_data(POLYBENCH_NAMES))

# intrastate copies
# optimization passes

# optimize end of loop
# optimize pull all first-copies to front

"""
I'm getting the following compiler failures:

**1) scattering_self_energies:** seems wcr related

error:
E   /home/finn/Desktop/ETH/BachelorThesis/dace/dace/codegen/../runtime/include/dace/reduction.h(65): error: no instance of overloaded function "atomicCAS" matches the argument list
E               argument types are: (dace::complex128 *, dace::complex128, dace::complex128)
E             detected during:
E               instantiation of "T dace::wcr_custom<T>::reduce_atomic(WCR, T *, const T &) [with T=dace::complex128, WCR=dace::_wcr_fixed<dace::ReductionType::Sum, dace::complex128>]" 
E   (538): here
E               instantiation of "T dace::wcr_fixed<REDTYPE, T, SFINAE>::reduce_atomic(T *, const T &) [with REDTYPE=dace::ReductionType::Sum, T=dace::complex128, SFINAE=void]" 
E   /home/finn/Desktop/ETH/BachelorThesis/.dacecache/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies/src/cuda/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies_cuda.cu(166): here
E   
E   1 error detected in the compilation of "/home/finn/Desktop/ETH/BachelorThesis/.dacecache/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies/src/cuda/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies_cuda.cu".
E   gmake[2]: *** [CMakeFiles/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies.dir/build.make:94: CMakeFiles/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies.dir/home/finn/Desktop/ETH/BachelorThesis/.dacecache/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies/src/cuda/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies_cuda.cu.o] Error 1
E   gmake[1]: *** [CMakeFiles/Makefile2:90: CMakeFiles/npbench_benchmarks_scattering_self_energies_scattering_self_energies_dace_scattering_self_energies.dir/all] Error 2
E   gmake: *** [Makefile:91: all] Error 2
''' 

additional warnings:
'''
E   /home/finn/Desktop/ETH/BachelorThesis/dace/dace/codegen/../runtime/include/dace/../../../external/moodycamel/concurrentqueue.h(3607): warning #68-D: integer conversion resulted in a change of sign
E   
/home/finn/Desktop/ETH/BachelorThesis/dace/dace/codegen/../runtime/include/dace/pi.h(58): warning #20208-D: 'long double' is treated as 'double' in device code
E   /home/finn/Desktop/ETH/BachelorThesis/dace/dace/codegen/../runtime/include/dace/nan.h(31): warning #20013-D: calling a constexpr __host__ function("quiet_NaN") from a __host__ __device__ function("operator double") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
'''

"""


"""
1) cholesky2                        fail due to bad OpenBlas
2) contour integral

3) lenet                            fail with compiler error and dimensionality mismatch
4) stockham_fft

5) test_scattering_self_energies    fail with wrc-related compiler error

6) vadv                             fail with data mismatch
"""