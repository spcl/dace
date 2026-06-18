import pytest
import numpy as np
import dace
import importlib
from dace.transformation.passes.offloading.OffloadToAcceleratorV2 import OffloadToAccelerator as OtA
from copy import deepcopy
from pathlib import Path

#########################################################
###                      Globals                      ###
#########################################################
pytest.mark.polybench = pytest.mark.polybench
pytest.mark.current = pytest.mark.current
pytest.mark.polybench_small = pytest.mark.polybench_small
pytest.mark.polybench_medium = pytest.mark.polybench_medium
pytest.mark.polybench_large = pytest.mark.polybench_large

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "polybench: mark test as NPBench/Polybench offload suite")
    config.addinivalue_line("markers", "current: tests of current interest")
    config.addinivalue_line("markers", "polybench_small: mark test as small Polybench SDFG")
    config.addinivalue_line("markers", "polybench_medium: mark test as medium Polybench SDFG")
    config.addinivalue_line("markers", "polybench_large: mark test as large Polybench SDFG")

VIEW_ORG = False
VIEW_MOD = False


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

    sdfg(**baseline_inputs)

    OtA().apply_pass(sdfg, {})
    sdfg.validate()
    if VIEW_MOD: sdfg.view()

    sdfg._recompile = True
    sdfg(**offload_inputs)

    for name, baseline_value in baseline_inputs.items():
        if isinstance(baseline_value, np.ndarray):
            assert np.allclose(baseline_value, offload_inputs[name], equal_nan=True), f"Mismatch in '{name}'"




########################################################
##                  Polybench Tests                  ###
########################################################

# helpers
def test_npbench_polybench_offload(short_name):
    module_name = f"npbench.benchmarks.polybench.{short_name}.{short_name}_dace"
    module = importlib.import_module(module_name)
    assert hasattr(module, "kernel"), f"{module_name} has no kernel!"
            
    sdfg = module.kernel.to_sdfg()
    _run_generic_offloading_test(sdfg)

# Small SDFGs
@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_atax(): test_npbench_polybench_offload("atax") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_bicg(): test_npbench_polybench_offload("bicg") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
@pytest.mark.current
def test_polybench_doitgen(): test_npbench_polybench_offload("doitgen") # FAILS with ValueError: matrix-matrix product only supported on matrices TODO

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_floyd_warshall(): test_npbench_polybench_offload("floyd_warshall") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_gemm(): test_npbench_polybench_offload("gemm") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_k3mm(): test_npbench_polybench_offload("k3mm") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_lu(): test_npbench_polybench_offload("lu") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_mvt(): test_npbench_polybench_offload("mvt") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_trisolv(): test_npbench_polybench_offload("trisolv") # fine

@pytest.mark.polybench
@pytest.mark.polybench_small
def test_polybench_trmm(): test_npbench_polybench_offload("trmm") # fine

"""
I'm getting multiple (three in my current testing subset) SDFGs with the following pattern (see also the screenshots):
Some arrays are connected to a lib node. One of them is also connected to a tasklet further down.
My pass now detects the tasklet and concludes the array is needed on the CPU, yet also detects the map and concludes its in the GPU.
At the moment, this throws a Not-Supported Error, as a copy is needed WITHIN a state and I only support interstate copies.

I believe in this case, even though the map is toplevel, it should not be offloaded, as the tasklet needs to run on CPU.
However, this case is difficult to distinguish from similar graphs where a top-level lib node is (indirectly) connected to map nodes and does need to be offloaded.
Essentially, if it was a map not a tasklet, the same lib node would need to be on GPU.
The difficulty detecting this comes from the multiple, potentially indirect, arbitrarily complex connections and intermediate steps between the array, lib node and/or tasklet.

It does seem to be a common pattern, so I think it should be supported.
1) I could analyse each lib node individually:
 - inside a map: do not offload
 - connected to a map: do offload
 - connected to a tasklet: do not offload
However, this a) adds overhead and b) unless every connection is traced through the entire graph, there is the possibility that indirect connections are missed

2) Another way is less analytical but simpler and perhaps more exhaustive:
- I offload all toplevel maps
- Whenever I run into the both-CPU-and-GPU-in-same-state-unsupported error, I check whether this state contains map nodes.
- I switch them to CPU and try again.
- If it works, move on. Else throw the error.

I prefer 2) even though its a bit hacky. What do you think?
"""

# Medium SDFGs
@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_cholesky(): test_npbench_polybench_offload("cholesky") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
@pytest.mark.current
def test_polybench_cholesky2(): test_npbench_polybench_offload("cholesky2") # fails due to improper edge handling! TODO

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_covariance(): test_npbench_polybench_offload("covariance") # fine, but requires unsupported intra-state copy

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_gesummv(): test_npbench_polybench_offload("gesummv") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_jacobi_1d(): test_npbench_polybench_offload("jacobi_1d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_jacobi_2d(): test_npbench_polybench_offload("jacobi_2d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_k2mm(): test_npbench_polybench_offload("k2mm") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_syrk(): test_npbench_polybench_offload("syrk") # fine, but requires unsupported intra-state copy

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_syr2k(): test_npbench_polybench_offload("syr2k") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_fdtd_2d(): test_npbench_polybench_offload("fdtd_2d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_medium
def test_polybench_gemver(): test_npbench_polybench_offload("gemver") # fine


# Large SDFGs
@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_nussinov(): test_npbench_polybench_offload("nussinov") # fine, I think...

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.current
def test_polybench_adi(): test_npbench_polybench_offload("adi") # incorrect, not enough copies TODO

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_correlation(): test_npbench_polybench_offload("correlation") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_deriche(): test_npbench_polybench_offload("deriche") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_durbin(): test_npbench_polybench_offload("durbin") # fine, no offload possible

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_gramschmidt(): test_npbench_polybench_offload("gramschmidt") #fine

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_heat_3d(): test_npbench_polybench_offload("heat_3d") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_ludcmp(): test_npbench_polybench_offload("ludcmp") # fine

@pytest.mark.polybench
@pytest.mark.polybench_large
@pytest.mark.current
def test_polybench_seidel_2d(): test_npbench_polybench_offload("seidel_2d") # fine (nested loop with copies! -> there's a slight inefficiency I can possibly fix at some point)

@pytest.mark.polybench
@pytest.mark.polybench_large
def test_polybench_symm(): test_npbench_polybench_offload("symm") # fine (has ineffeimcy at copy back like all loops)


#######################################################
##                       Main                       ###
#######################################################

VIEW_ORG = True
VIEW_MOD = True

if __name__ == "__main__":
    # Run with: python npbench_testsuite.py
    pytest.main([__file__, "-s", "-v", "--tb=short", "-m", "current"])
    # pytest.main([__file__, "-s", "-v", "--tb=short", "-m", "polybench_small"])
    #pytest.main([__file__, "-v", "--tb=short", "-m", "polybench"])

    # ISSUE Nr. 1: library nodes
    # ISSUE Nr. 2: __return_gpu
    # ISSUE Nr. 3: various small wierd shit

    # offload all toplevel library nodes -> 3 / 10 (mvt, gemm, floyd_warshall) PASS
    # don't do that at all               -> 7 / 10 (mvt, gemm, doitgen) FAIL
    # => fix mvt & gemm with library nodes
    # => doitgen is a different issue, sth about matrices in the generic tester
