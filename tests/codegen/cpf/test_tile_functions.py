# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF renders a vectorized kernel's tile nodes as calls of one ISA's tile functions, defined in the unit itself."""

import re

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from dace.libraries.tileops.dispatch import ISA as TileISA
from dace.libraries.tileops.dispatch import host_supported_isas, widest_function_isa
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone

#: Rendered ISA -> the suffix of its tile functions.
BACKENDS = {TileISA.SCALAR: "scalar", TileISA.AVX2: "avx2", TileISA.AVX512: "avx512", TileISA.ARM_NEON: "neon"}


def vectorized_fma_sdfg(name: str) -> dace.SDFG:
    """``C[i] = A[i] * B[i] + A[i]`` tiled by 8 with a masked tail, so it holds loads, stores, a mask and binops."""
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    for array in "ABC":
        sdfg.add_array(array, (N,), dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "axpy",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a * _b + _a",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    config = VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy=RemainderStrategy.MASKED_TAIL)
    VectorizeCPUMultiDim(config).apply_pass(sdfg, {})
    return sdfg


def render(isa: TileISA, language: str) -> cpf.Rendering:
    tag = "cpp" if language == "c++" else language
    sdfg = vectorized_fma_sdfg(f"cpf_tile_functions_{isa.name.lower()}_{tag}")
    runs_here = isa is TileISA.AUTO or isa in host_supported_isas()
    return cpf.render(sdfg, language=language, isa=isa, check_compiles=runs_here)


@pytest.mark.parametrize("language", ["c++", "c"])
@pytest.mark.parametrize("isa", [TileISA.SCALAR, TileISA.AVX2, TileISA.AVX512, TileISA.ARM_NEON, TileISA.AUTO])
def test_a_vectorized_kernel_renders_the_tile_functions_of_one_isa(isa: TileISA, language: str) -> None:
    """Every tile node calls a function of the chosen ISA (``AUTO``: the widest the host runs), defined once in the
    unit, and a host that runs the ISA computes the kernel."""
    rendering = render(isa, language)
    backend = BACKENDS[widest_function_isa() if isa is TileISA.AUTO else isa]
    called = set(re.findall(r"\b(tile_\w+?_(?:scalar|avx2|avx512|neon))\(", rendering.code))
    assert called and all(name.endswith(f"_{backend}") for name in called), called
    for name in called:
        assert len(re.findall(rf"static inline [^\n]*\b{name}\(", rendering.code)) == 1, name
    assert {"binop", "load", "store", "maskgen"} <= {name.split("_")[1] for name in called}
    if isa is not TileISA.AUTO and isa not in host_supported_isas():
        return
    assert_standalone(rendering.code, rendering.sdfg.name, language=language)
    library = build_standalone(rendering.code, rendering.sdfg.name, language=language)
    rng = np.random.default_rng(7)
    for n in (8, 21, 64):
        A, B, C = rng.random(n), rng.random(n), np.zeros(n)
        call_standalone(library, rendering.sdfg, {"A": A, "B": B, "C": C, "N": n}, rendering.arguments)
        np.testing.assert_allclose(C, A * B + A, rtol=1e-15)


def test_a_tile_isa_without_tile_functions_is_refused() -> None:
    with pytest.raises(ValueError, match="CPF renders tile nodes for"):
        cpf.render(vectorized_fma_sdfg("cpf_tile_functions_sve"), isa=TileISA.ARM_SVE, check_compiles=False)


if __name__ == "__main__":
    for language in ("c++", "c"):
        for isa in (TileISA.SCALAR, TileISA.AVX2, TileISA.AVX512, TileISA.ARM_NEON, TileISA.AUTO):
            test_a_vectorized_kernel_renders_the_tile_functions_of_one_isa(isa, language)
    test_a_tile_isa_without_tile_functions_is_refused()
