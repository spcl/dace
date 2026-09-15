# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every GPU BLAS expansion emits a REAL vendor symbol, under both dialects.

These nodes have no GPU test on either backend -- the suite covers ``pure``, MKL and OpenBLAS -- so
the vendor call each one emits was never checked by anything. That gap hid four wrong names at once
(``cublasDDnrm2``, ``cublasDDasum``, ``cublasDDcopy`` and ``cublasIDamax``, the last of them
pre-existing), all of which expand and generate happily and fail only when a CUDA or ROCm toolchain
tries to link them.

This needs no GPU: it expands the library node and reads the emitted text. What it pins is the one
property a shared-dialect refactor can silently break -- that the type letter appears EXACTLY ONCE.
A node whose routine name already includes the letter must go through ``dialect.routine``, and one
that does not must go through ``dialect.func``; using the wrong one doubles the letter into a symbol
neither vendor exports.
"""
import re

import pytest

import dace
from dace import Memlet
from dace.libraries import blas

N = 128
F = dace.float64

#: The routine names a vendor BLAS exports, so a doubled or transposed type letter cannot pass.
ROUTINES = (r"(axpy|copy|scal|swap|dot|nrm2|asum|amax|gemv|ger|symv|trmm|trsm|trmv|trsv|gemm|symm|"
            r"syrk|syr2k)")
#: ``<prefix><ONE type letter><routine>``, plus each vendor's non-typed helpers and the two
#: mixed-type norms (``Scnrm2``, ``Dzasum``).
VENDOR_SYMBOL = {
    "cuBLAS": re.compile(rf"^cublas(Set[A-Za-z]+|[SDCZ]{ROUTINES}|I[sdcz]amax|[SD][cz](nrm2|asum)|[A-Za-z]+Ex)$"),
    "rocBLAS": re.compile(rf"^rocblas_(set_[a-z_]+|[sdcz]{ROUTINES}|i[sdcz]amax|[sd][cz](nrm2|asum))$"),
}

#: node -> (factory, arrays, connector wiring). One entry per node converted to the shared dialect.
SPECS = {
    "axpy": (lambda: blas.axpy.Axpy("n", 2.0), [("x", [N], F), ("y", [N], F), ("r", [N], F)], [("_x", "x", True),
                                                                                               ("_y", "y", True),
                                                                                               ("_res", "r", False)]),
    "scal": (lambda: blas.scal.Scal("n", 3.0), [("x", [N], F), ("r", [N], F)], [("_x", "x", True),
                                                                                ("_res", "r", False)]),
    "nrm2": (lambda: blas.nrm2.Nrm2("n", N), [("x", [N], F), ("r", [1], F)], [("_x", "x", True),
                                                                              ("_result", "r", False)]),
    "asum": (lambda: blas.asum.Asum("n", N), [("x", [N], F), ("r", [1], F)], [("_x", "x", True),
                                                                              ("_result", "r", False)]),
    "iamax": (lambda: blas.iamax.Iamax("n", N), [("x", [N], F), ("r", [1], dace.int32)], [("_x", "x", True),
                                                                                          ("_result", "r", False)]),
    "swap": (lambda: blas.swap.Swap("n"), [("x", [N], F), ("y", [N], F)], [("_xin", "x", True), ("_yin", "y", True),
                                                                           ("_xout", "x", False),
                                                                           ("_yout", "y", False)]),
    "copy": (lambda: blas.copy.Copy("n"), [("x", [N], F), ("r", [N], F)], [("_x", "x", True), ("_y", "r", False)]),
    "ger": (lambda: blas.ger.Ger("n", alpha=1.0), [("x", [N], F), ("y", [N], F), ("a", [N, N], F),
                                                   ("r", [N, N], F)], [("_x", "x", True), ("_y", "y", True),
                                                                       ("_A", "a", True), ("_res", "r", False)]),
    "gemv": (lambda: blas.gemv.Gemv("n", alpha=1.0, beta=0.0), [("a", [N, N], F), ("x", [N], F),
                                                                ("y", [N], F)], [("_A", "a", True), ("_x", "x", True),
                                                                                 ("_y", "y", False)]),
}


def emitted_code(name: str, implementation: str) -> str:
    """The C++ one library node expands to, for one vendor."""
    factory, arrays, wiring = SPECS[name]
    sdfg = dace.SDFG(f"gpu_blas_dialect_{name}_{implementation}")
    state = sdfg.add_state("s")
    shapes = {arr: shape for arr, shape, _ in arrays}
    for arr, shape, dtype in arrays:
        sdfg.add_array(arr, shape, dtype, storage=dace.StorageType.GPU_Global)
    node = factory()
    node.implementation = implementation
    for conn, arr, is_input in wiring:
        # Any rank, not just vectors: a matrix operand written as `a[0:N]` is a rank-1 subset and
        # the node rejects it as "A must be a matrix" long before any code is emitted.
        subset = ", ".join("0" if extent == 1 else f"0:{extent}" for extent in shapes[arr])
        if is_input:
            state.add_memlet_path(state.add_read(arr), node, dst_conn=conn, memlet=Memlet(f"{arr}[{subset}]"))
        else:
            state.add_memlet_path(node, state.add_write(arr), src_conn=conn, memlet=Memlet(f"{arr}[{subset}]"))
    sdfg.expand_library_nodes()
    return "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive()
                     if isinstance(n, dace.sdfg.nodes.Tasklet) and n.code.as_string)


@pytest.mark.parametrize("name", sorted(SPECS))
@pytest.mark.parametrize("implementation", sorted(VENDOR_SYMBOL))
def test_every_emitted_call_is_a_real_vendor_symbol(name: str, implementation: str) -> None:
    code = emitted_code(name, implementation)
    calls = sorted({c.rstrip("(").strip() for c in re.findall(r"\b(?:cublas|rocblas)[A-Za-z_0-9]*\s*\(", code)})
    assert calls, f"{name}/{implementation} emitted no vendor call at all"
    unknown = [c for c in calls if not VENDOR_SYMBOL[implementation].match(c)]
    assert not unknown, (f"{name}/{implementation} emits {unknown}, which no {implementation} exports. A doubled "
                         "type letter means the expansion used dialect.func on a name that already carried it; "
                         "use dialect.routine instead.")


@pytest.mark.parametrize("name", sorted(SPECS))
def test_a_dialect_never_leaks_the_other_vendor(name: str) -> None:
    """The point of the split: a rocBLAS expansion must not name a cuBLAS symbol, or vice versa."""
    for implementation, foreign in (("cuBLAS", "rocblas"), ("rocBLAS", "cublas")):
        code = emitted_code(name, implementation)
        leaked = sorted(
            {c
             for c in re.findall(r"\b(?:cublas|rocblas)[A-Za-z_0-9]*", code) if c.lower().startswith(foreign)})
        assert not leaked, f"{name}/{implementation} leaks {leaked} from the other backend"


#: The host locals an expansion hands the vendor by address.
HOST_SCALARS = ("&__alpha", "&__beta", "&__tmp_idx")

#: How each dialect spells "read my scalar arguments from the host".
POINTER_HOST = {"cuBLAS": "CUBLAS_POINTER_MODE_HOST", "rocBLAS": "rocblas_pointer_mode_host"}


@pytest.mark.parametrize("name", sorted(SPECS))
@pytest.mark.parametrize("implementation", sorted(VENDOR_SYMBOL))
def test_a_host_scalar_is_only_passed_under_host_pointer_mode(name: str, implementation: str) -> None:
    """Both handles are created in DEVICE pointer mode, so ``&alpha`` on the stack is a GPU fault.

    ``dace_cublas.h`` / ``dace_rocblas.h`` set the mode once at handle creation. Every expansion
    that hands the vendor the address of a local -- an alpha, or the integer ``iamax`` writes back
    -- has to flip to host mode around the call. Passing one without flipping does not fail to
    compile: the GPU dereferences a host stack address at run time, which is how ``scal``, ``axpy``
    and ``iamax`` all faulted on gfx90a while ``asum`` and ``nrm2`` beside them were fine.
    """
    code = emitted_code(name, implementation)
    taken = min((code.find(local) for local in HOST_SCALARS if local in code), default=-1)
    if taken < 0:
        pytest.skip(f"{name} passes no host scalar by address")
    mode = code.find(POINTER_HOST[implementation])
    assert 0 <= mode < taken, (f"{name}/{implementation} passes a host address to the vendor without first setting "
                               f"{POINTER_HOST[implementation]}. The handle is in device pointer mode, so the GPU "
                               "dereferences a host stack address. Wrap the call in gpu_dialect.host_scalar_mode.")
