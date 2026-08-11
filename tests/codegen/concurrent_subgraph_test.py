# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace
import numpy as np
from dace import Memlet
from dace.sdfg import SDFG
from dace.sdfg.utils import concurrent_subgraphs
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA

N = dace.symbol('N')


@dace.program
def two_concurrent_components(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    C[:] = A + 1.0
    D[:] = B * 2.0


@dace.program
def elementwise_and_reduction(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], s: dace.float64[1]):
    C[:] = A + B
    s[0] = np.sum(A)


def generated_code(sdfg: SDFG) -> str:
    return '\n'.join(o.clean_code for o in copy.deepcopy(sdfg).generate_code())


def with_sections_requested(sdfg: SDFG) -> SDFG:
    """Copy of ``sdfg`` that asks every (nested) SDFG for ``#pragma omp parallel sections``."""
    probe = copy.deepcopy(sdfg)
    for sd in probe.all_sdfgs_recursive():
        sd.openmp_sections = True
    return probe


def test_duplicate_codegen():

    # Unfortunately I have to generate this graph manually, as doing it with the python
    # frontend wouldn't result in the node ordering that we want

    sdfg = dace.SDFG("dup")
    state = sdfg.add_state()

    c_task = state.add_tasklet("c_task", inputs={"c"}, outputs={"d"}, code='d = c')
    e_task = state.add_tasklet("e_task", inputs={"a", "d"}, outputs={"e"}, code="e = a + d")
    f_task = state.add_tasklet("f_task", inputs={"b", "d"}, outputs={"f"}, code="f = b + d")

    _, A_arr = sdfg.add_array("A", [
        1,
    ], dace.float32)
    _, B_arr = sdfg.add_array("B", [
        1,
    ], dace.float32)
    _, C_arr = sdfg.add_array("C", [
        1,
    ], dace.float32)
    _, D_arr = sdfg.add_array("D", [
        1,
    ], dace.float32)
    _, E_arr = sdfg.add_array("E", [
        1,
    ], dace.float32)
    _, F_arr = sdfg.add_array("F", [
        1,
    ], dace.float32)
    A = state.add_read("A")
    B = state.add_read("B")
    C = state.add_read("C")
    D = state.add_access("D")
    E = state.add_write("E")
    F = state.add_write("F")

    state.add_edge(C, None, c_task, "c", Memlet.from_array("C", C_arr))
    state.add_edge(c_task, "d", D, None, Memlet.from_array("D", D_arr))

    state.add_edge(A, None, e_task, "a", Memlet.from_array("A", A_arr))
    state.add_edge(B, None, f_task, "b", Memlet.from_array("B", B_arr))
    state.add_edge(D, None, f_task, "d", Memlet.from_array("D", D_arr))
    state.add_edge(D, None, e_task, "d", Memlet.from_array("D", D_arr))

    state.add_edge(e_task, "e", E, None, Memlet.from_array("E", E_arr, wcr="lambda x, y: x + y"))
    state.add_edge(f_task, "f", F, None, Memlet.from_array("F", F_arr, wcr="lambda x, y: x + y"))

    A = np.array([1], dtype=np.float32)
    B = np.array([1], dtype=np.float32)
    C = np.array([1], dtype=np.float32)
    D = np.array([1], dtype=np.float32)
    E = np.zeros_like(A)
    F = np.zeros_like(A)

    sdfg(A=A, B=B, C=C, D=D, E=E, F=F)

    assert E[0] == 2
    assert F[0] == 2


def test_cpu_target_honours_openmp_sections():
    """A plain CPU SDFG that asks for ``#pragma omp parallel sections`` gets them: the knob is
    live for every CPU target, not a Snitch-only no-op."""
    sdfg = two_concurrent_components.to_sdfg(simplify=True)
    assert [st for st in sdfg.states() if len(concurrent_subgraphs(st)) > 1], 'fixture has no components to wrap'

    code = generated_code(with_sections_requested(sdfg))
    assert '#pragma omp parallel sections' in code
    assert '#pragma omp section' in code
    # Gated: an SDFG that does not ask for it gets the components emitted back to back.
    off = copy.deepcopy(sdfg)
    for sd in off.all_sdfgs_recursive():
        sd.openmp_sections = False
    off_code = generated_code(off)
    assert 'omp parallel sections' not in off_code
    # Either way the parallelism has to still be there, in the maps -- one per component.
    assert code.count('#pragma omp parallel for') == 2
    assert off_code.count('#pragma omp parallel for') == 2

    rng = np.random.default_rng(1234)
    A, B = rng.random(64), rng.random(64)
    C, D = np.zeros(64), np.zeros(64)
    sdfg(A=A, B=B, C=C, D=D, N=64)
    # A single fp64 add / multiply-by-two is exact, so the oracle comparison needs no tolerance.
    assert np.array_equal(C, A + 1.0)
    assert np.array_equal(D, B * 2.0)


def test_canonicalize_opts_out_of_omp_sections():
    """Canonicalized output never takes the construct, even with the knob flipped globally AND
    on the input SDFG: the parallelism lives in the maps, and a map inside a section re-enters
    OpenMP at nesting level 2, where the default OMP_MAX_ACTIVE_LEVELS=1 gives it a team of one."""
    with dace.config.set_temporary('compiler', 'cpu', 'openmp_sections', value=True):
        sdfg = elementwise_and_reduction.to_sdfg(simplify=True)
        sdfg.openmp_sections = True
        canonicalize(sdfg)

        assert not any(sd.openmp_sections for sd in sdfg.all_sdfgs_recursive())
        assert 'omp parallel sections' not in generated_code(sdfg)
        # Not vacuous: the SAME canonicalized SDFG does emit the construct if the knob is
        # forced back on, so the opt-out is what keeps it out -- not the shape of the output.
        assert '#pragma omp parallel sections' in generated_code(with_sections_requested(sdfg))

        rng = np.random.default_rng(1234)
        A, B = rng.random(64), rng.random(64)
        C, s = np.zeros(64), np.zeros(1)
        sdfg(A=A, B=B, C=C, s=s, N=64)
        assert np.array_equal(C, A + B)
        assert np.isclose(s[0], np.sum(A), rtol=0, atol=1e-12)


def test_vectorize_opts_out_of_omp_sections():
    """Same enforcement at the vectorizer's own entry, which may run without canonicalize."""
    with dace.config.set_temporary('compiler', 'cpu', 'openmp_sections', value=True):
        sdfg = two_concurrent_components.to_sdfg(simplify=True)
        sdfg.openmp_sections = True
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})

        assert not any(sd.openmp_sections for sd in sdfg.all_sdfgs_recursive())
        assert 'omp parallel sections' not in generated_code(sdfg)
        # Not vacuous: the vectorizer moves the concurrent components into the nested map-body
        # SDFGs (top level keeps one tile-main and one remainder scope), so the state with
        # components to wrap is found recursively -- forcing the knob back on over the whole
        # tree does emit the construct.
        assert [st for sd in sdfg.all_sdfgs_recursive() for st in sd.states() if len(concurrent_subgraphs(st)) > 1]
        assert '#pragma omp parallel sections' in generated_code(with_sections_requested(sdfg))

        rng = np.random.default_rng(1234)
        A, B = rng.random(64), rng.random(64)
        C, D = np.zeros(64), np.zeros(64)
        sdfg(A=A, B=B, C=C, D=D, N=64)
        assert np.array_equal(C, A + 1.0)
        assert np.array_equal(D, B * 2.0)


if __name__ == "__main__":
    test_duplicate_codegen()
    test_cpu_target_honours_openmp_sections()
    test_canonicalize_opts_out_of_omp_sections()
    test_vectorize_opts_out_of_omp_sections()
