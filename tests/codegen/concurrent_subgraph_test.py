# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace
import numpy as np
from dace import Memlet
from dace.sdfg.utils import concurrent_subgraphs

N = dace.symbol('N')


@dace.program
def two_concurrent_components(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    C[:] = A + 1.0
    D[:] = B * 2.0


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


def test_concurrent_components_are_not_wrapped_in_omp_sections():
    """A state's independent components must be emitted one after the other, never in
    ``#pragma omp parallel sections``: the wrapper pushes each map's own
    ``#pragma omp parallel for`` a nesting level down, where OMP_MAX_ACTIVE_LEVELS=1 gives
    it a team of one, and OpenMP does not even promise the sections run concurrently."""
    sdfg = two_concurrent_components.to_sdfg(simplify=True)

    multi = [st for st in sdfg.states() if len(concurrent_subgraphs(st)) > 1]
    assert multi, 'fixture is vacuous: no state has concurrent components to wrap'

    # Poke the retired knob on a throwaway copy: the codegen must ignore it, not honour it.
    probe = copy.deepcopy(sdfg)
    probe.openmp_sections = True
    code = '\n'.join(o.clean_code for o in probe.generate_code())

    assert 'omp parallel sections' not in code
    assert '#pragma omp section' not in code
    # The parallelism has to still be there, in the maps -- one per component.
    assert code.count('#pragma omp parallel for') == 2
    # The knob is gone from the SDFG API, so nothing can ask for the construct back.
    assert 'openmp_sections' not in dace.SDFG.__properties__

    rng = np.random.default_rng(1234)
    A, B = rng.random(64), rng.random(64)
    C, D = np.zeros(64), np.zeros(64)
    sdfg(A=A, B=B, C=C, D=D, N=64)
    # A single fp64 add / multiply-by-two is exact, so the oracle comparison needs no tolerance.
    assert np.array_equal(C, A + 1.0)
    assert np.array_equal(D, B * 2.0)


if __name__ == "__main__":
    test_duplicate_codegen()
    test_concurrent_components_are_not_wrapped_in_omp_sections()
