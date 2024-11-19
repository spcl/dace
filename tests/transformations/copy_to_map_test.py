# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import CopyToMap
import copy
import pytest
import numpy as np
import re
from typing import Tuple, Optional


def _copy_to_map(storage: dace.StorageType):
    @dace
    def somecopy(a, b):
        b[:] = a

    desc_a = dace.data.Array(dace.float64, [3, 2], storage=storage, strides=(1, 32), total_size=64)
    desc_b = dace.data.Array(dace.float64, [3, 2], storage=storage, strides=(2, 1), total_size=6)
    A = dace.data.make_array_from_descriptor(desc_a, np.random.rand(3, 2))
    B = dace.data.make_array_from_descriptor(desc_b, np.random.rand(3, 2))
    sdfg = somecopy.to_sdfg(A, B)
    assert sdfg.apply_transformations(CopyToMap) == 1
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(A, B)

    assert np.allclose(B, A)


def _flatten_to_map(storage: dace.StorageType):
    @dace
    def somecopy(a, b):
        b[:] = a.flatten()

    desc_a = dace.data.Array(dace.float64, [3, 2], storage=storage, strides=(1, 32), total_size=64)
    desc_b = dace.data.Array(dace.float64, [6], storage=storage, total_size=6)
    A = dace.data.make_array_from_descriptor(desc_a, np.random.rand(3, 2))
    B = dace.data.make_array_from_descriptor(desc_b, np.random.rand(6))
    sdfg = somecopy.to_sdfg(A, B)

    sdfg.apply_transformations(CopyToMap)
    if storage == dace.StorageType.GPU_Global:
        sdfg.apply_gpu_transformations()

    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(A, B)

    assert np.allclose(B, A.flatten())


def test_copy_to_map():
    _copy_to_map(dace.StorageType.CPU_Heap)


@pytest.mark.gpu
def test_copy_to_map_gpu():
    _copy_to_map(dace.StorageType.GPU_Global)


def test_flatten_to_map():
    _flatten_to_map(dace.StorageType.CPU_Heap)


@pytest.mark.gpu
def test_flatten_to_map_gpu():
    _flatten_to_map(dace.StorageType.GPU_Global)


@pytest.mark.gpu
def test_preprocess():
    """ Tests preprocessing in the GPU code generator, adding CopyToMap automatically. """
    sdfg = dace.SDFG('copytest')

    # Create two arrays with different allocation structure
    desc_inp = dace.data.Array(dace.float64, [20, 21, 22], strides=(1, 32, 32 * 21), total_size=14784)
    desc_out = dace.data.Array(dace.float64, [20, 21, 22], start_offset=5, total_size=20 * 21 * 22 + 5)

    # Construct graph
    sdfg.add_datadesc('inp', desc_inp)
    sdfg.add_datadesc('out', desc_out)
    gpudesc_inp = copy.deepcopy(desc_inp)
    gpudesc_inp.storage = dace.StorageType.GPU_Global
    gpudesc_inp.transient = True
    gpudesc_out = copy.deepcopy(desc_out)
    gpudesc_out.storage = dace.StorageType.GPU_Global
    gpudesc_out.transient = True
    sdfg.add_datadesc('gpu_inp', gpudesc_inp)
    sdfg.add_datadesc('gpu_out', gpudesc_out)

    state = sdfg.add_state()
    a = state.add_read('inp')
    b = state.add_read('gpu_inp')
    c = state.add_read('gpu_out')
    d = state.add_read('out')
    state.add_nedge(a, b, dace.Memlet('inp'))
    state.add_nedge(b, c, dace.Memlet('gpu_inp'))
    state.add_nedge(c, d, dace.Memlet('gpu_out'))

    # Create arrays with matching layout
    inp = dace.data.make_array_from_descriptor(desc_inp, np.random.rand(20, 21, 22))
    out = dace.data.make_array_from_descriptor(desc_out, np.random.rand(20, 21, 22))

    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(inp=inp, out=out)

    assert np.allclose(out, inp)


def _perform_non_lin_delin_test(
        sdfg: dace.SDFG,
) -> bool:
    """Performs test for the special case CopyToMap that bypasses linearizing and delinearaziong.
    """
    assert sdfg.number_of_nodes() == 1
    state: dace.SDFGState = sdfg.states()[0]
    assert state.number_of_nodes() == 2
    assert state.number_of_edges() == 1
    assert all(isinstance(node, dace.nodes.AccessNode) for node in state.nodes())
    sdfg.validate()

    a = np.random.rand(*sdfg.arrays["a"].shape)
    b_unopt = np.random.rand(*sdfg.arrays["b"].shape)
    b_opt = b_unopt.copy()
    sdfg(a=a, b=b_unopt)

    nb_runs = sdfg.apply_transformations_repeated(CopyToMap, validate=True, options={"ignore_strides": True})
    assert nb_runs == 1, f"Expected 1 application, but {nb_runs} were performed."

    # Now looking for the tasklet and checking if the memlets follows the expected
    #  simple pattern.
    tasklet: dace.nodes.Tasklet = next(iter([node for node in state.nodes() if isinstance(node, dace.nodes.Tasklet)]))
    pattern: re.Pattern = re.compile(r"(__j[0-9])|(__j[0-9]+\s*\+\s*[0-9]+)|([0-9]+)")

    assert state.in_degree(tasklet) == 1
    assert state.out_degree(tasklet) == 1
    in_edge = next(iter(state.in_edges(tasklet)))
    out_edge = next(iter(state.out_edges(tasklet)))

    assert all(pattern.fullmatch(str(idxs[0]).strip()) for idxs in in_edge.data.src_subset), f"IN: {in_edge.data.src_subset}"
    assert all(pattern.fullmatch(str(idxs[0]).strip()) for idxs in out_edge.data.dst_subset), f"OUT: {out_edge.data.dst_subset}"

    # Now call it again after the optimization.
    sdfg(a=a, b=b_opt)
    assert np.allclose(b_unopt, b_opt)

    return True

def _make_non_lin_delin_sdfg(
        shape_a: Tuple[int, ...],
        shape_b: Optional[Tuple[int, ...]] = None
) -> Tuple[dace.SDFG, dace.SDFGState, dace.nodes.AccessNode, dace.nodes.AccessNode]:

    if shape_b is None:
        shape_b = shape_a

    sdfg = dace.SDFG("bypass1")
    state = sdfg.add_state(is_start_block=True)

    ac = []
    for name, shape in [('a', shape_a), ('b', shape_b)]:
        sdfg.add_array(
                name=name,
                shape=shape,
                dtype=dace.float64,
                transient=False,
        )
        ac.append(state.add_access(name))

    return sdfg, state, ac[0], ac[1]


def test_non_lin_delin_1():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((10, 10))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[0:10, 0:10] -> [0:10, 0:10]"),
    )
    _perform_non_lin_delin_test(sdfg)

def test_non_lin_delin_2():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((10, 10), (100, 100))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[0:10, 0:10] -> [50:60, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg)


def test_non_lin_delin_3():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 100), (100, 100))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[1:11, 20:30] -> [50:60, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg)


def test_non_lin_delin_4():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 4, 100), (100, 100))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[1:11, 2, 20:30] -> [50:60, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg)


def test_non_lin_delin_5():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 4, 100), (100, 10, 100))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[1:11, 2, 20:30] -> [50:60, 4, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg)


def test_non_lin_delin_6():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((100, 100), (100, 10, 100))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[1:11, 20:30] -> [50:60, 4, 40:50]"),
    )
    _perform_non_lin_delin_test(sdfg)


def test_non_lin_delin_7():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((10, 10), (20, 20))
    state.add_nedge(
            a,
            b,
            dace.Memlet("b[5:15, 6:16]"),
    )
    _perform_non_lin_delin_test(sdfg)


def test_non_lin_delin_8():
    sdfg, state, a, b = _make_non_lin_delin_sdfg((20, 20), (10, 10))
    state.add_nedge(
            a,
            b,
            dace.Memlet("a[5:15, 6:16]"),
    )
    _perform_non_lin_delin_test(sdfg)


if __name__ == '__main__':
    test_non_lin_delin_1()
    test_non_lin_delin_2()
    test_non_lin_delin_3()
    test_non_lin_delin_4()
    test_non_lin_delin_5()
    test_non_lin_delin_6()
    test_non_lin_delin_7()
    test_non_lin_delin_8()

    test_copy_to_map()
    test_flatten_to_map()
    try:
        import cupy
        test_copy_to_map_gpu()
        test_flatten_to_map_gpu()
        test_preprocess()
    except ModuleNotFoundError as E:
        if "'cupy'" not in str(E):
            raise
