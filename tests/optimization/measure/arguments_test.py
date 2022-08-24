import dace
import numpy as np

from dace.sdfg.analysis.cutout import cutout_state
from dace.optimization.measure import random_arguments, create_data_report, arguments_from_data_report


@dace.program
def two_maps(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[i, j]
            b >> B[i, j]

            b = a + 2


def test_random_arguments():
    sdfg = two_maps.to_sdfg()
    sdfg.simplify()

    arguments = random_arguments(sdfg)
    assert len(arguments) == 2

    assert "A" in arguments
    A = arguments["A"]
    assert A.shape == (10, 20)
    assert np.sum(A) != 0

    assert "B" in arguments
    B = arguments["B"]
    assert B.shape == (10, 20)
    assert np.all(B == 0.0)


def test_create_data_report():
    sdfg = two_maps.to_sdfg()
    sdfg.simplify()

    sdfg.clear_data_reports()

    arguments = random_arguments(sdfg)
    dreport = create_data_report(sdfg, arguments)

    assert len(dreport.keys()) == 2


def test_create_data_report_transients():
    sdfg = two_maps.to_sdfg()
    sdfg.simplify()

    sdfg.clear_data_reports()

    arguments = random_arguments(sdfg)
    dreport = create_data_report(sdfg, arguments, transients=True)

    assert len(dreport.keys()) == 3


def test_cutout_arguments_from_dreport():
    sdfg = two_maps.to_sdfg()
    sdfg.simplify()

    cutouts = []
    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue

            subgraph = state.scope_subgraph(node)
            cutout = cutout_state(state, *subgraph.nodes())
            cutouts.append(cutout)

    sdfg.clear_data_reports()

    arguments = random_arguments(sdfg)
    dreport = create_data_report(sdfg, arguments, transients=True)

    for cutout in cutouts:
        args = arguments_from_data_report(cutout, dreport)
        assert len(args) == 2

        first_map = False
        for node in cutout.start_state:
            if isinstance(node, dace.nodes.AccessNode) and node.data == "A":
                first_map = True
                break

        if first_map:
            assert "A" in args and "tmp" in args
        else:
            assert "tmp" in args and "B" in args


if __name__ == '__main__':
    test_random_arguments()
