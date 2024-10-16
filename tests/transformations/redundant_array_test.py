import os

from dace.transformation.dataflow import RedundantArray

from dace.cli import sdfg_diff

from dace import SDFG


def load_graph(relative_path: str):
    path = os.path.join(os.path.dirname(__file__), relative_path)
    print('Loading:', path)
    return SDFG.from_file(os.path.join(os.path.dirname(__file__), 'testdata/redundant-array-0.sdfg'))


def test_bug_1690():
    """Corresponds to https://github.com/spcl/dace/issues/1690"""

    # Make sure our input and golden data is good.
    g = load_graph('testdata/redundant-array-0.sdfg')
    g.validate()
    g.compile()
    g_corr = load_graph('testdata/redundant-array-0-correct.sdfg')
    g_corr.validate()
    g_corr.compile()

    # Apply and diff
    application_count = g.apply_transformations(RedundantArray)
    removed_keys, added_keys, changed_keys = sdfg_diff._sdfg_diff(g, g_corr)
    sdfg_diff._print_diff(g, g_corr, (removed_keys, added_keys, changed_keys))

    # Verify
    assert application_count == 1
    g.validate()
    g.compile()

    if __name__ == '__main__':
        test_bug_1690()
