import pytest

# nbconvert (which brings nbformat) is an optional test extra in setup.py, so importing it at module
# scope makes COLLECTION fail wherever it is not installed -- one uninstalled extra then breaks a
# whole-tree `pytest tests/` run. Skip this module instead.
nbformat = pytest.importorskip('nbformat')
preprocessors = pytest.importorskip('nbconvert.preprocessors')
ExecutePreprocessor = preprocessors.ExecutePreprocessor
CellExecutionError = preprocessors.CellExecutionError

BASE_PATH = "tutorials/"
NOTEBOOK_PATHS = [
    f"{BASE_PATH}getting_started.ipynb",
    f"{BASE_PATH}codegen.ipynb",
]


@pytest.mark.parametrize("notebook", NOTEBOOK_PATHS)
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)
        try:
            out = ep.preprocess(nb)
        except CellExecutionError:
            out = None
            msg = 'Error executing the notebook "%s".\n\n' % notebook
            print(msg)
            raise


if __name__ == '__main__':
    pytest.main(["-v", __file__])
