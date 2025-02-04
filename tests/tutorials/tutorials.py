import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

BASE_PATH = "tutorials/"
NOTEBOOK_PATHS = [f"{BASE_PATH}getting_started.ipynb",
                  f"{BASE_PATH}codegen.ipynb", ]

@pytest.mark.parametrize("notebook", NOTEBOOK_PATHS)
def test_notebook_exec(notebook):
  with open(notebook) as f:
      nb = nbformat.read(f, as_version=4)
      ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
      try:
        assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
      except Exception:
          assert False, f"Failed executing {notebook}"
