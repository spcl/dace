import dace
from dace.cli import dacelab
import os
import pytest


@pytest.mark.parametrize('filename', ['add', 'cholesky', 'forloop', 'matrix_scalar_add', 'mult', 'scalar_add'])
def test_dacelab(filename):
    path = os.path.dirname(os.path.abspath(__file__))
    dacelab.compile(os.path.join(path, filename + '.m'))
