import numpy as np

from utils.general import generate_arguments_fortran
from execute.parameters import ParametersProvider

generate_arguments_fortran('mwe_memlet_range', np.random.default_rng(42), ParametersProvider('cloudsc_vert_loop_10'))


