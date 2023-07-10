import numpy as np

from utils.general import generate_arguments_fortran
from execute.parameters import ParametersProvider

generate_arguments_fortran('mwe_map_similar_size', np.random.default_rng(42), ParametersProvider('cloudsc_vert_loop_10'))


