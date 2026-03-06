import numpy
from dace.dtypes import _FFI_CTYPES, typeclass


class Float32sr(typeclass):
    """
    32-bit floating-point type with stochastic rounding.

    Stochastic rounding randomly rounds to the nearest representable value
    with probability proportional to the distance, reducing systematic bias
    in repeated computations.

    Limitations of current implementation: library functions like blas fallback
    to round-to-nearest float32 for compatibility reasons; targets CPU only.
    """

    def __init__(self):
        self.type = numpy.float32
        self.bytes = 4
        self.dtype = self
        self.typename = "float"
        self.stochastically_rounded = True

    def to_json(self):
        return 'float32sr'

    @staticmethod
    def from_json(json_obj, context=None):
        from dace.symbolic import pystr_to_symbolic  # must be included!
        return Float32sr()

    @property
    def ctype(self):
        return "dace::float32sr"

    @property
    def ctype_unaligned(self):
        return self.ctype

    def as_ctypes(self):
        """ Returns the ctypes version of the typeclass. """
        return _FFI_CTYPES[self.type]

    def as_numpy_dtype(self):
        return numpy.dtype(self.type)

    @property
    def base_type(self):
        return self