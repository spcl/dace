import numpy
from dace.dtypes import _FFI_CTYPES, typeclass


class Float64sr(typeclass):
    """
    64-bit floating-point type with stochastic rounding.

    Stochastic rounding randomly rounds to the nearest representable value
    with probability proportional to the distance, reducing systematic bias
    in repeated computations.

    Limitations of current implementation: library functions like blas fallback
    to round-to-nearest float64 for compatibility reasons; targets CPU only.
    """

    def __init__(self):
        self.type = numpy.float64
        self.bytes = 8
        self.dtype = self
        self.typename = "double"
        self.stochastically_rounded = True

    def to_json(self):
        return 'float64sr'

    @staticmethod
    def from_json(json_obj, context=None):
        from dace.symbolic import pystr_to_symbolic  # must be included!
        return Float64sr()

    @property
    def ctype(self):
        return "dace::float64sr"

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