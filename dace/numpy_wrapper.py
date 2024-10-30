# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" NumPy type wrappers for interoperability in case NumPy is not installed """

try:
    import numpy
except (ImportError, ModuleNotFoundError):
    numpy = None

if numpy is not None:
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    intc = numpy.intc
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uintc = numpy.uintc
    uint64 = numpy.uint64
    float16 = numpy.float16
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    object_ = numpy.object_
else:
    # TODO
    pass