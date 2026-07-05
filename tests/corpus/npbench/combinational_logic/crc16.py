# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``crc16`` (combinational_logic) -- auto-ported from the npbench repo."""
import numpy as np
import dace
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'N': 1600}
INPUT_ARGS = ('N', )
ARRAY_ARGS = ('data', 'crc')
SCALARS = {}
OUTPUT_ARGS = ('crc', )

N = dace.symbol('N')

# CRC-16 polynomial constant (npbench port: this was a module constant upstream, not
# a kernel argument -- as a parameter named ``poly`` it both lacked a value source and
# collided with ``sympy.poly`` during canonicalization).
poly = 0x8408


def initialize(N, datatype=np.uint8):
    from numpy.random import default_rng
    rng = default_rng(42)
    data = rng.integers(0, 256, size=(N, ), dtype=np.uint8)
    crc = np.zeros(1, np.int64)
    return (data, crc)


def reference(data, crc):
    """
    CRC-16-CCITT Algorithm

    ``crc`` is a (1,) buffer; the 16-bit checksum is written in place.
    """
    c = 65535
    for b in data:
        cur_byte = 255 & b
        for _ in range(0, 8):
            if c & 1 ^ cur_byte & 1:
                c = c >> 1 ^ poly
            else:
                c >>= 1
            cur_byte >>= 1
    c = ~c & 65535
    c = c << 8 | c >> 8 & 255
    crc[0] = c & 65535


@dace.program
def kernel(data: dace.uint8[N], crc: dace.int64[1]):
    """
    CRC-16-CCITT Algorithm

    ``crc`` is a (1,) buffer; the 16-bit checksum is written in place.
    """
    c: dace.uint16 = 65535
    for i in range(N):
        b = data[i]
        cur_byte = 255 & b
        for _ in range(0, 8):
            if c & 1 ^ cur_byte & 1:
                c = c >> 1 ^ poly
            else:
                c >>= 1
            cur_byte >>= 1
    c = ~c & 65535
    c = c << 8 | c >> 8 & 255
    crc[0] = c & 65535


CORPUS = dict(name='crc16',
              dwarf='combinational_logic',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
