# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k03 complex conjugate -- the Zip/Block layout witness (AoS vs SoA vs AoSoA).

Conjugate-and-scale a complex array of N numbers modeled as two real float64 fields (re, im):

    re[i] =  SCALE * re[i]
    im[i] = -SCALE * im[i]      (negate imag = conjugate; SCALE = uniform scale on both fields)

The two fields can be laid out three ways, and that layout is the decision the sweep explores:

  * SoA (unzipped): re and im are two separate [N] arrays -- each field its own contiguous stream.
  * AoS (zipped)   : one [N, 2] array (re, im interleaved) -- ``ZipArrays`` fuses the two fields.
  * AoSoA (V)      : one [N/V, 2, V] array -- ``aosoa_layout`` = Zip after Block(V) of the particle
                     axis; V-contiguous runs per field per tile.

Conjugation touches the imaginary field, so AoS pulls the untouched real value into every cache line
while SoA multiplies prefetcher streams -- the classic AoS/SoA/AoSoA layout tradeoff. Every candidate
is transparent (the fields carry the same values; only the addressing changes), so all reproduce the
oracle -- the sweep picks the layout, the Zip/Block algebra guarantees correctness.

Source: SC26 layout paper SS IV-B1 (conjugation microbenchmark); R. Strzodka, "Abstraction for AoS
and SoA Layout in C++," GPU Computing Gems Jade, 2012; Sung, Liu & Hwu, "DL: a data layout
transformation system for heterogeneous computing," InPar 2012 (ASTA = AoSoA). Primitives: Zip, Block.
"""
import numpy
import dace

from dace.transformation.layout.zip_arrays import ZipArrays, aosoa_layout

N = dace.symbol("N")
SCALE = 2.0


@dace.program
def conjugate(re: dace.float64[N], im: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        re[i] = SCALE * re[i]
        im[i] = -SCALE * im[i]


def oracle(re, im):
    return {"re": SCALE * re, "im": -SCALE * im}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"re": rng.random(n), "im": rng.random(n)}


def candidates():
    """AoS / SoA / AoSoA layouts of the (re, im) complex array: Zip, and Zip-after-Block (AoSoA)."""

    def zipped(sdfg):
        ZipArrays(zip_map={"cplx": ["re", "im"]}).apply_pass(sdfg, {})

    def aosoa(vector_width):

        def apply(sdfg, vector_width=vector_width):
            aosoa_layout(sdfg, "cplx", ["re", "im"], vector_width)

        return apply

    return {
        "unzipped_soa": (lambda sdfg: None),
        "zipped_aos": zipped,
        "aosoa_4": aosoa(4),
        "aosoa_8": aosoa(8),
    }


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep.

    The complex array is in-place, so the closure packs the logical (re, im) fields into whatever
    physical layout the candidate produced -- separate ``re``/``im`` (SoA), a fused ``cplx`` of shape
    [N, 2] (AoS), or [N/V, 2, V] (AoSoA) -- runs, then reads the fields back out for comparison.
    """

    def run(sdfg):
        arrays = sdfg.arrays
        if "re" in arrays:  # SoA: two separate fields, no packing
            re = inputs["re"].copy()
            im = inputs["im"].copy()
            sdfg(re=re, im=im, N=n)
            return {"re": re, "im": im}

        # Zip fused the two fields into one array ``cplx``; pack per its descriptor shape.
        shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in arrays["cplx"].shape)
        aos = numpy.stack([inputs["re"], inputs["im"]], axis=-1)  # logical (N, 2) field-minor AoS
        if len(shape) == 2:  # AoS [N, 2]
            cplx = aos.copy()
        else:  # AoSoA [N/V, 2, V] = (N/V, V, 2) with field before lane
            vector_width = shape[-1]
            cplx = aos.reshape(-1, vector_width, 2).transpose(0, 2, 1).copy()
        sdfg(cplx=cplx, N=n)

        if len(shape) == 2:
            re, im = cplx[:, 0].copy(), cplx[:, 1].copy()
        else:
            back = cplx.transpose(0, 2, 1).reshape(n, 2)
            re, im = back[:, 0].copy(), back[:, 1].copy()
        return {"re": re, "im": im}

    return run
