"""A size the numpy frontend materializes as a data descriptor (``np.empty(Nt+1)`` -> ``Nt_plus_1``) and
then uses as that array's SHAPE must be promoted to a symbol when the next descriptor is added, not collide
with ``add_symbol`` (which raised ``FileExistsError: ... used by a data descriptor``)."""
import numpy as np

import dace

N = dace.symbol("N")


@dace.program
def size_scalar_shape_prog(a: dace.float64[N], Nt: dace.int64):
    b = np.empty(Nt + 1, dace.float64)
    for i in range(N):
        b[i] = a[i] * 2.0
    return b


def test_size_scalar_used_as_shape_is_promoted_to_symbol():
    sdfg = size_scalar_shape_prog.to_sdfg(simplify=True)  # pre-fix: FileExistsError on Nt_plus_1
    assert "Nt_plus_1" in sdfg.symbols, "the size scalar must become a symbol"
    assert "Nt_plus_1" not in sdfg.arrays, "the colliding data descriptor must be gone after promotion"
    sdfg.validate()


if __name__ == "__main__":
    test_size_scalar_used_as_shape_is_promoted_to_symbol()
    print("OK")
