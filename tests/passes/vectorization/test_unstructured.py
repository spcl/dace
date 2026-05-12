# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    _get_unstructured_access_cloudsc_sdfg,
)


@pytest.mark.parametrize("layout", ["C", "Fortran"])
def test_unstructured_access_pattern(layout: str):
    klon_val = 32
    klev_val = 32
    """
        arrays = {("iorder", dace.int64, ("klon", 5), (1, "klon")),
                    ("zqx", dace.float64, ("klon", "klev", 5), (1, "klon", "klon * klev")),
                    ("zsinksum", dace.float64, ("klon", 5), (1, "klon")),
                    ("zratio", dace.float64, ("klon", 5), (1, "klon"))}
    """
    zqx = numpy.random.rand(5, klev_val, klon_val).astype(numpy.float64)
    iorder = numpy.random.randint(1, 6, size=(5, klon_val), dtype=numpy.int64)
    zratio = numpy.zeros((5, klon_val), dtype=numpy.float64)
    zsinksum = numpy.zeros((5, klon_val), dtype=numpy.float64)
    run_vectorization_test(
        dace_func=_get_unstructured_access_cloudsc_sdfg(layout=layout),
        arrays={
            "zqx": zqx,
            "zratio": zratio,
            "zsinksum": zsinksum,
            "iorder": iorder,
        },
        params={
            "klon": klon_val,
            "klev": klev_val,
        },
        vector_width=8,
        save_sdfgs=True,
        from_sdfg=True,
        sdfg_name=f"unstructured_access_pattern_layout_{layout.lower()}",
    )
