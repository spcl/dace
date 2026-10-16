# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class Stockham:
    """
    The twiddle factor tables of native Stockham FFTs (see ``dace/libraries/fft/include/dace_stockham.h``), which a
    field of the program state computes on first use and frees when the program is finalized.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {"frame": ["../include/dace_stockham.h"], "cuda": ["../include/dace_stockham.h"]}
    state_fields = ["dace::fft::StockhamTwiddles stockham_twiddles;"]
    init_code = ""
    finalize_code = ""
    dependencies = []
