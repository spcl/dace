# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MKL's FFTW3 interface: the FFTW3 C API, declared by MKL's ``fftw/fftw3.h`` and served by ``libmkl_rt``."""

import dace.library
from dace.libraries.fft.environments.plans import FFTPlans


@dace.library.environment
class MKLFFTW3:
    """Headers of MKL's FFTW3 interface; include and link paths come with :class:`IntelMKL`, listed beside it."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = ["fftw/fftw3.h", "omp.h"]
    state_fields = ["dace::fft::PlanCache<fftw_plan> fftw_plans;", "dace::fft::PlanCache<fftwf_plan> fftwf_plans;"]
    init_code = ""
    finalize_code = ""
    dependencies = [FFTPlans]
