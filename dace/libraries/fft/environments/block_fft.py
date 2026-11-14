# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The device-side FFT an FFT library node inside a GPU kernel runs on one thread block."""

import dace.library


@dace.library.environment
class BlockFFT:
    """``dace/cuda/block_fft.cuh``: CUDA unit only, it holds ``__device__`` code."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = {"frame": [], "cuda": ["dace/cuda/block_fft.cuh"]}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []
