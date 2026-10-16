# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Vectorization report for the PAPI provider, on when instrumentation.papi.vectorization_analysis is.
# Each compiler spells the flag differently and CMake picks the compiler, so the choice lives here,
# not in Python. Guarded on CXX: add_compile_options is directory-wide, and these are host-only flags
# nvcc would reject on a program's .cu files. Only GCC writes a report file; the rest use stderr.
add_compile_options(
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopt-info-vec-optimized-missed=${CMAKE_BINARY_DIR}/../perf/vecreport.txt>"
  "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize;-Rpass-analysis=loop-vectorize>"
  "$<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-qopt-report=2;-qopt-report-phase=vec>"
  "$<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-Minfo=vect>"
)
