# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Vectorization reporting for the PAPI instrumentation provider. Included by the PAPI environment
# only when ``instrumentation.papi.vectorization_analysis`` is on.
#
# The flag is spelled differently by every compiler, and the compiler is chosen by CMake -- so the
# choice is made here, from ``CMAKE_CXX_COMPILER_ID``, rather than by a second detector in Python
# that would have to predict what CMake is about to pick and could drift from it.
#
# ``add_compile_options`` places these after ``CMAKE_CXX_FLAGS_<CONFIG>`` on the command line, so an
# optimization level from the build type cannot override them.
#
# Only GCC writes a report file; Clang and NVHPC emit their remarks on stderr, so the report lands in
# the build log rather than in ``perf/vecreport.txt``. That difference is the compilers', not ours --
# the alternative is offering the analysis to GCC users alone, which is what the previous
# hardcoded GCC flag effectively did.
set(DACE_VECREPORT "${CMAKE_BINARY_DIR}/../perf/vecreport.txt")

add_compile_options(
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopt-info-vec-optimized-missed=${DACE_VECREPORT}>"
  "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize;-Rpass-analysis=loop-vectorize>"
  "$<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-qopt-report=2;-qopt-report-phase=vec>"
  "$<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-Minfo=vect>"
)
