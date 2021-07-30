set(NCCL_LIB_NAME nccl)

# Find header path
find_path(NCCL_INCLUDE_DIR
NAMES nccl.h
PATHS $ENV{NCCL_ROOT}/include ${NCCL_ROOT}/include)

# Find library path
find_library(NCCL_LIBRARY
NAMES ${NCCL_LIB_NAME}
PATHS $ENV{NCCL_ROOT}/lib/ ${NCCL_ROOT}/lib)

message(STATUS "Using nccl library: ${NCCL_LIBRARY}")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nccl DEFAULT_MSG
                                NCCL_INCLUDE_DIR NCCL_LIBRARY)

# Append header and library to Environment variables
list(APPEND DACE_ENV_INCLUDES ${NCCL_INCLUDE_DIR})
list(APPEND DACE_ENV_LIBRARIES ${NCCL_LIBRARY})

mark_as_advanced(
NCCL_INCLUDE_DIR
NCCL_LIBRARY
)