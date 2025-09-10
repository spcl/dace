# Test SHCC compiler functionality
include(CheckCCompilerFlag)

check_c_compiler_flag("-Wall" SHCC_SUPPORTS_WALL)

if(SHCC_SUPPORTS_WALL)
    message(STATUS "SHCC supports -Wall flag")
else()
    message(FATAL_ERROR "SHCC compiler does not support -Wall flag")
endif()

# Add additional tests as needed to verify compiler compatibility
