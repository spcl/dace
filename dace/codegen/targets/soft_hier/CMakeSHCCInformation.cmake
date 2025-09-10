# Run a command to detect the version of the SHCC compiler
execute_process(
    COMMAND ${CMAKE_SHCC_COMPILER} --version
    OUTPUT_VARIABLE SHCC_COMPILER_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract the major, minor, and patch version numbers from the SHCC compiler version
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" CMAKE_SHCC_COMPILER_VERSION_MATCH "${SHCC_COMPILER_VERSION}")

if(CMAKE_SHCC_COMPILER_VERSION_MATCH)
    message(STATUS "Detected SHCC Compiler version: ${CMAKE_SHCC_COMPILER_VERSION_MATCH}")
    set(CMAKE_SHCC_COMPILER_VERSION "${CMAKE_SHCC_COMPILER_VERSION_MATCH}" CACHE INTERNAL "SHCC Compiler version")
else()
    message(WARNING "Could not detect SHCC Compiler version")
endif()

# Set the compiler features
set(CMAKE_SHCC_HAS_FEATURE_THREAD_SUPPORT TRUE CACHE INTERNAL "Indicates whether SHCC compiler has thread support")
set(CMAKE_SHCC_HAS_FEATURE_OPTIMIZATION_FLAGS TRUE CACHE INTERNAL "Indicates whether SHCC compiler supports optimization flags")

# Define standard SHCC optimization and warning flags
set(CMAKE_SHCC_STANDARD_FLAGS "-O3 -Wall" CACHE STRING "Standard SHCC compiler flags")
