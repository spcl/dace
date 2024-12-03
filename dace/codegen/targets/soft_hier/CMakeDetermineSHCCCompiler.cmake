# Determine SHCC compiler executable
set(SHCC_COMPILER_NAMES riscv32-unknown-elf-gcc)

# Find the SHCC compiler
find_program(SHCC_COMPILER 
    NAMES ${SHCC_COMPILER_NAMES} 
    PATHS 
        /scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin
        /usr/local/bin
        /usr/bin
        $ENV{PATH}
    NO_DEFAULT_PATH
    DOC "SHCC Compiler"
)

# Check if the compiler was found
if(SHCC_COMPILER)
    # Set CMake variables to use the SHCC compiler
    set(CMAKE_SHCC_COMPILER "${SHCC_COMPILER}" CACHE FILEPATH "SHCC compiler path" FORCE)
    set(CMAKE_SHCC_COMPILER_ENV_VAR "SHCC" CACHE INTERNAL "Environment variable for SHCC compiler")
    set(CMAKE_SHCC_COMPILER_WORKS TRUE CACHE INTERNAL "Flag indicating that SHCC compiler works")

    # Log a message to indicate the SHCC compiler was found
    message(STATUS "SHCC compiler found at: ${SHCC_COMPILER}")

    # Set default compiler flags (adjust as necessary for your use case)
    if(NOT CMAKE_SHCC_FLAGS_INIT)
        set(CMAKE_SHCC_FLAGS_INIT "-Wall" CACHE STRING "Initial SHCC compiler flags" FORCE)
    endif()

    # Add any specific additional settings for the SHCC compiler here
    set(CMAKE_SHCC_FLAGS "${CMAKE_SHCC_FLAGS_INIT}" CACHE STRING "Compiler flags for SHCC" FORCE)

    # Get CMake version components to generate the correct path for CMakeSHCCCompiler.cmake
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.0")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_VERSION})
        list(GET VERSION_LIST 0 CMAKE_VERSION_MAJOR)
        list(GET VERSION_LIST 1 CMAKE_VERSION_MINOR)
        list(GET VERSION_LIST 2 CMAKE_VERSION_PATCH)
    endif()

    # Construct the path using the exact version directory, including the patch version
    set(SHCC_COMPILER_FILE_PATH "${CMAKE_BINARY_DIR}/CMakeFiles/${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}/CMakeSHCCCompiler.cmake")

    # Generate the missing `CMakeSHCCCompiler.cmake` if needed
    if(NOT EXISTS "${SHCC_COMPILER_FILE_PATH}")
        message(STATUS "Generating missing CMakeSHCCCompiler.cmake at: ${SHCC_COMPILER_FILE_PATH}")
        file(WRITE "${SHCC_COMPILER_FILE_PATH}" "
            set(CMAKE_SHCC_COMPILER \"${SHCC_COMPILER}\")
            set(CMAKE_SHCC_COMPILER_WORKS TRUE)
            set(CMAKE_SHCC_FLAGS_INIT \"${CMAKE_SHCC_FLAGS_INIT}\")
        ")

        # Verify if the file was successfully generated
        if(EXISTS "${SHCC_COMPILER_FILE_PATH}")
            message(STATUS "Successfully generated CMakeSHCCCompiler.cmake")
        else()
            message(WARNING "Failed to generate CMakeSHCCCompiler.cmake")
        endif()
    endif()

else()
    # Fatal error if the compiler is not found
    message(FATAL_ERROR "SHCC compiler not found. Please install riscv32-unknown-elf-gcc and retry, or specify the path using the SHCC environment variable or the CMAKE_SHCC_COMPILER CMake option.")
endif()
