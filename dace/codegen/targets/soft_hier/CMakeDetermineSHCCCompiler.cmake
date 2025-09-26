# Determine SHCC compiler executable
set(SHCC_COMPILER_NAMES riscv32-unknown-elf-gcc)

# Find the SHCC compiler
find_program(SHCC_COMPILER
    NAMES ${SHCC_COMPILER_NAMES}
    PATHS
        ${GVSOC_PATH}/third_party/toolchain/install/bin
        ${GVSOC_DIR}/third_party/toolchain/install/bin
        $ENV{PATH}
    NO_DEFAULT_PATH
    DOC "SHCC Compiler"
)

# Check if the compiler was found
if(SHCC_COMPILER)
    # Set CMake variables to use the SHCC compiler
    set(CMAKE_SHCC_COMPILER "${SHCC_COMPILER}" CACHE FILEPATH "SHCC compiler path")
    set(CMAKE_SHCC_COMPILER_ENV_VAR "SHCC" CACHE INTERNAL "Environment variable for SHCC compiler")
    set(CMAKE_SHCC_COMPILER_WORKS TRUE CACHE INTERNAL "Flag indicating that SHCC compiler works")
    set(SOFTHIER_SW_BUILD_PATH /tmp CACHE PATH "Path to SoftHier software build directory")
    # Log a message to indicate the SHCC compiler was found
    message(STATUS "SHCC compiler found at: ${SHCC_COMPILER}")

    # Set default compiler flags (adjust as necessary for your use case)
    #if(NOT CMAKE_SHCC_FLAGS_INIT)
    #    set(CMAKE_SHCC_FLAGS_INIT "-Wall" CACHE STRING "Initial SHCC compiler flags")
    #endif()

    # Add any specific additional settings for the SHCC compiler here
    #set(CMAKE_SHCC_FLAGS "${CMAKE_SHCC_FLAGS_INIT}" CACHE STRING "Compiler flags for SHCC")

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
            # Auto-generated CMake SHCC Compiler Config
            set(CMAKE_SHCC_COMPILER \"${SHCC_COMPILER}\")
            set(CMAKE_SHCC_COMPILER_WORKS TRUE)
            set(CMAKE_SHCC_FLAGS_INIT \"${CMAKE_SHCC_FLAGS_INIT}\")
            set(SOFTHIER_INSTALL_PATH \"${SOFTHIER_INSTALL_PATH}\")
            # Compilation rule for SHCC source files to object files
            set(CMAKE_SHCC_COMPILE_OBJECT \"<CMAKE_SHCC_COMPILER> <FLAGS> -c <SOURCE> -o <OBJECT>\")

            # Linking rule for SHCC object files to executable (using the provided linker script)
            set(CMAKE_SHCC_LINK_EXECUTABLE \"<CMAKE_SHCC_COMPILER> <FLAGS> -T ${SOFTHIER_INSTALL_PATH}/flex_memory.ld <OBJECTS> -o <TARGET>\")

            # Additional variables for SRC_DIR and linker script
            set(CMAKE_SHCC_SRC_DIR \"${SRC_DIR}\")
            set(CMAKE_SHCC_LINKER_SCRIPT \"${LINKER_SCRIPT}\")
        ")

        # Verify if the file was successfully generated
        if(EXISTS "${SHCC_COMPILER_FILE_PATH}")
            message(STATUS "Successfully generated CMakeSHCCCompiler.cmake")
        else()
            message(WARNING "Failed to generate CMakeSHCCCompiler.cmake")
        endif()
    endif()

    # # Define custom commands for compiling and linking SHCC files
    # add_custom_command(
    #     OUTPUT softhier.elf
    #     COMMAND ${SHCC_COMPILER} -I ${SOFTHIER_INSTALL_PATH}/include -I ${INCLUDE_DIRS} -T ${CMAKE_SHCC_LINKER_SCRIPT} -nostartfiles -mabi=ilp32d -mcmodel=medany -march=rv32imafd -g -O3 -ffast-math -fno-builtin-printf -fno-common -ffunction-sections -Wl,--gc-sections -o softhier.elf ${SOFTHIER_INSTALL_PATH}/flex_start.s ${SOURCES}
    #     COMMENT "Generating softhier.elf"
    # )

    # add_custom_target(build_output ALL DEPENDS softhier.elf)

    # add_custom_command(
    #     OUTPUT softhier.dump
    #     DEPENDS softhier.elf
    #     COMMAND ${OBJDUMPER} -D softhier.elf > softhier.dump
    #     COMMENT "Creating disassembly output.dump"
    # )

    # add_custom_target(dump ALL DEPENDS softhier.dump)

else()
    # Fatal error if the compiler is not found
    message(FATAL_ERROR "SHCC compiler not found. Please install riscv32-unknown-elf-gcc and retry, or specify the path using the SHCC environment variable or the CMAKE_SHCC_COMPILER CMake option.")
endif()
