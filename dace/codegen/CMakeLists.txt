# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
cmake_minimum_required(VERSION 3.15)
project(dace_program)

# General options
set(DACE_PROGRAM_NAME "dace_program" CACHE STRING "Name of DaCe program")
set(DACE_SRC_DIR "" CACHE STRING "Root directory of generated code files")
set(DACE_FILES "" CACHE STRING "List of host code files relative to the root of the source directory")
set(DACE_LIBS "" CACHE STRING "Extra libraries")
set(HLSLIB_PART_NAME "${DACE_XILINX_PART_NAME}")

# FPGA specific
set(DACE_FPGA_AUTOBUILD_BITSTREAM OFF CACHE STRING "Automatically build bitstreams if they are not present.")

# Allow passing flags to various stages of Xilinx compilation process
set(DACE_XILINX_MODE "simulation" CACHE STRING "Type of compilation/execution [simulation/software_emulation/hardware_emulation/hardware].")
set(DACE_XILINX_HOST_FLAGS "" CACHE STRING "Extra flags to host code")
set(DACE_XILINX_SYNTHESIS_FLAGS "" CACHE STRING "Extra flags for performing high-level synthesis")
set(DACE_XILINX_BUILD_FLAGS "" CACHE STRING "Extra flags to xocc build phase")
set(DACE_XILINX_TARGET_CLOCK "" CACHE STRING "Target clock frequency of FPGA kernel")
set(DACE_XILINX_PART_NAME "xcu280-fsvh2892-2L-e" CACHE STRING "Xilinx chip to target from HLS")
set(DACE_XILINX_TARGET_PLATFORM "xilinx_u280_xdma_201920_1" CACHE STRING "Vitis platform to target")
set(DACE_XILINX_ENABLE_DEBUGGING OFF CACHE STRING "Inject debugging cores to kernel build (always on for simulation/emulation)")

# Intel FPGA options
set(DACE_INTELFPGA_MODE "simulation" CACHE STRING "Type of compilation/execution [emulator/simulator/hardare].")
set(DACE_INTELFPGA_HOST_FLAGS "" CACHE STRING "Extra flags to host compiler.")
set(DACE_INTELFPGA_KERNEL_FLAGS "" CACHE STRING "Extra flags to kernel compiler.")
set(DACE_INTELFPGA_TARGET_BOARD "a10gx" CACHE STRING "Target FPGA board.")
set(DACE_INTELFPGA_ENABLE_DEBUGGING OFF CACHE STRING "Enable debugging.")

# Target detection
set(DACE_ENABLE_MPI OFF)
set(DACE_ENABLE_CUDA OFF)
set(DACE_ENABLE_HIP OFF)
set(DACE_ENABLE_XILINX OFF)
set(DACE_ENABLE_INTELFPGA OFF)
set(DACE_ENABLE_RTL OFF)

# Split list by target
foreach(DACE_FILE ${DACE_FILES})
  # Extract the target from the folder name
  get_filename_component(DACE_FILE_NAME "${DACE_FILE}" NAME_WE)
  get_filename_component(DACE_FILE_EXT "${DACE_FILE}" EXT)
  get_filename_component(DACE_FILE_SUBDIR "${DACE_FILE}" DIRECTORY)
  get_filename_component(DACE_FILE_DIR "${DACE_FILE_SUBDIR}" DIRECTORY)
  get_filename_component(DACE_FILE_TARGET "${DACE_FILE_DIR}" NAME)
  get_filename_component(DACE_FILE_TARGET_TYPE "${DACE_FILE_SUBDIR}" NAME)
  if(DACE_FILE_TARGET STREQUAL "")
    # If there is no subtype, the directory of the file is the target directly
    set(DACE_FILE_TARGET ${DACE_FILE_TARGET_TYPE})
    set(DACE_FILE_TARGET_TYPE "")
  endif()
  # Make the path absolute
  set(DACE_FILE ${DACE_SRC_DIR}/${DACE_FILE})
  # Now treat the file according to the deduced target
  if(${DACE_FILE_TARGET} STREQUAL "cuda")
    if(${DACE_FILE_TARGET_TYPE} MATCHES "hip")
      set(DACE_ENABLE_HIP ON)
      set(DACE_HIP_FILES ${DACE_HIP_FILES} ${DACE_FILE})
    else()
      set(DACE_ENABLE_CUDA ON)
      set(DACE_CUDA_FILES ${DACE_CUDA_FILES} ${DACE_FILE})
    endif()
  elseif(${DACE_FILE_TARGET} STREQUAL "xilinx")
    set(DACE_ENABLE_XILINX ON)
    if(DACE_FILE_TARGET_TYPE MATCHES "host")
      set(DACE_XILINX_HOST_FILES ${DACE_XILINX_HOST_FILES} ${DACE_FILE})
    elseif (DACE_FILE_TARGET_TYPE MATCHES "ip")
      set(DACE_ENABLE_RTL ON)
      if (DACE_FILE_EXT MATCHES ".cpp")
        set(DACE_XILINX_IP_FILES ${DACE_XILINX_IP_FILES} ${DACE_FILE})
      endif()
    elseif(DACE_FILE_EXT MATCHES ".cpp")
      set(DACE_XILINX_KERNEL_FILES ${DACE_XILINX_KERNEL_FILES} ${DACE_FILE})
    elseif(DACE_FILE_EXT MATCHES ".cfg")
      set(DACE_XILINX_CONFIG_FILE ${DACE_FILE})
    endif()
  elseif(${DACE_FILE_TARGET} STREQUAL "intel_fpga")
    set(DACE_ENABLE_INTELFPGA ON)
    if(DACE_FILE_TARGET_TYPE MATCHES "host")
      set(DACE_INTELFPGA_HOST_FILES ${DACE_INTELFPGA_HOST_FILES} ${DACE_FILE})
    else()
      set(DACE_INTELFPGA_KERNEL_FILES ${DACE_INTELFPGA_KERNEL_FILES} ${DACE_FILE})
    endif()
  elseif(${DACE_FILE_TARGET} STREQUAL "mpi")
    set(DACE_ENABLE_MPI ON)
    set(DACE_CPP_FILES ${DACE_CPP_FILES} ${DACE_FILE})
  elseif(${DACE_FILE_TARGET} STREQUAL "rtl")
    set(DACE_ENABLE_RTL ON)
    if(DACE_FILE_EXT MATCHES ".v" OR DACE_FILE_EXT MATCHES ".sv")
      set(DACE_RTL_FILES ${DACE_RTL_FILES} ${DACE_FILE})
    elseif(DACE_FILE_EXT MATCHES ".cpp")
      set(DACE_HOST_FILES ${DACE_HOST_FILES} ${DACE_FILE})
    endif()
  else()
    set(DACE_CPP_FILES ${DACE_CPP_FILES} ${DACE_FILE})
  endif()
endforeach()

# Internal dependencies
set(DACE_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../runtime)
include_directories(${DACE_RUNTIME_DIR}/include)

# Global DaCe external dependencies
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
add_definitions(-DDACE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\")
set(DACE_LIBS ${DACE_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${OpenMP_CXX_LIBRARIES})
if(DACE_ENABLE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(DACE_LIBS ${DACE_LIBS} ${MPI_CXX_LIBRARIES})
endif()
if(DACE_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  include_directories(${CUDA_INCLUDE_DIRS})
  if (MSVC_IDE)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
  else()
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  endif()
  set(DACE_LIBS ${DACE_LIBS} ${CUDA_LIBRARIES})
  add_definitions(-DWITH_CUDA)

  if (MSVC_IDE)
    if (${CMAKE_VERSION} VERSION_LESS "3.15.0")
      message("WARNING: CMake versions older than 3.15 are known to cause issues with CUDA builds on Windows.")
    endif()
    cmake_policy(SET CMP0091 NEW)
  endif()
endif()

if(DACE_ENABLE_HIP)
  add_definitions(-DWITH_CUDA)
  add_definitions(-DWITH_HIP)

  # Load once to find HIP path... (due to some issue in FindHIP.cmake)
  find_package(HIP REQUIRED)
  get_filename_component(HIP_PATH "${HIP_HIPCC_EXECUTABLE}" DIRECTORY)
  set(HIP_PATH "${HIP_PATH}/..")
  get_filename_component(HIP_PATH "${HIP_PATH}" ABSOLUTE)
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake;${HIP_PATH}/hip/cmake" ${CMAKE_MODULE_PATH})
  # ...Then load again to get the macros
  find_package(HIP REQUIRED)

endif()

# Function for performing deferred variable expansion
function(expand_deferred_variables VAR_NAME)
  string(REGEX MATCHALL "_DACE_CMAKE_EXPAND{([^}]+)}" EXPAND_VARS
         "${${VAR_NAME}}")
  string(REPLACE " " ";" EXPAND_VARS "${EXPAND_VARS}")
  foreach(EXPAND_VAR ${EXPAND_VARS})
      # Extract only the variable name
      string(REGEX REPLACE "_DACE_CMAKE_EXPAND{(.+)}" "\\1"
             EXPAND_VAR ${EXPAND_VAR})
      # Now expand the variable and substitute it back into the original
      # variable
      string(REGEX REPLACE "_DACE_CMAKE_EXPAND{${EXPAND_VAR}}"
             "${${EXPAND_VAR}}" ${VAR_NAME} ${${VAR_NAME}})
  endforeach()
  # Have to explicitly set parent scope, otherwise this will have no effect
  set(${VAR_NAME} ${${VAR_NAME}} PARENT_SCOPE)
endfunction()

# Environment-specified external dependencies
if (${CMAKE_VERSION} VERSION_LESS ${DACE_ENV_MINIMUM_VERSION})
  message(FATAL_ERROR "Packages require CMake version >= ${DACE_ENV_MINIMUM_VERSION}.")
endif()
# Include any CMake files specified
foreach(CMAKE_FILE ${DACE_ENV_CMAKE_FILES})
  include(${CMAKE_FILE})
endforeach()
# Hideous way of "zipping" keys and values passed separately. Couldn't find a
# better way of doing this in CMake.
list(LENGTH DACE_ENV_VAR_KEYS NUM_ENV_VARS)
math(EXPR VARS_END "${NUM_ENV_VARS}-1")
if(${NUM_ENV_VARS} GREATER 0)
  foreach(i RANGE ${VARS_END})
    list(GET DACE_ENV_VAR_KEYS ${i} KEY)
    list(GET DACE_ENV_VAR_VALUES ${i} VAL)
    expand_deferred_variables(VAL)
    set(${KEY} ${VAL})
  endforeach()
endif()
string(REPLACE " " ";" DACE_ENV_PACKAGES "${DACE_ENV_PACKAGES}")
foreach(PACKAGE_NAME ${DACE_ENV_PACKAGES})
  find_package(${PACKAGE_NAME} REQUIRED)
endforeach()
# Un-escape and expand environment arguments, now that packages have been found
foreach(VAR_NAME DACE_ENV_INCLUDES DACE_ENV_LIBRARIES DACE_ENV_COMPILE_FLAGS
                 DACE_ENV_LINK_FLAGS DACE_ENV_VAR_VALUES)
    expand_deferred_variables(${VAR_NAME})
endforeach()
# Now evaluate variables again, in case some of them contained unexpanded
# values depending on packages
if(${NUM_ENV_VARS} GREATER 0)
  foreach(i RANGE ${VARS_END})
    list(GET DACE_ENV_VAR_KEYS ${i} KEY)
    list(GET DACE_ENV_VAR_VALUES ${i} VAL)
    set(${KEY} ${VAL})
  endforeach()
endif()
# Configure specified include directories, libraries, and flags
string(REPLACE " " ";" DACE_ENV_INCLUDES "${DACE_ENV_INCLUDES}")
string(REPLACE " " ";" DACE_ENV_LIBRARIES "${DACE_ENV_LIBRARIES}")
include_directories(${DACE_ENV_INCLUDES})
set(DACE_LIBS ${DACE_LIBS} ${DACE_ENV_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DACE_ENV_COMPILE_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")
set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")

if(DACE_ENABLE_XILINX OR DACE_ENABLE_INTELFPGA)
  set(DACE_HLSLIB_DIR ${CMAKE_SOURCE_DIR}/../external/hlslib)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DACE_HLSLIB_DIR}/cmake)
  include_directories(SYSTEM ${DACE_HLSLIB_DIR}/include)
endif()
if(DACE_ENABLE_XILINX)
  find_package(Vitis REQUIRED)
  include_directories(SYSTEM ${Vitis_INCLUDE_DIRS})
  add_definitions(-DDACE_XILINX -DDACE_VITIS_DIR=\"${VITIS_ROOT_DIR}\")
  set(DACE_LIBS ${DACE_LIBS} ${Vitis_LIBRARIES})
endif()
if(DACE_ENABLE_INTELFPGA)
  find_package(IntelFPGAOpenCL REQUIRED)
  include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})
  add_definitions(-DDACE_INTELFPGA)
  set(DACE_LIBS ${DACE_LIBS} ${IntelFPGAOpenCL_LIBRARIES})
endif()
if (DACE_ENABLE_RTL AND DACE_ENABLE_XILINX)
  set(DACE_RTLLIB_DIR ${CMAKE_SOURCE_DIR}/../external/rtllib)
  include ("${DACE_RTLLIB_DIR}/cmake/rtl_target.cmake")
endif()

# Create CUDA object files
if(DACE_ENABLE_CUDA)
  # Get local CUDA architectures
  if (NOT DEFINED LOCAL_CUDA_ARCHITECTURES)
      execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-ccbin" "${CMAKE_CXX_COMPILER}" "--run"
                      "${CMAKE_SOURCE_DIR}/tools/get_cuda_arch.cpp"
                      OUTPUT_VARIABLE _arch_out RESULT_VARIABLE _arch_res
                      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(_arch_res EQUAL 0)
        string(REGEX REPLACE "\n" ";" _arch_out "${_arch_out}")
        list(GET _arch_out -1 _local_arch)
        string(REGEX REPLACE " " ";" _local_arch "${_local_arch}")
        set(LOCAL_CUDA_ARCHITECTURES "${_local_arch}" CACHE STRING "Detected local GPUs for compilation")
        message(STATUS "Local CUDA architectures detected: ${LOCAL_CUDA_ARCHITECTURES}")
      else()
        set(LOCAL_CUDA_ARCHITECTURES "" CACHE STRING "Detected local GPUs for compilation")
        message(STATUS "No local CUDA-capable GPUs found")
      endif()
  endif()

  # Add flags to compile for local CUDA architectures
  foreach(var ${LOCAL_CUDA_ARCHITECTURES})
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${var},code=sm_${var})
  endforeach()

  cuda_include_directories(${DACE_RUNTIME_DIR}/include)
  cuda_compile(DACE_CUDA_OBJECTS ${DACE_CUDA_FILES})
  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_CUDA_OBJECTS})
endif() # DACE_ENABLE_CUDA


# Create HIP object files
if(DACE_ENABLE_HIP)
  # Get local AMD architectures
  if (NOT DEFINED LOCAL_HIP_ARCHITECTURES)
    # Compile and run a test program
    execute_process(COMMAND ${HIP_HIPCC_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/get_hip_arch.cpp" -o
      "${CMAKE_CURRENT_BINARY_DIR}/hiparch"
      OUTPUT_VARIABLE _arch_compout
      RESULT_VARIABLE _arch_res)
    if(_arch_res EQUAL 0)
      execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hiparch"
      OUTPUT_VARIABLE _arch_out
      RESULT_VARIABLE _arch_runres)
    endif()

    if((_arch_res EQUAL 0) AND (_arch_runres EQUAL 0))
      string(REGEX REPLACE "\n" ";" _arch_out "${_arch_out}")
      list(GET _arch_out -1 _local_arch)
      string(REGEX REPLACE " " ";" _local_arch "${_local_arch}")
      set(LOCAL_HIP_ARCHITECTURES "${_local_arch}" CACHE STRING "Detected local AMD GPUs for compilation")
      message(STATUS "Local AMD HIP architectures detected: ${LOCAL_HIP_ARCHITECTURES}")
    else()
      if(_arch_res EQUAL 0)
        set(LOCAL_HIP_ARCHITECTURES "" CACHE STRING "Detected local AMD GPUs for compilation")
      endif()
      message(STATUS "No local HIP-capable GPUs found")
    endif()
  endif()

  # Add flags to compile for local AMD architectures
  foreach(var ${LOCAL_HIP_ARCHITECTURES})
    list(APPEND HIP_HIPCC_FLAGS --offload-arch=gfx${var})
  endforeach()

  # Add flags from dace config
  list(APPEND HIP_HIPCC_FLAGS ${EXTRA_HIP_FLAGS})

  # Add include directories for other files
  set(DACE_LIBS ${DACE_LIBS} hip::host)

  set_source_files_properties(${DACE_HIP_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_prepare_target_commands(${DACE_PROGRAM_NAME} OBJ DACE_HIP_OBJECTS DACE_HIP_SOURCES ${DACE_HIP_FILES})
  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_HIP_OBJECTS})
endif() # DACE_ENABLE_HIP

# create verilator RTL simulation objects
if(DACE_ENABLE_RTL)
  if (DACE_ENABLE_XILINX AND (NOT (DACE_XILINX_MODE STREQUAL "simulation")))
    # Get all of the kernel names
    list(APPEND RTL_KERNELS "")
    foreach(RTL_FILE ${DACE_RTL_FILES})
        get_filename_component(RTL_KERNEL ${RTL_FILE} DIRECTORY)
        list(APPEND RTL_KERNELS ${RTL_KERNEL})
    endforeach()
    list(REMOVE_DUPLICATES RTL_KERNELS)

    # Prepare build folders
    set (RTL_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/rtl/generated")
    set (RTL_LOG_DIR       "${CMAKE_CURRENT_BINARY_DIR}/rtl/log")
    set (RTL_TEMP_DIR      "${CMAKE_CURRENT_BINARY_DIR}/rtl/tmp")
    file (MAKE_DIRECTORY
        ${RTL_GENERATED_DIR}
        ${RTL_LOG_DIR}
        ${RTL_TEMP_DIR})
    execute_process(COMMAND ${Vitis_PLATFORMINFO} -p ${DACE_XILINX_TARGET_PLATFORM} -jhardwarePlatform.board.part
      OUTPUT_VARIABLE RTL_PART
      RESULT_VARIABLE _platforminfo_res)

    if (NOT ${_platforminfo_res} EQUAL 0)
      message(FATAL_ERROR "No part was found for platform ${DACE_XILINX_TARGET_PLATFORM} after querying 'platforminfo -p ${DACE_XILINX_TARGET_PLATFORM} -j\"hardwarePlatform.board.part\"'")
    endif()

    # Generate all of the .xo targets
    foreach(RTL_SRC_DIR ${RTL_KERNELS})
        get_filename_component(RTL_KERNEL ${RTL_SRC_DIR} NAME)
        get_filename_component(RTL_SCRIPTS "${RTL_SRC_DIR}/../scripts" ABSOLUTE)
        set (RTL_XO "${RTL_KERNEL}.xo")
        rtllib_rtl_target(${RTL_KERNEL} ${RTL_SRC_DIR} ${RTL_SCRIPTS} ${RTL_GENERATED_DIR} ${RTL_LOG_DIR} ${RTL_TEMP_DIR} "${RTLLIB_DIR}/rtl" ${RTL_XO} ${RTL_PART} "" "")
        set (DACE_VITIS_KERNELS_HW_EMU ${DACE_VITIS_KERNELS_HW_EMU} ${RTL_XO})
        set (DACE_VITIS_KERNELS_HW ${DACE_VITIS_KERNELS_HW} ${RTL_XO})
    endforeach()
  else()
    # find verilator installation
    find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
    if (NOT verilator_FOUND)
      message(FATAL_ERROR "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable")
    endif()

    # check minimal version requirements
    set(VERILATOR_MIN_VERSION "4.028")
    if("${verilator_VERSION}" VERSION_LESS VERILATOR_MIN_VERSION)
      message(ERROR "Please upgrade verilator to version >=${VERILATOR_MIN_VERSION}")
    endif()

    # get verilator flags from dace.conf
    set(VERILATOR_FLAGS "${DACE_RTL_VERILATOR_FLAGS}")

    # add lint verilator flags
    if("${DACE_RTL_VERILATOR_LINT_WARNINGS}")
      # -Wall: Enable all style warnings
      # -Wno-fatal: Disable fatal exit on warnings
      set(VERILATOR_FLAGS "${VERILATOR_FLAGS}" "-Wall" "-Wno-fatal")
    endif()

    # add verilated.cpp source
    set(DACE_CPP_FILES "${DACE_CPP_FILES}" "${VERILATOR_ROOT}/include/verilated.cpp")

    foreach(RTL_FILE ${DACE_RTL_FILES})

      # extract design name
      get_filename_component(RTL_FILE_NAME "${RTL_FILE}" NAME_WE)

      # add verilated .cpp files to the dace cpp source file var
      set(VERILATOR_SRC "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.cpp")
      set(VERILATOR_SRC "${VERILATOR_SRC}" "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}__Syms.cpp")
      # add file generated by verilator >=v4.036
      if("${verilator_VERSION}" VERSION_GREATER_EQUAL "4.036")
          set(VERILATOR_SRC "${VERILATOR_SRC}" "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}__Slow.cpp")
      endif()
      set(DACE_CPP_FILES "${DACE_CPP_FILES}" "${VERILATOR_SRC}")

      # add verilated design
      add_library("${RTL_FILE_NAME}" OBJECT)

      # include verilator
      set(VERILATOR_INCLUDE "${VERILATOR_ROOT}/include" "${dace_program_BINARY_DIR}/CMakeFiles/${RTL_FILE_NAME}.dir/V${RTL_FILE_NAME}.dir")
      include_directories(${VERILATOR_INCLUDE})

      # verilate design
      verilate("${RTL_FILE_NAME}" SOURCES ${RTL_FILE} VERILATOR_ARGS "${VERILATOR_FLAGS}")

      # add object library for linking
      set(DACE_LIBS ${DACE_LIBS} ${${RTL_FILE_NAME}})

    endforeach()
  endif()
endif() # DACE_ENABLE_RTL


# Create Xilinx object files
if(DACE_ENABLE_XILINX)

  set(VITIS_BUILD_FLAGS)
  if((NOT (DACE_XILINX_MODE STREQUAL "hardware")) OR DACE_XILINX_ENABLE_DEBUGGING)
    set(VITIS_BUILD_FLAGS "-g")
    set(DACE_XILINX_HOST_FLAGS "${DACE_XILINX_HOST_FLAGS} -g")
    set(DACE_XILINX_SYNTHESIS_FLAGS "${DACE_XILINX_SYNTHESIS_FLAGS} -g")
  endif()

  set_source_files_properties(${DACE_XILINX_KERNEL_FILES} ${DACE_XILINX_HOST_FILES} PROPERTIES COMPILE_FLAGS "${DACE_XILINX_HOST_FLAGS}")
  set_source_files_properties(${DACE_XILINX_KERNEL_FILES} PROPERTIES COMPILE_FLAGS "-DDACE_XILINX_DEVICE_CODE ${DACE_XILINX_HOST_FLAGS}")
  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_XILINX_KERNEL_FILES} ${DACE_XILINX_HOST_FILES})

  if(((${Vitis_MAJOR_VERSION} LESS 2018) AND
      (${Vitis_MINOR_VERSION} LESS 3)) OR ${Vitis_MAJOR_VERSION} LESS 2017)
    add_definitions(-DHLSLIB_LEGACY_SDX=1)
  else()
    add_definitions(-DHLSLIB_LEGACY_SDX=0)
  endif()

  if(DACE_XILINX_MODE STREQUAL "simulation")
    # This will cause the OpenCL calls to instead call a simulation code
    # running on the host
    add_definitions(-DHLSLIB_SIMULATE_OPENCL)
  endif()

  set(DACE_XILINX_SYNTHESIS_FLAGS "${DACE_XILINX_SYNTHESIS_FLAGS} -DDACE_SYNTHESIS -DDACE_XILINX -DDACE_XILINX_DEVICE_CODE -DHLSLIB_SYNTHESIS -std=c++11")
  if(NOT Vitis_USE_VITIS_HLS)
    set(DACE_XILINX_SYNTHESIS_FLAGS "${DACE_XILINX_SYNTHESIS_FLAGS} -D__VIVADO_HLS__")
  else()
    set(DACE_XILINX_SYNTHESIS_FLAGS "${DACE_XILINX_SYNTHESIS_FLAGS} -D__VITIS_HLS__")
  endif()

  string(REPLACE " " ";" DACE_XILINX_BUILD_FLAGS_INTERNAL
         "${DACE_XILINX_BUILD_FLAGS}")

  set(VITIS_BUILD_FLAGS
    ${VITIS_BUILD_FLAGS}
    -s
    -O3
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_SOURCE_DIR}/../external/hlslib/include
    -I${CMAKE_SOURCE_DIR}/../runtime/include
    -I${CMAKE_BINARY_DIR}
    --platform ${DACE_XILINX_TARGET_PLATFORM}
    ${DACE_XILINX_BUILD_FLAGS_INTERNAL})

  # Optionally specify a target clock (otherwise default is used)
  if(NOT DACE_XILINX_TARGET_CLOCK STREQUAL "")
    set(VITIS_BUILD_FLAGS ${VITIS_BUILD_FLAGS} --kernel_frequency ${DACE_XILINX_TARGET_CLOCK})
  endif()

  # If the project uses generated IP cores (E.g. through double pumping)
  message("wat" ${DACE_XILINX_IP_FILES})
  if(DACE_XILINX_IP_FILES)
    set(VITIS_BUILD_FLAGS ${VITIS_BUILD_FLAGS} --user_ip_repo_paths ip_cores)
  endif()

  # Add synthesis and build commands
  set(DACE_SYNTHESIS_TARGETS)

  foreach(DACE_KERNEL_FILE ${DACE_XILINX_KERNEL_FILES})
    get_filename_component(DACE_KERNEL_NAME ${DACE_KERNEL_FILE} NAME)
    get_filename_component(DACE_KERNEL_DIRECTORY ${DACE_KERNEL_FILE} DIRECTORY)
    string(REGEX REPLACE "(.+).cpp" "\\1" DACE_KERNEL_NAME "${DACE_KERNEL_NAME}")
    string(REPLACE " " ";" DACE_XILINX_SYNTHESIS_FLAGS_INTERNAL ${DACE_XILINX_SYNTHESIS_FLAGS})
    set(DACE_VITIS_KERNEL_FILES ${DACE_VITIS_KERNEL_FILES} ${DACE_KERNEL_FILE})
    configure_file(${CMAKE_SOURCE_DIR}/Xilinx_HLS.tcl.in Synthesize_${DACE_KERNEL_NAME}.tcl)
    add_custom_target(xilinx_synthesis_${DACE_KERNEL_NAME} COMMAND ${Vitis_HLS} -f Synthesize_${DACE_KERNEL_NAME}.tcl)
    set_property(TARGET xilinx_synthesis_${DACE_KERNEL_NAME} APPEND PROPERTY
                 ADDITIONAL_CLEAN_FILES ${DACE_KERNEL_NAME} vitis_hls.log)
    set(DACE_SYNTHESIS_TARGETS ${DACE_SYNTHESIS_TARGETS} xilinx_synthesis_${DACE_KERNEL_NAME})

    if(Vitis_IS_LEGACY)
      set(VITIS_BUILD_FLAGS ${VITIS_BUILD_FLAGS}
      --xp prop:kernel.${DACE_KERNEL_NAME}.kernel_flags="${DACE_XILINX_SYNTHESIS_FLAGS}")
    else()
      set(VITIS_BUILD_FLAGS ${VITIS_BUILD_FLAGS}
          --advanced.prop kernel.${DACE_KERNEL_NAME}.kernel_flags="${DACE_XILINX_SYNTHESIS_FLAGS}")
    endif()

  endforeach()

  add_custom_target(xilinx_synthesis DEPENDS ${DACE_SYNTHESIS_TARGETS})

  if(Vitis_IS_LEGACY)
    set(VITIS_BUILD_FLAGS ${VITIS_BUILD_FLAGS} --max_memory_ports all)
  endif()

  if((NOT (DACE_XILINX_MODE STREQUAL "hardware")) OR DACE_XILINX_ENABLE_DEBUGGING)
    # TODO: add Chipscope debugging on memory interfaces. Need to pass
    # interfaces from codegen to CMake in order to do this.
    message(STATUS "Enabled debugging/profiling for Xilinx targets.")
    set(VITIS_BUILD_FLAGS ${VITIS_BUILD_FLAGS}
        --profile_kernel "data:all:all:all"
        --profile_kernel "stall:all:all"
        --profile_kernel "exec:all:all")
  endif()

  if(Vitis_MAJOR_VERSION LESS 2018 AND Vitis_MINOR_VERSION LESS 3)

    add_custom_command(
      OUTPUT ${DACE_PROGRAM_NAME}_sw_emu.xclbin
      COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
      ${VITIS_BUILD_FLAGS}
      -t sw_emu
      ${DACE_VITIS_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_sw_emu.xclbin
      DEPENDS ${DACE_VITIS_KERNEL_FILES})

    add_custom_target(xilinx_build_${DACE_PROGRAM_NAME}_software_emulation
                      DEPENDS ${DACE_PROGRAM_NAME}_sw_emu.xclbin)

    add_custom_command(
      OUTPUT ${DACE_PROGRAM_NAME}_hw_emu.xclbin
      COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
      ${VITIS_BUILD_FLAGS}
      -t hw_emu
      ${DACE_VITIS_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_hw_emu.xclbin
      DEPENDS ${DACE_VITIS_KERNEL_FILES})

    add_custom_target(xilinx_build_${DACE_PROGRAM_NAME}_hardware_emulation
                      DEPENDS ${DACE_PROGRAM_NAME}_hw_emu.xclbin)

    add_custom_command(
      OUTPUT ${DACE_PROGRAM_NAME}_hw.xclbin
      COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
      ${VITIS_BUILD_FLAGS}
      -t hw
      ${DACE_VITIS_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_hw.xclbin
      DEPENDS ${DACE_VITIS_KERNEL_FILES})

    add_custom_target(xilinx_build_${DACE_PROGRAM_NAME}_hardware
                      DEPENDS ${DACE_PROGRAM_NAME}_hw.xclbin)

  else()

    # Since with Vitis only one kernel can be compiled at a time,
    # for the general case we have to first compile each kernel
    # and then link together the generated .xo files

    foreach(DACE_IP ${DACE_XILINX_IP_FILES})
      get_filename_component(DACE_KERNEL_NAME ${DACE_IP} NAME_WE)
      get_filename_component(DACE_KERNEL_SRC ${DACE_IP} DIRECTORY)

      configure_file(${CMAKE_SOURCE_DIR}/Xilinx_IP.tcl.in Package_${DACE_KERNEL_NAME}.tcl)
      add_custom_command(
        OUTPUT ip_cores/${DACE_KERNEL_NAME}/impl/export.zip
        COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_HLS}
        -f Package_${DACE_KERNEL_NAME}.tcl
        DEPENDS ${DACE_IP}
      )

      execute_process(COMMAND ${Vitis_PLATFORMINFO} -p ${DACE_XILINX_TARGET_PLATFORM} -jhardwarePlatform.board.part
      OUTPUT_VARIABLE RTL_PART
      RESULT_VARIABLE _platforminfo_res
      OUTPUT_STRIP_TRAILING_WHITESPACE)

      set (RTL_XO "${DACE_KERNEL_NAME}.xo")
      rtllib_rtl_target(${DACE_KERNEL_NAME} ${DACE_KERNEL_SRC} ${DACE_KERNEL_SRC} ${DACE_KERNEL_SRC} log tmp ${DACE_KERNEL_SRC} ${RTL_XO} ${RTL_PART} ip_cores/${DACE_KERNEL_NAME}/impl/export.zip ip_cores)
      set (DACE_VITIS_KERNELS_HW_EMU ${DACE_VITIS_KERNELS_HW_EMU} ${RTL_XO})
      set (DACE_VITIS_KERNELS_HW ${DACE_VITIS_KERNELS_HW} ${RTL_XO})
    endforeach()

    foreach(DACE_KERNEL_FILE ${DACE_XILINX_KERNEL_FILES})
      # Compile .xo files

      # The kernel name is given by the file name
      get_filename_component(DACE_KERNEL_NAME ${DACE_KERNEL_FILE} NAME_WE)
      set(DACE_VITIS_KERNELS ${DACE_VITIS_KERNELS} ${DACE_KERNEL_NAME})

      add_custom_command(
        OUTPUT ${DACE_KERNEL_NAME}_sw_emu.xo
        COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
        ${VITIS_BUILD_FLAGS}
        -c
        -t sw_emu
        ${DACE_KERNEL_FILE}
        --kernel ${DACE_KERNEL_NAME}
        -o ${DACE_KERNEL_NAME}_sw_emu.xo
        DEPENDS ${DACE_VITIS_KERNEL_FILES})

      set(DACE_VITIS_KERNELS_SW_EMU ${DACE_VITIS_KERNELS_SW_EMU} ${DACE_KERNEL_NAME}_sw_emu.xo)

      add_custom_command(
        OUTPUT ${DACE_KERNEL_NAME}_hw_emu.xo
        COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
        ${VITIS_BUILD_FLAGS}
        -c
        -t hw_emu
        ${DACE_KERNEL_FILE}
        --kernel ${DACE_KERNEL_NAME}
        -o ${DACE_KERNEL_NAME}_hw_emu.xo
        DEPENDS ${DACE_VITIS_KERNEL_FILES})

      set(DACE_VITIS_KERNELS_HW_EMU ${DACE_VITIS_KERNELS_HW_EMU} ${DACE_KERNEL_NAME}_hw_emu.xo)

      add_custom_command(
        OUTPUT ${DACE_KERNEL_NAME}_hw.xo
        COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
        ${VITIS_BUILD_FLAGS}
        -c
        -t hw
        ${DACE_KERNEL_FILE}
        --kernel ${DACE_KERNEL_NAME}
        -o ${DACE_KERNEL_NAME}_hw.xo
        DEPENDS ${DACE_VITIS_KERNEL_FILES})

      set(DACE_VITIS_KERNELS_HW ${DACE_VITIS_KERNELS_HW} ${DACE_KERNEL_NAME}_hw.xo)

    endforeach()


    add_custom_target(xilinx_compile_${DACE_PROGRAM_NAME}_software_emulation
                      DEPENDS ${DACE_VITIS_KERNELS_SW_EMU})
    add_custom_target(xilinx_compile_software_emulation DEPENDS
                      xilinx_compile_${DACE_PROGRAM_NAME}_software_emulation)

    add_custom_target(xilinx_compile_${DACE_PROGRAM_NAME}_hardware_emulation
                      DEPENDS ${DACE_VITIS_KERNELS_HW_EMU})
    add_custom_target(xilinx_compile_hardware_emulation DEPENDS
                        xilinx_compile_${DACE_PROGRAM_NAME}_hardware_emulation)

    add_custom_target(xilinx_compile_${DACE_PROGRAM_NAME}_hardware DEPENDS
                      ${DACE_VITIS_KERNELS_HW})
    add_custom_target(xilinx_compile_hardware DEPENDS
                      xilinx_compile_${DACE_PROGRAM_NAME}_hardware)


    # Linking
    add_custom_command(
      OUTPUT ${DACE_PROGRAM_NAME}_sw_emu.xclbin
      COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
      ${VITIS_BUILD_FLAGS}
      --config "${DACE_XILINX_CONFIG_FILE}"
      -l
      -t sw_emu
      ${DACE_VITIS_KERNELS_SW_EMU}
      -o ${DACE_PROGRAM_NAME}_sw_emu.xclbin
      DEPENDS ${DACE_PROGRAM_NAME}_sw_emu.xo ${DACE_XILINX_CONFIG_FILE})

    add_custom_command(
      OUTPUT ${DACE_PROGRAM_NAME}_hw_emu.xclbin
      COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
      ${VITIS_BUILD_FLAGS}
      --config "${DACE_XILINX_CONFIG_FILE}"
      -l
      -t hw_emu
      ${DACE_VITIS_KERNELS_HW_EMU}
      -o ${DACE_PROGRAM_NAME}_hw_emu.xclbin
      DEPENDS ${DACE_VITIS_KERNELS_HW_EMU} ${DACE_XILINX_CONFIG_FILE})

    add_custom_command(
      OUTPUT emconfig.json
      COMMAND emconfigutil --platform ${DACE_XILINX_TARGET_PLATFORM}
      DEPENDS ${CMAKE_BINARY_DIR}/Synthesize_${DACE_KERNEL_NAME}.tcl)

    add_custom_command(
      OUTPUT ${DACE_PROGRAM_NAME}_hw.xclbin
      COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER}
      ${VITIS_BUILD_FLAGS}
      --config "${DACE_XILINX_CONFIG_FILE}"
      -l
      -t hw
      ${DACE_VITIS_KERNELS_HW}
      -o ${DACE_PROGRAM_NAME}_hw.xclbin
      DEPENDS ${DACE_VITIS_KERNELS_HW} ${DACE_XILINX_CONFIG_FILE})

  endif()

endif() # DACE_ENABLE_XILINX

# Create Intel FPGA object files
if(DACE_ENABLE_INTELFPGA)

  if((NOT (DACE_INTELFPGA_MODE STREQUAL "hardware")) OR DACE_INTELFPGA_ENABLE_DEBUGGING)
    set(DACE_INTELFPGA_HOST_FLAGS "${DACE_INTELFPGA_HOST_FLAGS} -g")
    set(DACE_INTELFPGA_SYNTHESIS_FLAGS "${DACE_INTELFPGA_KERNEL_FLAGS} -fast-compile -profile=all -g -fast-emulator")
  endif()

  set_source_files_properties(${DACE_INTELFPGA_KERNEL_FILES} ${DACE_INTELFPGA_HOST_FILES} PROPERTIES COMPILE_FLAGS "${DACE_INTELFPGA_HOST_FLAGS}")
  set_source_files_properties(${DACE_INTELFPGA_KERNEL_FILES} PROPERTIES COMPILE_FLAGS "-DDACE_INTELFPGA_DEVICE_CODE ${DACE_INTELFPGA_HOST_FLAGS}")
  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_INTELFPGA_KERNEL_FILES} ${DACE_INTELFPGA_HOST_FILES})

  # Add synthesis and build commands
  set(DACE_AOC_KERNEL_FILES)
  foreach(DACE_KERNEL_FILE ${DACE_INTELFPGA_KERNEL_FILES})

    get_filename_component(DACE_KERNEL_NAME ${DACE_KERNEL_FILE} NAME)
    string(REGEX REPLACE "kernel_(.+).cl" "\\1" DACE_KERNEL_NAME "${DACE_KERNEL_NAME}")
    set(DACE_AOC_KERNEL_FILES ${DACE_AOC_KERNEL_FILES} ${DACE_KERNEL_FILE})

    # Intel compiler does not allow to specify the output file if more than input file is used.
    # In this case, the output AOCX file will be named as the last OpenCL file given in input to the compiler.
    # We need to save the name of the last input file, so that later we can assign a proper name to the produced bitstream.
    get_filename_component(DACE_AOC_OUTPUT_FILE ${DACE_KERNEL_FILE} NAME_WE)
  endforeach()

  string(REPLACE " " ";" DACE_INTELFPGA_KERNEL_FLAGS_INTERNAL
         "${DACE_INTELFPGA_KERNEL_FLAGS}")

  set(DACE_AOC_BUILD_FLAGS
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_SOURCE_DIR}/../external/hlslib/include
    -I${CMAKE_SOURCE_DIR}/../runtime/include
    -I${CMAKE_BINARY_DIR}
    -board=${DACE_INTELFPGA_TARGET_BOARD}
    ${DACE_INTELFPGA_KERNEL_FLAGS_INTERNAL})

  add_custom_target(
    intelfpga_report_${DACE_PROGRAM_NAME}
    COMMAND
    ${IntelFPGAOpenCL_AOC}
    ${DACE_AOC_BUILD_FLAGS}
    ${DACE_AOC_KERNEL_FILES}
    -rtl
    -report
    COMMAND mv ${DACE_AOC_OUTPUT_FILE} ${DACE_PROGRAM_NAME})

  add_custom_command(
    OUTPUT ${DACE_PROGRAM_NAME}_emulator.aocx
    COMMAND ${IntelFPGAOpenCL_AOC}
    ${DACE_AOC_BUILD_FLAGS}
    -march=emulator
    ${DACE_AOC_KERNEL_FILES}
    COMMAND mv ${DACE_AOC_OUTPUT_FILE}.aocx  ${DACE_PROGRAM_NAME}_emulator.aocx
    DEPENDS ${DACE_AOC_KERNEL_FILES})

  add_custom_command(
    OUTPUT ${DACE_PROGRAM_NAME}_hardware.aocx
    COMMAND ${IntelFPGAOpenCL_AOC}
    ${DACE_AOC_BUILD_FLAGS}
    ${DACE_AOC_KERNEL_FILES}
    COMMAND mv ${DACE_AOC_OUTPUT_FILE}.aocx  ${DACE_PROGRAM_NAME}_hardware.aocx
    COMMAND mv ${DACE_AOC_OUTPUT_FILE} ${DACE_PROGRAM_NAME}
    DEPENDS ${DACE_AOC_KERNEL_FILES})

endif()

include("targets/mlir/mlir.cmake")

# Create DaCe library file
add_library(${DACE_PROGRAM_NAME} SHARED ${DACE_CPP_FILES} ${DACE_OBJECTS})
target_link_libraries(${DACE_PROGRAM_NAME} ${DACE_LIBS})

# Add additional required files
if(DACE_ENABLE_INTELFPGA)
  if(DACE_INTELFPGA_MODE STREQUAL "emulator")
      add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_emulator
                        ALL DEPENDS ${DACE_PROGRAM_NAME}_emulator.aocx)
  else()
      add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_emulator
                        DEPENDS ${DACE_PROGRAM_NAME}_emulator.aocx)
  endif()
  if(DACE_INTELFPGA_MODE STREQUAL "hardware" AND DACE_FPGA_AUTOBUILD_BITSTREAM)
      add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_hardware
                        ALL DEPENDS ${DACE_PROGRAM_NAME}_hardware.aocx)
  else()
      add_custom_target(intelfpga_compile_${DACE_PROGRAM_NAME}_hardware
                        DEPENDS ${DACE_PROGRAM_NAME}_hardware.aocx)
  endif()
endif()
if(DACE_ENABLE_XILINX)
    if(DACE_XILINX_MODE STREQUAL "software_emulation")
      add_custom_target(xilinx_link_${DACE_PROGRAM_NAME}_software_emulation
                        ALL DEPENDS ${DACE_PROGRAM_NAME}_sw_emu.xclbin)
    else()
      add_custom_target(xilinx_link_${DACE_PROGRAM_NAME}_software_emulation
                        DEPENDS ${DACE_PROGRAM_NAME}_sw_emu.xclbin)
    endif()
    add_custom_target(xilinx_link_software_emulation DEPENDS
                      xilinx_link_${DACE_PROGRAM_NAME}_software_emulation)
  if(DACE_XILINX_MODE STREQUAL "hardware_emulation")
    add_custom_target(xilinx_link_${DACE_PROGRAM_NAME}_hardware_emulation
                      ALL DEPENDS emconfig.json
                      ${DACE_PROGRAM_NAME}_hw_emu.xclbin)
  else()
    add_custom_target(xilinx_link_${DACE_PROGRAM_NAME}_hardware_emulation
                      DEPENDS ${DACE_PROGRAM_NAME}_hw_emu.xclbin)
  endif()
  add_custom_target(xilinx_link_hardware_emulation DEPENDS
                    xilinx_link_${DACE_PROGRAM_NAME}_hardware_emulation)
  if(DACE_XILINX_MODE STREQUAL "hardware" AND DACE_FPGA_AUTOBUILD_BITSTREAM)
      add_custom_target(xilinx_link_${DACE_PROGRAM_NAME}_hardware
                        ALL DEPENDS ${DACE_PROGRAM_NAME}_hw.xclbin)
  else()
      add_custom_target(xilinx_link_${DACE_PROGRAM_NAME}_hardware
                        DEPENDS ${DACE_PROGRAM_NAME}_hw.xclbin)
  endif()
  add_custom_target(xilinx_link_hardware DEPENDS
                    xilinx_link_${DACE_PROGRAM_NAME}_hardware)
endif()

# Create DaCe loader stub
add_library(dacestub_${DACE_PROGRAM_NAME} SHARED "${CMAKE_SOURCE_DIR}/tools/dacestub.cpp")
target_link_libraries(dacestub_${DACE_PROGRAM_NAME} ${CMAKE_THREAD_LIBS_INIT} ${OpenMP_CXX_LIBRARIES})

# Windows-specific fixes
if (MSVC_IDE)
    # Copy output DLL from the "Debug" and "Release" directories CMake adds
    # NOTE: The "|| (exit 0)" is added because copy sometimes fails due to the
    # stub library being already loaded.
    add_custom_target(CopyDLL ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${DACE_PROGRAM_NAME}> "${CMAKE_BINARY_DIR}/lib${DACE_PROGRAM_NAME}.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dacestub_${DACE_PROGRAM_NAME}> "${CMAKE_BINARY_DIR}/libdacestub_${DACE_PROGRAM_NAME}.dll" || (exit 0)
        DEPENDS ${DACE_PROGRAM_NAME}
        COMMENT "Copying binaries" VERBATIM)

    set_property(TARGET ${DACE_PROGRAM_NAME} PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
