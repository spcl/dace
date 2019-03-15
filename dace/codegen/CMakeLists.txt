cmake_minimum_required(VERSION 2.8.12)
project(dace_program)

# General options 
set(DACE_PROGRAM_NAME "dace_program" CACHE STRING "Name of DaCe program")
set(DACE_FILES "" CACHE STRING "Host code files")
set(DACE_LIBS "" CACHE STRING "Extra libraries")
set(HLSLIB_PART_NAME "${DACE_XILINX_PART_NAME}")

# Allow passing flags to various stages of Xilinx compilation process
set(DACE_XILINX_MODE "simulation" CACHE STRING "Type of compilation/execution [simulation/software_emulation/hardware_emulation/hardware].")
set(DACE_XILINX_HOST_FLAGS "" CACHE STRING "Extra flags to host code")
set(DACE_XILINX_SYNTHESIS_FLAGS "" CACHE STRING "Extra flags for performing high-level synthesis")
set(DACE_XILINX_BUILD_FLAGS "" CACHE STRING "Extra flags to xocc build phase")
set(DACE_XILINX_TARGET_CLOCK 200 CACHE STRING "Target clock frequency of FPGA kernel")
set(DACE_XILINX_PART_NAME "xcvu9p-fsgd2104-2l-e" CACHE STRING "Xilinx chip to target from HLS")
set(DACE_XILINX_TARGET_PLATFORM "xilinx_vcu1525_dynamic_5_1" CACHE STRING "SDAccel platform to target")
set(DACE_XILINX_ENABLE_DEBUGGING OFF CACHE STRING "Inject debugging cores to kernel build (always on for simulation/emulation)")

# Target detection
set(DACE_ENABLE_MPI OFF)
set(DACE_ENABLE_CUDA OFF)
set(DACE_ENABLE_XILINX OFF)

# Split list by target
foreach(DACE_FILE ${DACE_FILES})
  # Extract the target from the folder name
  get_filename_component(DACE_FILE_NAME ${DACE_FILE} NAME_WE)
  get_filename_component(DACE_FILE_TARGET ${DACE_FILE} DIRECTORY)
  get_filename_component(DACE_FILE_TARGET ${DACE_FILE_TARGET} NAME)
  if(${DACE_FILE_TARGET} STREQUAL "cuda")
    set(DACE_ENABLE_CUDA ON)
    set(DACE_CUDA_FILES ${DACE_CUDA_FILES} ${DACE_FILE})
  elseif(${DACE_FILE_TARGET} STREQUAL "xilinx")
    set(DACE_ENABLE_XILINX ON)
    if(DACE_FILE_NAME MATCHES ".+_host") 
      set(DACE_XILINX_HOST_FILES ${DACE_XILINX_HOST_FILES} ${DACE_FILE})
    else()
      set(DACE_XILINX_KERNEL_FILES ${DACE_XILINX_KERNEL_FILES} ${DACE_FILE})
    endif()
  elseif(${DACE_FILE_TARGET} STREQUAL "mpi")
    set(DACE_ENABLE_MPI ON)
    set(DACE_CPP_FILES ${DACE_CPP_FILES} ${DACE_FILE})
  else()
    set(DACE_CPP_FILES ${DACE_CPP_FILES} ${DACE_FILE})
  endif()
endforeach()

# Internal dependencies
set(DACE_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../runtime)
include_directories(${DACE_RUNTIME_DIR}/include)

# External dependencies 
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/" DACE_BINARY_DIR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DDACE_BINARY_DIR='\"${DACE_BINARY_DIR}\"'")
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
  set(DACE_LIBS ${DACE_LIBS} ${CUDA_LIBRARIES})
  add_definitions(-DWITH_CUDA)
endif()
if(DACE_ENABLE_XILINX)
  set(DACE_HLSLIB_DIR ${CMAKE_SOURCE_DIR}/../external/hlslib)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DACE_HLSLIB_DIR}/cmake)
  find_package(SDAccel REQUIRED)

  include_directories(SYSTEM ${SDAccel_INCLUDE_DIRS} ${DACE_HLSLIB_DIR}/include)
  add_definitions(-DDACE_XILINX)
  set(DACE_LIBS ${DACE_LIBS} ${SDAccel_LIBRARIES})

endif()

# Create CUDA object files
if(DACE_ENABLE_CUDA)
  # Get local CUDA architectures
  if (NOT DEFINED LOCAL_CUDA_ARCHITECTURES)
      execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run"
                      "${CMAKE_SOURCE_DIR}/tools/get_cuda_arch.cpp"
                      OUTPUT_VARIABLE _arch_out RESULT_VARIABLE _arch_res
                      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(_arch_res EQUAL 0)
        string(REGEX REPLACE "\n" ";" _arch_out "${_arch_out}")
        list(GET _arch_out -1 _local_arch)
        string(REGEX REPLACE " " ";" _local_arch "${_local_arch}")
        set(LOCAL_CUDA_ARCHITECTURES "${_local_arch}" CACHE STRING "Detected local GPUs for compilation")
        message("-- Local CUDA architectures detected: ${LOCAL_CUDA_ARCHITECTURES}")
      else()
        set(LOCAL_CUDA_ARCHITECTURES "" CACHE STRING "Detected local GPUs for compilation")
        message("-- No local CUDA-capable GPUs found")
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

# Create Xilinx object files
if(DACE_ENABLE_XILINX)
  if((NOT (DACE_XILINX_MODE STREQUAL "hardware")) OR DACE_XILINX_ENABLE_DEBUGGING)
    set(DACE_XILINX_HOST_FLAGS "${DACE_XILINX_HOST_FLAGS} -g")
    set(DACE_XILINX_SYNTHESIS_FLAGS "${DACE_XILINX_SYNTHESIS_FLAGS} -g")
  endif()

  set_source_files_properties(${DACE_XILINX_KERNEL_FILES} ${DACE_XILINX_HOST_FILES} PROPERTIES COMPILE_FLAGS "${DACE_XILINX_HOST_FLAGS}")
  set_source_files_properties(${DACE_XILINX_KERNEL_FILES} PROPERTIES COMPILE_FLAGS "-DDACE_XILINX_DEVICE_CODE ${DACE_XILINX_HOST_FLAGS}")
  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_XILINX_KERNEL_FILES} ${DACE_XILINX_HOST_FILES})

  if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND
      (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
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

  # Add synthesis and build commands
  set(DACE_SYNTHESIS_TARGETS)
  foreach(DACE_KERNEL_FILE ${DACE_XILINX_KERNEL_FILES})
    get_filename_component(DACE_KERNEL_NAME ${DACE_KERNEL_FILE} NAME)
    string(REGEX REPLACE "kernel_(.+).cpp" "\\1" DACE_KERNEL_NAME "${DACE_KERNEL_NAME}")
    string(REPLACE " " ";" DACE_XILINX_SYNTHESIS_FLAGS_INTERNAL ${DACE_XILINX_SYNTHESIS_FLAGS})
    set(DACE_XOCC_KERNEL_FILES ${DACE_XOCC_KERNEL_FILES} ${DACE_KERNEL_FILE})
    set(DACE_XOCC_KERNELS ${DACE_XOCC_KERNELS} --kernel ${DACE_KERNEL_NAME} --xp prop:kernel.${DACE_KERNEL_NAME}.kernel_flags=\"${DACE_XILINX_SYNTHESIS_FLAGS_INTERNAL}\")

    configure_file(${CMAKE_SOURCE_DIR}/Xilinx_HLS.tcl.in Synthesize_${DACE_KERNEL_NAME}.tcl)
    add_custom_target(xilinx_synthesis_${DACE_KERNEL_NAME} COMMAND ${SDAccel_VIVADO_HLS} -f Synthesize_${DACE_KERNEL_NAME}.tcl) 
    set(DACE_SYNTHESIS_TARGETS ${DACE_SYNTHESIS_TARGETS} xilinx_synthesis_${DACE_KERNEL_NAME})

  endforeach()

  add_custom_target(xilinx_synthesis DEPENDS ${DACE_SYNTHESIS_TARGETS})

  string(REPLACE " " ";" DACE_XILINX_BUILD_FLAGS_INTERNAL
         "${DACE_XILINX_BUILD_FLAGS}")

  set(XOCC_BUILD_FLAGS
    -s
    -O3
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_SOURCE_DIR}/../external/hlslib/include
    -I${CMAKE_SOURCE_DIR}/../runtime/include
    -I${CMAKE_BINARY_DIR}
    "${DACE_XOCC_KERNELS}"
    --platform ${DACE_XILINX_TARGET_PLATFORM}
    ${DACE_XILINX_BUILD_FLAGS_INTERNAL}
    --kernel_frequency ${DACE_XILINX_TARGET_CLOCK}
    --max_memory_ports all)

  if((NOT (DACE_XILINX_MODE STREQUAL "hardware")) OR DACE_XILINX_ENABLE_DEBUGGING)
    # TODO: add Chipscope debugging on memory interfaces. Need to pass
    # interfaces from codegen to CMake in order to do this.
    message(STATUS "Enabled debugging/profiling for Xilinx targets.")
    set(XOCC_BUILD_FLAGS ${XOCC_BUILD_FLAGS}
      --profile_kernel "data:all:all:all"
      --profile_kernel "stall:all:all"
      --profile_kernel "exec:all:all")
  endif()

  if(SDAccel_MAJOR_VERSION LESS 2018 AND SDAccel_MINOR_VERSION LESS 3)

    add_custom_target(
      xilinx_build_${DACE_PROGRAM_NAME}_software_emulation
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -t sw_emu
      ${DACE_XOCC_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_sw_emu.xclbin)

    add_custom_target(
      xilinx_build_${DACE_PROGRAM_NAME}_hardware_emulation
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -t hw_emu
      ${DACE_XOCC_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_hw_emu.xclbin)

    add_custom_target(
      xilinx_build_${DACE_PROGRAM_NAME}_hardware
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -t hw
      ${DACE_XOCC_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_hw.xclbin)

  else()

    add_custom_target(
      xilinx_compile_${DACE_PROGRAM_NAME}_software_emulation
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -c
      -t sw_emu
      ${DACE_XOCC_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_sw_emu.xo)

    add_custom_target(
      xilinx_compile_${DACE_PROGRAM_NAME}_hardware_emulation
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -c
      -t hw_emu
      ${DACE_XOCC_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_hw_emu.xo)

    add_custom_target(
      xilinx_compile_${DACE_PROGRAM_NAME}_hardware
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -c
      -t hw
      ${DACE_XOCC_KERNEL_FILES}
      -o ${DACE_PROGRAM_NAME}_hw.xo)

    add_custom_target(
      xilinx_build_${DACE_PROGRAM_NAME}_software_emulation
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -l
      -t sw_emu
      ${DACE_PROGRAM_NAME}_sw_emu.xo
      -o ${DACE_PROGRAM_NAME}_sw_emu.xclbin)

    add_custom_target(
      xilinx_build_${DACE_PROGRAM_NAME}_hardware_emulation
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -l
      -t hw_emu
      ${DACE_PROGRAM_NAME}_hw_emu.xo
      -o ${DACE_PROGRAM_NAME}_hw_emu.xclbin)

    add_custom_target(
      xilinx_build_${DACE_PROGRAM_NAME}_hardware
      COMMAND
      XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC}
      ${XOCC_BUILD_FLAGS}
      -l
      -t hw
      ${DACE_PROGRAM_NAME}_hw.xo
      -o ${DACE_PROGRAM_NAME}_hw.xclbin)

  endif()

endif() # DACE_ENABLE_XILINX

# Create DaCe library file 
add_library(${DACE_PROGRAM_NAME} SHARED ${DACE_CPP_FILES} ${DACE_OBJECTS})
target_link_libraries(${DACE_PROGRAM_NAME} ${DACE_LIBS})

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

    # Replace /MD with /MT so that CUDA links properly
    # https://stackoverflow.com/a/14172871/6489142
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_MINSIZEREL
        )
    foreach(CompilerFlag ${CompilerFlags})
      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endif()
