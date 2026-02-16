# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
cmake_minimum_required(VERSION 3.17)
project(dace_program)

# General options
set(DACE_PROGRAM_NAME "dace_program" CACHE STRING "Name of DaCe program")
set(DACE_SRC_DIR "" CACHE STRING "Root directory of generated code files")
set(DACE_FILES "" CACHE STRING "List of host code files relative to the root of the source directory")
set(DACE_LIBS "" CACHE STRING "Extra libraries")
set(DACE_CPP_STANDARD "20" CACHE STRING "C++ standard to use for compilation (e.g., 14, 17, 20, 23, 26)")
set(DACE_CMAKE_FILES "" CACHE STRING "List of additional CMake files to include")

# CUDA
set(DACE_CUDA_ARCHITECTURES_DEFAULT "" CACHE STRING "Default CUDA architectures in case native not found")

# Target detection
set(DACE_ENABLE_MPI OFF)
set(DACE_ENABLE_CUDA OFF)
set(DACE_ENABLE_HIP OFF)

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
      set(DACE_CPP_FILES ${DACE_CPP_FILES} ${DACE_FILE})
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

# Global DaCe external dependencies
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

list(APPEND DACE_LIBS Threads::Threads)
list(APPEND DACE_LIBS OpenMP::OpenMP_CXX)

add_definitions(-DDACE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\")

if(DACE_ENABLE_MPI)
  find_package(MPI REQUIRED)
  list(APPEND DACE_LIBS MPI::MPI_CXX)
endif()

if(DACE_ENABLE_CUDA)
  set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})

  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  # CMake 3.24: set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES native)
  if (NOT DEFINED LOCAL_CUDA_ARCHITECTURES)
      execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" "--run"
                      "${CMAKE_SOURCE_DIR}/tools/get_cuda_arch.cpp"
                      OUTPUT_VARIABLE _local_arch RESULT_VARIABLE _arch_res)

    if(_arch_res EQUAL 0)
      set(LOCAL_CUDA_ARCHITECTURES "${_local_arch}" CACHE STRING "Detected local GPUs for compilation")
      message(STATUS "Local CUDA architectures detected: ${LOCAL_CUDA_ARCHITECTURES}")
    else()
      set(LOCAL_CUDA_ARCHITECTURES "${DACE_CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING "Detected local GPUs for compilation")
      message(STATUS "No local CUDA-capable GPUs found. Using default: ${DACE_CUDA_ARCHITECTURES_DEFAULT}")
    endif()
  endif()

  set(CMAKE_CUDA_ARCHITECTURES "${LOCAL_CUDA_ARCHITECTURES}")
  enable_language(CUDA)
  list(APPEND DACE_LIBS CUDA::cudart CUDA::nvtx3)
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

  # Add libraries such as rocBLAS
  link_directories(${HIP_PATH}/../lib)
  if(ROCM_PATH)
    find_path(ROCTX_INCLUDE_DIR roctx.h HINTS ${ROCM_PATH}/include/roctracer ${ROCM_PATH}/roctracer/include)
    if(NOT ROCTX_INCLUDE_DIR)
      message(WARNING "Could not find roctx.h in ${ROCM_PATH}/include/roctracer or ${ROCM_PATH}/roctracer/include")
    endif()
  endif()
  if(ROCM_PATH AND ROCTX_INCLUDE_DIR)
    find_path(ROCTX_LIBRARY_DIR "libroctx64.so" HINTS ${ROCM_PATH}/lib)
    if(NOT ROCTX_LIBRARY_DIR)
      message(WARNING "Could not find libroctx64.so in ${ROCM_PATH}/lib")
    else()
      list(APPEND DACE_LIBS "-lroctx64 -L${ROCTX_LIBRARY_DIR}")
      include_directories(SYSTEM ${ROCTX_INCLUDE_DIR})
    endif()
  endif()
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
if (DEFINED DACE_ENV_MINIMUM_VERSION AND ${CMAKE_VERSION} VERSION_LESS ${DACE_ENV_MINIMUM_VERSION})
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
string(REPLACE "_DACE_CMAKE_EXPAND" "$" DACE_ENV_PACKAGES "${DACE_ENV_PACKAGES}")
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
string(REPLACE "_DACE_CMAKE_EXPAND" "$" DACE_ENV_INCLUDES "${DACE_ENV_INCLUDES}")
string(REPLACE "_DACE_CMAKE_EXPAND" "$" DACE_ENV_LIBRARIES "${DACE_ENV_LIBRARIES}")
string(REPLACE " " ";" DACE_ENV_INCLUDES "${DACE_ENV_INCLUDES}")
string(REPLACE " " ";" DACE_ENV_LIBRARIES "${DACE_ENV_LIBRARIES}")
include_directories(${DACE_ENV_INCLUDES})
set(DACE_LIBS ${DACE_LIBS} ${DACE_ENV_LIBRARIES})
string(REPLACE "_DACE_CMAKE_EXPAND" "$" DACE_ENV_LINK_FLAGS "${DACE_ENV_LINK_FLAGS}")
string(REPLACE "_DACE_CMAKE_EXPAND" "$" DACE_ENV_COMPILE_FLAGS "${DACE_ENV_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DACE_ENV_COMPILE_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")
set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${DACE_ENV_LINK_FLAGS}")


# Create HIP object files
if(DACE_ENABLE_HIP)
  enable_language(HIP)

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
    list(APPEND HIP_HIPCC_FLAGS --offload-arch=${var})
  endforeach()

  # Add flags from dace config
  list(APPEND HIP_HIPCC_FLAGS ${EXTRA_HIP_FLAGS})

  # Add include directories for other files
  set(DACE_LIBS ${DACE_LIBS} hip::host)

  set_source_files_properties(${DACE_HIP_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  set_source_files_properties(${DACE_HIP_FILES} PROPERTIES LANGUAGE HIP)
  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_HIP_FILES})
endif() # DACE_ENABLE_HIP

# Additional target-specific CMake files
foreach(CMAKE_FILE ${DACE_CMAKE_FILES})
  include(${CMAKE_FILE})
endforeach()

# Create DaCe library file
add_library(${DACE_PROGRAM_NAME} SHARED ${DACE_CPP_FILES} ${DACE_OBJECTS})
target_link_libraries(${DACE_PROGRAM_NAME} PUBLIC ${DACE_LIBS})

# Set C++ standard to C++20 (or the configured standard)
set_property(TARGET ${DACE_PROGRAM_NAME} PROPERTY CXX_STANDARD ${DACE_CPP_STANDARD})

# Create DaCe loader stub
add_library(dacestub_${DACE_PROGRAM_NAME} SHARED "${CMAKE_SOURCE_DIR}/tools/dacestub.cpp")
target_link_libraries(dacestub_${DACE_PROGRAM_NAME} Threads::Threads OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

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
