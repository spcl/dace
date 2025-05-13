set(HOST_TRIPLET "${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_HOST_SYSTEM_NAME}")
string(TOLOWER "${HOST_TRIPLET}" HOST_TRIPLET_LOWER)

find_program(CMAKE_CCE_COMPILER
    NAMES "ccec"
    PATHS
        "$ENV{ASCEND_HOME_PATH}/compiler/bin"
        "$ENV{ASCEND_INSTALL_PATH}/compiler/bin"
        "$ENV{ASCEND_HOME_PATH}/ccec_compiler/bin"
        "$ENV{ASCEND_INSTALL_PATH}/ccec_compiler/bin"
        "$ENV{ASCEND_HOME_PATH}/${HOST_TRIPLET_LOWER}/compiler/bin"
        "$ENV{ASCEND_INSTALL_PATH}/${HOST_TRIPLET_LOWER}/compiler/bin"
        "$ENV{ASCEND_HOME_PATH}/${HOST_TRIPLET_LOWER}/ccec_compiler/bin"
        "$ENV{ASCEND_INSTALL_PATH}/${HOST_TRIPLET_LOWER}/ccec_compiler/bin"
        "$ENV{PATH}"
    DOC "CCE Compiler"
)
set(ASCEND_INSTALL_PATH "$ENV{ASCEND_HOME_PATH}")

include(CMakeCCEFunction)

mark_as_advanced(CMAKE_CCE_COMPILER)

message(STATUS "CMAKE_CCE_COMPILER: " ${CMAKE_CCE_COMPILER})
set(CMAKE_CCE_SOURCE_FILE_EXTENSIONS cce)
set(CMAKE_CCE_COMPILER_ENV_VAR "CCE")
message(STATUS "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCCECompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeCCECompiler.cmake
    @ONLY
)

message(STATUS "ASCEND_PRODUCT_TYPE:\n" "  ${DACE_ASCEND_PRODUCT_TYPE}")
message(STATUS "ASCEND_CORE_TYPE:\n" "  ${DACE_ASCEND_CORE_TYPE}")
message(STATUS "ASCEND_INSTALL_PATH:\n" "  ${DACE_ASCEND_INSTALL_PATH}")

if(DEFINED ASCEND_INSTALL_PATH)
    set(_CMAKE_ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH})
else()
    message(FATAL_ERROR
        "no, installation path found, should passing -DASCEND_INSTALL_PATH=<PATH_TO_ASCEND_INSTALLATION> in cmake"
    )
    set(_CMAKE_ASCEND_INSTALL_PATH)
endif()


if(DEFINED DACE_ASCEND_PRODUCT_TYPE)
    set(_CMAKE_CCE_COMMON_COMPILE_OPTIONS "--cce-auto-sync")
    if(DACE_ASCEND_PRODUCT_TYPE STREQUAL "")
        message(FATAL_ERROR "ASCEND_PRODUCT_TYPE must be non-empty if set.")
    elseif(DACE_ASCEND_PRODUCT_TYPE AND NOT DACE_ASCEND_PRODUCT_TYPE MATCHES "^ascend[0-9][0-9][0-9][a-zA-Z]?[1-9]?$")
        message(FATAL_ERROR
            "ASCEND_PRODUCT_TYPE: ${DACE_ASCEND_PRODUCT_TYPE}\n"
            "is not one of the following: ascend910, ascend310p, ascend910B1, ascend910B4"
        )
    elseif(DACE_ASCEND_PRODUCT_TYPE STREQUAL "ascend910")
        if (DACE_ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c100")
        else()
            message(FATAL_ERROR, "only AiCore inside")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS)
    elseif(DACE_ASCEND_PRODUCT_TYPE STREQUAL "ascend310p")
        if (DACE_ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-m200")
        elseif(DACE_ASCEND_CORE_TYPE STREQUAL "VectorCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-m200-vec")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS
            "-mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-fp-ceiling=2 -mllvm -cce-aicore-record-overflow=false")
    elseif(DACE_ASCEND_PRODUCT_TYPE STREQUAL "ascend910B1")
        if (DACE_ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220-cube")
        elseif(DACE_ASCEND_CORE_TYPE STREQUAL "VectorCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220-vec")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS
            "-mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform"
        )
    elseif(DACE_ASCEND_PRODUCT_TYPE STREQUAL "ascend910B4")
        if (DACE_ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-soc-version=Ascend910B4 --cce-soc-core-type=VecCore")
        elseif(DACE_ASCEND_CORE_TYPE STREQUAL "VectorCore")
            message(FATAL_ERROR, "only AiCore inside")
        endif()
    endif()
endif()

product_dir(${DACE_ASCEND_PRODUCT_TYPE} PRODUCT_UPPER)
set(_CMAKE_CCE_HOST_IMPLICIT_LINK_DIRECTORIES
    ${_CMAKE_ASCEND_INSTALL_PATH}/runtime/lib64
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/simulator/${PRODUCT_UPPER}/lib
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${PRODUCT_UPPER}
)

# link library
set(_CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES stdc++)
if(DACE_ASCEND_RUN_MODE STREQUAL "ONBOARD")
    list(APPEND _CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES runtime)
elseif(DACE_ASCEND_RUN_MODE STREQUAL "SIMULATOR")
    message(FATAL_ERROR "DaCe currently only supports ONBOARD")
else()
    message(FATAL_ERROR
        "ASCEND_RUN_MODE: ${DACE_ASCEND_RUN_MODE}\n"
        "ASCEND_RUN_MODE must be one of the following: ONBOARD or SIMULATOR"
    )
endif()
list(APPEND _CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES ascendcl)

if(DACE_ASCEND_CUSTOM_TILING STREQUAL "CUSTOM_TILING")
elseif(DACE_ASCEND_CUSTOM_TILING STREQUAL "NO_CUSTOM_TILING")
else()
    message(FATAL_ERROR
        "ASCEND_CUSTOM_TILING: ${DACE_ASCEND_CUSTOM_TILING}\n"
        "ASCEND_CUSTOM_TILING must be one of the following: CUSTOM_TILING or NO_CUSTOM_TILING"
    )
endif()

set(__IMPLICIT_LINKS)
foreach(dir ${_CMAKE_CCE_HOST_IMPLICIT_LINK_DIRECTORIES})
  string(APPEND __IMPLICIT_LINKS " -L\"${dir}\"")
endforeach()
foreach(lib ${_CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES})
  if(${lib} MATCHES "/")
    string(APPEND __IMPLICIT_LINKS " \"${lib}\"")
  else()
    string(APPEND __IMPLICIT_LINKS " -l${lib}")
  endif()
endforeach()

set(_CMAKE_CCE_HOST_IMPLICIT_INCLUDE_DIRECTORIES
    ${_CMAKE_ASCEND_INSTALL_PATH}/acllib/include
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw/impl
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw/interface
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/include
)
set(__IMPLICIT_INCLUDES)
foreach(inc ${_CMAKE_CCE_HOST_IMPLICIT_INCLUDE_DIRECTORIES})
  string(APPEND __IMPLICIT_INCLUDES " -I\"${inc}\"")
endforeach()
