# ============================================================================
# CMakeLists.txt — hlfir_bridge
# ============================================================================
# Build targets:
#   hlfir_bridge_core    — static lib  (trace/extract)
#   hlfir_bridge_passes  — static lib  (MLIR passes)
#   hlfir_bridge         — Python ext  (nanobind; links both above)
#
# MLIR linking strategy (cf. pixel-druid.com/articles/how-to-link-against-mlir-with-cmake):
#   Debian/Ubuntu MLIR cmake configs are often broken (missing tool binaries
#   cause CMake export-set errors).  We avoid find_package(MLIR) entirely.
#   Instead we use find_package(LLVM) — which works — and locate MLIR
#   headers and libraries through LLVM's install prefix.  MLIR is always
#   co-installed alongside LLVM.
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# --- Default LLVM version ----------------------------------------------------
set(LLVM_VERSION "21" CACHE STRING "LLVM major version to use")

# --- Discover LLVM prefix from flang-new -------------------------------------
set(_LLVM_PREFIX "" CACHE PATH "LLVM installation prefix (auto-detected)")

if(NOT _LLVM_PREFIX)
    find_program(_FLANG_BIN flang-new-${LLVM_VERSION})
    if(_FLANG_BIN)
        get_filename_component(_FLANG_REAL "${_FLANG_BIN}" REALPATH)
        get_filename_component(_FLANG_BINDIR "${_FLANG_REAL}" DIRECTORY)
        get_filename_component(_LLVM_PREFIX "${_FLANG_BINDIR}" DIRECTORY)
        message(STATUS "Derived LLVM prefix from flang-new-${LLVM_VERSION}: ${_LLVM_PREFIX}")
    endif()
endif()

# --- Default compilers -------------------------------------------------------
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    if(_LLVM_PREFIX AND EXISTS "${_LLVM_PREFIX}/bin/clang-${LLVM_VERSION}")
        set(CMAKE_C_COMPILER "${_LLVM_PREFIX}/bin/clang-${LLVM_VERSION}" CACHE FILEPATH "")
    else()
        find_program(_DEFAULT_CC clang-${LLVM_VERSION})
        if(_DEFAULT_CC)
            set(CMAKE_C_COMPILER "${_DEFAULT_CC}" CACHE FILEPATH "")
        endif()
    endif()
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    if(_LLVM_PREFIX AND EXISTS "${_LLVM_PREFIX}/bin/clang++-${LLVM_VERSION}")
        set(CMAKE_CXX_COMPILER "${_LLVM_PREFIX}/bin/clang++-${LLVM_VERSION}" CACHE FILEPATH "")
    else()
        find_program(_DEFAULT_CXX clang++-${LLVM_VERSION})
        if(_DEFAULT_CXX)
            set(CMAKE_CXX_COMPILER "${_DEFAULT_CXX}" CACHE FILEPATH "")
        endif()
    endif()
endif()
if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
    if(_FLANG_BIN)
        set(CMAKE_Fortran_COMPILER "${_FLANG_BIN}" CACHE FILEPATH "")
    else()
        find_program(_DEFAULT_FC flang-new-${LLVM_VERSION})
        if(_DEFAULT_FC)
            set(CMAKE_Fortran_COMPILER "${_DEFAULT_FC}" CACHE FILEPATH "")
        endif()
    endif()
endif()

project(hlfir_bridge C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- LLVM (the only find_package we use) -------------------------------------
if(NOT LLVM_DIR)
    if(_LLVM_PREFIX AND EXISTS "${_LLVM_PREFIX}/lib/cmake/llvm/LLVMConfig.cmake")
        set(LLVM_DIR "${_LLVM_PREFIX}/lib/cmake/llvm" CACHE PATH "")
    else()
        set(LLVM_DIR "/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm"
            CACHE PATH "Path to LLVMConfig.cmake")
    endif()
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM library dir:  ${LLVM_LIBRARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# --- MLIR: located through LLVM, not find_package(MLIR) ----------------------
# MLIR headers live alongside LLVM's in the same include directory.
# MLIR libraries (.a / .so) live in LLVM_LIBRARY_DIR.
set(MLIR_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}" CACHE PATH "MLIR include dirs")

# Helper: find one MLIR (or Flang) library and create an imported target.
# Usage: find_mlir_lib(MLIRIR) → creates target MLIRIR if libMLIRIR.a exists.
macro(find_mlir_lib name)
    if(NOT TARGET ${name})
        find_library(_${name}_LIB ${name} HINTS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
        if(_${name}_LIB)
            add_library(${name} STATIC IMPORTED)
            set_target_properties(${name} PROPERTIES IMPORTED_LOCATION "${_${name}_LIB}")
        else()
            message(WARNING "Cannot find lib${name} in ${LLVM_LIBRARY_DIR}")
        endif()
        unset(_${name}_LIB CACHE)
    endif()
endmacro()

# The MLIR libraries we actually use.
set(_MLIR_LIBS
    MLIRIR MLIRParser MLIRSupport
    MLIRPass MLIRTransforms
    MLIRFuncDialect MLIRArithDialect MLIRSCFDialect
    MLIRControlFlowToSCF
    MLIROptLib
)
foreach(_lib ${_MLIR_LIBS})
    find_mlir_lib(${_lib})
endforeach()

# Monolithic shared libs (Debian ships these; handles transitive deps).
find_library(MLIR_MONO_LIB MLIR HINTS ${LLVM_LIBRARY_DIR})
set(LLVM_MONO_LIB "LLVM-${LLVM_VERSION_MAJOR}")
message(STATUS "MLIR monolithic: ${MLIR_MONO_LIB}")
message(STATUS "LLVM monolithic: ${LLVM_MONO_LIB}")

# --- Flang dialect libraries --------------------------------------------------
set(FLANG_DIALECT_LIBS
    HLFIRDialect FIRDialect FIRDialectSupport FIRSupport
    FIRBuilder FIRAnalysis CUFDialect CUFAttrs
    # ---- flang transforms / codegen ---------------------------------
    # Needed so the bridge can run ``fir-polymorphic-op``, which
    # statically devirtualises ``fir.dispatch`` / ``fir.select_type``
    # ops whose target is resolvable at compile time.  Surviving
    # dispatches after the conversion are caught loudly by our own
    # ``hlfir-reject-polymorphism`` pass.  The user-visible benefit:
    # CLASS-typed code that doesn't actually USE polymorphism (every
    # call site has a static target) lowers cleanly.
    #
    # The cascade of libs here is dictated by what
    # PolymorphicOpConversion.cpp actually references.  Each one is
    # noted with the symbol that pulls it in:
    #   FIRTransforms       fir::createPolymorphicOpConversion
    #   FIRCodeGenDialect   fir.cg.XDeclareOp (used by select_type
    #                                          → cf branch lowering)
    #   FIRCodeGen          fir::LLVMTypeConverter::convertBoxTypeAsStruct
    #   FortranSupport      Fortran::common::getFlangFullVersion (codegen
    #                                                              produced-by metadata)
    #   FIROpenACCSupport   fir::acc::registerTransformationalAttrs
    #   FIROpenMPSupport    fir::omp::registerTransformationalAttrs
    #                       (both are defensive dialect registrations
    #                       FIRCodeGen.a calls unconditionally; the
    #                       bridge doesn't use OpenACC or OpenMP, this
    #                       is purely transitive linker-cleanup.)
    FIRTransforms FIRCodeGenDialect FIRCodeGen FortranSupport
    FIROpenACCSupport FIROpenMPSupport
)
foreach(_lib ${FLANG_DIALECT_LIBS})
    find_mlir_lib(${_lib})
endforeach()

# --- Flang include dir --------------------------------------------------------
if(NOT FLANG_INCLUDE_DIR)
    set(FLANG_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}")
endif()

# --- Python + nanobind --------------------------------------------------------
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_STATUS)
if(NOT NB_STATUS EQUAL 0)
    message(FATAL_ERROR "nanobind not found.  Install with: pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# --- Common include paths -----------------------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${FLANG_INCLUDE_DIR}
)
add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_compile_definitions(${LLVM_DEFINITIONS_LIST})

# --- Core: trace + extraction -------------------------------------------------
add_library(hlfir_bridge_core STATIC
    bridge/trace_utils.cpp
    bridge/extract_vars.cpp
    bridge/extract_ast.cpp
    bridge/ast/expressions.cpp
    bridge/ast/assigns.cpp
    bridge/ast/elementals.cpp
    bridge/ast/control_flow.cpp
    bridge/ast/dispatch.cpp
)
target_link_libraries(hlfir_bridge_core PUBLIC
    MLIRIR MLIRParser MLIRSupport MLIRFuncDialect MLIRArithDialect
)

# --- Passes library -----------------------------------------------------------
add_library(hlfir_bridge_passes STATIC
    passes/Passes.cpp
    passes/PropagateShapes.cpp
    passes/InlineAll.cpp
    passes/LowerFirSelectCase.cpp
    passes/FlattenStructs.cpp
    passes/DefaultIntent.cpp
    passes/VerifyNoUnresolvedCalls.cpp
    passes/FoldElementAliases.cpp
    passes/ExpandVectorSubscriptGather.cpp
    passes/ExpandVectorSubscriptScatter.cpp
    passes/RejectPolymorphism.cpp
    passes/RewritePointerAssigns.cpp
    passes/RewriteSequenceAssociation.cpp
    passes/LiftReductionOperands.cpp
    passes/LiftAllocArrayOfRecords.cpp
)
target_link_libraries(hlfir_bridge_passes PUBLIC
    hlfir_bridge_core
    MLIRPass MLIRTransforms
    MLIRControlFlowToSCF
)

# --- Python extension ---------------------------------------------------------
nanobind_add_module(hlfir_bridge NB_STATIC bridge/bridge.cpp)
target_link_libraries(hlfir_bridge PRIVATE
    hlfir_bridge_core
    hlfir_bridge_passes
    -Wl,--whole-archive
    ${FLANG_DIALECT_LIBS}
    -Wl,--no-whole-archive
    ${MLIR_MONO_LIB}
    ${LLVM_MONO_LIB}
)
set_target_properties(hlfir_bridge PROPERTIES
    INSTALL_RPATH "${LLVM_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE)

# --- Install ------------------------------------------------------------------
install(TARGETS hlfir_bridge      LIBRARY DESTINATION .)
