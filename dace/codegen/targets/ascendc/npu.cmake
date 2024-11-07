# npu
file(GLOB KERNEL_FILES
    ${CMAKE_SOURCE_DIR}/*.cpp
)
list(REMOVE_ITEM KERNEL_FILES "${CMAKE_SOURCE_DIR}/main.cpp")
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE CCE)

file(GLOB MAIN_FILES
    ${CMAKE_SOURCE_DIR}/main.cpp
)
set_source_files_properties(${MAIN_FILES} PROPERTIES LANGUAGE CCE)

# ===================================================================
# exe mode: build a executable directly
add_executable(${smoke_testcase}_npu
    ${KERNEL_FILES}
    ${MAIN_FILES}
)

target_compile_options(${smoke_testcase}_npu PRIVATE
    -O2
    -std=c++17
)

target_compile_features(${smoke_testcase}_npu PUBLIC cxx_std_17)

target_compile_definitions(${smoke_testcase}_npu PRIVATE
    TILING_KEY_VAR=0
)

target_link_libraries(${smoke_testcase}_npu PRIVATE
    acl_cblas
)

set_target_properties(${smoke_testcase}_npu PROPERTIES
    OUTPUT_NAME ${smoke_testcase}_npu
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ===================================================================
# so mode: build a shared library first, and dynamic link to build a executable
file(GLOB KERNEL_FILES
    ${CMAKE_SOURCE_DIR}/*.cpp
)
list(REMOVE_ITEM KERNEL_FILES "${CMAKE_SOURCE_DIR}/main.cpp")

add_library(ascendc_kernels SHARED
    ${KERNEL_FILES}
)

target_compile_definitions(ascendc_kernels PRIVATE
    TILING_KEY_VAR=0
)

target_compile_options(ascendc_kernels PRIVATE
    -O2
    -std=c++17
)

target_compile_features(ascendc_kernels PUBLIC cxx_std_17)


set_target_properties(ascendc_kernels PROPERTIES
    OUTPUT_NAME ascendc_kernels
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
install(TARGETS ascendc_kernels
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}
)

# ===================================================================
add_executable(${smoke_testcase}_lib_npu
    ${MAIN_FILES}
)

target_compile_options(${smoke_testcase}_lib_npu PRIVATE
    -O2
    -std=c++17
)

target_compile_features(${smoke_testcase}_lib_npu PUBLIC cxx_std_17)

target_link_directories(${smoke_testcase}_lib_npu PRIVATE
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(${smoke_testcase}_lib_npu PRIVATE
        -Wl,--as-needed
    )
endif()

target_link_libraries(${smoke_testcase}_lib_npu PRIVATE
    ascendc_kernels
    acl_cblas
)

# add_dependencies(${smoke_testcase}_lib_npu ${smoke_testcase})
set_target_properties(${smoke_testcase}_lib_npu PROPERTIES
    OUTPUT_NAME ${smoke_testcase}_lib_npu
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
