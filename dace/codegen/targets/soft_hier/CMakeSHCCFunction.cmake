# Define a function to add an executable compiled by the SHCC compiler
function(add_shcc_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_compile_options(${target_name} PRIVATE "${CMAKE_SHCC_FLAGS}")
    target_link_options(${target_name} PRIVATE "${CMAKE_SHCC_FLAGS}")
    message(STATUS "Added SHCC executable target: ${target_name} with sources: ${ARGN}")
endfunction()

# Define a function to add custom SHCC library
function(add_shcc_library target_name)
    add_library(${target_name} STATIC ${ARGN})
    target_compile_options(${target_name} PRIVATE "${CMAKE_SHCC_FLAGS}")
    message(STATUS "Added SHCC library target: ${target_name} with sources: ${ARGN}")
endfunction()
