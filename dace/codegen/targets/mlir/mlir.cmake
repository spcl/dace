# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
set(DACE_MLIR_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/../src/mlir)
set(DACE_MLIR_WORK_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen/mlir)

file(GLOB DACE_MLIR_SRC_FILES ${DACE_MLIR_SRC_DIR}/*.mlir)

foreach(DACE_MLIR_SRC_FILE ${DACE_MLIR_SRC_FILES})
  file(MAKE_DIRECTORY ${DACE_MLIR_WORK_DIR})

  get_filename_component(DACE_MLIR_NAME ${DACE_MLIR_SRC_FILE} NAME_WLE)
  set(DACE_MLIR_MLIR_FILE ${DACE_MLIR_WORK_DIR}/${DACE_MLIR_NAME}.mlir)
  set(DACE_MLIR_LLVM_FILE ${DACE_MLIR_WORK_DIR}/${DACE_MLIR_NAME}.ll)
  set(DACE_MLIR_OBJECT_FILE ${DACE_MLIR_WORK_DIR}/${DACE_MLIR_NAME}.o)

  add_custom_command(
    OUTPUT ${DACE_MLIR_MLIR_FILE}
    COMMAND mlir-opt --lower-host-to-llvm ${DACE_MLIR_SRC_FILE} > ${DACE_MLIR_MLIR_FILE}
    DEPENDS ${DACE_MLIR_SRC_FILE}
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${DACE_MLIR_LLVM_FILE}
    COMMAND mlir-translate --mlir-to-llvmir ${DACE_MLIR_MLIR_FILE} > ${DACE_MLIR_LLVM_FILE}
    DEPENDS ${DACE_MLIR_MLIR_FILE}
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${DACE_MLIR_OBJECT_FILE}
    COMMAND llc -relocation-model=pic -filetype=obj ${DACE_MLIR_LLVM_FILE} -o ${DACE_MLIR_OBJECT_FILE}
    DEPENDS ${DACE_MLIR_LLVM_FILE}
    VERBATIM
  )

  set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_MLIR_OBJECT_FILE})
endforeach()
