import dace.library


@dace.library.environment
class cuDNN:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cudnn"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["cudnn.h"]
    init_code ='''
    #define checkCUDNN(expression)                                   \\
    {                                                                \\
      cudnnStatus_t status = (expression);                           \\
      if (status != CUDNN_STATUS_SUCCESS) {                          \\
        printf(\"%d: %s\\n\", __LINE__, cudnnGetErrorString(status));\\
      }                                                              \\
    }
    '''
    finalize_code = ""