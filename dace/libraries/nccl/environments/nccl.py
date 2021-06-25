# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class NCCL:

    cmake_minimum_version = None
    cmake_packages = ["NCCL"]
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["$nccl"]
    cmake_compile_flags = []
    cmake_link_flags = []

    headers = ["../inlcude/dace_nccl.h"]
    state_fields = ["dace::nccl::NCCLHandle nccl_handle;"]
    init_code = """\
        const int nGPUs = __state->gpu_context->size();
        _state.nccl_handle = NcclHandle(nGPUs);
        """
    # gpu_init_code = """\
    #     int nGPUs = 3;
    #     int[] gpu_ids = {0,1,2,};
    #     _state.nccl_handle = NcclHandle(nGPUs, gpu_ids);
    #     """


    finalize_code = """"""
    dependencies = []

#     @staticmethod
#     def handle_setup_code(node):
#         code = """\
# ncclComm_t comms[];
# ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) """