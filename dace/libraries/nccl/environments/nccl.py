# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class NCCL:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_files = ["../FindNCCL.cmake"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []

    headers = ["../include/dace_nccl.h"]
    state_fields = ["std::unordered_map<int, ncclComm_t> *ncclCommunicators;"]
    init_code = """\
        const int nGPUs = __state->gpu_context->size();
        __state->ncclCommunicators = new std::unordered_map<int, ncclComm_t> {nGPUs};
        int gpu_ids[nGPUs];
        for (int i = 0; i < nGPUs; i++){
            gpu_ids[i] = i;
        }
        ncclComm_t comms[nGPUs];
        dace::nccl::CheckNcclError(ncclCommInitAll(comms, nGPUs, gpu_ids));
        for (int i = 0; i< nGPUs; i++){
            __state->ncclCommunicators->insert({gpu_ids[i], comms[i]});
        }
        """

    finalize_code = """
        const int nGPUs = __state->ncclCommunicators->size();
        for (int i = 0; i < nGPUs; i++){
                dace::nccl::CheckNcclError(ncclCommDestroy(__state->ncclCommunicators->at(i)));
            }
        delete __state->ncclCommunicators;
    """
    dependencies = []
