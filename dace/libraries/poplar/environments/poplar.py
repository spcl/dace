import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings
from typing import Union

@dace.library.environment
class IPU:

    cmake_minimum_version = None
    cmake_packages = ["poplar"] # Find = POPLARConfig.cmake | poplar-config.cmake
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["poplar", "popops", "poplin", "poputil"]
    cmake_compile_flags = []
    cmake_link_flags = [] #-L/software/graphcore/poplar_sdk/3.3.0/poplar-ubuntu_20_04-3.3.0+7857-b67b751185/lib
    headers = [ "../include/poplar_dace_interface.h"]
    state_fields = [
            "// IPUModel APIs",
            "IPUModel ipuModel;",
            "Device device;",
            "Target target;",
            "Graph graph;",
            "Sequence prog;",
    ]
    init_code = """
        __state->device = __state->ipuModel.createDevice();
        __state->target = __state->device.getTarget();
        __state->graph = Graph(__state->target);
        popops::addCodelets(__state->graph);
        poplin::addCodelets(__state->graph);
    """
    finalize_code = """
        auto engine = Engine{__state->graph, __state->prog, {{"debug.retainDebugInformation", "true"}}};
        engine.load(__state->device);
        // Run the control program
        std::cout << "Running program";
        engine.run(0);
        std::cout << "Program complete";
        // engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
        return 0;
    """
    dependencies = []

