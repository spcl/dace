import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings
from typing import Union

@dace.library.environment
class IPU:

    cmake_minimum_version = None
    cmake_packages = ["IPU"]
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = ["-std=c++11"]
    cmake_link_flags = ["-lpoplar", "-lpoputil", "-lpoplin"]
    headers = ["poplar/Engine.hpp", "poplar/Graph.hpp", "poplar/IPUModel.hpp", "poplin/MatMul.hpp", "poplin/codelets.hpp", "popops/codelets.hpp", "poputil/TileMapping.hpp"]
    state_fields = [
        "// IPUModel APIs",
        "IPUModel ipuModel;",
        "Device device = ipuModel.createDevice();",
        "Target target = device.getTarget();",
        "// Create the Graph object",
        "Graph graph(target);",
        "popops::addCodelets(graph);",
        "poplin::addCodelets(graph);",
        "// Create a control program that is a sequence of steps",
        "Sequence prog;"
        
    ]
    init_code = """
        // IPUINIT.
        // Nothing for now.
    """
    finalize_code = """
        auto engine = Engine{__state->graph, __state->prog, {{"debug.retainDebugInformation", "true"}}};
        engine.load(__state->device);        
        // Run the control program
        std::cout << "Running program\n";
        engine.run(0);
        std::cout << "Program complete\n";
        engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
        return 0;
    """
    dependencies = []

