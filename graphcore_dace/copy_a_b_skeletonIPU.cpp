#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>


using namespace poplar;
using namespace poplar::program;

using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

// using ::poplar::FLOAT;
// using ::poplar::OptionFlags;
// using ::poplar::Tensor;
// using ::poplar::Graph;
// using ::poplar::Engine;
// using ::poplar::Device;
// using ::poplar::DeviceManager;
// using ::poplar::TargetType;
// using ::poplar::program::Program;
// using ::poplar::program::Sequence;
// using ::poplar::program::Copy;
// using ::poplar::program::Repeat;
// using ::poplar::program::Execute;


const auto NUM_DATA_ITEMS = 1;

auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        } else {
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

auto createGraphAndAddCodelets(const optional<Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());

    // Add our custom codelet, building from CPP source
    // with the given popc compiler options
    // graph.addCodelets({"codelets/SkeletonCodelets.cpp"}, "-O3 -I codelets");

    // Add the codelets for the popops librarys
    // popops::addCodelets(graph);
    return graph;
}

auto buildComputeGraph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs) {
    // Add tensors
    tensors["v1"] = graph.addVariable(poplar::FLOAT, {NUM_DATA_ITEMS}, "v1");
    poputil::mapTensorLinearly(graph, tensors["v1"]);

    tensors["v2"] = graph.addVariable(poplar::FLOAT, {NUM_DATA_ITEMS}, "v2");
    poputil::mapTensorLinearly(graph, tensors["v2"]);   // both v1 v2 will be on same tile

    // real magic happens here
    auto copyprogram =  Copy(tensors["v1"], tensors["v2"]); // tile to tile
    programs["main"] = copyprogram;

    // print_before = program::PrintTensor("v1-debug", v1);
    // programs["print_before"] = print_before;

    // print_after = program::PrintTensor("v2-debug", v2);
    // programs["print_after"] = print_after;

}

auto defineDataStreams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs) {
    auto toIpuStream = graph.addHostToDeviceFIFO("TO_IPU", FLOAT, NUM_DATA_ITEMS);
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", FLOAT, NUM_DATA_ITEMS);

    auto copyToIpuProgramv1 = Copy(toIpuStream, tensors["v1"]); // host->device
    auto copyToIpuProgramv2 = Copy(toIpuStream, tensors["v2"]);

    // print these tensors
    auto copyToHostProgramv1 = Copy(tensors["v1"], fromIpuStream);
    auto copyToHostProgramv2 = Copy(tensors["v2"], fromIpuStream);  // device->host

    // auto printit_v1  = PrintTensor("v1-debug", tensors["v1"]);
    // auto printit_v2  = PrintTensor("v2-debug", tensors["v2"]);    
    // auto printit_v1_after  = PrintTensor("v1-debug-after", tensors["v1"]);
    // auto printit_v2_after  = PrintTensor("v2-debug-after", tensors["v2"]);
    // programs["print_v1_before"] = printit_v1;
    // programs["print_v2_before"] = printit_v2;
    // programs["print_v1_after"] = printit_v1_after;
    // programs["print_v2_after"] = printit_v2_after;

    programs["copy_to_ipu_v1"] = copyToIpuProgramv1;
    programs["copy_to_ipu_v2"] = copyToIpuProgramv2;
    programs["copy_to_host_v1"] = copyToHostProgramv1;
    programs["copy_to_host_v2"] = copyToHostProgramv2;

}

auto serializeGraph(const Graph &graph) {
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
    graphSerOfs.close();
}

void print_data(std::vector<float> &v1_host, std::vector<float>& v2_host) {
        std::cout << "v1: ";
    for (auto i = 0; i < NUM_DATA_ITEMS; i++) {
        std::cout << v1_host[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "v2: ";
    for (auto i = 0; i < NUM_DATA_ITEMS; i++) {
        std::cout << v2_host[i] << " ";
    }
    std::cout << std::endl;

}

int main(int argc, char *argv[]) {
    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    auto device = getIpuDevice(1);
    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);


    std::cout << "STEP 3: Building the compute graph" << std::endl;
    auto tensors = map<string, Tensor>{};
    auto programs = map<string, Program>{};
    buildComputeGraph(graph, tensors, programs);

    std::cout << "STEP 4: Define data streams" << std::endl;
    defineDataStreams(graph, tensors, programs);

    std::cout << "STEP 5: Create engine and compile graph" << std::endl;
    auto ENGINE_OPTIONS = OptionFlags{
            {"target.saveArchive",                "archive.a"},
            {"autoReport.all",                    "true"},
            {"autoReport.outputSerializedGraph",  "true"},
    };

    auto programIds = map<string, int>();
    auto programsList = vector<Program>(programs.size());
    int index = 0;
    for (auto &nameToProgram: programs) {
        programIds[nameToProgram.first] = index;
        programsList[index] = nameToProgram.second;
        index++;
    }
    auto engine = Engine(graph, programsList, ENGINE_OPTIONS);

    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*device);
    engine.enableExecutionProfiling();

    std::cout << "STEP 7: Attach data streams(host to device data)" << std::endl;
    auto v1_host = vector<float>(NUM_DATA_ITEMS, 100.0f); // v1 = 1
    auto v2_host = vector<float>(NUM_DATA_ITEMS, 0.0f); // v2 = 0
    vector<float> vector_stream_in;
    vector_stream_in.insert(vector_stream_in.end(), v1_host.begin(),
                            v1_host.end());
    vector_stream_in.insert(vector_stream_in.end(), v2_host.begin(), v2_host.end());

    auto v1_host_out = vector<float>(NUM_DATA_ITEMS, 0.0f); // Output buffer for v1
    auto v2_host_out = vector<float>(NUM_DATA_ITEMS, 0.0f); // Output buffer for v2
    vector<float> vector_stream_out;
    vector_stream_out.insert(vector_stream_out.end(), v1_host_out.begin(), v1_host_out.end());
    vector_stream_out.insert(vector_stream_out.end(), v2_host_out.begin(), v2_host_out.end());

    // print before
    std::cout << "\nBefore: \n";
    print_data(v1_host, v2_host);   

    engine.connectStream("TO_IPU", vector_stream_in.data(), vector_stream_in.data() + vector_stream_in.size());
    engine.connectStream("FROM_IPU", vector_stream_out.data(), vector_stream_out.data() + vector_stream_out.size());

    std::cout << "\nSTEP 8: Run programs" << std::endl;
    engine.run(programIds["copy_to_ipu_v1"]); // Copy to IPU
    engine.run(programIds["copy_to_ipu_v2"]); // Copy to IPU
    // engine.run(programIds["print_v1_before"]); // Print v1
    // engine.run(programIds["print_v2_before"]); // Print v2
    engine.run(programIds["main"]); // Main program
    // engine.run(programIds["print_v1_after"]); // Print v1
    // engine.run(programIds["print_v2_after"]); // Print v2
    engine.run(programIds["copy_to_host_v1"]); // Copy from IPU
    engine.run(programIds["copy_to_host_v2"]); // Copy from IPU

    std::cout << "\nSTEP 9: Check results after\n" << std::endl;
    v1_host_out.assign(vector_stream_out.begin(), vector_stream_out.begin() + NUM_DATA_ITEMS);
    v2_host_out.assign(vector_stream_out.begin() + NUM_DATA_ITEMS, vector_stream_out.end());
    print_data(v1_host_out, v2_host_out);

    return EXIT_SUCCESS;
}

