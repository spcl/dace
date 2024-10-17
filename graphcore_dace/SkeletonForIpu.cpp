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


using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

using ::poplar::FLOAT;
using ::poplar::OptionFlags;
using ::poplar::Tensor;
using ::poplar::Graph;
using ::poplar::Engine;
using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::TargetType;
using ::poplar::program::Program;
using ::poplar::program::Sequence;
using ::poplar::program::Copy;
using ::poplar::program::Repeat;
using ::poplar::program::Execute;


const auto NUM_DATA_ITEMS = 10;
const auto HOW_MUCH_TO_ADD = 2.0f;
const auto NUM_TILES_IN_GC = 10;


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
    graph.addCodelets({"codelets/SkeletonCodelets.cpp"}, "-O3 -I codelets");

    // Add the codelets for the popops librarys
    popops::addCodelets(graph);
    return graph;
}

auto buildComputeGraph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int numTiles) {
    // Add tensors
    tensors["data"] = graph.addVariable(poplar::FLOAT, {NUM_DATA_ITEMS}, "data");
    poputil::mapTensorLinearly(graph, tensors["data"]);


    // Add programs and wire up data
    const auto NumElemsPerTile = NUM_DATA_ITEMS / numTiles;
    auto cs = graph.addComputeSet("loopBody");
    for (auto tileNum = 0; tileNum < numTiles; tileNum++) {
        const auto sliceEnd = std::min((tileNum + 1) * NumElemsPerTile, (int)   NUM_DATA_ITEMS);
        const auto sliceStart = tileNum * NumElemsPerTile;

        auto v = graph.addVertex(cs, "SkeletonVertex", {
                {"data", tensors["data"].slice(sliceStart, sliceEnd)}
        });
        graph.setInitialValue(v["howMuchToAdd"], HOW_MUCH_TO_ADD);
        // graph.setPerfEstimate(v, 100); // Ideally you'd get this as right as possible
        graph.setTileMapping(v, tileNum);
    }
    auto executeIncrementVertex = Execute(cs);

    // auto mainProgram = Repeat(1, executeIncrementVertex, "repeat1x");
    programs["main"] = executeIncrementVertex; // Program 0 will be the main program
}

auto defineDataStreams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs) {
    auto toIpuStream = graph.addHostToDeviceFIFO("TO_IPU", FLOAT, NUM_DATA_ITEMS);
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", FLOAT, NUM_DATA_ITEMS);

    auto copyToIpuProgram = Copy(toIpuStream, tensors["data"]);
    auto copyToHostProgram = Copy(tensors["data"], fromIpuStream);

    programs["copy_to_ipu"] = copyToIpuProgram;
    programs["copy_to_host"] = copyToHostProgram;
}

auto serializeGraph(const Graph &graph) {
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
    graphSerOfs.close();
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
    buildComputeGraph(graph, tensors, programs, NUM_TILES_IN_GC /* numTiles */);

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


    std::cout << "STEP 7: Attach data streams" << std::endl;
    auto hostData = vector<float>(NUM_DATA_ITEMS, 1.0f);
    // print before
    std::cout << "\nBefore: ";
    for (auto i = 0; i < NUM_DATA_ITEMS; i++) {
        std::cout << hostData[i] << " ";
    }
    std::cout << "\nHow much to add: " << HOW_MUCH_TO_ADD << std::endl;
    engine.connectStream("TO_IPU", hostData.data());
    engine.connectStream("FROM_IPU", hostData.data());

    std::cout << "\nSTEP 8: Run programs" << std::endl;
    engine.run(programIds["copy_to_ipu"]); // Copy to IPU
    engine.run(programIds["main"]); // Main program
    engine.run(programIds["copy_to_host"]); // Copy from IPU

    std::cout << "\nSTEP 9: Check results" << std::endl;
    // print hostData to see the result
    for (auto i = 0; i < NUM_DATA_ITEMS; i++) {
        std::cout << hostData[i] << " ";
    }
    

    std::cout << "\nSTEP 10: Capture debug and profile info" << std::endl;
    // serializeGraph(graph);
    // engine.printProfileSummary(std::cout,
                            //    OptionFlags{{"showExecutionSteps", "false"}});

    return EXIT_SUCCESS;
}
