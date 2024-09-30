// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <iostream>
#include <unordered_map>

// file headers
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <map>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using ::std::map;
using ::std::optional;
using ::std::string;
using ::std::vector;

using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::Engine;
using ::poplar::FLOAT;
using ::poplar::Graph;
using ::poplar::OptionFlags;
using ::poplar::TargetType;
using ::poplar::Tensor;
using ::poplar::program::Copy;
using ::poplar::program::Program;
using ::poplar::program::Execute;
using ::poplar::program::Repeat;

// Constants
const auto NUM_DATA_ITEMS = 200000;

// Struct
struct dace_poplar_context {
    optional<Device> device;
    Graph graph;
    map<string, Tensor> tensors;
    map<string, Program> programs;
    OptionFlags engineOptions;
    map<string, int> programIds;
    vector<Program> programsList;
    vector<float> hostData;
};