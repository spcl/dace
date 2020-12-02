// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include "hlslib/xilinx/DataPack.h"
#include <type_traits>

namespace dace {

template <typename T, unsigned width>
using vec =
    typename std::conditional<(width > 1), hlslib::DataPack<T, width>, T>::type;

// Don't distinguish aligned and unaligned on FPGA
template <typename T, unsigned width>
using vecu = vec<T, width>; 

} // End namespace dace
