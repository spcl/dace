// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
/**
    Support for additional math operators on Xilinx
*/

#pragma once

// fabs support for xilinx
template <typename T, int vector_length>
DACE_HDFI hlslib::DataPack<T, vector_length> fabs(const hlslib::DataPack<T, vector_length>& a) {
    hlslib::DataPack<T, vector_length> res;
    for (int i = 0; i < vector_length; ++i) {
        #pragma HLS UNROLL
        const auto elem = a[i];
        res[i] = elem < 0 ? -elem : elem;
    }
    return res;
}
