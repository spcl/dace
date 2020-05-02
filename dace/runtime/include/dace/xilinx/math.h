/**
    Additonal math function support for Xilinx
*/

#pragma once

// Make abs work on Xilinx with vector types and compatible with Intel
template <typename T>
DACE_CONSTEXPR DACE_HDFI T fabs(const T& a) {
    return (a < 0) ? a * -1 : a;
}