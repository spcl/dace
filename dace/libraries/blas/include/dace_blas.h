// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <complex>  // std::complex<float>, std::complex<double>

namespace dace {

namespace blas {

class BlasConstants {
public:
    static BlasConstants &Get() {
      static BlasConstants singleton;
      return singleton;
    }

    const void* Complex64Zero() const { return &complex64_zero_; }
    const void* Complex128Zero() const { return &complex128_zero_; }
    const void* Complex64Pone() const { return &complex64_pone_; }
    const void* Complex128Pone() const { return &complex128_pone_; }

private:
    BlasConstants() = default;

    const std::complex<float> complex64_zero_{0.0f, 0.0f};
    const std::complex<double> complex128_zero_{0.0, 0.0};
    const std::complex<float> complex64_pone_{1.0f, 0.0f};
    const std::complex<double> complex128_pone_{1.0, 0.0};
};

}   // namespace blas

}   // namespace
