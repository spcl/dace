#pragma once

#include <complex>  // std::complex<float>, std::complex<double>

namespace dacelib {

namespace blas {

class BlasConstants {
public:
    static constexpr const void* Complex64Zero() { return &complex64_zero_; }
    static constexpr const void* Complex128Zero() { return &complex128_zero_; }
    static constexpr const void* Complex64Pone() { return &complex64_pone_; }
    static constexpr const void* Complex128Pone() { return &complex128_pone_; }

private:
    static constexpr std::complex<float> complex64_zero_ = std::complex<float>(0.0f, 0.0f);
    static constexpr std::complex<double> complex128_zero_ = std::complex<double>(0.0, 0.0);
    static constexpr std::complex<float> complex64_pone_ = std::complex<float>(1.0f, 0.0f);
    static constexpr std::complex<double> complex128_pone_ = std::complex<double>(1.0, 0.0);
};

}   // namespace blas

}   // namespace dacelib