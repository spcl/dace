#pragma once

#include <complex>  // std::complex<float>, std::complex<double>

namespace dacelib {

namespace blas {

class BlasConstants {
    constexpr void* Complex64Zero() { return &complex64_zero_; }
    constexpr void* Complex128Zero() const { return &complex128_zero_; }
    constexpr void* Complex64Pone() const { return &complex64_pone_; }
    constexpr void* Complex128Pone() const { return &complex128_pone_; }

private:
    constexpr complex64_zero_ = std::complex<float>(0.0f, 0.0f);
    constexpr complex128_zero_ = std::complex<double>(0.0, 0.0);
    constexpr complex64_pone_ = std::complex<float>(1.0f, 0.0f);
    constexpr complex128_pone_ = std::complex<double>(1.0, 0.0);
};

}   // namespace blas

}   // namespace dacelib