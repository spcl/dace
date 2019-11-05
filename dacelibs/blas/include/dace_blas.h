#pragma once

#include <cstddef>    // size_t
#include <complex>  // std::complex<float>, std::complex<double>

namespace dacelib {

namespace blas {

template <typename Key = size_t>
class BlasHelper {
    void const* Complex64Zero() const { return &complex64_zero_; }
    void const* Complex128Zero() const { return &complex128_zero_; }
    void const* Complex64Pone() const { return &complex64_pone_; }
    void const* Complex128Pone() const { return &complex128_pone_; }

private:
    BlasHelper() {
        complex64_zero_ = std::complex<float>(0.0f, 0.0f);
        complex128_zero_ = std::complex<double>(0.0, 0.0);
        complex64_pone_ = std::complex<float>(1.0f, 0.0f);
        complex128_pone_ = std::complex<double>(1.0, 0.0);
    }

    std::complex<float> complex64_zero_;
    std::complex<double> complex128_zero_;
    std::complex<float> complex64_pone_;
    std::complex<double> complex128_pone_;
};

}   // namespace blas

}   // namespace dacelib