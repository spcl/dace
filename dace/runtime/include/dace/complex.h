#ifndef __DACE_COMPLEX_H
#define __DACE_COMPLEX_H

#include <complex>

#ifdef __CUDACC__
    #include <thrust/complex.h>
    #define dace_conj thrust::conj
#else
   #define dace_conj std::conj
#endif

// Contains a complex-j class to support the native complex type in Python

namespace dace 
{
    struct complexJ
    {
        int val;
        explicit complexJ(int v = 1) : val(v) {}
    };

    static inline int operator*(const complexJ& j1, const complexJ& j2)
    {
        return -j1.val * j2.val;
    }
    template<typename T>
    std::complex<T> operator*(const complexJ& j, const T& other)
    {
        return std::complex<T>(T(0), j.val * other);
    }
    template<typename T>
    std::complex<T> operator*(const T& other, const complexJ& j)
    {
        return std::complex<T>(T(0), j.val * other);
    }
    static inline complexJ operator*(const int& other, const complexJ& j)
    {
        return complexJ(j.val * other);
    }
    static inline complexJ operator*(const complexJ& j, const int& other)
    {
        return complexJ(j.val * other);
    }
    static inline complexJ operator-(const complexJ& j)
    {
        return complexJ(-j.val);
    }
}


// Complex-scalar multiplication functions

template<typename T>
std::complex<T> operator*(const std::complex<T>& a, const int& b) {
    return std::complex<T>(b*a.real(), b*a.imag());
}
template<typename T>
std::complex<T> operator*(const int& a, const std::complex<T>& b) {
    return std::complex<T>(a*b.real(), a*b.imag());
}

#endif  // __DACE_COMPLEX_H
