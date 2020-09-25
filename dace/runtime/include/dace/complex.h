// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_COMPLEX_H
#define __DACE_COMPLEX_H

#include <complex>
#include "types.h"

#ifdef __CUDACC__
    #include <thrust/complex.h>
    #define dace_conj thrust::conj

    template<typename T>
    using cmplx = thrust::complex<T>;
#else
   #define dace_conj std::conj

   template<typename T>
   using cmplx = std::complex<T>;
#endif

// Contains a complex-j class to support the native complex type in Python

namespace dace 
{
    struct complexJ
    {
        int val;
        explicit DACE_HDFI complexJ(int v = 1) : val(v) {}
    };

    static DACE_HDFI int operator*(const complexJ& j1, const complexJ& j2)
    {
        return -j1.val * j2.val;
    }
    template<typename T>
    cmplx<T> DACE_HDFI operator*(const complexJ& j, const T& other)
    {
        return cmplx<T>(T(0), j.val * other);
    }
    template<typename T>
    cmplx<T> DACE_HDFI operator*(const T& other, const complexJ& j)
    {
        return cmplx<T>(T(0), j.val * other);
    }
    template<typename T>
    cmplx<T> DACE_HDFI operator*(const complexJ& j, const cmplx<T>& other)
    {
        return cmplx<T>(T(0), j.val) * other;
    }
    template<typename T>
    cmplx<T> DACE_HDFI operator*(const cmplx<T>& other, const complexJ& j)
    {
        return cmplx<T>(T(0), j.val) * other;
    }
    static DACE_HDFI complexJ operator*(const int& other, const complexJ& j)
    {
        return complexJ(j.val * other);
    }
    static DACE_HDFI complexJ operator*(const complexJ& j, const int& other)
    {
        return complexJ(j.val * other);
    }
    static DACE_HDFI complexJ operator-(const complexJ& j)
    {
        return complexJ(-j.val);
    }
}


// Complex-scalar multiplication functions

template<typename T>
cmplx<T> operator*(const cmplx<T>& a, const int& b) {
    return cmplx<T>(b*a.real(), b*a.imag());
}
template<typename T>
cmplx<T> operator*(const int& a, const cmplx<T>& b) {
    return cmplx<T>(a*b.real(), a*b.imag());
}

#endif  // __DACE_COMPLEX_H
