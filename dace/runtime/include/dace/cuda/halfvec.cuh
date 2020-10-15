// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_HALFVEC_H
#define __DACE_HALFVEC_H

// Support for half-precision vector types in CUDA/HIP

// Scalar functions
DACE_DFI half max(half a, half b) {
    return __hgt(a, b) ? a : b;
}

DACE_DFI half tanh(half val) {
    return __float2half(tanhf(__half2float(val)));
}

DACE_DFI half exp(half val) {
    return hexp(val);
}

DACE_DFI half reciprocal(half val) {
    return hrcp(val);
}

// Vector types
struct __align__(8) half4 {
    enum { ELEMS = 4 };

    half h[ELEMS];
    
    DACE_HDFI half4() {}
    DACE_HDFI half4(const half4& other) {
        #pragma unroll
        for (int i = 0; i < ELEMS; ++i)
            h[i] = other.h[i];
    }

    DACE_HDFI half4& operator=(const half4& other) {
        #pragma unroll
        for (int i = 0; i < ELEMS; ++i)
            h[i] = other.h[i];
        return *this;
    }

    DACE_HDFI half4(const half& x, const half& y, const half& z, 
                    const half& w) {
        h[0] = x;
        h[1] = y;
        h[2] = z;
        h[3] = w;
    }

    DACE_HDFI half4(const half2& xy, const half2& zw) {
        h2<0>() = xy;
        h2<1>() = zw;
    }

    DACE_HDFI 
    static half4 fillall(half value) {
        half4 res;
        #if defined(__CUDA_ARCH__)
            half2 in = __half2half2(value);
            res.h2<0>() = in;
            res.h2<1>() = in;
        #else
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                res.h[i] = value;
            }
        #endif
        return res;
    }

    template <int stride>
    DACE_HDFI 
    static half4 load(half* ptr) {
        half4 res;
        if (stride == 1) {
            res = *(half4*)ptr;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                res.h[i] = ptr[i * stride];
            }
        }
        return res;
    }
    
    template <int stride>
    DACE_HDFI 
    void store(half* ptr) {
        if (stride == 1) {
            *(half4*)ptr = *this;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                ptr[i * stride] = h[i];
            }
        }
    }
    
    DACE_DFI
    void sum(float& res) {
        #pragma unroll
        for (int k = 0; k < ELEMS; k++) {
            res += __half2float(h[k]);
        }
    }
    
    template <int i>
    DACE_HDFI half2& h2() {
        return ((half2*)h)[i];
    }
    
    template <int i>
    DACE_HDFI const half2& h2() const {
        return ((const half2*)h)[i];
    }
    
    DACE_DFI
    half4 exp() {
        half4 res;
        res.h2<0>() = h2exp(h2<0>());
        res.h2<1>() = h2exp(h2<1>());
        return res;
    }
    
    DACE_DFI void operator*=(const half4& v) {
        h2<0>() = __hmul2(h2<0>(), v.h2<0>());
        h2<1>() = __hmul2(h2<1>(), v.h2<1>());
    }
    
    DACE_DFI void operator+=(const half4& v) {
        h2<0>() = __hadd2(h2<0>(), v.h2<0>());
        h2<1>() = __hadd2(h2<1>(), v.h2<1>());
    }
    
    DACE_DFI void operator-=(const half4& v) {
        h2<0>() = __hsub2(h2<0>(), v.h2<0>());
        h2<1>() = __hsub2(h2<1>(), v.h2<1>());
    }
    
    DACE_DFI void operator*=(const half2& v) {
        h2<0>() = __hmul2(h2<0>(), v);
        h2<1>() = __hmul2(h2<1>(), v);
    }
    
    DACE_DFI void operator+=(const half2& v) {
        h2<0>() = __hadd2(h2<0>(), v);
        h2<1>() = __hadd2(h2<1>(), v);
    }
    
    DACE_DFI void operator-=(const half2& v) {
        h2<0>() = __hsub2(h2<0>(), v);
        h2<1>() = __hsub2(h2<1>(), v);
    }
    
    DACE_DFI void operator*=(const half& v) {
        *this *= __half2half2(v);
    }
    
    DACE_DFI void operator+=(const half& v) {
        *this += __half2half2(v);
    }
    
    DACE_DFI void operator-=(const half& v) {
        *this -= __half2half2(v);
    }
};

struct __align__(16) half8 {
    enum { ELEMS = 8 };

    half h[ELEMS];

    DACE_HDFI half8() {}

    DACE_HDFI half8(const half8& other) {
        h2<0>() = other.h2<0>();
        h2<1>() = other.h2<1>();
        h2<2>() = other.h2<2>();
        h2<3>() = other.h2<3>();
    }

    DACE_HDFI half8& operator=(const half8& other) {
        h2<0>() = other.h2<0>();
        h2<1>() = other.h2<1>();
        h2<2>() = other.h2<2>();
        h2<3>() = other.h2<3>();
        return *this;
    }

    DACE_HDFI half8(half a, half b, half c, half d, half e, half f, half g, 
                    half _h) {
        h[0] = a;
        h[1] = b;
        h[2] = c;
        h[3] = d;
        h[4] = e;
        h[5] = f;
        h[6] = g;
        h[7] = _h;
    }

    DACE_HDFI half8(half2 ab, half2 cd, half2 ef, half2 gh) {
        h2<0>() = ab;
        h2<1>() = cd;
        h2<2>() = ef;
        h2<3>() = gh;
    }

    DACE_HDFI 
    static half8 fillall(half value) {
        half8 res;
        #if defined(__CUDA_ARCH__)
            half2 in = __half2half2(value);
            res.h2<0>() = in;
            res.h2<1>() = in;
            res.h2<2>() = in;
            res.h2<3>() = in;
        #else
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                res.h[i] = value;
            }
        #endif
        return res;
    }

    template <int stride>
    DACE_HDFI 
    static half8 load(half* ptr) {
        half8 res;
        if (stride == 1) {
            res = *(half8*)ptr;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                res.h[i] = ptr[i * stride];
            }
        }
        return res;
    }
    
    template <int stride>
    DACE_HDFI 
    void store(half* ptr) {
        if (stride == 1) {
            *(half8*)ptr = *this;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                ptr[i * stride] = h[i];
            }
        }
    }
    
    DACE_DFI
    void sum(float& res) {
        #pragma unroll
        for (int k = 0; k < ELEMS; k++) {
            res += __half2float(h[k]);
        }
    }
    
    template <int i>
    DACE_HDFI half2& h2() {
        return ((half2*)h)[i];
    }
    
    template <int i>
    DACE_HDFI const half2& h2() const {
        return ((const half2*)h)[i];
    }
    
    DACE_DFI
    half8 exp() {
        half8 res;
        res.h2<0>() = h2exp(h2<0>());
        res.h2<1>() = h2exp(h2<1>());
        res.h2<2>() = h2exp(h2<2>());
        res.h2<3>() = h2exp(h2<3>());
        return res;
    }
    
    DACE_DFI void operator*=(const half8& v) {
        h2<0>() = __hmul2(h2<0>(), v.h2<0>());
        h2<1>() = __hmul2(h2<1>(), v.h2<1>());
        h2<2>() = __hmul2(h2<2>(), v.h2<2>());
        h2<3>() = __hmul2(h2<3>(), v.h2<3>());
    }
    
    DACE_DFI void operator+=(const half8& v) {
        h2<0>() = __hadd2(h2<0>(), v.h2<0>());
        h2<1>() = __hadd2(h2<1>(), v.h2<1>());
        h2<2>() = __hadd2(h2<2>(), v.h2<2>());
        h2<3>() = __hadd2(h2<3>(), v.h2<3>());
    }
    
    DACE_DFI void operator-=(const half8& v) {
        h2<0>() = __hsub2(h2<0>(), v.h2<0>());
        h2<1>() = __hsub2(h2<1>(), v.h2<1>());
        h2<2>() = __hsub2(h2<2>(), v.h2<2>());
        h2<3>() = __hsub2(h2<3>(), v.h2<3>());
    }

    DACE_DFI void operator*=(const half4& v) {
        h2<0>() = __hmul2(h2<0>(), v.h2<0>());
        h2<1>() = __hmul2(h2<1>(), v.h2<1>());
        h2<2>() = __hmul2(h2<2>(), v.h2<0>());
        h2<3>() = __hmul2(h2<3>(), v.h2<1>());
    }
    
    DACE_DFI void operator+=(const half4& v) {
        h2<0>() = __hadd2(h2<0>(), v.h2<0>());
        h2<1>() = __hadd2(h2<1>(), v.h2<1>());
        h2<2>() = __hadd2(h2<2>(), v.h2<0>());
        h2<3>() = __hadd2(h2<3>(), v.h2<1>());
    }
    
    DACE_DFI void operator-=(const half4& v) {
        h2<0>() = __hsub2(h2<0>(), v.h2<0>());
        h2<1>() = __hsub2(h2<1>(), v.h2<1>());
        h2<2>() = __hsub2(h2<2>(), v.h2<0>());
        h2<3>() = __hsub2(h2<3>(), v.h2<1>());
    }

    DACE_DFI void operator*=(const half2& v) {
        h2<0>() = __hmul2(h2<0>(), v);
        h2<1>() = __hmul2(h2<1>(), v);
        h2<2>() = __hmul2(h2<2>(), v);
        h2<3>() = __hmul2(h2<3>(), v);
    }
    
    DACE_DFI void operator+=(const half2& v) {
        h2<0>() = __hadd2(h2<0>(), v);
        h2<1>() = __hadd2(h2<1>(), v);
        h2<2>() = __hadd2(h2<2>(), v);
        h2<3>() = __hadd2(h2<3>(), v);
    }
    
    DACE_DFI void operator-=(const half2& v) {
        h2<0>() = __hsub2(h2<0>(), v);
        h2<1>() = __hsub2(h2<1>(), v);
        h2<2>() = __hsub2(h2<2>(), v);
        h2<3>() = __hsub2(h2<3>(), v);
    }
    
    DACE_DFI void operator*=(const half& v) {
        *this *= __half2half2(v);
    }
    
    DACE_DFI void operator+=(const half& v) {
        *this += __half2half2(v);
    }
    
    DACE_DFI void operator-=(const half& v) {
        *this -= __half2half2(v);
    }
};


// Vector functions
DACE_DFI
half4 operator+(half4 a, const half4& b) {
    a += b;
    return a;
}

DACE_DFI
half4 operator-(half4 a, const half4& b) {
    a -= b;
    return a;
}

DACE_DFI
half4 operator*(half4 a, const half4& b) {
    a *= b;
    return a;
}

DACE_DFI
half8 operator+(half8 a, const half8& b) {
    a += b;
    return a;
}

DACE_DFI
half8 operator-(half8 a, const half8& b) {
    a -= b;
    return a;
}

DACE_DFI
half8 operator*(half8 a, const half8& b) {
    a *= b;
    return a;
}


// Vector functions for different lengths

DACE_DFI
half8 operator-(half8 a, half4 b) {
    a -= b;
    return a;
}

DACE_DFI
half8 operator+(half8 a, half4 b) {
    a -= b;
    return a;
}

DACE_DFI
half8 operator*(half8 a, half4 b) {
    a *= b;
    return a;
}


DACE_DFI
half4 operator-(half4 a, half2 b) {
    a -= b;
    return a;
}

DACE_DFI
half4 operator+(half4 a, half2 b) {
    a -= b;
    return a;
}

DACE_DFI
half4 operator*(half4 a, half2 b) {
    a *= b;
    return a;
}

DACE_DFI
half8 operator-(half8 a, half2 b) {
    a -= b;
    return a;
}

DACE_DFI
half8 operator+(half8 a, half2 b) {
    a -= b;
    return a;
}

DACE_DFI
half8 operator*(half8 a, half2 b) {
    a *= b;
    return a;
}



DACE_DFI
half4 operator-(half4 a, half b) {
    a -= b;
    return a;
}

DACE_DFI
half4 operator+(half4 a, half b) {
    a += b;
    return a;
}
DACE_DFI half4 operator+(half a, half4 b) { return b + a; }

DACE_DFI
half4 operator*(half4 a, half b) {
    a *= b;
    return a;
}
DACE_DFI half4 operator*(half a, half4 b) { return b * a; }

DACE_DFI
half8 operator-(half8 a, half b) {
    a -= b;
    return a;
}

DACE_DFI
half8 operator+(half8 a, half b) {
    a += b;
    return a;
}
DACE_DFI half8 operator+(half a, half8 b) { return b + a; }

DACE_DFI
half8 operator*(half8 a, half b) {
    a *= b;
    return a;
}
DACE_DFI half8 operator*(half a, half8 b) { return b * a; }

// Unary mathematical vector operations
#define HALF_VEC_UFUNC(op)                                                 \
DACE_DFI half2 op(half2 x) { return make_half2(op(x.x), op(x.y)); }        \
DACE_DFI half4 op(half4 x) { return half4(op(x.h2<0>()), op(x.h2<1>())); } \
DACE_DFI half8 op(half8 x) {                                               \
    return half8(op(x.h2<0>()), op(x.h2<1>()),                             \
                 op(x.h2<2>()), op(x.h2<3>()));                            \
}

namespace dace { namespace math {
    HALF_VEC_UFUNC(exp)
    HALF_VEC_UFUNC(tanh)
} }

// Vector comparison functions
DACE_DFI half2 max(half2 a, half2 b) {
    return make_half2(max(a.x, b.x), max(a.y, b.y));
}

DACE_DFI half4 max(half4 a, half b) {
    half2 bvec = __half2half2(b);
    return half4(max(a.h2<0>(), bvec), max(a.h2<1>(), bvec));
}
DACE_DFI half4 max(half a, half4 b) { return max(b, a); }

DACE_DFI half4 max(half4 a, half2 b) {
    return half4(max(a.h2<0>(), b), max(a.h2<1>(), b));
}
DACE_DFI half4 max(half2 a, half4 b) { return max(b, a); }

DACE_DFI half4 max(half4 a, half4 b) {
    return half4(max(a.h2<0>(), b.h2<0>()), max(a.h2<1>(), b.h2<1>()));
}

DACE_DFI half8 max(half8 a, half b) {
    half2 bvec = __half2half2(b);
    return half8(max(a.h2<0>(), bvec), 
                 max(a.h2<1>(), bvec),
                 max(a.h2<2>(), bvec),
                 max(a.h2<3>(), bvec));
}
DACE_DFI half8 max(half a, half8 b) { return max(b, a); }

DACE_DFI half8 max(half8 a, half2 b) {
    return half8(max(a.h2<0>(), b), 
                 max(a.h2<1>(), b),
                 max(a.h2<2>(), b),
                 max(a.h2<3>(), b));
}
DACE_DFI half8 max(half2 a, half8 b) { return max(b, a); }

DACE_DFI half8 max(half8 a, half4 b) {
    return half8(max(a.h2<0>(), b.h2<0>()), 
                 max(a.h2<1>(), b.h2<1>()),
                 max(a.h2<2>(), b.h2<0>()),
                 max(a.h2<3>(), b.h2<1>()));
}
DACE_DFI half8 max(half4 a, half8 b) { return max(b, a); }

DACE_DFI half8 max(half8 a, half8 b) {
    return half8(max(a.h2<0>(), b.h2<0>()), 
                 max(a.h2<1>(), b.h2<1>()),
                 max(a.h2<2>(), b.h2<2>()),
                 max(a.h2<3>(), b.h2<3>()));
}

#endif  // __DACE_HALFVEC_H
