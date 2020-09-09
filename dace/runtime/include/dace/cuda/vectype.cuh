// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
////////////////////////////////////////////////////////////////////////
// Define some operators on vector types

#define DEFINE_EXTTYPE1(T, NAME)                                       \
    struct exttype_##T##_##1 : NAME##1 {                               \
        DACE_HDFI exttype_##T##_##1 operator*(const exttype_##T##_##1 &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = other.x * x;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##1 operator+(const exttype_##T##_##1 &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = other.x + x;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##1 operator-(const exttype_##T##_##1 &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = x - other.x;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##1 operator/(const exttype_##T##_##1 &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = x / other.x;                                    \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##1 operator*(const U &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = other * x;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##1 operator+(const U &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = other + x;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##1 operator-(const U &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = x - other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##1 operator/(const U &other) const {  \
            exttype_##T##_##1 result;                                  \
            result.x = x / other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI T operator[](const U &index) const {                 \
            return x;                                                  \
        }                                                              \
    };
#define DEFINE_EXTTYPE2(T, NAME)                                       \
    struct exttype_##T##_##2 : NAME##2 {                               \
        DACE_HDFI exttype_##T##_##2 operator*(const exttype_##T##_##2 &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = other.x * x;                                    \
            result.y = other.y * y;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##2 operator+(const exttype_##T##_##2 &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = other.x + x;                                    \
            result.y = other.y + y;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##2 operator-(const exttype_##T##_##2 &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = x - other.x;                                    \
            result.y = y - other.y;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##2 operator/(const exttype_##T##_##2 &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = x / other.x;                                    \
            result.y = y / other.y;                                    \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##2 operator*(const U &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = other * x;                                      \
            result.y = other * y;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##2 operator+(const U &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = other + x;                                      \
            result.y = other + y;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##2 operator-(const U &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = x - other;                                      \
            result.y = y - other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##2 operator/(const U &other) const {  \
            exttype_##T##_##2 result;                                  \
            result.x = x / other;                                      \
            result.y = y / other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI T operator[](const U &index) const {                 \
            if (index == U(0)) return x;                               \
            return y;                                                  \
        }                                                              \
    };
#define DEFINE_EXTTYPE3(T, NAME)                                       \
    struct exttype_##T##_##3 : NAME##3 {                               \
        DACE_HDFI exttype_##T##_##3 operator*(const exttype_##T##_##3 &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = other.x * x;                                    \
            result.y = other.y * y;                                    \
            result.z = other.z * z;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##3 operator+(const exttype_##T##_##3 &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = other.x + x;                                    \
            result.y = other.y + y;                                    \
            result.z = other.z + z;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##3 operator-(const exttype_##T##_##3 &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = x - other.x;                                    \
            result.y = y - other.y;                                    \
            result.z = z - other.z;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##3 operator/(const exttype_##T##_##3 &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = x / other.x;                                    \
            result.y = y / other.y;                                    \
            result.z = z / other.z;                                    \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##3 operator*(const U &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = other * x;                                      \
            result.y = other * y;                                      \
            result.z = other * z;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##3 operator+(const U &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = other + x;                                      \
            result.y = other + y;                                      \
            result.z = other + z;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##3 operator-(const U &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = x - other;                                      \
            result.y = y - other;                                      \
            result.z = z - other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##3 operator/(const U &other) const {  \
            exttype_##T##_##3 result;                                  \
            result.x = x / other;                                      \
            result.y = y / other;                                      \
            result.z = z / other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI T operator[](const U &index) const {                 \
            if (index == U(0)) return x;                               \
            else if (index == U(1)) return y;                          \
            return z;                                                  \
        }                                                              \
    };
#define DEFINE_EXTTYPE4(T, NAME)                                       \
    struct exttype_##T##_##4 : NAME##4 {                               \
        DACE_HDFI exttype_##T##_##4 operator*(const exttype_##T##_##4 &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = other.x * x;                                    \
            result.y = other.y * y;                                    \
            result.z = other.z * z;                                    \
            result.w = other.w * w;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##4 operator+(const exttype_##T##_##4 &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = other.x + x;                                    \
            result.y = other.y + y;                                    \
            result.z = other.z + z;                                    \
            result.w = other.w + w;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##4 operator-(const exttype_##T##_##4 &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = x - other.x;                                    \
            result.y = y - other.y;                                    \
            result.z = z - other.z;                                    \
            result.w = w - other.w;                                    \
            return result;                                             \
        }                                                              \
        DACE_HDFI exttype_##T##_##4 operator/(const exttype_##T##_##4 &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = x / other.x;                                    \
            result.y = y / other.y;                                    \
            result.z = z / other.z;                                    \
            result.w = w / other.w;                                    \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##4 operator*(const U &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = other * x;                                      \
            result.y = other * y;                                      \
            result.z = other * z;                                      \
            result.w = other * w;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##4 operator+(const U &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = other + x;                                      \
            result.y = other + y;                                      \
            result.z = other + z;                                      \
            result.w = other + w;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##4 operator-(const U &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = x - other;                                      \
            result.y = y - other;                                      \
            result.z = z - other;                                      \
            result.w = w - other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI exttype_##T##_##4 operator/(const U &other) const {  \
            exttype_##T##_##4 result;                                  \
            result.x = x / other;                                      \
            result.y = y / other;                                      \
            result.z = z / other;                                      \
            result.w = w / other;                                      \
            return result;                                             \
        }                                                              \
        template <typename U>                                          \
        DACE_HDFI T operator[](const U &index) const {                 \
            if (index == U(0)) return x;                               \
            else if (index == U(1)) return y;                          \
            else if (index == U(2)) return z;                          \
            return w;                                                  \
        }                                                              \
    };

#define DEFINE_ALL_EXT_TYPES(T, NAME)                                  \
    DEFINE_EXTTYPE1(T, NAME);                                          \
    DEFINE_EXTTYPE2(T, NAME);                                          \
    DEFINE_EXTTYPE3(T, NAME);                                          \
    DEFINE_EXTTYPE4(T, NAME);
#define DEFINE_TWO_EXT_TYPES(T, NAME)                                  \
    DEFINE_EXTTYPE1(T, NAME);                                          \
    DEFINE_EXTTYPE2(T, NAME);

DEFINE_ALL_EXT_TYPES(int8,   char);
DEFINE_ALL_EXT_TYPES(uint8,  uchar);
DEFINE_ALL_EXT_TYPES(int16,  short);
DEFINE_ALL_EXT_TYPES(uint16, ushort);
DEFINE_ALL_EXT_TYPES(int32,  int);
DEFINE_ALL_EXT_TYPES(uint32, uint);
DEFINE_ALL_EXT_TYPES(int64,  longlong);
DEFINE_TWO_EXT_TYPES(uint64, ulonglong);
DEFINE_ALL_EXT_TYPES(float32,float);
DEFINE_ALL_EXT_TYPES(float64,double);

/////////////////////////////////////////////////////////////////////////////

#define DEFINE_VECTYPE(T, N)                                           \
    template<>                                                         \
        struct _vtype<T, N>                                            \
    {                                                                  \
        typedef exttype_##T##_##N aligned;                             \
        typedef aligned unaligned;                                     \
    };
#define DEFINE_ARRVECTYPE(T, N)                                        \
    template<>                                                         \
    struct _vtype<T, N>                                                \
    {                                                                  \
        typedef T aligned[N];                                          \
        typedef aligned unaligned;                                     \
    };


    DEFINE_VECTYPE(int8, 2);
    DEFINE_VECTYPE(int8, 3);
    DEFINE_VECTYPE(int8, 4);
    DEFINE_VECTYPE(uint8, 2);
    DEFINE_VECTYPE(uint8, 3);
    DEFINE_VECTYPE(uint8, 4);
    DEFINE_VECTYPE(int16, 2);
    DEFINE_VECTYPE(int16, 3);
    DEFINE_VECTYPE(int16, 4);
    DEFINE_VECTYPE(uint16, 2);
    DEFINE_VECTYPE(uint16, 3);
    DEFINE_VECTYPE(uint16, 4);
    DEFINE_VECTYPE(int32, 2);
    DEFINE_VECTYPE(int32, 3);
    DEFINE_VECTYPE(int32, 4);
    DEFINE_ARRVECTYPE(int32, 8);
    DEFINE_VECTYPE(uint32, 2);
    DEFINE_VECTYPE(uint32, 3);
    DEFINE_VECTYPE(uint32, 4);
    DEFINE_ARRVECTYPE(uint32, 8);
    DEFINE_VECTYPE(int64, 2);
    DEFINE_VECTYPE(uint64, 2);
    DEFINE_VECTYPE(float32, 2);
    DEFINE_VECTYPE(float32, 3);
    DEFINE_VECTYPE(float32, 4);
    DEFINE_VECTYPE(float64, 2);
