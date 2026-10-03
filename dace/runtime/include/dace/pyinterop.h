// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_INTEROP_H
#define __DACE_INTEROP_H

#include <type_traits>

#include "types.h"

// Various classes to simplify interoperability with python in code converted to C++

class range
{
public:
    class iterator
    {
        friend class range;
    public:
        DACE_HDFI int operator *() const { return i_; }
        DACE_HDFI const iterator &operator ++() { i_ += s_; return *this; }
        DACE_HDFI iterator operator ++(int) { iterator copy(*this); i_ += s_; return copy; }

        DACE_HDFI bool operator ==(const iterator &other) const { return i_ == other.i_; }
        DACE_HDFI bool operator !=(const iterator &other) const { return i_ != other.i_; }

    protected:
        DACE_HDFI iterator(int start, int skip = 1) : i_(start), s_(skip) { }

    private:
        int i_, s_;
    };

    DACE_HDFI iterator begin() const { return begin_; }
    DACE_HDFI iterator end() const { return end_; }
    DACE_HDFI range(int end) : begin_(0), end_(end) {}
    DACE_HDFI range(int begin, int end) : begin_(begin), end_(end) {}
    DACE_HDFI range(int begin, int end, int skip) : begin_(begin, skip), end_(end, skip) {}
private:
    iterator begin_;
    iterator end_;
};

typedef void *pyobject;

// Whether a CUDA half is among the arguments
template <typename... Ts>
struct _dace_has_half
#ifdef __CUDACC__
    : std::disjunction<std::is_same<Ts, dace::float16>...> {};
#else
    : std::false_type {};
#endif

// Sympy functions
template <typename U, typename... T>
static DACE_HDFI auto Min(U val, T... vals) {
    if constexpr (_dace_has_half<U, T...>::value) {
        using R = typename std::common_type<U, T...>::type;
        return min(R(val), R(vals)...);
    } else {
        return U(min(val, vals...));
    }
}
template <typename U, typename... T>
static DACE_HDFI auto Max(U val, T... vals) {
    if constexpr (_dace_has_half<U, T...>::value) {
        using R = typename std::common_type<U, T...>::type;
        return max(R(val), R(vals)...);
    } else {
        return U(max(val, vals...));
    }
}
// Deduced, not ``T``: ``abs`` of a complex value is real, so pinning the return to the argument
// type turns ``Abs(z)`` back into a complex and every comparison on it loses its candidate.
template <typename T>
static DACE_HDFI auto Abs(T val) {
    return abs(val);
}
template <typename T, typename U>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, U>::type IfExpr(bool condition, const T& iftrue, const U& iffalse)
{
    // Both arms in the result type: ``c ? h : 1.0`` with a half arm is ambiguous otherwise.
    using R = typename std::common_type<T, U>::type;
    return condition ? R(iftrue) : R(iffalse);
}

#endif  // __DACE_INTEROP_H
