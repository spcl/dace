// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_INTEROP_H
#define __DACE_INTEROP_H

#include <type_traits>

#include "types.h"

// Various classes to simplify interoperability with python in code converted to
// C++

class range {
 public:
  class iterator {
    friend class range;

   public:
    DACE_HDFI int operator*() const { return i_; }
    DACE_HDFI const iterator &operator++() {
      i_ += s_;
      return *this;
    }
    DACE_HDFI iterator operator++(int) {
      iterator copy(*this);
      i_ += s_;
      return copy;
    }

    DACE_HDFI bool operator==(const iterator &other) const {
      return i_ == other.i_;
    }
    DACE_HDFI bool operator!=(const iterator &other) const {
      return i_ != other.i_;
    }

   protected:
    DACE_HDFI iterator(int start, int skip = 1) : i_(start), s_(skip) {}

   private:
    int i_, s_;
  };

  DACE_HDFI iterator begin() const { return begin_; }
  DACE_HDFI iterator end() const { return end_; }
  DACE_HDFI range(int end) : begin_(0), end_(end) {}
  DACE_HDFI range(int begin, int end) : begin_(begin), end_(end) {}
  DACE_HDFI range(int begin, int end, int skip)
      : begin_(begin, skip), end_(end, skip) {}

 private:
  iterator begin_;
  iterator end_;
};

typedef void *pyobject;

// True when every argument type shares the same floating-point-ness, i.e. they
// are all floating-point or all integral. We reject the mixed case in Min/Max
// because the integer argument truncates the floating-point result (C++14-safe
// variadic recursion; no fold expression).
template <typename... Ts>
struct _dace_minmax_same_kind : std::true_type {};
template <typename T0, typename T1, typename... Ts>
struct _dace_minmax_same_kind<T0, T1, Ts...>
    : std::integral_constant<
          bool, (std::is_floating_point<T0>::value ==
                 std::is_floating_point<T1>::value) &&
                    _dace_minmax_same_kind<T1, Ts...>::value> {};

// Sympy functions. The return type follows ``std::common_type`` (matching the
// lowercase ``min``/``max`` templates) so the result is never narrowed to the
// first argument's type. The static_assert turns the silent int/double mixing
// bug into a clear compile-time error.
#define _DACE_MINMAX_MIXED_MSG                                              \
  "DaCe Min/Max: mixing floating-point and integer arguments is not "      \
  "allowed -- the integer argument truncates the floating-point result. "  \
  "Cast the operands to a common type (e.g. write min(x, 1.0) instead of " \
  "min(x, 1))."

template <typename U, typename... T>
static DACE_HDFI typename std::common_type<U, T...>::type Min(U val,
                                                              T... vals) {
  static_assert(_dace_minmax_same_kind<U, T...>::value, _DACE_MINMAX_MIXED_MSG);
  return min(val, vals...);
}
template <typename U, typename... T>
static DACE_HDFI typename std::common_type<U, T...>::type Max(U val,
                                                              T... vals) {
  static_assert(_dace_minmax_same_kind<U, T...>::value, _DACE_MINMAX_MIXED_MSG);
  return max(val, vals...);
}
template <typename T>
static DACE_HDFI T Abs(T val) {
  return abs(val);
}
template <typename T, typename U>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, U>::type IfExpr(
    bool condition, const T &iftrue, const U &iffalse) {
  return condition ? iftrue : iffalse;
}

#endif  // __DACE_INTEROP_H
