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
    DACE_HDFI const iterator& operator++() {
      i_ += s_;
      return *this;
    }
    DACE_HDFI iterator operator++(int) {
      iterator copy(*this);
      i_ += s_;
      return copy;
    }

    DACE_HDFI bool operator==(const iterator& other) const { return i_ == other.i_; }
    DACE_HDFI bool operator!=(const iterator& other) const { return i_ != other.i_; }

   protected:
    DACE_HDFI iterator(int start, int skip = 1) : i_(start), s_(skip) {}

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

// Sympy functions. The return type follows ``std::common_type`` (matching the lowercase ``min``/``max``
// templates) so the result is never narrowed to the first argument's type; operand kinds must not mix
// (``_dace_minmax_same_kind`` in math.h).
template <typename U, typename... T>
static DACE_HDFI typename std::common_type<U, T...>::type Min(U val, T... vals) {
  static_assert(_dace_minmax_same_kind<U, T...>::value, _DACE_MINMAX_MIXED_MSG);
  return min(val, vals...);
}
template <typename U, typename... T>
static DACE_HDFI typename std::common_type<U, T...>::type Max(U val, T... vals) {
  static_assert(_dace_minmax_same_kind<U, T...>::value, _DACE_MINMAX_MIXED_MSG);
  return max(val, vals...);
}
// Deduced, not ``T``: ``abs`` of a complex value is real, so pinning the return to the argument
// type turns ``Abs(z)`` back into a complex and every comparison on it loses its candidate.
template <typename T>
static DACE_HDFI auto Abs(T val) {
    return abs(val);
}
template <typename T, typename U>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, U>::type IfExpr(bool condition, const T& iftrue,
                                                                      const U& iffalse) {
  // Both arms in the result type: ``c ? h : 1.0`` with a half arm is ambiguous otherwise.
  using R = typename std::common_type<T, U>::type;
  return condition ? R(iftrue) : R(iffalse);
}

#endif  // __DACE_INTEROP_H
