// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#include <dace/dace.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <typeinfo>

struct add_wcr {
  template <typename T>
  static void resolve(T *ptr, const T &value) {
    *ptr += value;
  }
};

#define AW 1000
#define AH 1000

int main(int, char **) {
  /////////////////////////////////////////////////////////
  // Type checks
  static_assert(std::is_pod<dace::vec<float, 1> >::value == true,
                "Length-1 float vector should be POD");

  static_assert(std::is_scalar<dace::vec<double, 1> >::value == true,
                "Length-1 double vector should be scalar");

  static_assert(std::is_scalar<dace::vec<float, 2> >::value == false,
                "Length-2 float vector should not be scalar");

  static_assert(
      std::is_scalar<dace::vec<std::complex<float>, 1> >::value == false,
      "Length-1 complex vector should not be scalar");

  static_assert(std::is_pod<dace::vec<std::complex<float>, 1> >::value == false,
                "Length-1 complex vector should not be POD");

  /////////////////////////////////////////////////////////
  // Streams
  dace::Stream<uint64_t> s;
  s.push(1);
  std::atomic<uint64_t> result(0ULL);

  // Stream consume (async, but useless operation)
  dace::Consume<1>::consume(s, 8, [&](int /*pe*/, uint64_t &u) {
    if (u < 256ULL) {
      result += u;
      s.push(2 * u);
      s.push(2 * u + 1);
    }
  });

  assert(result == 255 * 128);

  // Stream with array as a sink
  double *bla2 = new double[10];
  memset(bla2, 0xAE, sizeof(double) * 10);
  dace::ArrayStreamView<double> s_bla2(bla2);
  dace::vec<double, 2> vec = dace::vec<double, 2>{3.0, 2.0};
  dace::vec<double, 2> vec_arr[2] = {0};

  s_bla2.push<2>(vec);
  assert(bla2[0] == 3.0);
  assert(bla2[1] == 2.0);

  s_bla2.push<2>(vec_arr, 2);
  assert(bla2[2] == 0.0);
  assert(bla2[3] == 0.0);
  assert(bla2[4] == 0.0);
  assert(bla2[5] == 0.0);

  s_bla2.push(9.0);
  assert(bla2[6] == 9.0);

  delete[] bla2;

  /////////////////////////////////////////////////////////
  // Integer power (dace::math::ipow). Regression for the exponent-0 case,
  // which must be the multiplicative identity 1 (the loop used to start at
  // `a` and skip b==0, returning `a` instead of 1).
  assert(dace::math::ipow(7, 0u) == 1);     // <-- the bug: was 7
  assert(dace::math::ipow(7, 1u) == 7);
  assert(dace::math::ipow(2, 3u) == 8);
  assert(dace::math::ipow(5, 2u) == 25);
  assert(dace::math::ipow(1, 0u) == 1);
  assert(dace::math::ipow(0, 0u) == 1);     // 0**0 == 1 by this convention
  assert(dace::math::ipow(0, 3u) == 0);
  assert(dace::math::ipow(3.0, 0u) == 1.0);
  assert(dace::math::ipow(2.0, 4u) == 16.0);

  printf("Success!\n");
  return 0;
}
