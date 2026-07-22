// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Host-side ``dace::bfloat16`` conformance test, the counterpart of
// tests/cpp/half_ops_test.cpp. Driven by tests/bfloat16_ops_cpp_test.py.
//
// bfloat16 is the leading 16 bits of an IEEE-754 binary32: same 8-bit exponent,
// mantissa truncated from 23 explicit bits to 7. There is no native-type branch to
// switch between (see the commentary in <dace/types.h>), so unlike the float16 test
// this is compiled once per ISA configuration rather than once per backend -- what
// varies is the compiler's freedom to vectorize and reassociate, not the algorithm.
//
// Prints "FAIL: ..." for each failure and returns non-zero if there was any.

#include <dace/types.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

using dace::bfloat16;

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_BITS(expr, want)                                                 \
  do {                                                                         \
    uint16_t got_ = raw(expr);                                                 \
    if (got_ != (uint16_t)(want)) {                                            \
      printf("FAIL: %s:%d: %s -> 0x%04x, want 0x%04x\n", __FILE__, __LINE__,   \
             #expr, got_, (unsigned)(want));                                   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static uint16_t raw(bfloat16 x) {
  uint16_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

static uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

static float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// ---------------------------------------------------------------------------
// Storage: size, alignment and bit layout. Identical to what every serialized
// DaCe array and every host<->device copy assumes, and -- critically -- identical
// to __nv_bfloat16 / __hip_bfloat16, which are what dace::bfloat16 becomes inside
// a .cu / .hip translation unit of the very same program.
// ---------------------------------------------------------------------------
static void test_layout() {
  CHECK(sizeof(bfloat16) == 2);
  CHECK(alignof(bfloat16) == 2);
  CHECK(std::is_trivially_copyable<bfloat16>::value);
  CHECK(std::is_standard_layout<bfloat16>::value);

  struct Packed {
    bfloat16 a;
    bfloat16 b;
  };
  CHECK(sizeof(Packed) == 4);
  bfloat16 arr[8];
  CHECK(sizeof(arr) == 16);

  // bf16 encodings. Each is simply the top half of the float32 encoding, which is
  // what makes these checkable by inspection: 1.0f is 0x3f800000, so 0x3f80.
  CHECK_BITS(bfloat16(0.0f), 0x0000);
  CHECK_BITS(bfloat16(-0.0f), 0x8000);
  CHECK_BITS(bfloat16(1.0f), 0x3f80);
  CHECK_BITS(bfloat16(-1.0f), 0xbf80);
  CHECK_BITS(bfloat16(2.0f), 0x4000);
  CHECK_BITS(bfloat16(-2.5f), 0xc020);
  CHECK_BITS(bfloat16(0.5f), 0x3f00);
  CHECK_BITS(bfloat16(INFINITY), 0x7f80);
  CHECK_BITS(bfloat16(-INFINITY), 0xff80);

  // The exponent range is the SAME as float32's -- the headline difference from
  // binary16, which overflows at 65504. Nothing in normal float range overflows.
  CHECK_BITS(bfloat16(65536.0f), 0x4780);
  CHECK_BITS(bfloat16(3.0e38f), 0x7f62);
  CHECK(!std::isinf((float)bfloat16(3.0e38f)));
  // Largest finite bf16 is 0x7f7f; the float32 max rounds UP past it to infinity,
  // which is the correct IEEE round-to-nearest result and not an emulation bug.
  CHECK_BITS(bfloat16(3.38953139e38f), 0x7f7f);
  CHECK_BITS(bfloat16(3.40282347e38f), 0x7f80);

  // Subnormals: bf16's smallest normal is 2^-126 (float32's), and it has 7
  // mantissa bits of subnormal range below that.
  CHECK_BITS(bfloat16(1.17549435e-38f), 0x0080);  // smallest normal
  CHECK_BITS(bfloat16(9.18354962e-41f), 0x0001);  // smallest subnormal
  CHECK_BITS(bfloat16(4.59177481e-41f), 0x0000);  // half of it: ties-to-even -> 0

  // Round-to-nearest-EVEN, not round-half-away. These float32 values sit exactly
  // halfway between two bf16 neighbours (the discarded low 16 bits are 0x8000).
  CHECK_BITS(bfloat16(u2f(0x3f808000u)), 0x3f80);  // tie -> even (down)
  CHECK_BITS(bfloat16(u2f(0x3f818000u)), 0x3f82);  // tie -> even (up)
  // And one ulp either side of a tie rounds normally.
  CHECK_BITS(bfloat16(u2f(0x3f808001u)), 0x3f81);
  CHECK_BITS(bfloat16(u2f(0x3f807fffu)), 0x3f80);

  // NaN stays NaN in both directions and must never round into an infinity --
  // the reason the conversion tests for NaN before applying the rounding bias.
  CHECK(std::isnan((float)bfloat16(NAN)));
  CHECK(std::isnan((float)bfloat16(-NAN)));
  // A NaN whose payload lives entirely in the discarded low bits is the case that
  // a naive "add bias and shift" gets wrong, producing +inf.
  CHECK(std::isnan((float)bfloat16(u2f(0x7f800001u))));
  CHECK(std::isnan((float)bfloat16(u2f(0xff800001u))));
}

// ---------------------------------------------------------------------------
// Exhaustive: every one of the 2^16 bit patterns.
//
// bf16 -> float is a pure 16-bit shift, so it is exact for every pattern, and the
// round trip back must be the identity for all of them (NaN payloads excepted --
// see below, where they are checked separately rather than skipped).
// ---------------------------------------------------------------------------
static void test_exhaustive_roundtrip() {
  int nan_count = 0;
  for (uint32_t i = 0; i < 65536; ++i) {
    uint16_t bits = (uint16_t)i;
    bfloat16 b;
    b.h = bits;  // not memcpy: bfloat16 has a user-provided default ctor, so
                 // -Wclass-memaccess (correctly) objects to memcpy INTO one
    float f = (float)b;

    // The widening is a shift: assert that literally, for every pattern.
    if (f2u(f) != ((uint32_t)bits << 16)) {
      printf("FAIL: widen 0x%04x -> 0x%08x, want 0x%08x\n", bits, f2u(f), (uint32_t)bits << 16);
      ++failures;
      return;
    }

    if (std::isnan(f)) {
      // Must come back as SOME NaN, and must not become an infinity.
      ++nan_count;
      if (!std::isnan((float)bfloat16(f))) {
        printf("FAIL: NaN 0x%04x did not survive the round trip\n", bits);
        ++failures;
        return;
      }
      continue;
    }
    if (raw(bfloat16(f)) != bits) {
      printf("FAIL: round trip 0x%04x -> %g -> 0x%04x\n", bits, (double)f, raw(bfloat16(f)));
      ++failures;
      return;
    }
  }
  // 2 signs * 127 payloads: sanity that the NaN arm above was actually exercised
  // and the loop did not silently skip it.
  CHECK(nan_count == 254);
}

// ---------------------------------------------------------------------------
// Exhaustive: every one of the 2^32 float bit patterns, against an independent
// reference implementation of round-to-nearest-even.
//
// The reference computes the rounding decision from the discarded bits directly
// (guard bit + sticky bits + tie-to-even on the retained lsb) rather than by the
// add-a-bias trick the header uses, so an error in the bias arithmetic cannot be
// mirrored in the reference and cancel out.
// ---------------------------------------------------------------------------
static uint16_t reference_from_float(uint32_t u) {
  uint32_t exp = (u >> 23) & 0xff;
  uint32_t mant = u & 0x7fffffu;
  if (exp == 0xff && mant != 0) {  // NaN: quiet it, never round it
    return (uint16_t)((u >> 16) | 0x0040u);
  }
  uint16_t hi = (uint16_t)(u >> 16);
  uint32_t low = u & 0xffffu;
  if (low > 0x8000u) return (uint16_t)(hi + 1);           // above the midpoint: up
  if (low < 0x8000u) return hi;                           // below: down
  return (uint16_t)(hi + (hi & 1u));                      // exact tie: to even
}

static void test_exhaustive_narrowing() {
  uint32_t u = 0;
  do {
    uint16_t want = reference_from_float(u);
    uint16_t got = raw(bfloat16(u2f(u)));
    if (got != want) {
      printf("FAIL: narrow 0x%08x -> 0x%04x, want 0x%04x\n", u, got, want);
      ++failures;
      return;
    }
  } while (++u != 0);
}

// ---------------------------------------------------------------------------
// constexpr folding: dace::bfloat16(<constant>) must still be a constant
// expression, exactly like float32(<constant>) / float64(<constant>).
// Under a pre-C++20 standard DACE_LOWP_CE degrades to ``inline`` on purpose,
// so this only applies where std::bit_cast is available.
// ---------------------------------------------------------------------------
#if defined(__cpp_lib_bit_cast)
static void test_constexpr() {
  constexpr bfloat16 one = bfloat16(1.0f);
  constexpr bfloat16 neg = bfloat16(-2.5f);
  constexpr bfloat16 zero = bfloat16();
  // Forcing them into static_assert / array bounds proves compile-time folding:
  // a runtime-only conversion would not compile here at all.
  static_assert((float)one == 1.0f, "bfloat16(1.0f) must fold");
  static_assert((float)neg == -2.5f, "bfloat16(-2.5f) must fold");
  static_assert((float)zero == 0.0f, "default-constructed bfloat16 must fold to 0");
  static_assert((float)bfloat16(256.0f) == 256.0f, "exact power of two must fold");
  int folded[(int)(float)bfloat16(3.0f)];
  CHECK(sizeof(folded) / sizeof(folded[0]) == 3);
}
#else
static void test_constexpr() {}
#endif

// ---------------------------------------------------------------------------
// Operator surface parity with float32 / float64.
//
// The point of this function is as much that it COMPILES as what it asserts:
// every expression below is a shape DaCe's codegen can emit, and each one has to
// resolve to exactly one candidate. `bfloat16` has an implicit `operator float()`,
// so a badly chosen overload set makes mixed-type expressions ambiguous rather
// than wrong -- which shows up here as a build failure.
// ---------------------------------------------------------------------------
template <typename T>
static double surface(double av, double bv) {
  T a = (T)av, b = (T)bv;
  double acc = 0.0;

  // Binary arithmetic, same type.
  acc += (double)(float)(a + b);
  acc += (double)(float)(a - b);
  acc += (double)(float)(a * b);
  acc += (double)(float)(a / b);

  // Mixed with float, double and integer literals, both operand orders.
  acc += (double)(a + 1.5f) + (double)(1.5f + a);
  acc += (double)(a - 1.5) + (double)(1.5 - a);
  acc += (double)(a * 3) + (double)(3 * a);
  acc += (double)(a / 2) + (double)(2 / (a + (T)1.0f));

  // Unary.
  acc += (double)(float)(-a);
  acc += (double)(float)(+a);

  // Comparisons, same type and mixed.
  acc += (a < b) + (a > b) + (a <= b) + (a >= b) + (a == b) + (a != b);
  acc += (a < 1.5f) + (1.5f < a) + (a > 1) + (a == 0.0);
  acc += (bool)(a != (T)0.0f) ? 1.0 : 0.0;

  // Compound assignment.
  T c = a;
  c += b;
  c -= (T)0.5f;
  c *= 2.0f;
  c /= 4.0;
  c += 1;
  acc += (double)(float)c;

  // Increment / decrement, prefix and postfix.
  T d = (T)1.0f;
  ++d;
  d++;
  --d;
  d--;
  acc += (double)(float)d;

  // Ternary and assignment through a conversion.
  T e = (a < b) ? a : b;
  acc += (double)(float)e;

  return acc;
}

static void test_parity() {
  // The same expression tree, evaluated for bfloat16 / float32 / float64.
  //
  // bfloat16 vs float32 is asserted EXACTLY. That is the real invariant of this
  // design: `dace::bfloat16` evaluates every arithmetic operator in `float` through
  // the implicit conversion, so as long as the values are chosen such that each
  // value stored back into a bfloat16 variable is representable in bf16, the two
  // evaluations must agree bit for bit. It is also what catches an accidental
  // change of evaluation type -- if an `operator+(bfloat16, bfloat16)` returning
  // `bfloat16` were added, intermediates would round to 8 significant bits and this
  // comparison would break immediately.
  //
  // float64 legitimately differs (it carries more precision through the same tree),
  // so it is only checked to float precision: that leg exists to prove the surface
  // COMPILES and computes the same quantity for a primitive type.
  const double cases[][2] = {{1.0, 2.0}, {0.5, 0.25}, {-3.0, 4.0}, {2.0, -0.5}, {8.0, 16.0}};
  for (const auto &c : cases) {
    double h = surface<bfloat16>(c[0], c[1]);
    double f = surface<float>(c[0], c[1]);
    double d = surface<double>(c[0], c[1]);
    if (h != f) {
      printf("FAIL: parity (%g,%g): bfloat16=%.17g != float32=%.17g\n", c[0], c[1], h, f);
      ++failures;
    }
    // 2^-20 ~ 1e-6: a few float roundings' worth of relative slack.
    if (std::fabs(f - d) > 1e-6 * std::fabs(d)) {
      printf("FAIL: parity (%g,%g): float32=%.17g vs float64=%.17g\n", c[0], c[1], f, d);
      ++failures;
    }
  }
}

// ---------------------------------------------------------------------------
// Compound assignment semantics: `b OP= x` must equal `bfloat16(float(b) OP x)`,
// i.e. compute in float and round back once -- the same thing `b = b OP x` does
// through the implicit conversion.
// ---------------------------------------------------------------------------
static void test_compound_semantics() {
  const float vals[] = {0.0f, 1.0f, -1.0f, 0.1f, 3.7f, -12.25f, 1024.0f, 6e-5f, 0.333f, 1e30f};
  for (float a : vals) {
    for (float b : vals) {
      // Both operands go through bf16 first: `x OP= bfloat16(b)` must equal
      // rounding a and b to bf16, computing in float, and rounding once.
      const float ha = (float)bfloat16(a), hb = (float)bfloat16(b);
      bfloat16 x;

      x = bfloat16(a); x += bfloat16(b);
      CHECK_BITS(x, raw(bfloat16(ha + hb)));
      x = bfloat16(a); x -= bfloat16(b);
      CHECK_BITS(x, raw(bfloat16(ha - hb)));
      x = bfloat16(a); x *= bfloat16(b);
      CHECK_BITS(x, raw(bfloat16(ha * hb)));
      if (hb != 0.0f) {
        x = bfloat16(a); x /= bfloat16(b);
        CHECK_BITS(x, raw(bfloat16(ha / hb)));
      }
      // A float right-hand side must NOT be pre-rounded to bf16: the single
      // ``operator+=(float)`` overload takes it at full float precision.
      x = bfloat16(a); x += b;
      CHECK_BITS(x, raw(bfloat16(ha + b)));
      x = bfloat16(a); x *= b;
      CHECK_BITS(x, raw(bfloat16(ha * b)));

      // Equivalence with the spelling `b = b OP x`, which is what the implicit
      // conversion has always given and what these operators must not change.
      bfloat16 y = bfloat16(a);
      y = bfloat16((float)y + (float)bfloat16(b));
      x = bfloat16(a); x += bfloat16(b);
      CHECK_BITS(x, raw(y));
    }
  }
  // Pre/post increment return values.
  bfloat16 p = bfloat16(1.0f);
  CHECK_BITS(++p, 0x4000);            // 2.0, returns the new value
  CHECK_BITS(p++, 0x4000);            // returns the old value...
  CHECK_BITS(p, 0x4040);              // ...and leaves 3.0
  CHECK_BITS(--p, 0x4000);
  CHECK_BITS(p--, 0x4000);
  CHECK_BITS(p, 0x3f80);              // 1.0
}

// ---------------------------------------------------------------------------
// OpenMP reductions. This is the pattern ExpandReduceOpenMP emits for a
// bfloat16 Reduce node:
//     #pragma omp parallel for reduction(+: _out[0])
//     for (...) _out[0] += _in[_i0 * 1];
//
// Every case below is EXACT and order-independent by construction, so it can be
// asserted with zero tolerance:
//   - sum: bf16 has 8 significant bits, so it represents every integer up to 256
//     exactly; a sum of N <= 256 ones is exact however the runtime splits it.
//     (This is the one place bf16 needs a smaller N than binary16's 2048 -- it
//     trades mantissa bits for exponent range.)
//   - product: products of powers of two are exact while in range, and bf16's
//     range is float32's, so there is a lot of room.
//   - min/max: selection, never rounds.
// A tolerance-based bf16 sum test would be worthless: the worst-case bound for
// RNE summation is gamma_n = n*u/(1-n*u) with u = 2^-8, vacuous well before the
// N needed to engage several threads.
// ---------------------------------------------------------------------------
static void test_openmp_reductions() {
#ifdef _OPENMP
  const int N = 256;
  static bfloat16 in[256];

  printf("omp_get_max_threads = %d\n", omp_get_max_threads());
  CHECK(omp_get_max_threads() > 1);

  // --- sum: N ones -> exactly N ---
  for (int i = 0; i < N; ++i) in[i] = bfloat16(1.0f);
  bfloat16 out[1];
  out[0] = bfloat16(0.0f);
#pragma omp parallel for reduction(+ : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] += in[_i0 * 1];
  }
  CHECK_BITS(out[0], raw(bfloat16((float)N)));

  // --- sum with non-uniform data: small integers, partial sums stay exact ---
  int exact = 0;
  uint32_t seed = 12345u;
  for (int i = 0; i < N; ++i) {
    seed = seed * 1103515245u + 12345u;
    int v = (int)((seed >> 16) & 0x3) - 1;  // -1, 0, 1 or 2
    exact += v;
    in[i] = bfloat16((float)v);
  }
  out[0] = bfloat16(0.0f);
#pragma omp parallel for reduction(+ : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] += in[_i0 * 1];
  }
  CHECK_BITS(out[0], raw(bfloat16((float)exact)));

  // --- subtraction: OpenMP's `-` reduction, which negates within each private
  // copy and then SUMS them. Must agree exactly with the same clause on float.
  //
  // The `-` reduction identifier is deprecated in OpenMP 5.2 and GCC 16 warns about
  // it, which -Werror turns into a build failure. It is still tested, and the
  // warning suppressed rather than the case dropped, because ExpandReduceOpenMP
  // emits exactly this clause for a Sub node -- so as long as DaCe generates it,
  // dace::bfloat16 has to support it. ---
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 16
#pragma GCC diagnostic ignored "-Wdeprecated-openmp"
#endif
  for (int i = 0; i < N; ++i) in[i] = bfloat16(1.0f);
  static float fin[256];
  for (int i = 0; i < N; ++i) fin[i] = 1.0f;
  out[0] = bfloat16(0.0f);
#pragma omp parallel for reduction(- : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] -= in[_i0 * 1];
  }
  float fout = 0.0f;
#pragma omp parallel for reduction(- : fout)
  for (int _i0 = 0; _i0 < N; ++_i0) {
    fout -= fin[_i0];
  }
#pragma GCC diagnostic pop
  CHECK((float)out[0] == fout);
  CHECK_BITS(out[0], raw(bfloat16(-(float)N)));

  // --- product: powers of two, exact and in range ---
  for (int i = 0; i < N; ++i) in[i] = bfloat16((i % 2) ? 2.0f : 0.5f);
  out[0] = bfloat16(1.0f);
#pragma omp parallel for reduction(* : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] *= in[_i0 * 1];
  }
  CHECK_BITS(out[0], raw(bfloat16(1.0f)));  // equal numbers of 2.0 and 0.5

  // --- min / max: selection, exact ---
  for (int i = 0; i < N; ++i) in[i] = bfloat16((float)((i * 37) % 1001) - 500.0f);
  float wantmin = 1e30f, wantmax = -1e30f;
  for (int i = 0; i < N; ++i) {
    float v = (float)in[i];
    if (v < wantmin) wantmin = v;
    if (v > wantmax) wantmax = v;
  }
  out[0] = bfloat16(wantmax);  // identity is overwritten by the clause anyway
#pragma omp parallel for reduction(min : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] = ((float)in[_i0] < (float)out[0]) ? in[_i0] : out[0];
  }
  CHECK_BITS(out[0], raw(bfloat16(wantmin)));

  out[0] = bfloat16(wantmin);
#pragma omp parallel for reduction(max : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] = ((float)in[_i0] > (float)out[0]) ? in[_i0] : out[0];
  }
  CHECK_BITS(out[0], raw(bfloat16(wantmax)));

  // --- logical && / || : truthiness, normalized to exactly 1.0 / 0.0.
  // Asserted against the identical clause on `float`, which is the definition of
  // "parity" here -- including that a truthy 5.0 reduces to 1.0, not to 5.0. ---
  {
    static float lfin[256];
    for (int i = 0; i < N; ++i) {
      in[i] = bfloat16(1.0f);
      lfin[i] = 1.0f;
    }
    // all true
    out[0] = bfloat16(123.0f);  // non-identity start must still be folded in
    float fl = 123.0f;
#pragma omp parallel for reduction(&& : out[0])
    for (int _i0 = 0; _i0 < N; ++_i0) out[0] = out[0] && in[_i0];
#pragma omp parallel for reduction(&& : fl)
    for (int _i0 = 0; _i0 < N; ++_i0) fl = fl && lfin[_i0];
    CHECK((float)out[0] == fl);
    CHECK_BITS(out[0], raw(bfloat16(1.0f)));

    // one false element anywhere must win
    in[N / 2] = bfloat16(0.0f);
    lfin[N / 2] = 0.0f;
    out[0] = bfloat16(123.0f);
    fl = 123.0f;
#pragma omp parallel for reduction(&& : out[0])
    for (int _i0 = 0; _i0 < N; ++_i0) out[0] = out[0] && in[_i0];
#pragma omp parallel for reduction(&& : fl)
    for (int _i0 = 0; _i0 < N; ++_i0) fl = fl && lfin[_i0];
    CHECK((float)out[0] == fl);
    CHECK_BITS(out[0], raw(bfloat16(0.0f)));

    // all false
    for (int i = 0; i < N; ++i) {
      in[i] = bfloat16(0.0f);
      lfin[i] = 0.0f;
    }
    out[0] = bfloat16(0.0f);
    fl = 0.0f;
#pragma omp parallel for reduction(|| : out[0])
    for (int _i0 = 0; _i0 < N; ++_i0) out[0] = out[0] || in[_i0];
#pragma omp parallel for reduction(|| : fl)
    for (int _i0 = 0; _i0 < N; ++_i0) fl = fl || lfin[_i0];
    CHECK((float)out[0] == fl);
    CHECK_BITS(out[0], raw(bfloat16(0.0f)));

    // a single truthy 5.0 must normalize to 1.0, exactly as float does
    in[7] = bfloat16(5.0f);
    lfin[7] = 5.0f;
    out[0] = bfloat16(0.0f);
    fl = 0.0f;
#pragma omp parallel for reduction(|| : out[0])
    for (int _i0 = 0; _i0 < N; ++_i0) out[0] = out[0] || in[_i0];
#pragma omp parallel for reduction(|| : fl)
    for (int _i0 = 0; _i0 < N; ++_i0) fl = fl || lfin[_i0];
    CHECK((float)out[0] == fl);
    CHECK_BITS(out[0], raw(bfloat16(1.0f)));
  }

  // --- Div is lowered by ExpandReduceOpenMP as a `*` reduction over the divisors
  // followed by one divide, because `/` is not an OpenMP reduction identifier.
  // Powers of two keep the product and the quotient exact and order-independent. ---
  for (int i = 0; i < N; ++i) in[i] = bfloat16(1.0f);
  in[3] = in[9] = in[77] = in[200] = bfloat16(2.0f);
  {
    bfloat16 acc = bfloat16(1.0f);
#pragma omp parallel for reduction(* : acc)
    for (int _i0 = 0; _i0 < N; ++_i0) {
      acc *= in[_i0 * 1];
    }
    out[0] = bfloat16(1024.0f);
    out[0] = out[0] / acc;
    CHECK_BITS(out[0], raw(bfloat16(64.0f)));  // 1024 / 2^4
  }

  // --- the private copies really are identity-initialized: reducing into a
  // non-zero starting value must ADD to it, not clobber it ---
  for (int i = 0; i < N; ++i) in[i] = bfloat16(1.0f);
  out[0] = bfloat16(7.0f);
#pragma omp parallel for reduction(+ : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] += in[_i0];
  }
  CHECK_BITS(out[0], raw(bfloat16((float)N + 7.0f)));
#else
  printf("FAIL: built without OpenMP\n");
  ++failures;
#endif
}

int main() {
  printf("dace::bfloat16 backend: EMULATED (host)\n");
  test_layout();
  test_exhaustive_roundtrip();
  test_exhaustive_narrowing();
  test_constexpr();
  test_parity();
  test_compound_semantics();
  test_openmp_reductions();
  printf(failures ? "FAILED (%d)\n" : "OK (%d failures)\n", failures);
  return failures != 0;
}
