// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Host-side ``dace::float16`` conformance test. Driven by tests/half_ops_cpp_test.py,
// which compiles it twice -- once letting <dace/types.h> pick the native binary16
// backing type for the conversions, and once with -DDACE_HALF_NO_NATIVE forcing the
// software emulation -- so both paths are covered on any machine.
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

using dace::float16;

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

static uint16_t raw(float16 x) {
  uint16_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

// ---------------------------------------------------------------------------
// Storage: size, alignment and bit layout must be identical on both paths, and
// identical to what every serialized DaCe array and every host<->device copy
// already assumes.
// ---------------------------------------------------------------------------
static void test_layout() {
  CHECK(sizeof(float16) == 2);
  CHECK(alignof(float16) == 2);
  CHECK(std::is_trivially_copyable<float16>::value);
  CHECK(std::is_standard_layout<float16>::value);

  struct Packed {
    float16 a;
    float16 b;
  };
  CHECK(sizeof(Packed) == 4);
  float16 arr[8];
  CHECK(sizeof(arr) == 16);

  // IEEE-754 binary16 encodings (same values NumPy's float16 produces).
  CHECK_BITS(float16(0.0f), 0x0000);
  CHECK_BITS(float16(-0.0f), 0x8000);
  CHECK_BITS(float16(1.0f), 0x3c00);
  CHECK_BITS(float16(-1.0f), 0xbc00);
  CHECK_BITS(float16(2.0f), 0x4000);
  CHECK_BITS(float16(-2.5f), 0xc100);
  CHECK_BITS(float16(0.5f), 0x3800);
  CHECK_BITS(float16(65504.0f), 0x7bff);   // largest finite binary16
  CHECK_BITS(float16(65536.0f), 0x7c00);   // overflows to +inf
  CHECK_BITS(float16(6.1035156e-5f), 0x0400);  // smallest normal
  CHECK_BITS(float16(5.9604645e-8f), 0x0001);  // smallest subnormal
  CHECK_BITS(float16(2.9802322e-8f), 0x0000);  // half of it: ties-to-even -> 0
  CHECK_BITS(float16(INFINITY), 0x7c00);
  CHECK_BITS(float16(-INFINITY), 0xfc00);

  // Round-to-nearest-EVEN, not round-half-away. 2049 and 2051 are exactly
  // halfway between representable binary16 neighbours (spacing 2 above 2048).
  CHECK_BITS(float16(2049.0f), 0x6800);  // -> 2048 (even mantissa)
  CHECK_BITS(float16(2051.0f), 0x6802);  // -> 2052 (even mantissa)

  // NaN stays NaN in both directions (the payload is deliberately not asserted:
  // the hardware path quiets NaNs per IEEE-754, the emulation canonicalizes).
  CHECK(std::isnan((float)float16(NAN)));

  // Every one of the 2^16 bit patterns must survive a half->float->half round
  // trip unchanged, except NaNs (whose payload may be canonicalized) and the
  // fact that both zeros keep their sign.
  for (uint32_t i = 0; i < 65536; ++i) {
    uint16_t bits = (uint16_t)i;
    float16 h;
    h.h = bits;  // not memcpy: half has a user-provided default ctor, so
                 // -Wclass-memaccess (correctly) objects to memcpy INTO one
    float f = (float)h;
    if (std::isnan(f)) continue;
    if (raw(float16(f)) != bits) {
      printf("FAIL: round trip 0x%04x -> %g -> 0x%04x\n", bits, (double)f, raw(float16(f)));
      ++failures;
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// constexpr folding: dace::float16(<constant>) must still be a constant
// expression, exactly like float32(<constant>) / float64(<constant>).
// Under a pre-C++20 standard DACE_HALF_CE degrades to ``inline`` on purpose,
// so this only applies where std::bit_cast is available.
// ---------------------------------------------------------------------------
#if defined(__cpp_lib_bit_cast)
static void test_constexpr() {
  constexpr float16 one = float16(1.0f);
  constexpr float16 neg = float16(-2.5f);
  constexpr float16 zero = float16();
  // Forcing them into static_assert / array bounds proves compile-time folding:
  // a runtime-only conversion would not compile here at all.
  static_assert((float)one == 1.0f, "float16(1.0f) must fold");
  static_assert((float)neg == -2.5f, "float16(-2.5f) must fold");
  static_assert((float)zero == 0.0f, "default-constructed float16 must fold to 0");
  static_assert((float)float16(65504.0f) == 65504.0f, "max finite must fold");
  int folded[(int)(float)float16(3.0f)];
  CHECK(sizeof(folded) / sizeof(folded[0]) == 3);
}
#else
static void test_constexpr() {}
#endif

// ---------------------------------------------------------------------------
// Operator surface parity with float32 / float64.
//
// The point of this function is as much that it COMPILES as what it asserts:
// every expression below is a shape DaCe's codegen can emit, and each one has
// to resolve to exactly one candidate. `half` has an implicit `operator float()`,
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

  // Compound assignment -- the operators added for this fix.
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
  // The same expression tree, evaluated for float16 / float32 / float64.
  //
  // float16 vs float32 is asserted EXACTLY. That is the real invariant of this
  // design: `dace::half` evaluates every arithmetic operator in `float` through
  // the implicit conversion, so as long as the values are chosen such that each
  // value stored back into a float16 variable is representable in binary16, the
  // two evaluations must agree bit for bit. It is also what catches an
  // accidental change of evaluation type -- if an `operator+(half, half)`
  // returning `half` were added, intermediates would round to 16 bits and this
  // comparison would break.
  //
  // float64 legitimately differs (it carries more precision through the same
  // tree), so it is only checked to float-precision: this leg exists to prove
  // the expression surface COMPILES and computes the same quantity for a
  // primitive type, not to assert bit equality.
  const double cases[][2] = {{1.0, 2.0}, {0.5, 0.25}, {-3.0, 4.0}, {2.0, -0.5}, {8.0, 16.0}};
  for (const auto &c : cases) {
    double h = surface<float16>(c[0], c[1]);
    double f = surface<float>(c[0], c[1]);
    double d = surface<double>(c[0], c[1]);
    if (h != f) {
      printf("FAIL: parity (%g,%g): float16=%.17g != float32=%.17g\n", c[0], c[1], h, f);
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
// Compound assignment semantics: `h OP= x` must equal `half(float(h) OP x)`,
// i.e. compute in float and round back once -- the same thing `h = h OP x`
// has always done through the implicit conversion.
// ---------------------------------------------------------------------------
static void test_compound_semantics() {
  const float vals[] = {0.0f, 1.0f, -1.0f, 0.1f, 3.7f, -12.25f, 1024.0f, 6e-5f, 0.333f};
  for (float a : vals) {
    for (float b : vals) {
      // Both operands go through binary16 first: `x OP= float16(b)` must equal
      // rounding a and b to binary16, computing in float, and rounding once.
      const float ha = (float)float16(a), hb = (float)float16(b);
      float16 x;

      x = float16(a); x += float16(b);
      CHECK_BITS(x, raw(float16(ha + hb)));
      x = float16(a); x -= float16(b);
      CHECK_BITS(x, raw(float16(ha - hb)));
      x = float16(a); x *= float16(b);
      CHECK_BITS(x, raw(float16(ha * hb)));
      if (hb != 0.0f) {
        x = float16(a); x /= float16(b);
        CHECK_BITS(x, raw(float16(ha / hb)));
      }
      // A float right-hand side must NOT be pre-rounded to 16 bits: the single
      // ``operator+=(float)` overload takes it at full float precision.
      x = float16(a); x += b;
      CHECK_BITS(x, raw(float16(ha + b)));
      x = float16(a); x *= b;
      CHECK_BITS(x, raw(float16(ha * b)));

      // Equivalence with the pre-existing spelling `h = h OP x`, which is what
      // the emulation has always done and what this fix must not change.
      float16 y = float16(a);
      y = float16((float)y + (float)float16(b));
      x = float16(a); x += float16(b);
      CHECK_BITS(x, raw(y));
    }
  }
  // Pre/post increment return values.
  float16 p = float16(1.0f);
  CHECK_BITS(++p, 0x4000);            // 2.0, returns the new value
  CHECK_BITS(p++, 0x4000);            // returns the old value...
  CHECK_BITS(p, 0x4200);              // ...and leaves 3.0
  CHECK_BITS(--p, 0x4000);
  CHECK_BITS(p--, 0x4000);
  CHECK_BITS(p, 0x3c00);              // 1.0
}

// ---------------------------------------------------------------------------
// OpenMP reductions. This is the pattern ExpandReduceOpenMP emits for a
// float16 Reduce node:
//     #pragma omp parallel for reduction(+: _out[0])
//     for (...) _out[0] += _in[_i0 * 1];
//
// Every case below is EXACT and order-independent by construction, so it can be
// asserted with zero tolerance:
//   - sum: binary16 represents every integer up to 2048 exactly, so a sum of
//     N<=2048 ones is exact no matter how the runtime splits and combines it.
//     A wrong identity, a dropped chunk or a double-counted chunk all change
//     the integer result and are caught immediately.
//   - product: products of powers of two are exact while in range.
//   - min/max: selection, never rounds.
// A tolerance-based fp16 sum test would be worthless here: the standard
// worst-case bound for RNE summation is gamma_n = n*u/(1-n*u) with u = 2^-11,
// which is already vacuous (>1) for the N needed to engage several threads.
// ---------------------------------------------------------------------------
static void test_openmp_reductions() {
#ifdef _OPENMP
  const int N = 2048;
  static float16 in[2048];

  printf("omp_get_max_threads = %d\n", omp_get_max_threads());
  CHECK(omp_get_max_threads() > 1);

  // --- sum: N ones -> exactly N ---
  for (int i = 0; i < N; ++i) in[i] = float16(1.0f);
  float16 out[1];
  out[0] = float16(0.0f);
#pragma omp parallel for reduction(+ : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] += in[_i0 * 1];
  }
  CHECK_BITS(out[0], raw(float16((float)N)));

  // --- sum with non-uniform data: small integers, partial sums stay exact ---
  int exact = 0;
  uint32_t seed = 12345u;
  for (int i = 0; i < N; ++i) {
    seed = seed * 1103515245u + 12345u;
    int v = (int)((seed >> 16) & 0x3) - 1;  // -1, 0, 1 or 2
    exact += v;
    in[i] = float16((float)v);
  }
  out[0] = float16(0.0f);
#pragma omp parallel for reduction(+ : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] += in[_i0 * 1];
  }
  CHECK_BITS(out[0], raw(float16((float)exact)));

  // --- product: powers of two, exact and in range ---
  for (int i = 0; i < N; ++i) in[i] = float16((i % 2) ? 2.0f : 0.5f);
  out[0] = float16(1.0f);
#pragma omp parallel for reduction(* : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] *= in[_i0 * 1];
  }
  CHECK_BITS(out[0], raw(float16(1.0f)));  // equal numbers of 2.0 and 0.5

  // --- min / max: selection, exact ---
  for (int i = 0; i < N; ++i) in[i] = float16((float)((i * 37) % 1001) - 500.0f);
  float wantmin = 1e30f, wantmax = -1e30f;
  for (int i = 0; i < N; ++i) {
    float v = (float)in[i];
    if (v < wantmin) wantmin = v;
    if (v > wantmax) wantmax = v;
  }
  out[0] = float16(wantmax);  // identity is overwritten by the clause anyway
#pragma omp parallel for reduction(min : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] = ((float)in[_i0] < (float)out[0]) ? in[_i0] : out[0];
  }
  CHECK_BITS(out[0], raw(float16(wantmin)));

  out[0] = float16(wantmin);
#pragma omp parallel for reduction(max : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] = ((float)in[_i0] > (float)out[0]) ? in[_i0] : out[0];
  }
  CHECK_BITS(out[0], raw(float16(wantmax)));

  // --- the private copies really are identity-initialized: reducing into a
  // non-zero starting value must ADD to it, not clobber it ---
  for (int i = 0; i < N; ++i) in[i] = float16(1.0f);
  out[0] = float16(7.0f);
#pragma omp parallel for reduction(+ : out[0])
  for (int _i0 = 0; _i0 < N; ++_i0) {
    out[0] += in[_i0];
  }
  CHECK_BITS(out[0], raw(float16((float)N + 7.0f)));
#else
  printf("FAIL: built without OpenMP\n");
  ++failures;
#endif
}

int main() {
#if defined(DACE_HALF_NATIVE_T)
  printf("dace::half conversion backend: NATIVE\n");
#else
  printf("dace::half conversion backend: EMULATED\n");
#endif
  test_layout();
  test_constexpr();
  test_parity();
  test_compound_semantics();
  test_openmp_reductions();
  printf(failures ? "FAILED (%d)\n" : "OK (%d failures)\n", failures);
  return failures != 0;
}
