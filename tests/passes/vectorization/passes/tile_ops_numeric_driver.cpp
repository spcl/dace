// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Numeric differential driver for the per-ISA K=1 tile-op backend headers. The
// sibling ``tile_ops_all_ops_driver.cpp`` only proves a backend COMPILES; this
// one proves it COMPUTES the scalar contract. One ``#include
// TILE_OPS_BACKEND_HEADER`` is chosen by the test, the binary is run, and every
// tile op result is dumped as raw hex on stdout. The scalar backend's dump is
// the oracle: an ISA backend must be BIT-IDENTICAL (the headers document
// bit-for-bit agreement -- ``std::fma`` everywhere, the same pairwise reduce
// tree, the same py_mod).
//
// Inputs are generated from the case index alone, so the scalar and ISA runs
// see identical operands without sharing any state.
#include <dace/dace.h>

#ifndef TILE_OPS_BACKEND_HEADER
#error "define TILE_OPS_BACKEND_HEADER=<dace/tile_ops/<isa>.h>"
#endif
#include TILE_OPS_BACKEND_HEADER

#include <cstdint>
#include <cstdio>
#include <limits>

using namespace dace::tileops;

namespace {

// Deterministic value stream: a splitmix64 finalizer over the lane index and a
// per-case salt, so both binaries build the same operands from nothing.
std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

template <typename T>
T gen(std::uint64_t salt, int lane) {
  const std::int64_t h = static_cast<std::int64_t>(mix(salt * 1000003ULL + std::uint64_t(lane)) % 4001ULL) - 2000;
  if constexpr (std::is_floating_point_v<T>) return T(h) / T(8);
  return T(h % 251);
}

// Non-zero variant for divisors: integer ``/`` and ``%`` by zero is UB.
template <typename T>
T gen_nz(std::uint64_t salt, int lane) {
  const T v = gen<T>(salt, lane);
  return v == T(0) ? T(3) : v;
}

const char* type_name(float) { return "f32"; }
const char* type_name(double) { return "f64"; }
const char* type_name(std::int32_t) { return "i32"; }
const char* type_name(std::int64_t) { return "i64"; }

void emit(const char* label, const void* p, std::size_t n) {
  std::printf("%s ", label);
  const unsigned char* b = static_cast<const unsigned char*>(p);
  for (std::size_t i = 0; i < n; ++i) std::printf("%02x", b[i]);
  std::printf("\n");
}

// One buffer set; ``out`` is poisoned before every call so a lane the op fails
// to write shows up as a diff instead of silently inheriting the previous case.
template <typename T, int VLEN>
struct Buf {
  T out[VLEN];
  T a[VLEN];
  T b[VLEN];
  T c[VLEN];
  T cond[VLEN];
  bool mask[VLEN];
  std::int32_t idx32[VLEN];
  std::int64_t idx64[VLEN];

  explicit Buf(std::uint64_t salt) {
    for (int i = 0; i < VLEN; ++i) {
      a[i] = gen<T>(salt + 1, i);
      b[i] = gen_nz<T>(salt + 2, i);
      c[i] = gen<T>(salt + 3, i);
      cond[i] = T(mix(salt + 4 + std::uint64_t(i)) % 3ULL);  // 0 / 1 / 2 -> falsy + two truthy
      mask[i] = (mix(salt + 5 + std::uint64_t(i)) % 3ULL) != 0ULL;
      idx32[i] = std::int32_t(mix(salt + 6 + std::uint64_t(i)) % std::uint64_t(VLEN));
      idx64[i] = std::int64_t(mix(salt + 7 + std::uint64_t(i)) % std::uint64_t(VLEN));
    }
  }

  void poison() {
    for (int i = 0; i < VLEN; ++i) out[i] = T(-99);
  }
};

// ------------------------------- binop -------------------------------
template <typename T, int VLEN, char Op, bool BA, bool BB, bool MK>
void run_binop(std::uint64_t salt) {
  Buf<T, VLEN> z(salt);
  z.poison();
  tile_binop<T, VLEN, Op, BA, BB, MK>(z.out, z.a, z.b, z.mask);
  char label[128];
  std::snprintf(label, sizeof(label), "binop:%s:v%d:%c:%d%d:%d", type_name(T()), VLEN, Op, int(BA), int(BB), int(MK));
  emit(label, z.out, sizeof(z.out));
}

template <typename T, int VLEN, char Op>
void run_binop_flags(std::uint64_t salt) {
  run_binop<T, VLEN, Op, false, false, false>(salt);
  run_binop<T, VLEN, Op, false, false, true>(salt + 11);
  run_binop<T, VLEN, Op, true, false, true>(salt + 22);
  run_binop<T, VLEN, Op, false, true, true>(salt + 33);
  run_binop<T, VLEN, Op, true, true, false>(salt + 44);
}

template <typename T, int VLEN>
void run_all_binops(std::uint64_t salt) {
  run_binop_flags<T, VLEN, '+'>(salt + 100);
  run_binop_flags<T, VLEN, '-'>(salt + 200);
  run_binop_flags<T, VLEN, '*'>(salt + 300);
  run_binop_flags<T, VLEN, '/'>(salt + 400);
  run_binop_flags<T, VLEN, '%'>(salt + 500);
  run_binop_flags<T, VLEN, 'm'>(salt + 600);
  run_binop_flags<T, VLEN, 'M'>(salt + 700);
  run_binop_flags<T, VLEN, '<'>(salt + 800);
  run_binop_flags<T, VLEN, 'l'>(salt + 900);
  run_binop_flags<T, VLEN, '>'>(salt + 1000);
  run_binop_flags<T, VLEN, 'g'>(salt + 1100);
  run_binop_flags<T, VLEN, '='>(salt + 1200);
  run_binop_flags<T, VLEN, '!'>(salt + 1300);
  run_binop_flags<T, VLEN, '&'>(salt + 1400);
  run_binop_flags<T, VLEN, '|'>(salt + 1500);
}

// -------------------------------- unop --------------------------------
template <typename T, int VLEN, char Op, bool BC, bool MK>
void run_unop(std::uint64_t salt) {
  Buf<T, VLEN> z(salt);
  z.poison();
  tile_unop<T, VLEN, Op, BC, MK>(z.out, z.a, z.mask);
  char label[128];
  std::snprintf(label, sizeof(label), "unop:%s:v%d:%c:%d:%d", type_name(T()), VLEN, Op, int(BC), int(MK));
  emit(label, z.out, sizeof(z.out));
}

template <typename T, int VLEN, char Op>
void run_unop_flags(std::uint64_t salt) {
  run_unop<T, VLEN, Op, false, false>(salt);
  run_unop<T, VLEN, Op, false, true>(salt + 11);
  run_unop<T, VLEN, Op, true, true>(salt + 22);
}

template <typename T, int VLEN>
void run_all_unops(std::uint64_t salt) {
  run_unop_flags<T, VLEN, 'n'>(salt + 100);
  run_unop_flags<T, VLEN, '!'>(salt + 200);
  run_unop_flags<T, VLEN, 'a'>(salt + 300);
  run_unop_flags<T, VLEN, 'f'>(salt + 400);
  run_unop_flags<T, VLEN, 'c'>(salt + 500);
  if constexpr (std::is_floating_point_v<T>) {
    run_unop_flags<T, VLEN, 'e'>(salt + 600);
    run_unop_flags<T, VLEN, 'l'>(salt + 700);
    run_unop_flags<T, VLEN, 's'>(salt + 800);
    run_unop_flags<T, VLEN, 'S'>(salt + 900);
    run_unop_flags<T, VLEN, 'C'>(salt + 1000);
    run_unop_flags<T, VLEN, 't'>(salt + 1100);
  }
}

// --------------------------------- fma ---------------------------------
template <typename T, int VLEN, bool BA, bool BB, bool BC, bool MK>
void run_fma(std::uint64_t salt) {
  Buf<T, VLEN> z(salt);
  z.poison();
  tile_fma<T, VLEN, BA, BB, BC, MK>(z.out, z.a, z.b, z.c, z.mask);
  char label[128];
  std::snprintf(label, sizeof(label), "fma:%s:v%d:%d%d%d:%d", type_name(T()), VLEN, int(BA), int(BB), int(BC), int(MK));
  emit(label, z.out, sizeof(z.out));
}

template <typename T, int VLEN>
void run_all_fma(std::uint64_t salt) {
  run_fma<T, VLEN, false, false, false, false>(salt);
  run_fma<T, VLEN, false, false, false, true>(salt + 11);
  run_fma<T, VLEN, true, false, false, true>(salt + 22);
  run_fma<T, VLEN, false, true, false, true>(salt + 33);
  run_fma<T, VLEN, false, false, true, true>(salt + 44);
  run_fma<T, VLEN, true, true, true, false>(salt + 55);
}

// --------------------------------- ite ---------------------------------
template <typename T, int VLEN, bool BT, bool BE, bool MK>
void run_ite(std::uint64_t salt) {
  Buf<T, VLEN> z(salt);
  z.poison();
  tile_ite<T, T, VLEN, BT, BE, MK>(z.out, z.cond, z.a, z.b, z.mask);
  char label[128];
  std::snprintf(label, sizeof(label), "ite:%s:v%d:%d%d:%d", type_name(T()), VLEN, int(BT), int(BE), int(MK));
  emit(label, z.out, sizeof(z.out));
}

template <typename T, int VLEN>
void run_all_ite(std::uint64_t salt) {
  run_ite<T, VLEN, false, false, false>(salt);
  run_ite<T, VLEN, false, false, true>(salt + 11);
  run_ite<T, VLEN, true, false, true>(salt + 22);
  run_ite<T, VLEN, false, true, true>(salt + 33);
}

// ---------------------------- load / store ----------------------------
// ``dst`` for a strided store is 2*VLEN wide and pre-seeded, so a lane an ISA
// path wrongly writes (or wrongly skips) is visible.
template <typename T, int VLEN, bool MK>
void run_load(std::uint64_t salt, std::int64_t stride) {
  Buf<T, VLEN> z(salt);
  T src[4 * VLEN];
  for (int i = 0; i < 4 * VLEN; ++i) src[i] = gen<T>(salt + 9, i);
  z.poison();
  const T* __restrict__ srcp = src;  // codegen emits a restrict-qualified pointer, not an array
  tile_load<T, VLEN, MK>(z.out, srcp, z.mask, stride);
  char label[128];
  std::snprintf(label, sizeof(label), "load:%s:v%d:s%lld:%d", type_name(T()), VLEN, (long long)stride, int(MK));
  emit(label, z.out, sizeof(z.out));
}

template <typename T, int VLEN, bool MK>
void run_store(std::uint64_t salt, std::int64_t stride) {
  Buf<T, VLEN> z(salt);
  T dst[4 * VLEN];
  for (int i = 0; i < 4 * VLEN; ++i) dst[i] = gen<T>(salt + 13, i);
  T* __restrict__ dstp = dst;
  tile_store<T, VLEN, MK>(dstp, z.a, z.mask, stride);
  char label[128];
  std::snprintf(label, sizeof(label), "store:%s:v%d:s%lld:%d", type_name(T()), VLEN, (long long)stride, int(MK));
  emit(label, dst, sizeof(dst));
}

template <typename T, int VLEN>
void run_all_load_store(std::uint64_t salt) {
  run_load<T, VLEN, false>(salt, 1);
  run_load<T, VLEN, true>(salt + 11, 1);
  run_load<T, VLEN, false>(salt + 22, 3);
  run_load<T, VLEN, true>(salt + 33, 3);
  run_store<T, VLEN, false>(salt + 44, 1);
  run_store<T, VLEN, true>(salt + 55, 1);
  run_store<T, VLEN, false>(salt + 66, 3);
  run_store<T, VLEN, true>(salt + 77, 3);
}

// --------------------------- gather / scatter ---------------------------
template <typename T, typename IdxT, int VLEN, bool MK>
void run_gather(std::uint64_t salt, const IdxT* idx) {
  Buf<T, VLEN> z(salt);
  z.poison();
  const T* __restrict__ srcp = z.a;
  tile_gather<T, IdxT, VLEN, MK>(z.out, srcp, idx, z.mask);
  char label[128];
  std::snprintf(label, sizeof(label), "gather:%s:i%d:v%d:%d", type_name(T()), int(sizeof(IdxT) * 8), VLEN, int(MK));
  emit(label, z.out, sizeof(z.out));
}

template <typename T, typename IdxT, int VLEN, bool MK>
void run_scatter(std::uint64_t salt, const IdxT* idx) {
  Buf<T, VLEN> z(salt);
  T dst[VLEN];
  for (int i = 0; i < VLEN; ++i) dst[i] = gen<T>(salt + 17, i);
  T* __restrict__ dstp = dst;
  tile_scatter<T, IdxT, VLEN, MK>(dstp, z.a, idx, z.mask);
  char label[128];
  std::snprintf(label, sizeof(label), "scatter:%s:i%d:v%d:%d", type_name(T()), int(sizeof(IdxT) * 8), VLEN, int(MK));
  emit(label, dst, sizeof(dst));
}

// Gather / scatter with NEGATIVE indices: codegen biases the base pointer, so a
// lane index below zero is legal. All-false / all-true masks pin the two mask
// extremes an ISA masked form is most likely to get wrong.
template <typename T, int VLEN>
void run_gather_scatter_edges(std::uint64_t salt) {
  T buf[3 * VLEN];
  for (int i = 0; i < 3 * VLEN; ++i) buf[i] = gen<T>(salt + 21, i);
  const T* __restrict__ mid = buf + VLEN;  // negative lane indices stay in-bounds
  std::int64_t idx[VLEN];
  std::int32_t idx32[VLEN];
  for (int i = 0; i < VLEN; ++i) {
    idx[i] = std::int64_t(i % 2 ? -(i % VLEN) : (i % VLEN));
    idx32[i] = std::int32_t(idx[i]);
  }
  for (int m = 0; m < 2; ++m) {
    bool mask[VLEN];
    for (int i = 0; i < VLEN; ++i) mask[i] = (m != 0);
    T out[VLEN];
    for (int i = 0; i < VLEN; ++i) out[i] = T(-99);
    char label[128];
    tile_gather<T, std::int64_t, VLEN, true>(out, mid, idx, mask);
    std::snprintf(label, sizeof(label), "gather_neg:%s:i64:v%d:m%d", type_name(T()), VLEN, m);
    emit(label, out, sizeof(out));
    for (int i = 0; i < VLEN; ++i) out[i] = T(-99);
    tile_gather<T, std::int32_t, VLEN, true>(out, mid, idx32, mask);
    std::snprintf(label, sizeof(label), "gather_neg:%s:i32:v%d:m%d", type_name(T()), VLEN, m);
    emit(label, out, sizeof(out));
    T dst[3 * VLEN];
    for (int i = 0; i < 3 * VLEN; ++i) dst[i] = gen<T>(salt + 23, i);
    T* __restrict__ dmid = dst + VLEN;
    tile_scatter<T, std::int64_t, VLEN, true>(dmid, buf, idx, mask);
    std::snprintf(label, sizeof(label), "scatter_neg:%s:i64:v%d:m%d", type_name(T()), VLEN, m);
    emit(label, dst, sizeof(dst));
    // All-mask load / store extremes.
    T lout[VLEN];
    for (int i = 0; i < VLEN; ++i) lout[i] = T(-99);
    const T* __restrict__ srcp = buf;
    tile_load<T, VLEN, true>(lout, srcp, mask, 1);
    std::snprintf(label, sizeof(label), "load_edge:%s:v%d:m%d:s1", type_name(T()), VLEN, m);
    emit(label, lout, sizeof(lout));
    for (int i = 0; i < VLEN; ++i) lout[i] = T(-99);
    tile_load<T, VLEN, true>(lout, srcp, mask, 2);
    std::snprintf(label, sizeof(label), "load_edge:%s:v%d:m%d:s2", type_name(T()), VLEN, m);
    emit(label, lout, sizeof(lout));
    T sdst[3 * VLEN];
    for (int i = 0; i < 3 * VLEN; ++i) sdst[i] = gen<T>(salt + 27, i);
    T* __restrict__ sdstp = sdst;
    tile_store<T, VLEN, true>(sdstp, buf, mask, 2);
    std::snprintf(label, sizeof(label), "store_edge:%s:v%d:m%d:s2", type_name(T()), VLEN, m);
    emit(label, sdst, sizeof(sdst));
  }
}

template <typename T, int VLEN>
void run_all_gather_scatter(std::uint64_t salt) {
  run_gather_scatter_edges<T, VLEN>(salt + 700);
  Buf<T, VLEN> z(salt);
  run_gather<T, std::int32_t, VLEN, false>(salt, z.idx32);
  run_gather<T, std::int32_t, VLEN, true>(salt + 11, z.idx32);
  run_gather<T, std::int64_t, VLEN, false>(salt + 22, z.idx64);
  run_gather<T, std::int64_t, VLEN, true>(salt + 33, z.idx64);
  run_scatter<T, std::int32_t, VLEN, false>(salt + 44, z.idx32);
  run_scatter<T, std::int32_t, VLEN, true>(salt + 55, z.idx32);
  run_scatter<T, std::int64_t, VLEN, false>(salt + 66, z.idx64);
  run_scatter<T, std::int64_t, VLEN, true>(salt + 77, z.idx64);
}

// -------------------------------- reduce --------------------------------
template <typename T, int VLEN, char Op>
void run_reduce(std::uint64_t salt) {
  Buf<T, VLEN> z(salt);
  const T r = tile_reduce<T, VLEN, Op>(z.a);
  char label[128];
  std::snprintf(label, sizeof(label), "reduce:%s:v%d:%c", type_name(T()), VLEN, Op);
  emit(label, &r, sizeof(r));
}

template <typename T, int VLEN>
void run_all_reduce(std::uint64_t salt) {
  run_reduce<T, VLEN, '+'>(salt);
  run_reduce<T, VLEN, '*'>(salt + 11);
  run_reduce<T, VLEN, 'm'>(salt + 22);
  run_reduce<T, VLEN, 'M'>(salt + 33);
}

// ------------------------------- mask_gen -------------------------------
template <typename IdxT, int VLEN>
void run_mask_gen(IdxT base, IdxT ub) {
  bool out[VLEN];
  for (int i = 0; i < VLEN; ++i) out[i] = true;
  tile_mask_gen<IdxT, VLEN>(out, base, ub);
  char label[128];
  std::snprintf(label, sizeof(label), "maskgen:i%d:v%d:b%lld:u%lld", int(sizeof(IdxT) * 8), VLEN, (long long)base,
                (long long)ub);
  emit(label, out, sizeof(out));
}

template <typename IdxT, int VLEN>
void run_all_mask_gen() {
  run_mask_gen<IdxT, VLEN>(IdxT(0), IdxT(VLEN));
  run_mask_gen<IdxT, VLEN>(IdxT(0), IdxT(VLEN / 2));
  run_mask_gen<IdxT, VLEN>(IdxT(3), IdxT(VLEN));
  run_mask_gen<IdxT, VLEN>(IdxT(VLEN), IdxT(VLEN));  // fully inactive
  run_mask_gen<IdxT, VLEN>(IdxT(-2), IdxT(VLEN - 1));
}

// --------------------------- adversarial values ---------------------------
// Random operands never hit the values where a SIMD instruction and the scalar
// contract are allowed to disagree: NaN (compare-unordered, min/max operand
// selection), signed zero (``std::min(+0,-0)`` keeps ``a``, ``minps`` keeps
// ``b``), infinities, and the integer extremes.
template <typename T>
T special(int k) {
  if constexpr (std::is_floating_point_v<T>) {
    const T v[12] = {T(0),
                     -T(0),
                     T(1),
                     T(-1),
                     std::numeric_limits<T>::infinity(),
                     -std::numeric_limits<T>::infinity(),
                     std::numeric_limits<T>::quiet_NaN(),
                     std::numeric_limits<T>::denorm_min(),
                     T(2.5),
                     T(-2.5),
                     std::numeric_limits<T>::max(),
                     std::numeric_limits<T>::lowest()};
    return v[k % 12];
  } else {
    const T v[8] = {
        T(0), T(1), T(-1), T(2), T(-2), T(7), std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
    return v[k % 8];
  }
}

template <typename T, int VLEN, char Op, bool MK>
void run_binop_special(int shift) {
  T out[VLEN], a[VLEN], b[VLEN];
  bool mask[VLEN];
  for (int i = 0; i < VLEN; ++i) {
    a[i] = special<T>(i);
    b[i] = special<T>(i + shift);
    mask[i] = (i % 3) != 1;
    out[i] = T(-99);
  }
  tile_binop<T, VLEN, Op, false, false, MK>(out, a, b, mask);
  char label[128];
  std::snprintf(label, sizeof(label), "binop_sp:%s:v%d:%c:sh%d:%d", type_name(T()), VLEN, Op, shift, int(MK));
  emit(label, out, sizeof(out));
}

template <typename T, int VLEN, char Op>
void run_binop_special_shifts() {
  run_binop_special<T, VLEN, Op, false>(1);
  run_binop_special<T, VLEN, Op, true>(1);
  run_binop_special<T, VLEN, Op, false>(3);
  run_binop_special<T, VLEN, Op, false>(6);
}

template <typename T, int VLEN>
void run_all_specials() {
  run_binop_special_shifts<T, VLEN, 'm'>();
  run_binop_special_shifts<T, VLEN, 'M'>();
  run_binop_special_shifts<T, VLEN, '<'>();
  run_binop_special_shifts<T, VLEN, 'l'>();
  run_binop_special_shifts<T, VLEN, '>'>();
  run_binop_special_shifts<T, VLEN, 'g'>();
  run_binop_special_shifts<T, VLEN, '='>();
  run_binop_special_shifts<T, VLEN, '!'>();
  run_binop_special_shifts<T, VLEN, '&'>();
  run_binop_special_shifts<T, VLEN, '|'>();
  run_binop_special_shifts<T, VLEN, '+'>();
  run_binop_special_shifts<T, VLEN, '-'>();
  run_binop_special_shifts<T, VLEN, '*'>();
  // ite / reduce over the same operand table.
  T out[VLEN], a[VLEN], b[VLEN], cond[VLEN];
  bool mask[VLEN];
  for (int i = 0; i < VLEN; ++i) {
    a[i] = special<T>(i);
    b[i] = special<T>(i + 3);
    cond[i] = special<T>(i + 1);
    mask[i] = (i % 3) != 1;
    out[i] = T(-99);
  }
  tile_ite<T, T, VLEN, false, false, false>(out, cond, a, b, mask);
  char label[128];
  std::snprintf(label, sizeof(label), "ite_sp:%s:v%d", type_name(T()), VLEN);
  emit(label, out, sizeof(out));
  const T rm = tile_reduce<T, VLEN, 'm'>(a);
  std::snprintf(label, sizeof(label), "reduce_sp:%s:v%d:m", type_name(T()), VLEN);
  emit(label, &rm, sizeof(rm));
  const T rM = tile_reduce<T, VLEN, 'M'>(a);
  std::snprintf(label, sizeof(label), "reduce_sp:%s:v%d:M", type_name(T()), VLEN);
  emit(label, &rM, sizeof(rM));
  const T rs = tile_reduce<T, VLEN, '+'>(a);
  std::snprintf(label, sizeof(label), "reduce_sp:%s:v%d:+", type_name(T()), VLEN);
  emit(label, &rs, sizeof(rs));
}

template <typename T, int VLEN>
void run_all_ops(std::uint64_t salt) {
  run_all_binops<T, VLEN>(salt);
  run_all_unops<T, VLEN>(salt + 10000);
  run_all_fma<T, VLEN>(salt + 20000);
  run_all_ite<T, VLEN>(salt + 30000);
  run_all_load_store<T, VLEN>(salt + 40000);
  run_all_gather_scatter<T, VLEN>(salt + 50000);
  run_all_reduce<T, VLEN>(salt + 60000);
  run_all_specials<T, VLEN>();
}

}  // namespace

int main() {
  run_all_ops<double, 8>(1);
  run_all_ops<double, 3>(2);
  run_all_ops<double, 17>(3);
  run_all_ops<float, 16>(4);
  run_all_ops<float, 5>(5);
  run_all_ops<float, 9>(6);
  run_all_ops<std::int32_t, 8>(7);
  run_all_ops<std::int32_t, 11>(8);
  run_all_ops<std::int64_t, 4>(9);
  run_all_ops<std::int64_t, 7>(10);
  run_all_mask_gen<std::int64_t, 8>();
  run_all_mask_gen<std::int64_t, 17>();
  run_all_mask_gen<std::int32_t, 8>();
  run_all_mask_gen<std::int32_t, 5>();
  return 0;
}
