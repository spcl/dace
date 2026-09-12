// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Syntax-only driver for the per-ISA K=1 tile-op backend headers
// (``dace/tile_ops/<isa>.h``). ONE ``#include TILE_OPS_BACKEND_HEADER`` is
// chosen by the test via ``-DTILE_OPS_BACKEND_HEADER=<dace/tile_ops/avx2.h>``;
// the body below odr-uses EVERY tile op with EVERY operand present (out / a / b
// / c / cond / idx / mask), over each element type, both broadcast flags, both
// masked flags, every op char, and a spread of widths (power-of-two, odd, and
// non-multiple-of-lane-width so the interior + scalar-tail split is exercised).
//
// The point is coverage of INSTANTIATION, not execution: compiled with
// ``-fsyntax-only`` (per ISA, with the matching ``-march`` / ``-m<isa>`` and
// compiler) so a backend that fails to compile on its target -- an undefined
// helper, a missing include, a wrong intrinsic name -- is caught WITHOUT an ARM
// host. Every function template a real kernel could emit is forced here, so a
// header cannot silently ship an uninstantiable op.
#include <dace/dace.h>

#ifndef TILE_OPS_BACKEND_HEADER
#error "define TILE_OPS_BACKEND_HEADER=<dace/tile_ops/<isa>.h>"
#endif
#include TILE_OPS_BACKEND_HEADER

#include <cstdint>

using namespace dace::tileops;

// One buffer set wide enough for the largest VLEN driven below.
template <typename T, int VLEN>
struct Buf {
  T out[VLEN];
  T a[VLEN];
  T b[VLEN];
  T c[VLEN];
  T cond[VLEN];  // condition tile stored as the element type (1/0)
  bool mask[VLEN];
};

template <typename T, int VLEN, char Op>
inline void drive_one_binop() {
  Buf<T, VLEN> z{};
  tile_binop<T, VLEN, Op, false, false, false>(z.out, z.a, z.b, z.mask);
  tile_binop<T, VLEN, Op, true, false, true>(z.out, z.a, z.b, z.mask);   // broadcast A, masked
  tile_binop<T, VLEN, Op, false, true, true>(z.out, z.a, z.b, z.mask);   // broadcast B, masked
  tile_binop<T, VLEN, Op, true, true, false>(z.out, z.a, z.b, z.mask);   // both broadcast
}

template <typename T, int VLEN>
inline void drive_all_binops() {
  drive_one_binop<T, VLEN, '+'>();
  drive_one_binop<T, VLEN, '-'>();
  drive_one_binop<T, VLEN, '*'>();
  drive_one_binop<T, VLEN, '/'>();
  drive_one_binop<T, VLEN, '%'>();
  drive_one_binop<T, VLEN, 'm'>();
  drive_one_binop<T, VLEN, 'M'>();
  drive_one_binop<T, VLEN, '<'>();
  drive_one_binop<T, VLEN, 'l'>();
  drive_one_binop<T, VLEN, '>'>();
  drive_one_binop<T, VLEN, 'g'>();
  drive_one_binop<T, VLEN, '='>();
  drive_one_binop<T, VLEN, '!'>();
  drive_one_binop<T, VLEN, '&'>();
  drive_one_binop<T, VLEN, '|'>();
}

template <typename T, int VLEN, char Op>
inline void drive_one_unop() {
  Buf<T, VLEN> z{};
  tile_unop<T, VLEN, Op, false, false>(z.out, z.a, z.mask);
  tile_unop<T, VLEN, Op, true, true>(z.out, z.a, z.mask);  // broadcast, masked
}

template <typename T, int VLEN>
inline void drive_all_unops() {
  drive_one_unop<T, VLEN, 'n'>();
  drive_one_unop<T, VLEN, '!'>();
  drive_one_unop<T, VLEN, 'a'>();
  drive_one_unop<T, VLEN, 'e'>();
  drive_one_unop<T, VLEN, 'l'>();
  drive_one_unop<T, VLEN, 's'>();
  drive_one_unop<T, VLEN, 'S'>();
  drive_one_unop<T, VLEN, 'C'>();
  drive_one_unop<T, VLEN, 'f'>();
  drive_one_unop<T, VLEN, 'c'>();
  drive_one_unop<T, VLEN, 't'>();
}

template <typename T, int VLEN>
inline void drive_fma() {
  Buf<T, VLEN> z{};
  tile_fma<T, VLEN, false, false, false, false>(z.out, z.a, z.b, z.c, z.mask);
  tile_fma<T, VLEN, true, false, true, true>(z.out, z.a, z.b, z.c, z.mask);
  tile_fma<T, VLEN, false, true, false, true>(z.out, z.a, z.b, z.c, z.mask);
}

template <typename T, int VLEN>
inline void drive_ite() {
  Buf<T, VLEN> z{};
  tile_ite<T, T, VLEN, false, false, false>(z.out, z.cond, z.a, z.b, z.mask);
  tile_ite<T, T, VLEN, true, false, true>(z.out, z.cond, z.a, z.b, z.mask);
  tile_ite<T, T, VLEN, false, true, true>(z.out, z.cond, z.a, z.b, z.mask);
}

template <typename T, int VLEN>
inline void drive_load_store() {
  Buf<T, VLEN> z{};
  tile_load<T, VLEN, false>(z.out, z.a, z.mask, 1);   // unit-stride dense
  tile_load<T, VLEN, true>(z.out, z.a, z.mask, 1);     // unit-stride masked
  tile_load<T, VLEN, false>(z.out, z.a, z.mask, 2);    // strided
  tile_load<T, VLEN, true>(z.out, z.a, z.mask, 2);
  tile_store<T, VLEN, false>(z.out, z.a, z.mask, 1);
  tile_store<T, VLEN, true>(z.out, z.a, z.mask, 1);
  tile_store<T, VLEN, false>(z.out, z.a, z.mask, 2);
  tile_store<T, VLEN, true>(z.out, z.a, z.mask, 2);
}

template <typename T, int VLEN, typename IdxT>
inline void drive_gather_scatter() {
  Buf<T, VLEN> z{};
  IdxT idx[VLEN]{};
  tile_gather<T, IdxT, VLEN, false>(z.out, z.a, idx, z.mask);
  tile_gather<T, IdxT, VLEN, true>(z.out, z.a, idx, z.mask);
  tile_scatter<T, IdxT, VLEN, false>(z.out, z.a, idx, z.mask);
  tile_scatter<T, IdxT, VLEN, true>(z.out, z.a, idx, z.mask);
}

template <typename T, int VLEN>
inline void drive_reduce() {
  Buf<T, VLEN> z{};
  (void)tile_reduce<T, VLEN, '+'>(z.a);
  (void)tile_reduce<T, VLEN, '*'>(z.a);
  (void)tile_reduce<T, VLEN, 'm'>(z.a);
  (void)tile_reduce<T, VLEN, 'M'>(z.a);
}

template <typename IdxT, int VLEN>
inline void drive_mask_gen() {
  bool out[VLEN]{};
  tile_mask_gen<IdxT, VLEN>(out, IdxT(0), IdxT(VLEN));
}

template <typename T, int VLEN>
inline void drive_all_ops() {
  drive_all_binops<T, VLEN>();
  drive_all_unops<T, VLEN>();
  drive_fma<T, VLEN>();
  drive_ite<T, VLEN>();
  drive_load_store<T, VLEN>();
  drive_gather_scatter<T, VLEN, std::int32_t>();
  drive_gather_scatter<T, VLEN, std::int64_t>();
  drive_reduce<T, VLEN>();
}

// The single odr-use root; never called (syntax-only), but its body forces every
// instantiation above to be semantically analysed.
void drive_everything() {
  drive_all_ops<double, 8>();
  drive_all_ops<float, 16>();
  drive_all_ops<std::int32_t, 8>();
  drive_all_ops<std::int64_t, 4>();
  drive_all_ops<double, 3>();    // odd, shorter than any lane group
  drive_all_ops<double, 17>();   // non-multiple of every lane width (interior + tail)
  drive_all_ops<float, 5>();     // non-multiple, narrow
  drive_mask_gen<std::int64_t, 8>();
  drive_mask_gen<std::int64_t, 17>();
  drive_mask_gen<std::int32_t, 8>();
}
