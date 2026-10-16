// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#ifdef WITH_CUDA
#if defined(WITH_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#endif

namespace dace {

namespace fft {

/**
 * Twiddle factor tables of Stockham FFTs (see ``dace/libraries/fft/algorithms/stockham.py``), computed when a program
 * first requests them and shared by all transforms that need the same table. A table is identified by its kind, its
 * sizes, the direction of the transform, its element type, and whether it is in host or GPU memory. The cache is a
 * field of the program state and frees its tables when the program is finalized.
 *
 * Each entry is a root of unity ``exp(sign * 2 pi i * t / n)`` for an integer ``t``, evaluated in double precision.
 */
class StockhamTwiddles {
 public:
  StockhamTwiddles() = default;
  StockhamTwiddles(const StockhamTwiddles &) = delete;
  StockhamTwiddles &operator=(const StockhamTwiddles &) = delete;
  ~StockhamTwiddles() { Clear(); }

  /**
   * The twiddle factors of a stage of radix ``R`` and span ``L``: a ``[R - 1, L]`` table with
   * ``exp(sign * 2 pi i * a * b / (L * R))`` at ``(a - 1, b)``.
   */
  template <typename T>
  T *Stage(int64_t radix, int64_t span, int sign, bool device = false) {
    return Get<T>(kStage, radix, span, 0, sign, device);
  }

  /** The ``n`` roots of unity ``exp(sign * 2 pi i * t / n)``. */
  template <typename T>
  T *Roots(int64_t n, int sign, bool device = false) {
    return Get<T>(kRoots, n, 0, 0, sign, device);
  }

  /** The ``S`` roots ``exp(sign * 2 pi i * t / Q)`` of the low part of ``t`` in a group of stages. */
  template <typename T>
  T *GroupLow(int64_t modulus, int64_t split, int sign, bool device = false) {
    return Get<T>(kGroupLow, modulus, split, 0, sign, device);
  }

  /** The ``ceil(Q / S)`` roots ``exp(sign * 2 pi i * t * S / Q)`` of the high part of ``t`` in a group of stages. */
  template <typename T>
  T *GroupHigh(int64_t modulus, int64_t split, int sign, bool device = false) {
    return Get<T>(kGroupHigh, modulus, split, 0, sign, device);
  }

  /** A ``[P, lanes]`` table with ``exp(sign * 2 pi i * a * l / Q)`` at ``(a, l)``, for adjacent transforms. */
  template <typename T>
  T *GroupLane(int64_t modulus, int64_t length, int64_t lanes, int sign, bool device = false) {
    return Get<T>(kGroupLane, modulus, length, lanes, sign, device);
  }

  /** Frees all tables. */
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &entry : tables_) {
      if (std::get<6>(entry.first)) {
#ifdef WITH_CUDA
#if defined(WITH_HIP)
        (void)hipFree(entry.second);
#else
        (void)cudaFree(entry.second);
#endif
#endif
      } else {
        delete[] static_cast<char *>(entry.second);
      }
    }
    tables_.clear();
  }

 private:
  enum Kind { kStage, kRoots, kGroupLow, kGroupHigh, kGroupLane };
  // Kind, three sizes, sign, element size, and whether the table is in GPU memory
  using Key = std::tuple<int, int64_t, int64_t, int64_t, int, size_t, bool>;

  template <typename T>
  T *Get(Kind kind, int64_t p0, int64_t p1, int64_t p2, int sign, bool device) {
    const Key key{kind, p0, p1, p2, sign, sizeof(T), device};
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = tables_.find(key);
    if (found != tables_.end()) return static_cast<T *>(found->second);

    std::vector<T> values = Compute<T>(kind, p0, p1, p2, sign);
    const size_t bytes = values.size() * sizeof(T);
    void *table = nullptr;
    if (device) {
#ifdef WITH_CUDA
#if defined(WITH_HIP)
      (void)hipMalloc(&table, bytes);
      (void)hipMemcpy(table, values.data(), bytes, hipMemcpyHostToDevice);
#else
      (void)cudaMalloc(&table, bytes);
      (void)cudaMemcpy(table, values.data(), bytes, cudaMemcpyHostToDevice);
#endif
#endif
    } else {
      table = new char[bytes];
      std::memcpy(table, values.data(), bytes);
    }
    tables_[key] = table;
    return static_cast<T *>(table);
  }

  template <typename T>
  static T Root(int64_t t, int64_t n, int sign) {
    const double pi = 3.14159265358979323846;
    const double angle = sign * 2.0 * pi * static_cast<double>(t % n) / static_cast<double>(n);
    using Real = typename T::value_type;
    return T(static_cast<Real>(std::cos(angle)), static_cast<Real>(std::sin(angle)));
  }

  template <typename T>
  static std::vector<T> Compute(Kind kind, int64_t p0, int64_t p1, int64_t p2, int sign) {
    std::vector<T> values;
    switch (kind) {
      case kStage:  // p0 = R, p1 = L
        values.resize((p0 - 1) * p1);
        for (int64_t a = 1; a < p0; ++a)
          for (int64_t b = 0; b < p1; ++b) values[(a - 1) * p1 + b] = Root<T>(a * b, p0 * p1, sign);
        break;
      case kRoots:  // p0 = n
        values.resize(p0);
        for (int64_t t = 0; t < p0; ++t) values[t] = Root<T>(t, p0, sign);
        break;
      case kGroupLow:  // p0 = Q, p1 = S
        values.resize(p1);
        for (int64_t t = 0; t < p1; ++t) values[t] = Root<T>(t, p0, sign);
        break;
      case kGroupHigh:  // p0 = Q, p1 = S
        values.resize((p0 + p1 - 1) / p1);
        for (int64_t t = 0; t < static_cast<int64_t>(values.size()); ++t) values[t] = Root<T>(t * p1, p0, sign);
        break;
      case kGroupLane:  // p0 = Q, p1 = P, p2 = lanes
        values.resize(p1 * p2);
        for (int64_t a = 0; a < p1; ++a)
          for (int64_t l = 0; l < p2; ++l) values[a * p2 + l] = Root<T>(a * l, p0, sign);
        break;
    }
    return values;
  }

  std::mutex mutex_;
  std::map<Key, void *> tables_;
};

}  // namespace fft

}  // namespace dace
