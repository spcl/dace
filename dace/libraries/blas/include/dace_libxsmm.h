// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cblas.h>
#include <libxsmm.h>

#include <cstdint>

namespace dace {

namespace blas {

inline void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, double alpha, const double* a, int lda,
                 const double* b, int ldb, double beta, double* c, int ldc) {
  cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha, const float* a, int lda,
                 const float* b, int ldb, float beta, float* c, int ldc) {
  cblas_sgemm(CblasColMajor, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// One real GEMM shape with its LIBXSMM kernel looked up once. ``libxsmm_?gemm`` resolves the kernel on every call
// and hands a transposed operand to BLAS: 180 ns a call against 20 to 60 ns for the 4x8 to 5x5 products a batch
// repeats per element. It also computes with ``alpha = 1`` and ``beta`` 0 or 1 whatever it is passed (2.1.0), so
// it is not called at all. A transposed operand of at most ``kPack`` elements is packed untransposed on the stack
// and the kernel serves it too; a shape past either bound, or coefficients a kernel has no form for, is the BLAS
// call ``libxsmm_?gemm`` would make.
template <typename T>
class SmallGemm {
 public:
  // Most elements of a transposed operand packed per call.
  static constexpr int64_t kPack = 1024;
  // Largest ``m * n * k`` LIBXSMM generates a kernel for by default; BLAS takes a larger product.
  static constexpr int64_t kKernel = 64 * 64 * 64;

  SmallGemm(char ta, char tb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint lda,
            libxsmm_blasint ldb, libxsmm_blasint ldc, T alpha, T beta)
      : ta_(ta), tb_(tb), m_(m), n_(n), k_(k), lda_(lda), ldb_(ldb), ldc_(ldc), alpha_(alpha), beta_(beta) {
    const bool pack_a = ta != 'N', pack_b = tb != 'N';
    if (alpha != T(1) || (beta != T(0) && beta != T(1)) || m < 1 || n < 1 || k < 1 ||
        int64_t(m) * n * k > kKernel || (pack_a && int64_t(m) * k > kPack) || (pack_b && int64_t(k) * n > kPack)) {
      return;
    }
    const libxsmm_datatype type = sizeof(T) == sizeof(double) ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
    kernel_ = libxsmm_dispatch_gemm(
        libxsmm_create_gemm_shape(m, n, k, pack_a ? m : lda, pack_b ? k : ldb, ldc, type, type, type, type),
        beta == T(0) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE);
  }

  // ``c = alpha * op(a) * op(b) + beta * c``, column-major.
  void operator()(const T* a, const T* b, T* c) const {
    if (kernel_ == nullptr) {
      gemm(ta_ == 'N' ? CblasNoTrans : CblasTrans, tb_ == 'N' ? CblasNoTrans : CblasTrans, m_, n_, k_, alpha_, a,
           lda_, b, ldb_, beta_, c, ldc_);
      return;
    }
    T packed_a[kPack], packed_b[kPack];
    if (ta_ != 'N') {
      for (libxsmm_blasint l = 0; l < k_; ++l) {
        for (libxsmm_blasint i = 0; i < m_; ++i) {
          packed_a[i + l * m_] = a[l + i * lda_];
        }
      }
      a = packed_a;
    }
    if (tb_ != 'N') {
      for (libxsmm_blasint j = 0; j < n_; ++j) {
        for (libxsmm_blasint l = 0; l < k_; ++l) {
          packed_b[l + j * k_] = b[j + l * ldb_];
        }
      }
      b = packed_b;
    }
    libxsmm_gemm_param args;
    args.a.primary = const_cast<T*>(a);
    args.b.primary = const_cast<T*>(b);
    args.c.primary = c;
    kernel_(&args);
  }

 private:
  const char ta_, tb_;
  const libxsmm_blasint m_, n_, k_, lda_, ldb_, ldc_;
  const T alpha_, beta_;
  libxsmm_gemmfunction kernel_ = nullptr;
};

}  // namespace blas

}  // namespace dace
