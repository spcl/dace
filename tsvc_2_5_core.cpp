// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// TSVC-2.5 extension corpus: native C++ implementations of the kernels
// in ``tsvc_2_5_core.py``. One ``<kernel>_run_timed`` function
// per kernel; signature mirrors the Python counterpart so a single
// driver can dispatch into either backend by name.
//
// Recurring theme: symbolic-step / symbolic-offset / quasi-affine
// kernels where the front-end's view of the dependence vector still
// carries a runtime symbol (``SSYM``, ``K``, ``M``). The compiler must
// either prove the symbol non-zero / non-negative at runtime or emit
// scalar fallback code.

#include <chrono>
#include <cmath>
#include <cstdint>
using clock_highres = std::chrono::high_resolution_clock;

extern "C" {

// ext_strided_load_ssym: dst[i] = src[i * ssym] * scale
void ext_strided_load_ssym_run_timed(double *__restrict__ dst, const double *__restrict__ src, const double scale,
                                     const int len_1d, const int ssym, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    dst[i] = src[i * ssym] * scale;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_strided_load_2: dst[i] = src[i * 2] * scale (constant-stride sibling)
void ext_strided_load_2_run_timed(double *__restrict__ dst, const double *__restrict__ src, const double scale,
                                  const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    dst[i] = src[i * 2] * scale;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_strided_store_ssym: dst[i * ssym] = src[i] * scale
void ext_strided_store_ssym_run_timed(double *__restrict__ dst, const double *__restrict__ src, const double scale,
                                      const int len_1d, const int ssym, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    dst[i * ssym] = src[i] * scale;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_strided_store_2: dst[i * 2] = src[i] * scale
void ext_strided_store_2_run_timed(double *__restrict__ dst, const double *__restrict__ src, const double scale,
                                   const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    dst[i * 2] = src[i] * scale;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_gather_load: dst[i] = src[idx[i]] * scale
void ext_gather_load_run_timed(double *__restrict__ dst, const double *__restrict__ src,
                               const std::int64_t *__restrict__ idx, const double scale, const int len_1d,
                               std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    dst[i] = src[idx[i]] * scale;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_scatter_store: dst[idx[i]] = src[i] * scale (permutation idx)
void ext_scatter_store_run_timed(double *__restrict__ dst, const double *__restrict__ src,
                                 const std::int64_t *__restrict__ idx, const double scale, const int len_1d,
                                 std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    dst[idx[i]] = src[i] * scale;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_floordiv_offset: a[i] = a[i + len_1d / 2] + b[i]
void ext_floordiv_offset_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                                   std::int64_t *time_ns) {
  const int half = len_1d / 2;
  auto t1 = clock_highres::now();
  for (int i = 0; i < half; ++i) {
    a[i] = a[i + half] + b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_floordiv_offset_m: a[i] = a[i + len_1d / m] + b[i]
void ext_floordiv_offset_m_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                                     const int m, std::int64_t *time_ns) {
  const int chunk = len_1d / m;
  auto t1 = clock_highres::now();
  for (int i = 0; i < chunk; ++i) {
    a[i] = a[i + chunk] + b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_modular_wrap: a[(i + k) % len_1d] = b[i]
void ext_modular_wrap_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d, const int k,
                                std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    a[(i + k) % len_1d] = b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_war_unit: a[i] = a[i+1] + b[i] (s121 shape)
void ext_war_unit_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d - 1; ++i) {
    a[i] = a[i + 1] + b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_war_sym: a[i] = a[i + k] + b[i] (symbolic-offset WAR)
void ext_war_sym_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d, const int k,
                           std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d - k; ++i) {
    a[i] = a[i + k] + b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_peel_multi_back: multi-front conflict-write loop
void ext_peel_multi_back_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                                   std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    a[i] = b[i] * 2.0;
    if (i == len_1d - 1) {
      a[len_1d - 2] = a[len_1d - 2] + 1.0;
    } else if (i == len_1d - 2) {
      a[len_1d - 3] = a[len_1d - 3] + 1.0;
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_tile_2d_sym: two-axis tile with symbolic tile size s
void ext_tile_2d_sym_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_2d, const int s,
                               std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int ti = 0; ti < len_2d; ti += s) {
    for (int tj = 0; tj < len_2d; tj += s) {
      for (int i = ti; i < ti + s; ++i) {
        for (int j = tj; j < tj + s; ++j) {
          b[i * len_2d + j] = a[i * len_2d + j] * 2.0;
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -------------------------------------------------------------------------
// TSVC-named symbolic-step variants
// -------------------------------------------------------------------------

// s121_sym_k: a[i] = a[i + k] + b[i] (TSVC s121 with symbolic offset)
void s121_sym_k_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d, const int k,
                          std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d - k; ++i) {
    a[i] = a[i + k] + b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s4113_ssym: a[ip[i * ssym]] = b[ip[i * ssym]] + c[i] (TSVC s4113 with
// symbolic stride on the index array)
void s4113_ssym_run_timed(double *__restrict__ a, const double *__restrict__ b, const double *__restrict__ c,
                          const std::int64_t *__restrict__ ip, const int len_1d, const int ssym,
                          std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d / ssym; ++i) {
    a[ip[i * ssym]] = b[ip[i * ssym]] + c[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// vas_ssym: a[ip[i * ssym]] = b[i] (TSVC vas with symbolic-stride scatter)
void vas_ssym_run_timed(double *__restrict__ a, const double *__restrict__ b, const std::int64_t *__restrict__ ip,
                        const int len_1d, const int ssym, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d / ssym; ++i) {
    a[ip[i * ssym]] = b[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -------------------------------------------------------------------------
// Loop-fission family
// -------------------------------------------------------------------------

// fission_indep_2body: two independent writes sharing three reads
void fission_indep_2body_run_timed(double *__restrict__ a, double *__restrict__ b, const double *__restrict__ x,
                                   const double *__restrict__ y, const double *__restrict__ z, const int len_1d,
                                   std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    a[i] = x[i] * y[i] + z[i];
    b[i] = x[i] - y[i] * z[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fission_dep_then_indep: body A carries a unit-offset dep on `a`, body B independent
void fission_dep_then_indep_run_timed(double *__restrict__ a, double *__restrict__ b, const double *__restrict__ x,
                                      const double *__restrict__ y, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  a[0] = x[0];
  for (int i = 1; i < len_1d; ++i) {
    a[i] = a[i - 1] + x[i];
    b[i] = y[i] * 2.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fission_dep_const_offset: body A carries a constant-offset-2 dep, body B independent
void fission_dep_const_offset_run_timed(double *__restrict__ a, double *__restrict__ b, const double *__restrict__ x,
                                        const double *__restrict__ y, const double *__restrict__ z, const int len_1d,
                                        std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  a[0] = x[0];
  a[1] = x[1];
  for (int i = 2; i < len_1d; ++i) {
    a[i] = a[i - 2] + x[i];
    b[i] = y[i] * z[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fission_dep_sym_offset: symbolic-offset (k) version of fission_dep_const_offset
void fission_dep_sym_offset_run_timed(double *__restrict__ a, double *__restrict__ b, const double *__restrict__ x,
                                      const double *__restrict__ y, const double *__restrict__ z, const int len_1d,
                                      const int k, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = k; i < len_1d; ++i) {
    a[i] = a[i - k] + x[i];
    b[i] = y[i] * z[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -------------------------------------------------------------------------
// Already-tiled stencils
// -------------------------------------------------------------------------

// jacobi2d_tiled_const: 2D 5-point Jacobi pre-tiled with constant tile size 64
void jacobi2d_tiled_const_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_2d,
                                    std::int64_t *time_ns) {
  const int t = 64;
  auto t1 = clock_highres::now();
  for (int ii = 1; ii < len_2d - 1 - t; ii += t) {
    for (int jj = 1; jj < len_2d - 1 - t; jj += t) {
      for (int i = ii; i < ii + t; ++i) {
        for (int j = jj; j < jj + t; ++j) {
          b[i * len_2d + j] = 0.2 * (a[i * len_2d + j] + a[(i - 1) * len_2d + j] + a[(i + 1) * len_2d + j] +
                                     a[i * len_2d + (j - 1)] + a[i * len_2d + (j + 1)]);
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// jacobi2d_tiled_sym: 2D 5-point Jacobi pre-tiled with symbolic tile size t
void jacobi2d_tiled_sym_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_2d, const int t,
                                  std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int ii = 1; ii < len_2d - 1 - t; ii += t) {
    for (int jj = 1; jj < len_2d - 1 - t; jj += t) {
      for (int i = ii; i < ii + t; ++i) {
        for (int j = jj; j < jj + t; ++j) {
          b[i * len_2d + j] = 0.2 * (a[i * len_2d + j] + a[(i - 1) * len_2d + j] + a[(i + 1) * len_2d + j] +
                                     a[i * len_2d + (j - 1)] + a[i * len_2d + (j + 1)]);
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// jacobi2d_double_tiled_const: 2D 5-point Jacobi with constant outer (64) and inner (8) tiles
void jacobi2d_double_tiled_const_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_2d,
                                           std::int64_t *time_ns) {
  const int t1_v = 64;
  const int t2_v = 8;
  auto t1 = clock_highres::now();
  for (int ii = 1; ii < len_2d - 1 - t1_v; ii += t1_v) {
    for (int jj = 1; jj < len_2d - 1 - t1_v; jj += t1_v) {
      for (int iii = ii; iii < ii + t1_v; iii += t2_v) {
        for (int jjj = jj; jjj < jj + t1_v; jjj += t2_v) {
          for (int i = iii; i < iii + t2_v; ++i) {
            for (int j = jjj; j < jjj + t2_v; ++j) {
              b[i * len_2d + j] = 0.2 * (a[i * len_2d + j] + a[(i - 1) * len_2d + j] + a[(i + 1) * len_2d + j] +
                                         a[i * len_2d + (j - 1)] + a[i * len_2d + (j + 1)]);
            }
          }
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// jacobi2d_double_tiled_sym: two-level Jacobi with symbolic outer t1 and inner t2
void jacobi2d_double_tiled_sym_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_2d,
                                         const int t1_v, const int t2_v, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int ii = 1; ii < len_2d - 1 - t1_v; ii += t1_v) {
    for (int jj = 1; jj < len_2d - 1 - t1_v; jj += t1_v) {
      for (int iii = ii; iii < ii + t1_v; iii += t2_v) {
        for (int jjj = jj; jjj < jj + t1_v; jjj += t2_v) {
          for (int i = iii; i < iii + t2_v; ++i) {
            for (int j = jjj; j < jjj + t2_v; ++j) {
              b[i * len_2d + j] = 0.2 * (a[i * len_2d + j] + a[(i - 1) * len_2d + j] + a[(i + 1) * len_2d + j] +
                                         a[i * len_2d + (j - 1)] + a[i * len_2d + (j + 1)]);
            }
          }
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// heat3d_tiled_const: 3D 7-point heat stencil pre-tiled with constant tile size 8
void heat3d_tiled_const_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_3d,
                                  std::int64_t *time_ns) {
  const int t = 8;
  const int n = len_3d;
  auto t1 = clock_highres::now();
  for (int kk = 1; kk < n - 1 - t; kk += t) {
    for (int jj = 1; jj < n - 1 - t; jj += t) {
      for (int ii = 1; ii < n - 1 - t; ii += t) {
        for (int k = kk; k < kk + t; ++k) {
          for (int j = jj; j < jj + t; ++j) {
            for (int i = ii; i < ii + t; ++i) {
              b[(k * n + j) * n + i] =
                  0.125 * (a[((k + 1) * n + j) * n + i] - 2.0 * a[(k * n + j) * n + i] + a[((k - 1) * n + j) * n + i]) +
                  0.125 * (a[(k * n + (j + 1)) * n + i] - 2.0 * a[(k * n + j) * n + i] + a[(k * n + (j - 1)) * n + i]) +
                  0.125 *
                      (a[(k * n + j) * n + (i + 1)] - 2.0 * a[(k * n + j) * n + i] + a[(k * n + j) * n + (i - 1)]) +
                  a[(k * n + j) * n + i];
            }
          }
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// heat3d_tiled_sym: 3D 7-point heat stencil pre-tiled with symbolic tile size t
void heat3d_tiled_sym_run_timed(double *__restrict__ b, const double *__restrict__ a, const int len_3d, const int t,
                                std::int64_t *time_ns) {
  const int n = len_3d;
  auto t1 = clock_highres::now();
  for (int kk = 1; kk < n - 1 - t; kk += t) {
    for (int jj = 1; jj < n - 1 - t; jj += t) {
      for (int ii = 1; ii < n - 1 - t; ii += t) {
        for (int k = kk; k < kk + t; ++k) {
          for (int j = jj; j < jj + t; ++j) {
            for (int i = ii; i < ii + t; ++i) {
              b[(k * n + j) * n + i] =
                  0.125 * (a[((k + 1) * n + j) * n + i] - 2.0 * a[(k * n + j) * n + i] + a[((k - 1) * n + j) * n + i]) +
                  0.125 * (a[(k * n + (j + 1)) * n + i] - 2.0 * a[(k * n + j) * n + i] + a[(k * n + (j - 1)) * n + i]) +
                  0.125 *
                      (a[(k * n + j) * n + (i + 1)] - 2.0 * a[(k * n + j) * n + i] + a[(k * n + j) * n + (i - 1)]) +
                  a[(k * n + j) * n + i];
            }
          }
        }
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -------------------------------------------------------------------------
// ECRAD-style clamped reduction
// -------------------------------------------------------------------------

// ecrad_clamped_reduction: clamp(exp(-sqrt(max(x*x+y*y, 1e-12)) * d), 0, 1)
void ecrad_clamped_reduction_run_timed(double *__restrict__ out, const double *__restrict__ x,
                                       const double *__restrict__ y, const double *__restrict__ d, const int len_1d,
                                       std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    double k_val = std::sqrt(std::fmax(x[i] * x[i] + y[i] * y[i], 1e-12));
    double e = std::exp(-k_val * d[i]);
    double clamped = e < 1.0 ? e : 1.0;
    out[i] = clamped > 0.0 ? clamped : 0.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -------------------------------------------------------------------------
// Masked stores
// -------------------------------------------------------------------------

// masked_store_const: predicated store keyed on int mask
void masked_store_const_run_timed(double *__restrict__ a, const double *__restrict__ b,
                                  const std::int64_t *__restrict__ mask, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    if (mask[i] > 0) {
      a[i] = b[i];
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// masked_store_sym: predicated store keyed on double comparison against scalar k
void masked_store_sym_run_timed(double *__restrict__ a, const double *__restrict__ b,
                                const double *__restrict__ threshold_data, const int len_1d, const double k,
                                std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    if (threshold_data[i] > k) {
      a[i] = b[i];
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -------------------------------------------------------------------------
// Quasi-affine subscript ranges (even/odd, pairwise, mod-K, floor-div)
// -------------------------------------------------------------------------

// quasi_affine_reduce_even: sum a[i] for i in 0..len_1d step 2
void quasi_affine_reduce_even_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_1d,
                                        std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double acc = 0.0;
  for (int i = 0; i < len_1d; i += 2) {
    acc += a[i];
  }
  out[0] = acc;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// quasi_affine_reduce_odd: sum a[i] for i in 1..len_1d step 2
void quasi_affine_reduce_odd_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_1d,
                                       std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double acc = 0.0;
  for (int i = 1; i < len_1d; i += 2) {
    acc += a[i];
  }
  out[0] = acc;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// quasi_affine_pairwise_sum: b[i] = a[2*i] + a[2*i + 1]
void quasi_affine_pairwise_sum_run_timed(const double *__restrict__ a, double *__restrict__ b, const int len_1d,
                                         std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    b[i] = a[2 * i] + a[2 * i + 1];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// quasi_affine_mod_k_stripe: a[i] = b[i] * 2.0 if i % k == 0 else c[i]
void quasi_affine_mod_k_stripe_run_timed(double *__restrict__ a, const double *__restrict__ b,
                                         const double *__restrict__ c, const int len_1d, const int k,
                                         std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    if ((i % k) == 0) {
      a[i] = b[i] * 2.0;
    } else {
      a[i] = c[i];
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// quasi_affine_floor_div_scatter: b[i / 2] += a[i] (pair-stripe reduction)
void quasi_affine_floor_div_scatter_run_timed(const double *__restrict__ a, double *__restrict__ b, const int len_1d,
                                              std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < 2 * len_1d; ++i) {
    b[i / 2] += a[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// wavefront2d: a[i,j] = 0.25 * (a[i,j] + a[i-1,j] + a[i,j-1] + a[i-1,j-1])
void wavefront2d_run_timed(double *__restrict__ a, const int len_2d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 1; i < len_2d; ++i) {
    for (int j = 1; j < len_2d; ++j) {
      a[i * len_2d + j] = 0.25 * (a[i * len_2d + j] + a[(i - 1) * len_2d + j] + a[i * len_2d + (j - 1)] +
                                  a[(i - 1) * len_2d + (j - 1)]);
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_break_find_first (s481): if d[i] < 0 break; a[i] = a[i] + b[i]*c[i]
void ext_break_find_first_run_timed(double *__restrict__ a, const double *__restrict__ b,
                                    const double *__restrict__ c, const double *__restrict__ d, const int len_1d,
                                    std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    if (d[i] < 0.0) break;
    a[i] = a[i] + b[i] * c[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_break_post_body (s482): a[i] = a[i] + b[i]*c[i]; if c[i] > b[i] break
void ext_break_post_body_run_timed(double *__restrict__ a, const double *__restrict__ b,
                                   const double *__restrict__ c, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    a[i] = a[i] + b[i] * c[i];
    if (c[i] > b[i]) break;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ext_break_capture (s332): first i with a[i] > k -> capture index + value, break
void ext_break_capture_run_timed(const double *__restrict__ a, std::int64_t *__restrict__ out_index,
                                 double *__restrict__ out_value, const int len_1d, const double k,
                                 std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  out_index[0] = -1;
  out_value[0] = -1.0;
  for (int i = 0; i < len_1d; ++i) {
    if (a[i] > k) {
      out_index[0] = i;
      out_value[0] = a[i];
      break;
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// cond_reduce_sum (s3111): if a[i] > 0 out += a[i]
void cond_reduce_sum_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_1d,
                               std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  out[0] = 0.0;
  for (int i = 0; i < len_1d; ++i) {
    if (a[i] > 0.0) {
      out[0] = out[0] + a[i];
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// cond_reduce_sym: if a[i] > k out += a[i]
void cond_reduce_sym_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_1d,
                               const double k, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  out[0] = 0.0;
  for (int i = 0; i < len_1d; ++i) {
    if (a[i] > k) {
      out[0] = out[0] + a[i];
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// iv_additive: s = 0; for i in [0, len_1d): s += 1.5; out[0] = s
void iv_additive_run_timed(double *__restrict__ out, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double s = 0.0;
  for (int i = 0; i < len_1d; ++i) {
    s = s + 1.5;
  }
  out[0] = s;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// iv_multiplicative: s = 1; for i in [0, len_1d): s *= 0.99; out[0] = s
void iv_multiplicative_run_timed(double *__restrict__ out, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double s = 1.0;
  for (int i = 0; i < len_1d; ++i) {
    s = s * 0.99;
  }
  out[0] = s;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// argmax_value (s314): x = a[0]; for i: if a[i] > x: x = a[i]; out[0] = x
void argmax_value_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_1d,
                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double x = a[0];
  for (int i = 1; i < len_1d; ++i) {
    if (a[i] > x) {
      x = a[i];
    }
  }
  out[0] = x;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// argmin_value (s316): x = a[0]; for i: if a[i] < x: x = a[i]; out[0] = x
void argmin_value_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_1d,
                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double x = a[0];
  for (int i = 1; i < len_1d; ++i) {
    if (a[i] < x) {
      x = a[i];
    }
  }
  out[0] = x;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// neg_stride_rev (s112): for i = len_1d-1 .. 0: a[i] = b[i] + 1
void neg_stride_rev_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                              std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = len_1d - 1; i >= 0; --i) {
    a[i] = b[i] + 1.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// reroll_saxpy7 (s351): 7x (prime) hand-unrolled saxpy over a step-7 loop
void reroll_saxpy7_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                             std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; i += 7) {
    a[i] = a[i] + b[i] * 2.0;
    a[i + 1] = a[i + 1] + b[i + 1] * 2.0;
    a[i + 2] = a[i + 2] + b[i + 2] * 2.0;
    a[i + 3] = a[i + 3] + b[i + 3] * 2.0;
    a[i + 4] = a[i + 4] + b[i + 4] * 2.0;
    a[i + 5] = a[i + 5] + b[i + 5] * 2.0;
    a[i + 6] = a[i + 6] + b[i + 6] * 2.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// scan_strided_2: a[i] = a[i-2] + x[i] (stride-2 prefix sum -> two scans)
void scan_strided_2_run_timed(double *__restrict__ a, const double *__restrict__ x, const int len_1d,
                              std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 2; i < len_1d; ++i) {
    a[i] = a[i - 2] + x[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// scan_strided_sym: a[i] = a[i-k] + x[i] (stride-k prefix sum -> k scans)
void scan_strided_sym_run_timed(double *__restrict__ a, const double *__restrict__ x, const int len_1d, const int k,
                                std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = k; i < len_1d; ++i) {
    a[i] = a[i - k] + x[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// scan_multi_carry: a[i] = a[i-1] + x[i]; b[i] = b[i-1] * y[i] (two scans, add + mul)
void scan_multi_carry_run_timed(double *__restrict__ a, double *__restrict__ b, const double *__restrict__ x,
                                const double *__restrict__ y, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 1; i < len_1d; ++i) {
    a[i] = a[i - 1] + x[i];
    b[i] = b[i - 1] * y[i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// scan_conditional: out[i] = (mask[i] > 0) ? out[i-1] + delta[i] : out[i-1]
void scan_conditional_run_timed(double *__restrict__ out, const double *__restrict__ delta,
                                const std::int64_t *__restrict__ mask, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 1; i < len_1d; ++i) {
    if (mask[i] > 0) {
      out[i] = out[i - 1] + delta[i];
    } else {
      out[i] = out[i - 1];
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// scan_multi_5carry: five independent prefix sums acc[r][i] = acc[r][i-1] + delta[r][i]
void scan_multi_5carry_run_timed(double *__restrict__ acc, const double *__restrict__ delta, const int len_1d,
                                 std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 1; i < len_1d; ++i) {
    acc[0 * len_1d + i] = acc[0 * len_1d + (i - 1)] + delta[0 * len_1d + i];
    acc[1 * len_1d + i] = acc[1 * len_1d + (i - 1)] + delta[1 * len_1d + i];
    acc[2 * len_1d + i] = acc[2 * len_1d + (i - 1)] + delta[2 * len_1d + i];
    acc[3 * len_1d + i] = acc[3 * len_1d + (i - 1)] + delta[3 * len_1d + i];
    acc[4 * len_1d + i] = acc[4 * len_1d + (i - 1)] + delta[4 * len_1d + i];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// argmax_with_index (s315): running max carrying value + index
void argmax_with_index_run_timed(const double *__restrict__ a, double *__restrict__ out_value,
                                 std::int64_t *__restrict__ out_index, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  double x = a[0];
  std::int64_t idx = 0;
  for (int i = 1; i < len_1d; ++i) {
    if (a[i] > x) {
      x = a[i];
      idx = i;
    }
  }
  out_value[0] = x;
  out_index[0] = idx;
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// reroll_gather (s353): 7x (prime) hand-unrolled gather saxpy a[i+k] += b[ip[i+k]] * 2
void reroll_gather_run_timed(double *__restrict__ a, const double *__restrict__ b,
                             const std::int64_t *__restrict__ ip, const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; i += 7) {
    a[i] = a[i] + b[ip[i]] * 2.0;
    a[i + 1] = a[i + 1] + b[ip[i + 1]] * 2.0;
    a[i + 2] = a[i + 2] + b[ip[i + 2]] * 2.0;
    a[i + 3] = a[i + 3] + b[ip[i + 3]] * 2.0;
    a[i + 4] = a[i + 4] + b[ip[i + 4]] * 2.0;
    a[i + 5] = a[i + 5] + b[ip[i + 5]] * 2.0;
    a[i + 6] = a[i + 6] + b[ip[i + 6]] * 2.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// thomas_solve: tridiagonal forward elimination + backward substitution
void thomas_solve_run_timed(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c,
                            double *__restrict__ d, double *__restrict__ x, const int len_1d,
                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  c[0] = c[0] / b[0];
  d[0] = d[0] / b[0];
  for (int i = 1; i < len_1d; ++i) {
    double m = b[i] - a[i] * c[i - 1];
    c[i] = c[i] / m;
    d[i] = (d[i] - a[i] * d[i - 1]) / m;
  }
  x[len_1d - 1] = d[len_1d - 1];
  for (int i = len_1d - 2; i >= 0; --i) {
    x[i] = d[i] - c[i] * x[i + 1];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// reduce_inner_carry: out[i] = sum_j a[i][j] (outer parallel, inner carried)
void reduce_inner_carry_run_timed(const double *__restrict__ a, double *__restrict__ out, const int len_2d,
                                  std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_2d; ++i) {
    double s = 0.0;
    for (int j = 0; j < len_2d; ++j) {
      s = s + a[i * len_2d + j];
    }
    out[i] = s;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// config_select_branch: per-element guard (if-inside form) selects output array
void config_select_branch_run_timed(double *__restrict__ out_a, double *__restrict__ out_b,
                                    const double *__restrict__ src, const int len_1d, const int k,
                                    std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    if (k > 0) {
      out_a[i] = src[i] * 2.0;
    } else {
      out_b[i] = src[i] + 1.0;
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// move_if_data_dep_nest: data-dependent guard cond[i] in the MIDDLE of a 2D nest
void move_if_data_dep_nest_run_timed(double *__restrict__ out, const double *__restrict__ src,
                                     const double *__restrict__ cond, const int len_2d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_2d; ++i) {
    if (cond[i] > 0.0) {
      for (int j = 0; j < len_2d; ++j) {
        out[i * len_2d + j] = src[i * len_2d + j] * 2.0;
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fuse_move_ifs: two guarded nests (data-dep cond[i], then loop-invariant k) that fuse after moving ifs in
void fuse_move_ifs_run_timed(double *__restrict__ a, double *__restrict__ b, const double *__restrict__ src,
                             const double *__restrict__ cond, const int len_2d, const int k,
                             std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_2d; ++i) {
    if (cond[i] > 0.0) {
      for (int j = 0; j < len_2d; ++j) {
        a[i * len_2d + j] = src[i * len_2d + j] * 2.0;
      }
    }
  }
  if (k > 0) {
    for (int i = 0; i < len_2d; ++i) {
      for (int j = 0; j < len_2d; ++j) {
        b[i * len_2d + j] = src[i * len_2d + j] + 1.0;
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fuse_stencil_through_transient: tmp[i]=a[i-1]+a[i]+a[i+1]; out[i]=tmp[i]*tmp[i+1] (fused, offset-corrected)
void fuse_stencil_through_transient_run_timed(double *__restrict__ out, const double *__restrict__ a,
                                              const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 1; i < len_1d - 2; ++i) {
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fuse_diamond: t=a*a; u=t+1; v=t-1; out=u*v (fused diamond)
void fuse_diamond_run_timed(double *__restrict__ out, const double *__restrict__ a, const int len_1d,
                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    double t = a[i] * a[i];
    out[i] = (t + 1.0) * (t - 1.0);
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// loop_to_map_disjoint_strided: a[2*i] = b[i]+1; a[2*i+1] = b[i]*2 (disjoint, parallel)
void loop_to_map_disjoint_strided_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    a[2 * i] = b[i] + 1.0;
    a[2 * i + 1] = b[i] * 2.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// loop_to_map_overlap_seq: a[5*i] = b[i]+1; a[3*i] = b[i]*2 (overlap -> sequential)
void loop_to_map_overlap_seq_run_timed(double *__restrict__ a, const double *__restrict__ b, const int len_1d,
                                       std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d / 5; ++i) {
    a[5 * i] = b[i] + 1.0;
    a[3 * i] = b[i] * 2.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// loop_to_map_threshold_gather: per (i,k) threshold on gathered w[idx[i],k] selects the update
void loop_to_map_threshold_gather_run_timed(double *__restrict__ out, const double *__restrict__ x,
                                            const double *__restrict__ y, const double *__restrict__ w,
                                            const std::int64_t *__restrict__ idx, const int len_2d,
                                            std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_2d; ++i) {
    for (int k = 0; k < len_2d; ++k) {
      if (w[idx[i] * len_2d + k] > 0.5) {
        out[i * len_2d + k] = x[i * len_2d + k] * 2.0;
      } else {
        out[i * len_2d + k] = y[i * len_2d + k] + 1.0;
      }
    }
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fission_gather_2body: b[i] = a[idx[i]]; e[i] = c[idx[i]] (two independent gathers)
void fission_gather_2body_run_timed(double *__restrict__ b, double *__restrict__ e, const double *__restrict__ a,
                                    const double *__restrict__ c, const std::int64_t *__restrict__ idx,
                                    const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    b[i] = a[idx[i]];
    e[i] = c[idx[i]];
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// fission_scatter_2body: b[idx[i]] = a[i]*2; e[idx[i]] = c[i]+1 (two independent scatters, idx perm)
void fission_scatter_2body_run_timed(double *__restrict__ b, double *__restrict__ e, const double *__restrict__ a,
                                     const double *__restrict__ c, const std::int64_t *__restrict__ idx,
                                     const int len_1d, std::int64_t *time_ns) {
  auto t1 = clock_highres::now();
  for (int i = 0; i < len_1d; ++i) {
    b[idx[i]] = a[i] * 2.0;
    e[idx[i]] = c[i] + 1.0;
  }
  auto t2 = clock_highres::now();
  time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

}  // extern "C"
