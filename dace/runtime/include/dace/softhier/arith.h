#pragma once

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
#include "flex_dump.h"
#define floor(x) ((x))
#define Mod(x, y) ((x) % (y))
#define Min(x, y) ((x) < (y) ? (x) : (y))
#define dace_cluster_index(x,y,dim_x)         ((y)*(dim_x)+(x))
#define dace_remote_xy(x,y,offset,dim_x)       (ARCH_CLUSTER_TCDM_REMOTE+dace_cluster_index(x,y,dim_x)*ARCH_CLUSTER_TCDM_SIZE+offset)


// SoftHier Global Code
#ifndef _SOFTHIER_MACROS_DEFINED
#define _SOFTHIER_MACROS_DEFINED
#define STR(x) #x
#define XSTR(x) STR(x)
#endif _SOFTHIER_MACROS_DEFINED

inline void _softhier_vadd_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width)
{
    //flex_intra_cluster_sync();
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfadd.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
    //flex_intra_cluster_sync();
}

/*vc = va * vb*/
inline void _softhier_vmul_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    //flex_intra_cluster_sync();
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfmul.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
    //flex_intra_cluster_sync();
}

/*vc = va - vb*/
inline void _softhier_vsub_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    //flex_intra_cluster_sync();
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfsub.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
    //flex_intra_cluster_sync();
}


// Vector addition with scalar: vc = va + scalar
inline void _softhier_vadd_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t avl;
        while(vector_width > 0) {
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma": "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfadd.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}


// Vector multiplication with scalar: vc = va * scalar
inline void _softhier_vmul_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t avl;
        while(vector_width > 0) {
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfmul.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

// Vector division: vc = va / vb
inline void _softhier_vdiv_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t avl;
        while(vector_width > 0) {
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle32.v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfdiv.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

// Vector division with scalar: vc = va / scalar
inline void _softhier_vdiv_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t avl;
        while(vector_width > 0) {
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfdiv.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

// Vector exponent: vb = exp(va)
static inline void _softhier_vexp_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vector_width)
{
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;

        while(vlen > 0) {
            /* Set vector length */
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            /* Load input vector into v0 (x@v0) */
            asm volatile("vle" XSTR(32) ".v v8, (%0)" :: "r"(va_addr));
            asm volatile(".word %0"::"i"(0x32041857));  // vfexp.vv v16, v8
            // Store result
            asm volatile("vse" XSTR(32) ".v v16, (%0)" :: "r"(vb_addr));

            /* Update pointers and length */
            vlen -= avl;
            va_addr += 4 * avl;
            vb_addr += 4 * avl;
        }
    }
}


// Vector exponent: vb = log(va)
static inline void _softhier_vlog_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vector_width)
{
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        const float SQRTHF = 0.707106781186547524f;
        const float PX1logf = 7.0376836292E-2f;
        const float PX2logf = -1.1514610310E-1f;
        const float PX3logf = 1.1676998740E-1f;
        const float PX4logf = -1.2420140846E-1f;
        const float PX5logf = 1.4249322787E-1f;
        const float PX6logf = -1.6668057665E-1f;
        const float PX7logf = 2.0000714765E-1f;
        const float PX8logf = -2.4999993993E-1f;
        const float PX9logf = 3.3333331174E-1f;
        const float const_c1 = 0.693359375f;
        const float const_c2 = -2.12194440e-4f;
        const float const_half = 0.5f;
        const float neg_const_half = -0.5f;
        const float const_1 = 1.0f;
        const uint32_t exp_mask = 0xff;
        const uint32_t exp_bias = 127;
        const uint32_t exp_shift = 23;
        const uint32_t mant_mask = 0x807fffff;
        const uint32_t mant_set = 0x3f000000;


        while(vlen > 0) {
            /* Set vector length */
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));

            /* Load input vector into v0 */
            // C: x = get_mant_exponent_f(x, &fe);
            asm volatile("vle" XSTR(32) ".v v0, (%0)" :: "r"(va_addr));

            /* ===== EXTRACT MANTISSA AND EXPONENT ===== */
            
            /* v8 = v0 (copy for bit manipulation) */
            // C: u.f = x;
            asm volatile("vmv.v.v v8, v0");
            
            /* Reinterpret as integers for bit ops */
            // C: (u.i >> 23) & 0xff
            asm volatile("vsrl.vx v16, v8, %0" :: "r"(exp_shift));  // Shift right by 23
            asm volatile("vand.vx v16, v16, %0" :: "r"(exp_mask));  // Mask to get exponent
            
            /* v16 now contains biased exponent, convert to fe (unbiased) */
            // C: exp -= 127; *fe = (float)exp;
            asm volatile("vsub.vx v16, v16, %0" :: "r"(exp_bias));  // Remove bias
            asm volatile("vfcvt.f.x.v v16, v16");                   // Convert to float -> fe in v16
            
            /* Extract mantissa: set exponent bits to bias (127) */
            // C: u.i = (u.i & 0x807fffffU) | 0x3f000000U;
            asm volatile("vand.vx v8, v8, %0" :: "r"(mant_mask));   // Keep sign and mantissa
            asm volatile("vor.vx v8, v8, %0" :: "r"(mant_set));     // Set exponent to 0 (bias 127)
            // v8 now contains mantissa in [0.5, 1.0) -> this is x
            
            /* ===== BLENDING ===== */
            
            /* Check if x > SQRTHF */
            // C: if (x > SQRTHF) { fe += 1.0f; } else { x += x; }
            asm volatile("vmfgt.vf v0, v8, %0" :: "r"(SQRTHF));     // Mask: v0 = (x > SQRTHF)
            
            /* For elements where mask is true: fe += 1.0 */
            // C: fe += 1.0f;
            asm volatile("vfadd.vf v24, v16, %0" :: "r"(const_1));  // v24 = fe + 1.0
            asm volatile("vmerge.vvm v16, v16, v24, v0");           // v16 = mask ? v24 : v16
            
            /* For elements where mask is false: x += x (i.e., x *= 2) */
            // C: x += x;
            asm volatile("vfadd.vv v24, v8, v8");                   // v24 = x + x
            asm volatile("vmnot.m v1, v0");                         // v1 = ~v0 (inverse mask)
            asm volatile("vmerge.vvm v8, v8, v24, v1");             // v8 = ~mask ? v24 : v8
            
            /* x -= 1.0 */
            // C: x -= 1.0f;
            asm volatile("vfsub.vf v8, v8, %0" :: "r"(const_1));    // v8 = x - 1.0
            
            /* ===== COMPUTE x^2 ===== */
            
            /* x2 = x * x */
            // C: const float x2 = x * x;
            asm volatile("vfmul.vv v24, v8, v8");                   // v24 = x^2
            
            /* ===== POLYNOMIAL EVALUATION ===== */
            
            /* res = get_log_poly_f(x) */
            // C: float y = x * PX1logf;
            asm volatile("vfmul.vf v32, v8, %0" :: "r"(PX1logf));
            
            // C: y += PX2logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX2logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX3logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX3logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX4logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX4logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX5logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX5logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX6logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX6logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX7logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX7logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX8logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX8logf));
            
            // C: y *= x;
            asm volatile("vfmul.vv v32, v32, v8");
            
            // C: y += PX9logf;
            asm volatile("vfadd.vf v32, v32, %0" :: "r"(PX9logf));
            
            /* res *= x2 * x (i.e., res *= x^3) */
            // C: res *= x2 * x;
            asm volatile("vfmul.vv v32, v32, v24");                 // res *= x2
            asm volatile("vfmul.vv v32, v32, v8");                  // res *= x
            
            /* ===== FINAL FORMULA ===== */
            
            /* res += -2.12194440e-4f * fe */
            // C: res += -2.12194440e-4f * fe;
            asm volatile("vfmul.vf v40, v16, %0" :: "r"(const_c2)); // v40 = c2 * fe
            asm volatile("vfadd.vv v32, v32, v40");                 // res += v40
            
            /* res += -0.5f * x2 */
            // C: res += -0.5f * x2;
            asm volatile("vfmul.vf v40, v24, %0" :: "r"(neg_const_half)); // v40 = -0.5 * x2
            asm volatile("vfadd.vv v32, v32, v40");                    // res += v40
            
            /* res = x + res */
            // C: res = x + res;
            asm volatile("vfadd.vv v32, v8, v32");                  // res = x + res
            
            /* res += 0.693359375f * fe */
            // C: res += 0.693359375f * fe;
            asm volatile("vfmul.vf v40, v16, %0" :: "r"(const_c1)); // v40 = c1 * fe
            asm volatile("vfadd.vv v32, v32, v40");                 // res += v40

            // Store result
            asm volatile("vse" XSTR(32) ".v v32, (%0)" :: "r"(vb_addr));

            /* Update pointers and length */
            vlen -= avl;
            va_addr += 4 * avl;
            vb_addr += 4 * avl;
        }
    }
}


inline void _softhier_vadd_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfadd.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vmin_vs_(uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width, float scalar) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(vb_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfmin.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vmax_vs_(uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width, float scalar) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(vb_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfmax.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vgt_vs_(uint32_t va_addr, uint32_t vc_addr, uint32_t vector_width, float threshold, float true_val, float false_val) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v16, %0" ::"f"(threshold));
            asm volatile("vfmv.v.f v24, %0" ::"f"(true_val));
            asm volatile("vfmv.v.f v0, %0" ::"f"(false_val));
            asm volatile("vmfgt.vv v1, v8, v16");
            asm volatile("vmerge.vvm v8, v0, v24, v1");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vdiv_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfdiv.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vmul_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfmul.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vsub_sv_(uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width, float scalar) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(vb_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfsub.vv v8, v0, v8");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vmin_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfmin.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vmax_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfmax.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}

inline void _softhier_vgt_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width, float true_val, float false_val) {
    if (flex_is_first_core()) {
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v16, (%0)" ::"r"(vb_addr));
            asm volatile("vfmv.v.f v24, %0" ::"f"(true_val));
            asm volatile("vfmv.v.f v0, %0" ::"f"(false_val));
            asm volatile("vmfgt.vv v1, v8, v16");
            asm volatile("vmerge.vvm v8, v0, v24, v1");
            asm volatile("vse" XSTR(32) ".v v8, (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }
    }
}


void softhier_device_0_0_11(const uint32_t local_ZA, const uint32_t local_ZICEFRAC, const uint32_t local_ZLI, const uint32_t local_ZLIQFRAC, const uint32_t local_ZQX) {
    size_t local_ZA_tile_height = 4;
    size_t local_ZA_tile_width = 8192;
    size_t local_ZICEFRAC_tile_height = 4;
    size_t local_ZICEFRAC_tile_width = 8192;
    size_t local_ZLI_tile_height = 4;
    size_t local_ZLI_tile_width = 8192;
    size_t local_ZLIQFRAC_tile_height = 4;
    size_t local_ZLIQFRAC_tile_width = 8192;
    size_t local_ZQX_tile_height = 4;
    size_t local_ZQX_tile_width = 8192;
    {
        // TEST KERNEL SCOPE
        const uint32_t cluster_id = flex_get_cluster_id();
        {
            for (int sdi = 0; sdi < 1; sdi += 1) {
                for (int sdj = 0; sdj < 1; sdj += 1) {
                    {
                        // TEST DEVICE SCOPE
                        int sbi = cluster_id % 1;
                        int sbj = cluster_id / 1;
                        if (sbj <= 0) {
                            if (sbi <= 0) {
                                // Minels: [0, 0], Maxels: [0, 0]
                                {
                                    for (int jk = 0; jk < 4; jk += 1) {
                                        flex_intra_cluster_sync();
                                        // Nested SDFG cloud_fraction_update_113_4_114_8 begin
                                        cloud_fraction_update_113_4_114_8_0_0_8(local_ZQX, local_ZA, local_ZICEFRAC, local_ZLI, local_ZLIQFRAC, jk);
                                    }
                                }
                            }
                        }
                    }
                    flex_intra_cluster_sync();
                    // Finished deivelevel scope
                }
            }
        }
    }
}