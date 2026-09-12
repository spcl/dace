# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""RV-1a: compile + numerically validate the scalar/common-header
``horizontal_reduce_<op>`` primitives.

The vectorized reduction expansion (RV-1b) emits
``horizontal_reduce_<op><T, W>(buf)`` to fold a W-wide vector
accumulator to a scalar. The portable baseline is the log-depth
pairwise tree in ``cpu_vectorizable_math_common.h``; the scalar backend
delegates straight to it. This test compiles the real codegen header
(scalar-fallback path) with ``g++`` and checks every supported op
against a reference fold, including the odd-width and width-1 edges.
The avx512 / neon / sve single-instruction variants COMPILE wherever their target compiler
exists -- that needs no such CPU, and gating the compile on one is how an x86 box without
AVX-512 came to skip the whole intrinsic path and report green. Only EXECUTION is gated,
by mark (``avx512`` / ``arm_cross``), never by a runtime skip.
"""
import platform
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

import dace

INCLUDE = str(Path(dace.__file__).parent / "runtime" / "include")
#: The standard DaCe builds generated code with. Pinning c++17 here un-gates the day the runtime
#: headers reach for a C++20 feature -- ``dace/codegen/common.py`` clamps generated code to >= 20.
STD = f"-std=c++{dace.Config.get('compiler', 'cpp_standard')}"
HOST_IS_X86 = platform.machine().lower() in ("x86_64", "amd64", "x64")

SRC = textwrap.dedent("""
    #include "dace/cpu_vectorizable_math.h"
    #include <cstdio>
    #include <cmath>
    int main() {
      double a[8] = {1,3,2,5,4,0.5,8,6};
      double s=0,p=1,mx=a[0],mn=a[0];
      for(int i=0;i<8;i++){s+=a[i];p*=a[i];mx=std::max(mx,a[i]);mn=std::min(mn,a[i]);}
      int ok=1;
      if(std::fabs(horizontal_reduce_add<double,8>(a)-s)>1e-9){printf("add FAIL\\n");ok=0;}
      if(std::fabs(horizontal_reduce_mul<double,8>(a)-p)>1e-9){printf("mul FAIL\\n");ok=0;}
      if(horizontal_reduce_max<double,8>(a)!=mx){printf("max FAIL\\n");ok=0;}
      if(horizontal_reduce_min<double,8>(a)!=mn){printf("min FAIL\\n");ok=0;}
      double a5[5]={3,1,4,1,5};
      if(horizontal_reduce_min<double,5>(a5)!=1.0){printf("min5 FAIL\\n");ok=0;}
      if(horizontal_reduce_max<double,5>(a5)!=5.0){printf("max5 FAIL\\n");ok=0;}
      double a1[1]={42};
      if(horizontal_reduce_add<double,1>(a1)!=42.0){printf("w1 FAIL\\n");ok=0;}
      int bi[4]={11,6,12,5};
      int rb=bi[0],ro=bi[0],rx=bi[0];
      for(int i=1;i<4;i++){rb&=bi[i];ro|=bi[i];rx^=bi[i];}
      if(horizontal_reduce_band<int,4>(bi)!=rb){printf("band FAIL\\n");ok=0;}
      if(horizontal_reduce_bor<int,4>(bi)!=ro){printf("bor FAIL\\n");ok=0;}
      if(horizontal_reduce_bxor<int,4>(bi)!=rx){printf("bxor FAIL\\n");ok=0;}
      printf(ok?"ALL OK\\n":"FAILED\\n");
      return ok?0:1;
    }
    """)

AVX512_SRC = textwrap.dedent("""
    #define __DACE_USE_INTRINSICS 1
    #define __DACE_USE_AVX512 1
    #include "dace/cpu_vectorizable_math.h"
    #include <cstdio>
    #include <cmath>
    int main() {
      double a[8]={1,3,2,5,4,0.5,8,6};
      double s=0,p=1,mx=a[0],mn=a[0];
      for(int i=0;i<8;i++){s+=a[i];p*=a[i];mx=std::max(mx,a[i]);mn=std::min(mn,a[i]);}
      int ok=1;
      if(std::fabs(horizontal_reduce_add<double,8>(a)-s)>1e-9){printf("add FAIL\\n");ok=0;}
      if(std::fabs(horizontal_reduce_mul<double,8>(a)-p)>1e-9){printf("mul FAIL\\n");ok=0;}
      if(horizontal_reduce_max<double,8>(a)!=mx){printf("max FAIL\\n");ok=0;}
      if(horizontal_reduce_min<double,8>(a)!=mn){printf("min FAIL\\n");ok=0;}
      double a13[13]; for(int i=0;i<13;i++) a13[i]=i*0.5+1;
      double s13=0; for(int i=0;i<13;i++) s13+=a13[i];
      if(std::fabs(horizontal_reduce_add<double,13>(a13)-s13)>1e-9){printf("add13 FAIL\\n");ok=0;}
      double a3[3]={2,7,4};
      if(horizontal_reduce_max<double,3>(a3)!=7.0){printf("max3 FAIL\\n");ok=0;}
      float f[16]; for(int i=0;i<16;i++) f[i]=i+1; float fs=0; for(int i=0;i<16;i++) fs+=f[i];
      if(std::fabs(horizontal_reduce_add<float,16>(f)-fs)>1e-3f){printf("addf FAIL\\n");ok=0;}
      int bi[4]={11,6,12,5}; int rb=bi[0]; for(int i=1;i<4;i++) rb&=bi[i];
      if(horizontal_reduce_band<int,4>(bi)!=rb){printf("band FAIL\\n");ok=0;}
      printf(ok?"ALL OK\\n":"FAILED\\n"); return ok?0:1;
    }
    """)


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
def test_scalar_horizontal_reduce_compiles_and_is_correct(tmp_path):
    src = tmp_path / "hreduce_check.cpp"
    src.write_text(SRC)
    exe = tmp_path / "hreduce_check"
    compile_res = subprocess.run(["g++", STD, "-I", INCLUDE, str(src), "-o", str(exe)], capture_output=True, text=True)
    assert compile_res.returncode == 0, f"compile failed:\n{compile_res.stderr}"
    run_res = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"runtime check failed:\n{run_res.stdout}"
    assert "ALL OK" in run_res.stdout, run_res.stdout


def build_avx512(tmp_path):
    """Build the AVX-512 driver; returns the executable path and the compiler result."""
    src = tmp_path / "hred_avx512.cpp"
    src.write_text(AVX512_SRC)
    exe = tmp_path / "hred_avx512"
    res = subprocess.run(["g++", STD, "-mavx512f", "-I", INCLUDE,
                          str(src), "-o", str(exe)],
                         capture_output=True,
                         text=True)
    return exe, res


@pytest.mark.skipif(not HOST_IS_X86, reason="AVX-512 intrinsics need an x86-targeting compiler")
def test_avx512_horizontal_reduce_compiles(tmp_path):
    """Building the AVX-512 header needs the compiler, NOT the instruction set: an x86 box without
    an AVX-512 CPU used to skip this whole file's intrinsic path and report green."""
    _, res = build_avx512(tmp_path)
    assert res.returncode == 0, f"compile failed:\n{res.stderr}"


@pytest.mark.avx512
@pytest.mark.skipif(not HOST_IS_X86, reason="AVX-512 intrinsics need an x86-targeting compiler")
def test_avx512_horizontal_reduce_is_correct(tmp_path):
    """Executing those instructions is what needs the hardware, so only this half is mark-gated."""
    exe, res = build_avx512(tmp_path)
    assert res.returncode == 0, f"compile failed:\n{res.stderr}"
    run_res = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"runtime check failed:\n{run_res.stdout}"
    assert "ALL OK" in run_res.stdout, run_res.stdout


ARM_TU = textwrap.dedent("""
    #define __DACE_USE_INTRINSICS 1
    #include "dace/cpu_vectorizable_math.h"
    template <typename T> void use_fp() {
      T a[16] = {};
      volatile T r = T(0);
      r += horizontal_reduce_add<T,16>(a); r += horizontal_reduce_mul<T,16>(a);
      r += horizontal_reduce_max<T,16>(a); r += horizontal_reduce_min<T,16>(a);
      (void)r;
    }
    template <typename T> void use_int() {
      T a[8] = {};
      volatile T r = T(0);
      r += horizontal_reduce_add<T,8>(a);  r += horizontal_reduce_band<T,8>(a);
      r += horizontal_reduce_bor<T,8>(a);  r += horizontal_reduce_bxor<T,8>(a);
      (void)r;
    }
    int main() { use_fp<double>(); use_fp<float>(); use_int<int>();
                 use_int<long long>(); return 0; }
    """)


def _aarch64_cxx():
    """An aarch64 C++ driver able to find a C++/glibc sysroot, or None."""
    g = shutil.which("aarch64-linux-gnu-g++")
    if g:
        return [g]
    return None


def _syntax_only_ok(driver, march_flags, tmp_path, name):
    src = tmp_path / f"{name}.cpp"
    src.write_text(ARM_TU)
    res = subprocess.run(driver + [STD, "-fsyntax-only"] + march_flags + ["-I", INCLUDE, str(src)],
                         capture_output=True,
                         text=True)
    return res.returncode == 0, res.stderr


@pytest.mark.arm_cross
@pytest.mark.parametrize("variant,flags", [
    ("neon", ["-march=armv8-a"]),
    ("sve", ["-march=armv8-a+sve", "-D__DACE_USE_SVE=1"]),
])
def test_arm_horizontal_reduce_syntax_only(variant, flags, tmp_path):
    """Mark-gated rather than runtime-skipped: an absent cross toolchain is a deselection the
    selection line has to state, not a hole that reports green on every x86 box."""
    driver = _aarch64_cxx()
    assert driver is not None, "aarch64-linux-gnu-g++ is required by the arm_cross mark"
    ok, err = _syntax_only_ok(driver, flags, tmp_path, f"hred_{variant}")
    assert ok, f"{variant} -fsyntax-only failed:\n{err}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
