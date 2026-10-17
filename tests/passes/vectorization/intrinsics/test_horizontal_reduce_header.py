# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compile and numerically validate the ``horizontal_reduce_<op>`` primitives of ``dace/horizontal_reduce.h``.

The vectorized reduction expansion emits ``horizontal_reduce_<op><T, W>(buf)`` to fold a W-wide vector accumulator to a
scalar. This test compiles the header with ``g++`` and checks every op against a reference fold, including the odd-width
and width-1 edges.
"""

import subprocess
import textwrap
from pathlib import Path

import dace

INCLUDE = str(Path(dace.__file__).parent / "runtime" / "include")
#: The standard DaCe builds generated code with. Pinning c++17 here un-gates the day the runtime
#: headers reach for a C++20 feature -- ``dace/codegen/common.py`` clamps generated code to >= 20.
STD = f"-std=c++{dace.Config.get('compiler', 'cpp_standard')}"

SRC = textwrap.dedent("""
    #include "dace/horizontal_reduce.h"
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


def test_scalar_horizontal_reduce_compiles_and_is_correct(tmp_path):
    src = tmp_path / "hreduce_check.cpp"
    src.write_text(SRC)
    exe = tmp_path / "hreduce_check"
    compile_res = subprocess.run(["g++", STD, "-I", INCLUDE, str(src), "-o", str(exe)], capture_output=True, text=True)
    assert compile_res.returncode == 0, f"compile failed:\n{compile_res.stderr}"
    run_res = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"runtime check failed:\n{run_res.stdout}"
    assert "ALL OK" in run_res.stdout, run_res.stdout


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_scalar_horizontal_reduce_compiles_and_is_correct(Path(directory))
