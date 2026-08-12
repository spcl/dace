# Canonicalize speedup vs `auto-opt`

host `primrose-pc` · OMP `4` · best-of `7` (warmup 2) · kernels measured `54` · generated `2026-07-15T12:20:40`

Speedup = `auto-opt_min / pipeline_min` (best-of-N); **>1× means faster than auto-opt**. Only numerically-correct pipelines are timed — ✓ verified, ✗ miscompile, · not measured.

## preset `S`

| suite | kernel | shape | auto-opt ms | canon ms | ✓ | canon× |
|:--|:--|:--|--:|--:|:-:|--:|
| poly | adi | tsteps=20 N=20 | 0.071 | 0.064 | ✓ | 1.11× |
| poly | atax | M=38 N=42 | 0.001 | 0.002 | ✓ | 0.72× |
| poly | bicg | M=38 N=42 | 0.001 | 0.002 | ✓ | 0.70× |
| poly | cholesky | N=40 | 0.005 | 0.005 | ✓ | 1.12× |
| poly | correlation | M=28 N=32 | 0.013 | 0.020 | ✓ | 0.67× |
| poly | covariance | M=28 N=32 | 0.042 | 0.044 | ✓ | 0.96× |
| poly | deriche | W=64 H=64 | 0.077 | 0.010 | ✓ | 7.40× |
| poly | doitgen | NQ=8 NR=10 NP=12 | 0.002 | 0.003 | ✓ | 0.55× |
| poly | durbin | N=40 | 0.129 | 0.123 | ✓ | 1.05× |
| poly | fdtd_2d | TMAX=20 NX=20 NY=30 | 0.089 | 0.082 | ✓ | 1.09× |
| poly | floyd_warshall | N=60 | 0.153 | 0.151 | ✓ | 1.01× |
| poly | gemm | NI=20 NJ=25 NK=30 | 0.005 | 0.005 | ✓ | 1.03× |
| poly | gemver | N=40 | 0.007 | 0.008 | ✓ | 0.84× |
| poly | gesummv | N=30 | 0.005 | 0.004 | ✓ | 1.06× |
| poly | gramschmidt | M=30 N=20 | 0.274 | 0.067 | ✓ | 4.11× |
| poly | heat_3d | tsteps=20 N=10 | 0.077 | 0.077 | ✓ | 0.99× |
| poly | jacobi_1d | tsteps=20 N=30 | 0.038 | 0.041 | ✓ | 0.94× |
| poly | jacobi_2d | tsteps=20 N=30 | 0.045 | 0.046 | ✓ | 0.97× |
| poly | k2mm | NI=16 NJ=18 NK=22 NL=24 | 0.004 | 0.004 | ✓ | 1.00× |
| poly | k3mm | NI=16 NJ=18 NK=20 NL=22 NM=24 | 0.002 | 0.002 | ✓ | 0.88× |
| poly | lu | N=40 | 0.961 | 0.962 | ✓ | 1.00× |
| poly | ludcmp | N=40 | 1.434 | 1.467 | ✓ | 0.98× |
| poly | mvt | N=40 | 0.003 | 0.004 | ✓ | 0.81× |
| poly | nussinov | N=60 | 0.014 | 0.070 | ✓ | 0.20× |
| poly | seidel_2d | tsteps=20 N=40 | 1.728 | 1.600 | ✓ | 1.08× |
| poly | symm | M=20 N=30 | 0.677 | 0.066 | ✓ | 10.26× |
| poly | syr2k | M=20 N=30 | 0.004 | 0.011 | ✓ | 0.36× |
| poly | syrk | M=20 N=30 | 0.586 | 0.147 | ✓ | 4.00× |
| poly | trisolv | N=40 | 0.001 | 0.002 | ✓ | 0.71× |
| poly | trmm | M=20 N=30 | 0.004 | 0.004 | ✓ | 1.06× |
| np | crc16 | N=32 | 0.008 | 0.001 | ✓ | 8.48× |
| np | cholesky2 | N=32 | 0.008 | 0.007 | ✓ | 1.14× |
| np | contour_integral | NR=8 NM=12 slab_per_bc=2 num_int_pts=32 | 0.297 | ERR | · | — |
| np | covariance2 | M=32 N=32 | 0.010 | 0.012 | ✓ | 0.83× |
| np | permute_3d | N=32 | 0.015 | 0.014 | ✓ | 1.02× |
| np | scattering_self_energies | Nkz=2 NE=4 Nqz=2 Nw=2 N3D=2 NA=6 NB=2 Norb=3 | 4.238 | 0.539 | ✓ | 7.86× |
| np | arc_distance | N=32 | 0.002 | 0.003 | ✓ | 0.73× |
| np | azimint_hist | N=32 npt=32 | 0.026 | 0.013 | ✓ | 1.99× |
| np | azimint_naive | N=32 npt=32 | 0.008 | 0.005 | ✓ | 1.74× |
| np | compute | M=32 N=32 | 0.003 | 0.002 | ✓ | 1.27× |
| np | go_fast | N=32 | 0.003 | 0.004 | ✓ | 0.84× |
| np | mandelbrot1 | xmin=-1.75 xmax=0.25 xn=32 XN=32 ymin=-1.0 ymax=1.0 yn=32 YN=32 maxiter=32 horizon=2.0 | 0.034 | 0.236 | ✓ | 0.14× |
| np | mandelbrot2 | xmin=-2.0 xmax=0.5 XN=32 ymin=-1.25 ymax=1.25 YN=32 maxiter=32 horizon=2.0 | 0.284 | 0.297 | ✓ | 0.95× |
| np | lenet | N=4 H=16 W=16 C_before_fc1=16 | 0.888 | 0.045 | ✓ | 19.81× |
| np | mlp | C_in=3 N=8 S0=32 S1=32 S2=32 | 0.041 | 0.014 | ✓ | 3.00× |
| np | resnet | N=8 W=14 H=14 C1=32 C2=8 | 2.983 | 2.132 | ✓ | 1.40× |
| np | softmax | N=16 H=16 SM=32 | 22.971 | 2.558 | ✓ | 8.98× |
| np | nbody | N=25 tEnd=2.0 dt=0.05 softening=0.1 G=1.0 Nt=3 | 0.034 | 0.052 | ✓ | 0.65× |
| np | spmv | M=32 N=32 nnz=32 | 0.003 | 0.004 | ✓ | 0.77× |
| np | stockham_fft | R=2 K=15 | 8.722 | 11.353 | ✓ | 0.77× |
| np | cavity_flow | ny=32 nx=32 nt=25 nit=5 rho=1.0 nu=0.1 | 0.696 | WRONG | ✗ | — |
| np | channel_flow | ny=32 nx=32 nit=5 rho=1.0 nu=0.1 F=1.0 | 0.092 | 0.037 | ✓ | 2.47× |
| np | hdiff | I=32 J=32 K=32 | 0.040 | 0.044 | ✓ | 0.92× |
| np | vadv | dtr_stage=1.0 I=32 J=32 K=32 | 0.383 | 0.453 | ✓ | 0.85× |

**geomean speedup** — canon `1.257×` (n=52) · **correctness** — canon 52✓ 1✗ 1·

## preset `paper`

| suite | kernel | shape | auto-opt ms | canon ms | ✓ | canon× |
|:--|:--|:--|--:|--:|:-:|--:|
| poly | adi | tsteps=100 N=200 | 7.261 | 7.394 | ✓ | 0.98× |
| poly | atax | M=390 N=410 | 0.030 | 0.027 | ✓ | 1.11× |
| poly | bicg | M=390 N=410 | 0.031 | 0.027 | ✓ | 1.12× |
| poly | cholesky | N=400 | 0.151 | 0.154 | ✓ | 0.98× |
| poly | correlation | M=240 N=260 | 2.251 | 1.278 | ✓ | 1.76× |
| poly | covariance | M=240 N=260 | 1.120 | 1.216 | ✓ | 0.92× |
| poly | deriche | W=720 H=480 | 5.924 | 0.457 | ✓ | 12.97× |
| poly | doitgen | NQ=40 NR=50 NP=60 | 0.223 | 0.060 | ✓ | 3.73× |
| poly | durbin | N=400 | WRONG | WRONG | ✗ | — |
| poly | fdtd_2d | TMAX=100 NX=200 NY=240 | 2.894 | 2.069 | ✓ | 1.40× |
| poly | floyd_warshall | N=500 | 48.095 | 48.964 | ✓ | 0.98× |
| poly | gemm | NI=200 NJ=220 NK=240 | 0.143 | 0.110 | ✓ | 1.30× |
| poly | gemver | N=400 | 0.106 | 0.131 | ✓ | 0.81× |
| poly | gesummv | N=250 | 0.045 | 0.049 | ✓ | 0.91× |
| poly | gramschmidt | M=200 N=240 | WRONG | WRONG | ✗ | — |
| poly | heat_3d | tsteps=100 N=40 | 10.512 | 8.611 | ✓ | 1.22× |
| poly | jacobi_1d | tsteps=100 N=400 | 0.212 | 0.219 | ✓ | 0.97× |
| poly | jacobi_2d | tsteps=100 N=250 | 1.880 | 1.890 | ✓ | 0.99× |
| poly | k2mm | NI=180 NJ=190 NK=210 NL=220 | 0.213 | 0.198 | ✓ | 1.08× |
| poly | k3mm | NI=180 NJ=190 NK=200 NL=210 NM=220 | 0.260 | 0.251 | ✓ | 1.03× |
| poly | lu | N=400 | 230.197 | 241.863 | ✓ | 0.95× |
| poly | ludcmp | N=400 | 308.176 | 136.239 | ✓ | 2.26× |
| poly | mvt | N=400 | 0.029 | 0.027 | ✓ | 1.04× |
| poly | nussinov | N=500 | 9.575 | 4.951 | ✓ | 1.93× |
| poly | seidel_2d | tsteps=100 N=400 | 135.702 | 104.544 | ✓ | 1.30× |
| poly | symm | M=200 N=240 | 51.480 | 1.718 | ✓ | 29.96× |
| poly | syr2k | M=200 N=240 | 1.572 | 6.913 | ✓ | 0.23× |
| poly | syrk | M=200 N=240 | 47.534 | 30.217 | ✓ | 1.57× |
| poly | trisolv | N=400 | 0.012 | 0.015 | ✓ | 0.82× |
| poly | trmm | M=200 N=240 | 0.635 | 0.323 | ✓ | 1.97× |
| np | crc16 | N=256 | 0.003 | 0.003 | ✓ | 1.01× |
| np | cholesky2 | N=256 | 0.203 | 0.202 | ✓ | 1.01× |
| np | contour_integral | NR=8 NM=12 slab_per_bc=2 num_int_pts=32 | 0.295 | ERR | · | — |
| np | covariance2 | M=256 N=256 | 0.317 | 0.315 | ✓ | 1.01× |
| np | permute_3d | N=128 | 2.943 | 2.702 | ✓ | 1.09× |
| np | scattering_self_energies | Nkz=2 NE=4 Nqz=2 Nw=2 N3D=2 NA=6 NB=2 Norb=3 | 3.323 | 0.606 | ✓ | 5.48× |
| np | arc_distance | N=256 | 0.004 | 0.006 | ✓ | 0.78× |
| np | azimint_hist | N=256 npt=256 | 0.129 | 0.015 | ✓ | 8.86× |
| np | azimint_naive | N=256 npt=256 | 0.039 | 0.013 | ✓ | 2.94× |
| np | compute | M=256 N=256 | 0.012 | 0.013 | ✓ | 0.93× |
| np | go_fast | N=256 | 0.009 | 0.009 | ✓ | 1.06× |
| np | mandelbrot1 | xmin=-1.75 xmax=0.25 xn=125 XN=125 ymin=-1.0 ymax=1.0 yn=125 YN=125 maxiter=60 horizon=2.0 | WRONG | WRONG | ✗ | — |
| np | mandelbrot2 | xmin=-2.0 xmax=0.5 XN=200 ymin=-1.25 ymax=1.25 YN=200 maxiter=40 horizon=2.0 | 10.443 | 6.590 | ✓ | 1.58× |
| np | lenet | N=4 H=16 W=16 C_before_fc1=16 | 0.924 | 0.048 | ✓ | 19.44× |
| np | mlp | C_in=3 N=8 S0=256 S1=256 S2=256 | 0.263 | 0.050 | ✓ | 5.27× |
| np | resnet | N=8 W=14 H=14 C1=32 C2=8 | 2.991 | 2.851 | ✓ | 1.05× |
| np | softmax | N=16 H=16 SM=128 | 423.988 | 12.424 | ✓ | 34.13× |
| np | nbody | N=25 tEnd=2.0 dt=0.05 softening=0.1 G=1.0 Nt=3 | 0.039 | 0.049 | ✓ | 0.81× |
| np | spmv | M=256 N=256 nnz=256 | 0.004 | 0.004 | ✓ | 0.95× |
| np | stockham_fft | R=2 K=15 | 7.270 | 10.516 | ✓ | 0.69× |
| np | cavity_flow | ny=61 nx=61 nt=25 nit=5 rho=1.0 nu=0.1 | 1.218 | 0.957 | ✓ | 1.27× |
| np | channel_flow | ny=61 nx=61 nit=5 rho=1.0 nu=0.1 F=1.0 | 0.251 | 0.070 | ✓ | 3.60× |
| np | hdiff | I=64 J=64 K=60 | 0.205 | 0.222 | ✓ | 0.92× |
| np | vadv | dtr_stage=1.0 I=64 J=64 K=60 | 2.611 | 2.587 | ✓ | 1.01× |

**geomean speedup** — canon `1.615×` (n=50) · **correctness** — canon 50✓ 3✗ 1·
