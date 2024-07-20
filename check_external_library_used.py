from dace.libraries import blas

print('BLAS calls will expand by default to', blas.default_implementation)

if blas.IntelMKL.is_installed():
    blas.default_implementation = 'MKL'
elif blas.cuBLAS.is_installed():
    blas.default_implementation = 'cuBLAS'
elif blas.OpenBLAS.is_installed():
    blas.default_implementation = 'OpenBLAS'
elif not blas.BLAS.is_installed():
    # No BLAS library found, use the unoptimized native SDFG fallback
    blas.default_implementation = 'pure'
