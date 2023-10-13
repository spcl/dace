# Instructions to prepare CLOUDSC repository

1. Clone the [repository](https://github.com/ecmwf-ifs/dwarf-p-cloudsc/tree/main)
1. Build it using `./cloudsc-bundle create` and `./cloudsc-bundle build --clean --build-type release` to build the
   fortran version. To build the CUDA version use
1. To build the CUDA version (you might want to do this in another repository) use
```bash
./cloudsc-bundle create
./cloudsc-bundle build \
    --clean \
    --build-type=Release \
    --cloudsc-fortran=OFF \
    --cloudsc-cuda ON \
    --with-serialbox \
    --with-cuda \
    --build-dir build_cuda
cd build_cuda
make dwarf-cloudsc-cuda dwarf-cloudsc-cuda-k-caching
cd ../..
```
1. Copy the files `CMakeLists.txt`, `cloudsc_driver_mod.F90.template` and `working.cc` from this folder into
   `src/cloudsc_fortran` inside the cloned CLOUDSC repository.
1. Replace `<PathToYourRepository>` inside `CMakeLists.txt` by the absolute path to this repository.
