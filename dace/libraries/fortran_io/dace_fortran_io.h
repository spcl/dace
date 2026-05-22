/* Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved. */
/* C prototypes for the shipped iso_c_binding Fortran I/O wrappers
 * (``dace_fortran_io.f90``).  The fortran_io library nodes expand to C++
 * tasklets that call these; ``dace_fio_open`` returns a fresh unit the
 * matching reads/writes/close reuse. */
#ifndef DACE_FORTRAN_IO_H
#define DACE_FORTRAN_IO_H

#ifdef __cplusplus
extern "C" {
#endif

int dace_fio_open(const char *path, int path_len, int for_write);
void dace_fio_close(int unit);

void dace_fio_read_f64(int unit, double *x);
void dace_fio_read_f64_arr(int unit, double *x, int n);
void dace_fio_read_f32(int unit, float *x);
void dace_fio_read_f32_arr(int unit, float *x, int n);
void dace_fio_read_i32(int unit, int *x);
void dace_fio_read_i32_arr(int unit, int *x, int n);
void dace_fio_read_i64(int unit, long long *x);
void dace_fio_read_i64_arr(int unit, long long *x, int n);

void dace_fio_write_f64(int unit, const double *x);
void dace_fio_write_f64_arr(int unit, const double *x, int n);
void dace_fio_write_f32(int unit, const float *x);
void dace_fio_write_f32_arr(int unit, const float *x, int n);
void dace_fio_write_i32(int unit, const int *x);
void dace_fio_write_i32_arr(int unit, const int *x, int n);
void dace_fio_write_i64(int unit, const long long *x);
void dace_fio_write_i64_arr(int unit, const long long *x, int n);

/* Generic namelist read: open a (file, group), fetch members by name, close. */
int dace_nml_open(const char *path, int path_len, const char *group, int group_len);
void dace_nml_close(int handle);
void dace_nml_get_f64(int handle, const char *name, int name_len, double *x);
void dace_nml_get_f64_arr(int handle, const char *name, int name_len, double *x, int n);
void dace_nml_get_f32(int handle, const char *name, int name_len, float *x);
void dace_nml_get_f32_arr(int handle, const char *name, int name_len, float *x, int n);
void dace_nml_get_i32(int handle, const char *name, int name_len, int *x);
void dace_nml_get_i32_arr(int handle, const char *name, int name_len, int *x, int n);

#ifdef __cplusplus
}
#endif

#endif /* DACE_FORTRAN_IO_H */
