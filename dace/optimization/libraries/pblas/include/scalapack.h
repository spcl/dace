#ifndef SCALAPACK_H
#define SCALAPACK_H

#include <complex>
typedef std::complex<float> complex_s;
typedef std::complex<double> complex_d;

extern "C" {

	void Cblacs_pinfo( int* mypnum, int* nprocs);
	void Cblacs_get(int context, int request, int* value);
	int Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
	void Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
	void Cblacs_gridexit( int context);
	void Cblacs_exit( int error_code);
	void Cblacs_gridmap( int* context, int* map, int ld_usermap, int np_row, int np_col);

	int npreroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
	int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

	void descinit_(int *idescal, int *m, int *n, int *mb, int *nb, int *dummy1 , int *dummy2 , int *icon, int *procRows, int *info);

    void psgemv_(
        char *trans, int *m, int *n,
        float *alpha, float *a, int *ia, int *ja, int *desca,
        float *x, int *ix, int *jx, int *descx, int *incx,
        float *beta, float *y, int *iy, int *jy, int *descy, int *incy
    );
    void pdgemv_(
        char *trans, int *m, int *n,
        double *alpha, double *a, int *ia, int *ja, int *desca,
        double *x, int *ix, int *jx, int *descx, int *incx,
        double *beta, double *y, int *iy, int *jy, int *descy, int *incy
    );
    void pcgemv_(
        char *trans, int *m, int *n,
        complex_s *alpha, complex_s *a, int *ia, int *ja, int *desca,
        complex_s *x, int *ix, int *jx, int *descx, int *incx,
        complex_s *beta, complex_s *y, int *iy, int *jy, int *descy, int *incy
    );
    void pzgemv_(
        char *trans, int *m, int *n,
        complex_d *alpha, complex_d *a, int *ia, int *ja, int *desca,
        complex_d *x, int *ix, int *jx, int *descx, int *incx,
        complex_d *beta, complex_d *y, int *iy, int *jy, int *descy, int *incy
    );

    void psgemm_(
        char *transa, char *transb, int *m, int *n, int *k,
        float *alpha, float *a, int *ia, int *ja, int *desca,
        float *b, int *ib, int *jb, int *descb,
        float *beta, float *c, int *ic, int *jc, int *descc
    );
    void pdgemm_(
        char *transa, char *transb, int *m, int *n, int *k,
        double *alpha, double *a, int *ia, int *ja, int *desca,
        double *b, int *ib, int *jb, int *descb,
        double *beta, double *c, int *ic, int *jc, int *descc
    );
    void pcgemm_(
        char *transa, char *transb, int *m, int *n, int *k,
        complex_s *alpha, complex_s *a, int *ia, int *ja, int *desca,
        complex_s *b, int *ib, int *jb, int *descb,
        complex_s *beta, complex_s *c, int *ic, int *jc, int *descc
    );
    void pzgemm_(
        char *transa, char *transb, int *m, int *n, int *k,
        complex_d *alpha, complex_d *a, int *ia, int *ja, int *desca,
        complex_d *b, int *ib, int *jb, int *descb,
        complex_d *beta, complex_d *c, int *ic, int *jc, int *descc
    );

}

#endif
