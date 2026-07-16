// LAYOUT PROBE: five BYTE-IDENTICAL copies of the experimental heat_3d stencil, timed
// sequentially with per-twin destination buffers (the layout-EXPOSING method). If source
// form mattered, identical twins would tie. Any spread here is pure code placement +
// destination alignment -- i.e. the confound. Recompile with -falign-* to move it.
#include "bench.h"
static inline long long A_idx(long long d0,long long d1,long long d2,long long N){return N*d1+d0*(N*N)+d2;}
static inline long long O_idx(long long d0,long long d1,long long d2,long long N){return d0*((N-2)*(N-2))+d1*(N-2)+d2;}
#define IDX_FN(i,j,k) A_idx((i),(j),(k),N)

#define BODY(A,B,tmp10,tmp13,N)                                                                   \
    for (int i=0;i<N-2;++i) for(int j=0;j<N-2;++j) for(int k=0;k<N-2;++k){                        \
        double B0[1];                                                                            \
        const double t5[1]={(double)((2.0*A[IDX_FN(i+1,j+1,k+1)]))};                             \
        const double am2_1[1]={(double)((A[IDX_FN(i+1,j+1,k+2)]-t5[0]))};                        \
        const double a2_1[1]={(double)((am2_1[0]+A[IDX_FN(i+1,j+1,k)]))};                        \
        const double t6[1]={(double)((0.125*a2_1[0]))};                                          \
        const double t2[1]={(double)((2.0*A[IDX_FN(i+1,j+1,k+1)]))};                             \
        const double am2_0[1]={(double)((A[IDX_FN(i+1,j+2,k+1)]-t2[0]))};                        \
        const double a2_0[1]={(double)((am2_0[0]+A[IDX_FN(i+1,j,k+1)]))};                        \
        const double t3[1]={(double)((0.125*a2_0[0]))};                                          \
        const double t0[1]={(double)((2.0*A[IDX_FN(i+1,j+1,k+1)]))};                             \
        const double am2[1]={(double)((A[IDX_FN(i+2,j+1,k+1)]-t0[0]))};                          \
        const double a2[1]={(double)((am2[0]+A[IDX_FN(i,j+1,k+1)]))};                            \
        const double t1[1]={(double)((0.125*a2[0]))};                                            \
        const double t4[1]={(double)((t1[0]+t3[0]))};                                            \
        const double t7[1]={(double)((t4[0]+t6[0]))};                                            \
        const double Bv[1]={(double)((t7[0]+A[IDX_FN(i+1,j+1,k+1)]))};                           \
        B0[0]=Bv[0]; tmp10[O_idx(i,j,k,N)]=(2.0*Bv[0]);                                          \
        B0[0]=B[IDX_FN(i+1,j+1,k+1)]; tmp13[O_idx(i,j,k,N)]=(2.0*B0[0]);                         \
    }
__attribute__((noinline)) void T0(double*A,double*B,double*t10,double*t13,int N){BODY(A,B,t10,t13,N)}
__attribute__((noinline)) void T1(double*A,double*B,double*t10,double*t13,int N){BODY(A,B,t10,t13,N)}
__attribute__((noinline)) void T2(double*A,double*B,double*t10,double*t13,int N){BODY(A,B,t10,t13,N)}
__attribute__((noinline)) void T3(double*A,double*B,double*t10,double*t13,int N){BODY(A,B,t10,t13,N)}
__attribute__((noinline)) void T4(double*A,double*B,double*t10,double*t13,int N){BODY(A,B,t10,t13,N)}

int main(){
    const int N=120; const long long NN=(long long)N*N*N, OO=(long long)(N-2)*(N-2)*(N-2);
    std::vector<double> A(NN),B(NN); for(long long i=0;i<NN;++i){A[i]=double((i%97)+1)*0.5;B[i]=double((i%89)+1)*0.25;}
    // per-twin destination buffers (exposes destination-placement confound too)
    std::vector<std::vector<double>> o(5),d(5); for(int f=0;f<5;++f){o[f].assign(OO,0);d[f].assign(OO,0);}
    void(*fn[5])(double*,double*,double*,double*,int)={T0,T1,T2,T3,T4};
    const char*nm[5]={"T0","T1","T2","T3","T4"};
    Stat s[5];
    for(int f=0;f<5;++f) s[f]=bench([&]{fn[f](A.data(),B.data(),o[f].data(),d[f].data(),N);});
    double best=s[0].median; for(int f=1;f<5;++f) best=std::min(best,s[f].median);
    std::printf("LAYOUT PROBE: 5 byte-identical twins, sequential + per-twin buffers\n");
    for(int f=0;f<5;++f) row(nm[f],s[f],best);
    double mx=s[0].median,mn=s[0].median; for(int f=1;f<5;++f){mx=std::max(mx,s[f].median);mn=std::min(mn,s[f].median);}
    std::printf("  SPREAD across identical twins = %.2fx (this is the layout floor)\n",mx/mn);
    return 0;
}
