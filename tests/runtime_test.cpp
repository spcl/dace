#include <cstdio>
#include <typeinfo>
#include <cassert>
#include <cstring>
#include <iostream>

#include <dace/dace.h>

struct add_wcr
{
    template<typename T>
    static void resolve(T *ptr, const T& value)
    {
        *ptr += value;
    }
};


#define AW 1000
#define AH 1000

int main(int, char **)
{
    /////////////////////////////////////////////////////////
    // Type checks
    static_assert(std::is_pod<dace::vec<float, 1> >::value == true,
                  "Length-1 float vector should be POD");

    static_assert(std::is_scalar<dace::vec<double, 1> >::value == true,
                  "Length-1 double vector should be scalar");

    static_assert(std::is_scalar<dace::vec<float, 2> >::value == false,
                  "Length-2 float vector should not be scalar");

    static_assert(
        std::is_scalar<dace::vec<std::complex<float>, 1> >::value == false,
        "Length-1 complex vector should not be scalar");

    static_assert(
        std::is_pod<dace::vec<std::complex<float>, 1> >::value == false,
        "Length-1 complex vector should not be POD");

    /////////////////////////////////////////////////////////
    // ArrayView

    float *bla = new float[AW*AH];
    for (int y = 0; y < AH; ++y)
        for (int x = 0; x < AW; ++x)
            bla[y*AW + x] = y * 10000 + x;
    
    dace::ArrayViewOut<float, 2> arr (bla, AW, 1);
    dace::ArrayViewOut<float, 2, 2> arrv2(bla, AW, 1);

    // Normal access
    float v = arr(50, 50);
    assert(v == 500050.0f);

    // Vectorized access
    dace::vec<float, 2> v2 = arrv2(50, 50);
    assert(v2[0] == 500100.0f);
    assert(v2[1] == 500101.0f);

    // ROI
    dace::ArrayViewIn<float, 2> arroi(bla + 250*AW + 500, AW, 1);
    assert(arroi(-1, 4) == 2490504.0f);

    // Ordinary write
    arr.write(1337.0f, 1, 3);
    assert(bla[1*AW+3] == 1337.0f);
    
    // Vector write
    arrv2.write(dace::vec<float,2>{ 1337.0f, 7331.0f }, 5, 5);
    assert(bla[5 * AW + 10] == 1337.0f);
    assert(bla[5 * AW + 11] == 7331.0f);

    // Write + Conflict resolution
    arr.write_and_resolve(1, [](auto a, auto b) { return a + b; }, 5, 10);
    arr.write_and_resolve<dace::ReductionType::Sum>(1, 5, 10);
    assert(bla[5 * AW + 10] == 1339.0f);

    // Scalar access
    dace::ArrayViewOut<float, 0> scalar(bla + 25 * AW + 90);
    dace::ArrayViewOut<float, 0, 2> scalar_vec(bla + 25 * AW + 90);
    float val = scalar;
    assert(val == 250090.0f);
    
    dace::vec<float, 2> val_vec = scalar_vec;
    assert(val_vec[0] == 250090.0f);
    assert(val_vec[1] == 250091.0f);

    // Scalar writes
    scalar.write(val + 1.0f);
    assert(fabs(bla[25*AW+90] - 250091.0f) <= 1e-6);

    scalar_vec.write_and_resolve(val_vec, [](auto a, auto b) { return a + b; });
    scalar_vec.write_and_resolve<dace::ReductionType::Sum>(val_vec);
    assert(fabs(bla[25 * AW + 90] - (250091.0f + 2*250090.0f)) <= 1e-6);
    assert(fabs(bla[25 * AW + 91] - (250091.0f + 2*250091.0f)) <= 1e-6);
    
    // ArrayView with skips
    dace::ArrayViewIn<float, 2> arr_skip(bla + 3*AW + 6, AW*2, 3);
    assert(arr_skip(0, 0) == 30006.0);
    assert(arr_skip(0, 1) == 30009.0);
    assert(arr_skip(1, 2) == 50012.0);

    delete[] bla;

    /////////////////////////////////////////////////////////
    // Streams
    dace::Stream<uint64_t> s;
    s.push(1);
    std::atomic<uint64_t> result (0ULL);

    // Stream consume (async, but useless operation)
    dace::Consume<1>::consume(s, 8, [&](int /*pe*/, uint64_t& u)
    {
        if (u < 256ULL) {
            result += u;
            s.push(2 * u);
            s.push(2 * u + 1);
        }
    });

    assert(result == 255 * 128);

    // Stream with array as a sink
    double *bla2 = new double[10];
    memset(bla2, 0xAE, sizeof(double) * 10);
    dace::ArrayStreamView<double> s_bla2 (bla2);
    dace::vec<double, 2> vec = dace::vec<double, 2>{ 3.0, 2.0 };
    dace::vec<double, 2> vec_arr[2] = {0};

    s_bla2.push<2>(vec);
    assert(bla2[0] == 3.0);
    assert(bla2[1] == 2.0);

    s_bla2.push<2>(vec_arr, 2);
    assert(bla2[2] == 0.0);
    assert(bla2[3] == 0.0);
    assert(bla2[4] == 0.0);
    assert(bla2[5] == 0.0);

    s_bla2.push(9.0);
    assert(bla2[6] == 9.0);

    delete[] bla2;

    printf("Success!\n");
    return 0;
}

