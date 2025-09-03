// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
