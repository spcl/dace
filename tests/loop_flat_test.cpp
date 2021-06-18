// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#include <omp.h>
#include <array>
#include <cassert>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include <dace/intset.h>

using dace::const_int_range;
using dace::make_range;

int main(int argc, char **argv) {
    int runtime_size = 3;
    if (argc > 1) {
        runtime_size = std::stoi(argv[1]);
    }

    std::cout << "int_range test (no int range)" << std::endl;
    int sum_regression = 0, sum;

    #pragma omp parallel for
    for (auto i0 = 3; i0 < 4; i0++)
        for (auto i1 = -4; i1 < 18; i1 += 2)
            for (auto i2 = 0; i2 < 10; i2 += 3) {
                std::cout << "Thread " << omp_get_thread_num() << ": i0 = " << i0
                    << ", i1 = " << i1 << " and i2 = " << i2 << std::endl;
                sum_regression += i0 * 10000 + i1 * 100 + i2;
            }

    std::cout << "int_range test" << std::endl;

    auto range = make_range(std::make_tuple(runtime_size, 4, 1),
                            std::make_tuple(-4, 18, 2),
                            std::make_tuple(0, 10, runtime_size));
    sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (auto i = 0; i < range.size(); ++i) {
        auto i0 = range.index_value(i, 0);
        auto i1 = range.index_value(i, 1);
        auto i2 = range.index_value(i, 2);

        sum += i0 * 10000 + i1 * 100 + i2;

        #pragma omp critical
        std::cout << "Thread " << omp_get_thread_num() << ": i0 = " << i0
            << ", i1 = " << i1 << " and i2 = " << i2 << std::endl;
    }

    if (sum != sum_regression) {
        std::cout << "int_range regression failed (" << sum
            << " != " << sum_regression << ")" << std::endl;
        return 1;
    }

    std::cout << "const_int_range test" << std::endl;

    sum = 0;

    typedef const_int_range<3, 4, 1, -4, 18, 2, 0, 10, 3> myrange;
    #pragma omp parallel for reduction(+:sum)
    for (auto i = 0; i < myrange::size; ++i) {
        auto i0 = myrange::index_value(i, 0);
        auto i1 = myrange::index_value(i, 1);
        auto i2 = myrange::index_value(i, 2);

        sum += i0 * 10000 + i1 * 100 + i2;

        #pragma omp critical
        printf("Thread %d: i0 = %d, i1 = %d, i2 = %d\n", omp_get_thread_num(), i0,
               i1, i2);
    }


    if (sum != sum_regression) {
        std::cout << "const_int_range regression failed (" << sum
            << " != " << sum_regression << ")" << std::endl;
        return 2;
    }

    std::cout << "Correctness test" << std::endl;
    std::vector<std::tuple<int, int, int>> valid_range;
    for (auto i0 = 3; i0 < 4; i0++)
        for (auto i1 = -4; i1 < 18; i1 += 2)
            for (auto i2 = 0; i2 < 10; i2 += 3)
                valid_range.emplace_back(i0, i1, i2);
    auto test_range = make_range(
        std::make_tuple(3, 4, 1),
        std::make_tuple(-4, 18, 2),
        std::make_tuple(0, 10, 3)
    );
    assert((int)valid_range.size() == test_range.size());
    assert(valid_range.size() == myrange::size);
    for (auto i = 0; i < test_range.size(); ++i) {
        auto i0 = test_range.index_value(i, 0);
        auto i1 = test_range.index_value(i, 1);
        auto i2 = test_range.index_value(i, 2);
        assert(valid_range[i] == std::make_tuple(i0, i1, i2));
        auto i_all = test_range.index_values(i);
        auto i00 = i_all[0];
        auto i10 = i_all[1];
        auto i20 = i_all[2];
        assert(valid_range[i] == std::make_tuple(i00, i10, i20));
        auto i01 = myrange::index_value(i, 0);
        auto i11 = myrange::index_value(i, 1);
        auto i21 = myrange::index_value(i, 2);
        assert(valid_range[i] == std::make_tuple(i01, i11, i21));
        auto i_all1 = myrange::index_values(i);
        auto i02 = i_all1[0];
        auto i12 = i_all1[1];
        auto i22 = i_all1[2];
        assert(valid_range[i] == std::make_tuple(i02, i12, i22));
    }

    std::cout << "PASS" << std::endl;

    return 0;
}
