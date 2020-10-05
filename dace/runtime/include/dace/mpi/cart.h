// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_MPI_CART_H
#define __DACE_MPI_CART_H

#include <vector>

class Cart {
    int ndims;
    int size;
    std::vector<int> strides;
public:
    Cart(int n, const int dims[]) {
        ndims = n;
        size = 1;
        strides.resize(ndims);
        for (auto i = ndims - 1; i >= 0; --i) {
            strides[i] = size;
            size *= dims[i];
        }
    }
    void coords(int rank, int coords[]) {
        int rem = rank;
        for (auto i = 0; i < ndims; ++i) {
            coords[i] = rem / strides[i];
            rem = rem % strides[i];
        }
    }
    void rank(int coords[], int& rank) {
        rank = 0;
        for (auto i = 0; i < ndims; ++i) {
            rank += coords[i] * strides[i];
        }
    }
};

#endif  // __DACE_MPI_CART_H