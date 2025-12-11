#pragma once

template<typename T>
inline void multiplex_elements(const T* __restrict__ in_ptr,
                               T* __restrict__ out_ptr,
                               int N, int D)
{
    int out_idx = 0;
    for (int i = 0; i < N; ++i) {
        T v = in_ptr[i];
        for (int r = 0; r < D; ++r) {
            out_ptr[out_idx++] = v;
        }
    }
}