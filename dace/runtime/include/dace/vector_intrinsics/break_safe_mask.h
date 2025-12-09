#pragma once

template <int VLEN, typename T>
void compute_break_safe_prefix_mask(const T* a, const T* b, T* outmask)
{
    // First lane
    T first = (a[0] > b[0]) ? T(1.0) : T(0.0);
    outmask[0] = first;

    T active = first;   // stays 1.0 until first failure

    for (int i = 1; i < VLEN; ++i) {
        T cur = (a[i] > b[i]) ? T(1.0) : T(0.0);
        active = (active == T(1.0) && cur == T(1.0)) ? T(1.0) : T(0.0);
        outmask[i] = active;
    }
}


