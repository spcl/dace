#include <iostream>
#include <cstdlib> // For aligned_alloc
#include <dace/dace.h>
#include "../../include/hash.h"

int main() {
    size_t num_elements = 153600; // Number of floats
    size_t total_size = num_elements * sizeof(float); // Total size in bytes

    std::cout << "Attempting to allocate an array of " << total_size / (1024.0 * 1024.0)
              << " MB (" << num_elements << " floats)...\n";

    // Try with normal allocation
    try {
        float *normal_array = new float[num_elements];
        std::cout << "Normal allocation succeeded.\n";
        delete[] normal_array;
    } catch (std::bad_alloc &e) {
        std::cerr << "Normal allocation failed: " << e.what() << "\n";
    }

    // Try with aligned allocation
    float __tmp5;
    float __tmp7;
    double __tmp14;
    double __tmp15;
    float *p;
    p = new float DACE_ALIGN(64)[1024];
    float *q;
    q = new float DACE_ALIGN(64)[1024];
    double DY;
    double DT;
    double B2;
    double d;
    double e;
    float *gradient_u_tmp;
    gradient_u_tmp = new float DACE_ALIGN(64)[1024];
    float *gradient_q;
    gradient_q = new float DACE_ALIGN(64)[1024];
    memset(gradient_q, 0, sizeof(float)*1024);
    float *gradient_p;
    gradient_p = new float DACE_ALIGN(64)[1024];
    memset(gradient_p, 0, sizeof(float)*1024);
    double gradient_e = 0;
    double gradient_d = 0;
    float *stored_u;
    stored_u = new float DACE_ALIGN(64)[153600];
    memset(stored_u, 0, sizeof(float)*153600);
    double *stored___tmp23;
    stored___tmp23 = new double DACE_ALIGN(64)[150];
    memset(stored___tmp23, 0, sizeof(double)*150);
    float *stored___tmp25;
    stored___tmp25 = new float DACE_ALIGN(64)[4500];
    memset(stored___tmp25, 0, sizeof(float)*4500);
    float *stored_p;
    stored_p = new float DACE_ALIGN(64)[153600];
    memset(stored_p, 0, sizeof(float)*153600);
    int64_t t;
    int64_t j_3;
    int64_t j_4;
    if (stored_u) {
        std::cout << "Aligned allocation succeeded.\n";
        free(stored_u); // Use free for aligned_alloc
    } else {
        std::cerr << "Aligned allocation failed.\n";
    }

    return 0;
}
