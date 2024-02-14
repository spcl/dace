#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if _OPENMP
#include <omp.h>
#endif
#define TACO_MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
    int32_t order;            // tensor order (number of modes)
    int32_t* dimensions;      // tensor dimensions
    int32_t csize;            // component size
    int32_t* mode_ordering;   // mode storage ordering
    taco_mode_t* mode_types;  // mode storage types
    uint8_t*** indices;       // tensor index data (per mode)
    uint8_t* vals;            // tensor values
    uint8_t* fill_value;      // tensor fill value
    int32_t vals_size;        // values array size
} taco_tensor_t;
#endif
#if !_OPENMP
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
#endif
int cmp(const void* a, const void* b) {
    return *((const int*)a) - *((const int*)b);
}
int taco_gallop(int* array, int arrayStart, int arrayEnd, int target) {
    if (array[arrayStart] >= target || arrayStart >= arrayEnd) {
        return arrayStart;
    }
    int step = 1;
    int curr = arrayStart;
    while (curr + step < arrayEnd && array[curr + step] < target) {
        curr += step;
        step = step * 2;
    }

    step = step / 2;
    while (step > 0) {
        if (curr + step < arrayEnd && array[curr + step] < target) {
            curr += step;
        }
        step = step / 2;
    }
    return curr + 1;
}
int taco_binarySearchAfter(int* array, int arrayStart, int arrayEnd,
                           int target) {
    if (array[arrayStart] >= target) {
        return arrayStart;
    }
    int lowerBound = arrayStart;  // always < target
    int upperBound = arrayEnd;    // always >= target
    while (upperBound - lowerBound > 1) {
        int mid = (upperBound + lowerBound) / 2;
        int midValue = array[mid];
        if (midValue < target) {
            lowerBound = mid;
        } else if (midValue > target) {
            upperBound = mid;
        } else {
            return mid;
        }
    }
    return upperBound;
}
int taco_binarySearchBefore(int* array, int arrayStart, int arrayEnd,
                            int target) {
    if (array[arrayEnd] <= target) {
        return arrayEnd;
    }
    int lowerBound = arrayStart;  // always <= target
    int upperBound = arrayEnd;    // always > target
    while (upperBound - lowerBound > 1) {
        int mid = (upperBound + lowerBound) / 2;
        int midValue = array[mid];
        if (midValue < target) {
            lowerBound = mid;
        } else if (midValue > target) {
            upperBound = mid;
        } else {
            return mid;
        }
    }
    return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
    taco_tensor_t* t = (taco_tensor_t*)malloc(sizeof(taco_tensor_t));
    t->order = order;
    t->dimensions = (int32_t*)malloc(order * sizeof(int32_t));
    t->mode_ordering = (int32_t*)malloc(order * sizeof(int32_t));
    t->mode_types = (taco_mode_t*)malloc(order * sizeof(taco_mode_t));
    t->indices = (uint8_t***)malloc(order * sizeof(uint8_t***));
    t->csize = csize;
    for (int32_t i = 0; i < order; i++) {
        t->dimensions[i] = dimensions[i];
        t->mode_ordering[i] = mode_ordering[i];
        t->mode_types[i] = mode_types[i];
        switch (t->mode_types[i]) {
            case taco_mode_dense:
                t->indices[i] = (uint8_t**)malloc(1 * sizeof(uint8_t**));
                break;
            case taco_mode_sparse:
                t->indices[i] = (uint8_t**)malloc(2 * sizeof(uint8_t**));
                break;
        }
    }
    return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
    for (int i = 0; i < t->order; i++) {
        free(t->indices[i]);
    }
    free(t->indices);
    free(t->dimensions);
    free(t->mode_ordering);
    free(t->mode_types);
    free(t);
}
#endif