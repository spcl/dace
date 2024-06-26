// nvcc -O3 -lineinfo tensor_core_example.cu -arch=sm_86 && ncu -fo 123 --set detailed ./a.out
// -maxrregcount=255 required without launch_bounds
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define M 4096
#define N 4096
#define K 4096
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// number of tiles processed by block BLOCK_M x BLOCK_N
#define BLOCK_M 4
#define BLOCK_N 2
// number of tiles processed by warp WARP_M x WARP_N
#define WARP_M 2
#define WARP_N 4
#define WARP_SIZE 32
#define GRID_M (M / (BLOCK_M * WARP_M * WMMA_M))
#define GRID_N (N / (BLOCK_N * WARP_N * WMMA_N))
#define STEP_K 4
#define GRID_K 4
#define TILES_K (K / (GRID_K * STEP_K * WMMA_K))
#define HALF_VEC 8 // (sizeof(int4) / sizeof(half))
#define FLOAT_VEC 4 // (sizeof(int4) / sizeof(float))
#define PAD_HALF 16

#define THREADS_PER_BLOCK (WARP_SIZE * BLOCK_M * BLOCK_N)


#define REDUCE_K_THREADS_PER_BLOCK 512

__forceinline__
__device__ void memcpy2d(int4* dst, int dst_stride, const int4* src, int src_stride, int rows, int cols) {
    // this is block-local (all warps of a block participate)
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         dst[i * dst_stride + j] = src[i * src_stride + j];
    //     }
    // }
    int warps = BLOCK_M * BLOCK_N;//blockDim.x / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (cols <= WARP_SIZE) {
        // if (threadIdx.x == 0 and blockIdx.x == 0) printf("rows: %d cols %d\n", rows, cols);

        int warp_rows = WARP_SIZE / cols;
        int col = lane % cols;
        int warp_row = lane / cols;
        int rows_per_warp = rows / warps;  // 128 / 8 = 16
        // if (threadIdx.x == 0 and blockIdx.x == 0) printf("warp_rows: %d\n", warp_rows);
        int warp_row_offset = warp * rows_per_warp;  // 0, 16, ..., 112
        for (int row_in_chunk = 0; row_in_chunk < rows_per_warp; row_in_chunk += warp_rows) {
            int row = (warp_row_offset + row_in_chunk + warp_row);
            dst[row * dst_stride + col] = src[row * src_stride + col];
        }
    } else {
        // row size (cols) is larger than warp size
        // therefore we do not need to subdivide the warp

        int warp_cols = cols / WARP_SIZE;
        int rows_per_warp = rows / warps;
        int warp_row_offset = warp * rows_per_warp;
        for (int row_in_chunk = 0; row_in_chunk < rows_per_warp; row_in_chunk++) {
            int row = warp_row_offset + row_in_chunk;
            for (int warp_col = 0; warp_col < warp_cols; warp_col++) {
                int col = warp_col * WARP_SIZE + lane;
                dst[row * dst_stride + col] = src[row * src_stride + col];
            }
        }
    }
}


__global__ void 
__launch_bounds__(THREADS_PER_BLOCK, 1)
compute_gemm(const half *a, const half *b, float *c) {
    extern __shared__ float cs[];
    half* as = (half*)cs;
    half* bs = as + BLOCK_M * WARP_M * WMMA_M * (STEP_K * WMMA_K + PAD_HALF);

    int warp = threadIdx.x / WARP_SIZE;
    int block_m = warp / BLOCK_N;  // 0 to BLOCK_M
    int block_n = warp % BLOCK_N;  // 0 to BLOCK_N


    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> ar[WARP_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> br[WARP_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cr[WARP_M][WARP_N];

    for (int grid_kmn = blockIdx.x; grid_kmn < GRID_K * GRID_M * GRID_N; grid_kmn += gridDim.x) {
        int grid_k =  grid_kmn / (GRID_M * GRID_N);
        int grid_m = (grid_kmn / GRID_N) % GRID_M;
        int grid_n = grid_kmn % GRID_N;

        #pragma unroll
        for (int warp_m = 0; warp_m < WARP_M; warp_m++) {
            #pragma unroll
            for (int warp_n = 0; warp_n < WARP_N; warp_n++) {
                wmma::fill_fragment(cr[warp_m][warp_n], 0.0f);
            }
        }

        // Go through the global K dimension by a fixed step at a time.
        #pragma unroll
        for (int tile_k = 0; tile_k < TILES_K; tile_k += 1) {

            memcpy2d(
                (int4*)as,
                (STEP_K * WMMA_K + PAD_HALF) / HALF_VEC, 
                (int4*)&a[(grid_k * TILES_K * STEP_K * WMMA_K) + (tile_k * STEP_K * WMMA_K) + (grid_m * BLOCK_M * WARP_M * WMMA_M) * K],
                K / HALF_VEC,
                BLOCK_M * WARP_M * WMMA_M,
                STEP_K * WMMA_K / HALF_VEC
            );

            memcpy2d(
                (int4*)bs,
                (STEP_K * WMMA_K + PAD_HALF) / HALF_VEC, 
                (int4*)&b[(grid_k * TILES_K * STEP_K * WMMA_K) + (tile_k * STEP_K * WMMA_K) + (grid_n * BLOCK_N * WARP_N * WMMA_N) * K],
                K / HALF_VEC,
                BLOCK_N * WARP_N * WMMA_N,
                STEP_K * WMMA_K / HALF_VEC
            );

            __syncthreads();

            #pragma unroll
            for (int step_k = 0; step_k < STEP_K; step_k++) {
                #pragma unroll
                for (int warp_m = 0; warp_m < WARP_M; warp_m++) {
                    wmma::load_matrix_sync(ar[warp_m], 
                        &as[(step_k * WMMA_K) + (block_m * WARP_M * WMMA_M + warp_m * WMMA_M) * (STEP_K * WMMA_K + PAD_HALF)],
                        (STEP_K * WMMA_K + PAD_HALF)
                    );
                }
                #pragma unroll
                for (int warp_n = 0; warp_n < WARP_N; warp_n++) {
                    wmma::load_matrix_sync(br[warp_n], 
                        &bs[(step_k * WMMA_K) + (block_n * WARP_N * WMMA_N + warp_n * WMMA_N) * (STEP_K * WMMA_K + PAD_HALF)],
                        (STEP_K * WMMA_K + PAD_HALF)
                    );
                    #pragma unroll
                    for (int warp_m = 0; warp_m < WARP_M; warp_m++) {
                        wmma::mma_sync(cr[warp_m][warp_n], ar[warp_m], br[warp_n], cr[warp_m][warp_n]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
        #pragma unroll
        for (int warp_m = 0; warp_m < WARP_M; warp_m++) {
            #pragma unroll
            for (int warp_n = 0; warp_n < WARP_N; warp_n++) {

                float *tile_ptr = cs +
                    (block_m * WARP_M * WMMA_M + warp_m * WMMA_M) * (BLOCK_N * WARP_N * WMMA_N)
                            + (block_n * WARP_N * WMMA_N + warp_n * WMMA_N);

                wmma::store_matrix_sync(tile_ptr, cr[warp_m][warp_n], WMMA_N * BLOCK_N * WARP_N, wmma::mem_row_major);
            }
        }2

        __syncthreads();

        // shape of c [GRID_K, M, N]
        memcpy2d(
            (int4*)&c[grid_k * M * N + (grid_m * BLOCK_M * WARP_M * WMMA_M) * N + (grid_n * BLOCK_N * WARP_N * WMMA_N)],
            // (int4*)&c[(grid_m * BLOCK_M * WARP_M * WMMA_M) * N + (grid_n * BLOCK_N * WARP_N * WMMA_N)],
            N / FLOAT_VEC,
            (int4*)cs,
            BLOCK_N * WARP_N * WMMA_N / FLOAT_VEC,
            BLOCK_M * WARP_M * WMMA_M,
            BLOCK_N * WARP_N * WMMA_N / FLOAT_VEC
        );

        __syncthreads();
    }
}


__global__ void
__launch_bounds__(REDUCE_K_THREADS_PER_BLOCK, 1)
reduce_k(float *c) {
    int m = blockIdx.y;
    int n = (threadIdx.x + blockIdx.x * REDUCE_K_THREADS_PER_BLOCK) * FLOAT_VEC;
    float sum[FLOAT_VEC] = {0};
    float rc[FLOAT_VEC] = {0};
    for (int grid_k = 0; grid_k < GRID_K; grid_k++) {
        *(int4*)rc = *(int4*)&c[grid_k * M * N + m * N + n];
        for (int v = 0; v < FLOAT_VEC; v++) {
            sum[v] += rc[v];
        }
    }
    *(int4*)&c[m * N + n] = *(int4*)sum;
}


#define checkCudaErrors(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        printf("Error in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)


int main() {
    cudaDeviceProp deviceProp;
    if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
        printf("Error getting device properties\n");
        return 1;
    }
    printf("Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    half *a;
    half *b;
    float *c;
    cudaMallocManaged(&a, sizeof(half) * K * M);
    cudaMallocManaged(&b, sizeof(half) * K * N);
    cudaMallocManaged(&c, sizeof(float) * M * N * GRID_K);
    // initialize
    for (int i = 0; i < M * K; i++) {
        a[i] = __float2half(((float)rand() / RAND_MAX) - 0.5);
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = __float2half(((float)rand() / RAND_MAX) - 0.5);
    }
    for (int i = 0; i < M * N * GRID_K; i++) {
        c[i] = 0;
    }
    int cs = BLOCK_M * WARP_M * WMMA_M * BLOCK_N * WARP_N * WMMA_N * sizeof(float); // size of C tile
    int as = BLOCK_M * WARP_M * WMMA_M * (STEP_K * WMMA_K + PAD_HALF) * sizeof(half); // size of A tile
    int bs = BLOCK_N * WARP_N * WMMA_N * (STEP_K * WMMA_K + PAD_HALF) * sizeof(half); // size of B tile
    int shared_size = max(as + bs, cs);
    checkCudaErrors(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // launch
    int warmup = 3;
    int repeats = 10;
    int number = 10;
    // int warmup = 1;
    // int repeats = 1;
    // int number = 1;
    std::vector<double> times;
    for (int r = 0; r < warmup + repeats; r++) {
        checkCudaErrors(cudaEventRecord(start));
        for (int num = 0; num < number; num++) {
            compute_gemm<<<deviceProp.multiProcessorCount, WARP_SIZE * BLOCK_M * BLOCK_N, shared_size>>>(a, b, c);
            reduce_k<<<dim3(N / (REDUCE_K_THREADS_PER_BLOCK * FLOAT_VEC), M), REDUCE_K_THREADS_PER_BLOCK>>>(c);
            checkCudaErrors(cudaGetLastError());
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double time = milliseconds / 1e3 / number;
        if (r >= warmup) {
            times.push_back(time);
        }
    }
    double min_time = 1e10;
    double max_time = 0;
    double avg_time = 0;
    for (int i = 0; i < times.size(); i++) {
        if (times[i] < min_time) {
            min_time = times[i];
        }
        if (times[i] > max_time) {
            max_time = times[i];
        }
        avg_time += times[i];
    }
    avg_time /= times.size();
    printf("Time [ms] min: %.3f, avg: %.3f, max: %.3f\n", min_time * 1e3, avg_time * 1e3, max_time * 1e3);
    printf("tflop/s: %.3f\n", 1.0 / (avg_time * 1e12) * M * N * K);
    // (random) verify
    int elems_to_verify = 10000;
    for (int i = 0; i < elems_to_verify; i++) {
        int m = rand() % M;
        int n = rand() % N;
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (float)a[m * K + k] * (float)b[n * K + k];
        }
        float eps = 1e-3;
        float val = (float)c[m * N + n];
        if (fabs(val - sum) > eps && fabs(val - sum)/(fabs(sum)+eps) > eps) {
            printf("Error at (%d, %d): %f != %f\n", m, n, c[m * N + n], sum);
            return 1;
        }
    }
    printf("Verification passed\n");
}
