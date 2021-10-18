#include <dace/xilinx/device.h>
#include <dace/math.h>
#include <dace/complex.h>
constexpr long long P = 16;
constexpr long long M = 32;
constexpr long long N = 32;
constexpr long long K = 32;

struct gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t {

};


void module_read_A_4_0(float const *__A_device_in, dace::FIFO<float, 1, P> &A_pipe) {

    {
        for (int n0 = 0; n0 < (N / P); n0 += 1) {
            for (int k = 0; k < K; k += 1) {
                for (int n1 = 0; n1 < P; n1 += 1) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_FLATTEN
                    {
                        float from_memory = __A_device_in[((K * ((P * n0) + n1)) + k)];


                        ///////////////////
                        // Tasklet code (read_A)
                        A_pipe.push(from_memory);
                        ///////////////////

                    }
                }
            }
        }
    }
}

void module_read_B_9_0(dace::vec<float, 8> const *__B_device_in, dace::FIFO<float, 8, 1> &B_pipe) {

    {
        for (int n = 0; n < (N / P); n += 1) {
            for (int k = 0; k < K; k += 1) {
                for (int m = 0; m < (M / 8); m += 1) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_FLATTEN
                    {
                        dace::vec<float, 8> from_memory = __B_device_in[(m + ((M * k) / 8))];


                        ///////////////////
                        // Tasklet code (read_B)
                        B_pipe.push(from_memory);
                        ///////////////////

                    }
                }
            }
        }
    }
}

// [Double pumped] void write_c_31_multiply_add_30_buffer_a_29(dace::FIFO<float, 1, P> A_pipe[(P + 1)], dace::FIFO<float, 8, 1> B_pipe[(P + 1)], dace::FIFO<float, 8, 1> C_pipe[(P + 1)]);

void module_write_C_39_0(dace::vec<float, 8> const *__C_device_in, dace::vec<float, 8> *__C_device_out, dace::FIFO<float, 8, 1> &C_pipe) {

    {
        for (int n = 0; n < N; n += 1) {
            for (int m = 0; m < (M / 8); m += 1) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                {
                    dace::vec<float, 8> from_kernel = C_pipe.pop();
                    dace::vec<float, 8> prev_c = __C_device_in[(m + ((M * n) / 8))];
                    dace::vec<float, 8> to_memory;

                    ///////////////////
                    // Tasklet code (write_C)
                    to_memory = (from_kernel + prev_c);
                    ///////////////////

                    *(__C_device_out + (m + ((M * n) / 8))) = to_memory;
                }
            }
        }
    }
}

DACE_EXPORTED void gemm_0_0(float *__A_device_in_0, dace::vec<float, 8> *__B_device_in_0, dace::vec<float, 8> *__C_device_in_0, dace::vec<float, 8> *__C_device_out_0, dace::FIFO<float, 1, P> &A_pipe, dace::FIFO<float, 8, 1> &B_pipe, dace::FIFO<float, 8, 1> &C_pipe) {
    #pragma HLS INTERFACE m_axi port=__A_device_in_0 offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=__B_device_in_0 offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=__C_device_in_0 offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=__C_device_out_0 offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=__A_device_in_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=__B_device_in_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=__C_device_in_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=__C_device_out_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE axis port=A_pipe
    #pragma HLS INTERFACE axis port=B_pipe
    #pragma HLS INTERFACE axis port=C_pipe
    
    #pragma HLS DATAFLOW
    
    HLSLIB_DATAFLOW_INIT();
    HLSLIB_DATAFLOW_FUNCTION(module_read_A_4_0, __A_device_in_0, A_pipe);
    HLSLIB_DATAFLOW_FUNCTION(module_read_B_9_0, __B_device_in_0, B_pipe);
    // [Double pumped] HLSLIB_DATAFLOW_FUNCTION(write_c_31_multiply_add_30_buffer_a_29, A_pipe, B_pipe, C_pipe);
    HLSLIB_DATAFLOW_FUNCTION(module_write_C_39_0, __C_device_in_0, __C_device_out_0, C_pipe);
    HLSLIB_DATAFLOW_FINALIZE();
}
