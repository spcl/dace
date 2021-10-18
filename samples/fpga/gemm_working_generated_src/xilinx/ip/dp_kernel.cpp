#include <dace/xilinx/device.h>
#include <dace/math.h>
#include <dace/complex.h>
constexpr long long P = 16;
constexpr long long M = 32;
constexpr long long N = 32;
constexpr long long K = 32;

struct gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t {

};

void core (dace::FIFO<float, 1, P> &A_pipe_in, dace::FIFO<float, 4, 1> &B_pipe_in, dace::FIFO<float, 4, 1> &C_pipe_out, dace::FIFO<float, 1, P> A_pipe[P+1], dace::FIFO<float, 4, 1> B_pipe[P+1], dace::FIFO<float, 4, 1> C_pipe[P+1], int p) {
    for (int n0 = 0; n0 < (N / P); n0 += 1) {
        dace::vec<float, 4> C_buffer[(M / 4)];
        {
            for (int k = 0; k < K; k += 1) {
                #pragma HLS DEPENDENCE variable=C_buffer false
                float A_reg;
                {
                    for (int n1 = 0; n1 < P; n1 += 1) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_FLATTEN
                        {
                            float a_in = p > 0 ? (A_pipe[p]).pop() : A_pipe_in.pop();

                            dace::FIFO<float, 1, 16> &a_out = A_pipe[(p + 1)];

                            ///////////////////
                            // Tasklet code (buffer_a)
                            if (n1 == ((P - p) - 1)) {
                                A_reg = a_in;
                            }
                            if (p < (P - 1)) {
                                a_out.push(a_in);
                            }
                            ///////////////////

                        }
                    }
                }
                {
                    for (int m = 0; m < (M / 4); m += 1) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_FLATTEN
                        #pragma HLS DEPENDENCE variable=C_buffer false
                        {
                            float a_in = A_reg;
                            dace::vec<float, 4> b_in = p > 0 ? B_pipe[p].pop() : B_pipe_in.pop();
                            dace::vec<float, 4> c_in = C_buffer[m];
                            dace::FIFO<float, 4, 1> &b_out = B_pipe[(p + 1)];
                            dace::vec<float, 4> c_out;

                            ///////////////////
                            // Tasklet code (multiply_add)
                            auto c_prev = c_in;
                            if (k == 0) {
                                c_prev = 0;
                            }
                            c_out = (c_prev + (a_in * b_in));
                            if (p < (P - 1)) {
                                b_out.push(b_in);
                            }
                            ///////////////////

                            dace::Write<float, 4>(&C_buffer[m], c_out);
                            #pragma HLS DEPENDENCE variable=c_out false
                        }
                    }
                }
            }
        }
        {
            for (int n1 = 0; n1 < P; n1 += 1) {
                for (int m = 0; m < (M / 4); m += 1) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_FLATTEN
                    {
                        dace::vec<float, 4> const &buffer_in = C_buffer[m];
                        dace::FIFO<float, 4, 1> &forward_in = C_pipe[(p - 1)];
                        dace::FIFO<float, 4, 1> &c_out = p < P-1 ? C_pipe[p] : C_pipe_out;

                        ///////////////////
                        // Tasklet code (write_c)
                        if (n1 <= p) {
                            if (p < P-1)
                                C_pipe[p].push(
                                    (p > 0) && (n1 > 0) ? 
                                        forward_in.pop() : 
                                        buffer_in);
                            else
                                C_pipe_out.push(n1 > 0 ? forward_in.pop() : buffer_in);
                        }
                        ///////////////////

                    }
                }
            }
        }
    }
}

DACE_EXPORTED void dp_kernel(dace::FIFO<float, 1, P> &A_pipe_in, dace::FIFO<float, 4, 1> &B_pipe_in, dace::FIFO<float, 4, 1> &C_pipe_out) 
{
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis port=A_pipe_in
    #pragma HLS INTERFACE axis port=B_pipe_in
    #pragma HLS INTERFACE axis port=C_pipe_out
    #pragma HLS DATAFLOW
    {
        dace::FIFO<float, 1, 16> A_pipe[(P + 1)];
        //#pragma HLS array_partition variable=A_pipe dim=1 complete
        dace::SetNames(A_pipe, "A_pipe", (P + 1));
        dace::FIFO<float, 4, 1> B_pipe[(P + 1)];
        //#pragma HLS array_partition variable=B_pipe dim=1 complete
        dace::SetNames(B_pipe, "B_pipe", (P + 1));
        dace::FIFO<float, 4, 1> C_pipe[(P + 1)];
        //#pragma HLS array_partition variable=C_pipe dim=1 complete
        dace::SetNames(C_pipe, "C_pipe", (P + 1));
        {
            for (int p = 0; p < P; p++) {
                #pragma HLS UNROLL
                {
                    core(A_pipe_in, B_pipe_in, C_pipe_out, A_pipe, B_pipe, C_pipe, p);
                }
            }
        }
    }
}
