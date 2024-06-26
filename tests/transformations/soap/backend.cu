#include <mpi.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define DATATYPE fp16

#define P ..... (known at code generation time)
#define M N K .... (known at code generation time, sizes of tensors)

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // allocate memory for local tensors
    DATATYPE* A, *B, *C = .....
    contract_tensors_mpi(rank, A, B, C, M, N, K);
}


void contract_tensors_mpi(int p, float *A, float *B, float *C, int n, int m, int k) {
    std::tie(p_x, p_y, p_z) = get_nD_grid(p); // map rank p to 3D grid (all optimal, static)

    // get local domain ranges
    std::tie(i_start, i_end, j_start, j_end, k_start, k_end) = \
        get_mpi_domain_range(p_x, p_y, p_z); // get domain range for rank p

    #define Dom_size_I, Dom_size_J, Dom_size_K .... // known at code generation time.
    #define K_step .... // known at code generation time. If K_step == Dom_size_K, then we have 
    // one round of communcation. Otherwise, we have Dom_size_K/K_step rounds of communication.
    
    // assume k is the reduction dimension. Preferably, we should have k = Dom_size_K - one CUDA invocation.
    for (int k = 0; k < Dom_size_K; k += K_step) {
        // inter-node communication
        MPI_Put(&A_local, ..., dest_rank,...);
        MPI_Put(&B_local, ..., dest_rank,...);
        ...
        // launch CUDA kernel
        #define n_threadblocks .... // known at code generation time.
        # define block_size .... // known at code generation time.
        // creagte CUDA grid object
        dim3 grid(n_threadblocks, 1, 1);
        dim3 block(block_size, block_size, 1);
        contract_tensors_cuda<<<grid, block>>>(A_local, B_local, C_local); // no need to pass
        // the size of the local tensors, as they are known at code generation time.
        cudaDeviceSynchronize();

        // maybe reduction is needed
    }
}



// CUDA kernel
__global__ void contract_tensors_cuda(float *A, float *B, float *C) {
    // p is the threadblock id. In our parallel model, this is this "single parallel processor".
    int p = blockIdx.x;
    std::tie(p_x, p_y, p_z) = get_nD_grid_gpu(p); // map rank p to 3D grid (all optimal, static)

    // get local domain ranges
    std::tie(i_start, i_end, j_start, j_end, k_start, k_end) = \
        get_cuda_domain_range(p_x, p_y, p_z,, P); // get domain range for rank p
    
    // move pointers A, B, C to the start of the local domain
    // NOTE: we operate on MPI local domains, we never materialize the global tensors of size M,N,K
    A += i_start * Dom_size_K  + k_start;
    B += j_start + k_start* Dom_size_J;
    C += i_start * Dom_size_J + j_start;

    // copy-paste the nice Tensor Core CUDA code here, taken from e.g., Bruce Lee repo
    // or any other close-to-optimal GEMM code. The generated code just says:
    // YOU ARE THREADBLOCK P. PLEASE COMPUTE C = A x B, where A, B, C are local tensors of size 
    // Dom_size_I x Dom_size_J x Dom_size_K, GIVEN THE POINTERS A, B, C 
    // (which we manually moved to the start of your patch of local domain).
    code = generated_code_that_receives(A, B, C, i_start, i_end, j_start, j_end, k_start, k_end);

    // and now, the Tensor Contraction Multiprocess Optimizations (TCMO) magic happens...

}