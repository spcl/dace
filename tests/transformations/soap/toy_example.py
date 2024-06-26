import concurrent.futures
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

# Example function to apply
def func(x):
    return x * x

# Example outer loop function to be parallelized
def process_tile(i_outer, j_outer, tile_size, A):
    for i_inner in range(i_outer, min(i_outer + tile_size, len(A))):
        for j_inner in range(j_outer, min(j_outer + tile_size, len(A))):
            A[i_inner, j_inner] = func(A[i_inner, j_inner])

def parallel_outer_sequential_inner(A, tile_size):
    N = A.shape[0]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_tile, i_outer, j_outer, tile_size, A) for i_outer in range(0, N, tile_size) for j_outer in range(0, N, tile_size)]
        concurrent.futures.wait(futures)

# Define the function to compute the values
def compute_value(i, j):
    return i + j

# Example usage
N = 8
tile_size = 2
A = np.fromfunction(compute_value, (N, N), dtype=int)

parallel_outer_sequential_inner(A, tile_size)

print(A)




def sequential_kernel_0(A, K, O1, O2, Q, V, W, p):

    # we are threadblock p, simiarly to the CUDA counterpart:
    # p = blockIdx.x
    # get the n-D mapping of p to process grid
    p_remain = p
    p_i0 = p_remain // 4
    p_remain = p_remain % 4
    p_i1 = p_remain // 4
    p_remain = p_remain % 4
    p_i2 = p_remain // 1
    p_remain = p_remain % 1


    # calculate the offsets given the self.outer_tile
    i0_outer = p_i0 * 4
    i1_outer = p_i1 * 16
    i2_outer = p_i2 * 4

    # each threadblock computes self.num_domains_per_proc local domains
    for domain_nr in range(16):
        # calculate the offsets given the loc_domain_dims
        domain_remain = domain_nr
        i0_inner = domain_remain // 16
        domain_remain = domain_remain % 16
        i1_inner = domain_remain // 2
        domain_remain = domain_remain % 2
        i2_inner = domain_remain // 1
        domain_remain = domain_remain % 1

        # calculate the global indices
        i0_start = i0_outer + i0_inner*4
        i1_start = i1_outer + i1_inner*2
        i2_start = i2_outer + i2_inner*2

        # sequential schedule within the tile (streaming dimension)
        for i0 in range(i0_start, min(4, i0_start + 4)):
            for i1 in range(i1_start, min(16, i1_start + 2)):
                for i2 in range(i2_start, min(16, i2_start + 2)):
                     A[i2,i1] += K[i2,i0] * Q[i0,i1]

        # take non-reduction part of the computation (loops i1 and i2).
        # we calculate this part of A (A[i2,i1]).
        # We now want to immediately consume this value in the next kernel.
            for i1 in range(i1_start, min(16, i1_start + 2)):
                for i2 in range(i2_start, min(16, i2_start + 2)):

            
        # sequential schedule within the tile (streaming dimension)
        # use i1 range from the previous kernel for the i0 reduction range
        for i0 in range(i1_start, min(16, i1_start + 2)):
            for i1 in range(i1_start, min(4, i1_start + 2)):
                for i2 in range(i2_start, min(16, i2_start + 2)):
                     O1[i2,i1] += A[i2,i0] * V[i0,i1]


def sequential_kernel_2(A, K, O1, O2, Q, V, W, p):

    # we are threadblock p, simiarly to the CUDA counterpart:
    # p = blockIdx.x
    # get the n-D mapping of p to process grid
    p_remain = p
    p_i0 = p_remain // 4
    p_remain = p_remain % 4
    p_i1 = p_remain // 2
    p_remain = p_remain % 2
    p_i2 = p_remain // 1
    p_remain = p_remain % 1


    # calculate the offsets given the self.outer_tile
    i0_outer = p_i0 * 4
    i1_outer = p_i1 * 2
    i2_outer = p_i2 * 8

    # each threadblock computes self.num_domains_per_proc local domains
    for domain_nr in range(4):
        # calculate the offsets given the loc_domain_dims
        domain_remain = domain_nr
        i0_inner = domain_remain // 4
        domain_remain = domain_remain % 4
        i1_inner = domain_remain // 4
        domain_remain = domain_remain % 4
        i2_inner = domain_remain // 1
        domain_remain = domain_remain % 1

        # calculate the global indices
        i0_start = i0_outer + i0_inner*4
        i1_start = i1_outer + i1_inner*2
        i2_start = i2_outer + i2_inner*2

        # sequential schedule within the tile (streaming dimension)
        for i0 in range(i0_start, min(4, i0_start + 4)):
            for i1 in range(i1_start, min(4, i1_start + 2)):
                for i2 in range(i2_start, min(16, i2_start + 2)):
                     O2[i2,i1] += O1[i2,i0] * W[i0,i1]




def parallel_kernel_1(A, K, O1, O2, Q, V, W):
    '''
    Host code. Invocations of two kernels, given gridDim. CUDA counterpart:
    dim3 block(WARP_SIZE);
    dim3 grid(P, 1, 1);

    sequential_kernel_0<<<grid, block>>>(K, Q, W);
    sequential_kernel_1<<<grid, block>>>(K, Q, W);
    '''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(sequential_kernel_0, K, Q, W, grid) for grid in range(P)]
        concurrent.futures.wait(futures)

        futures = [executor.submit(sequential_kernel_1, K, Q, W, grid) for grid in range(P)]
        concurrent.futures.wait(futures)

        futures = [executor.submit(sequential_kernel_2, A, K, O1, O2, Q, V, W, p) for p in range(4)]
        concurrent.futures.wait(futures)
        
        
def parallel_kernel_2(A, K, O1, O2, Q, V, W):
    for p in range(4):
        sequential_kernel_1(A, K, O1, O2, Q, V, W, p)
        
        
# test the parallel kernel above on an example with random input matrices
# K, Q, V, W of sizes D and L

D = 4
L = 16

K = np.random.rand(L, D)
Q = np.random.rand(D, L)
V = np.random.rand(L, D)
W = np.random.rand(D, D)

# allocate memory for the output matrices
A = np.zeros((L, L))
O1 = np.zeros((L, D))
O2 = np.zeros((L, D))


A_local = A[i_start:i_end; j_start:j_end]

def do_local_computations(A_local, B_local):
    # mpi communication
    A_rcv_buffer = np.zeros((L, D))
    

do_local_computations(A_local, B_local)

parallel_kernel_1(A, K, O1, O2, Q, V, W)

# validate the result
O2_ref = K @ Q @ V @ W

np.allclose(O2, O2_ref)
a = 1