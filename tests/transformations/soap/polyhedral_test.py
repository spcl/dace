import copy
import numpy as np
import dace as dc
import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis, perform_soap_analysis_einsum
from dace.transformation.estimator.soap.utils import d2sp
import numpy as np
import sympy as sp

dim_m, dim_n, dim_k = (dc.symbol(s, dtype=dc.int64) for s in ('dim_m', 'dim_n', 'dim_k'))
grid_size, grid_jump = (dc.symbol(s, dtype=dc.int64) for s in ('grid_size', 'grid_jump'))
N,K,D, FIN, FOUT, P_init = (dc.symbol(s, dtype=dc.int64) 
        for s in ('N', 'K', 'D', 'FIN', 'FOUT', 'P_init'))

@dc.program
def jacobi1d(TSTEPS: dc.int32, A: dc.float32[dim_m], B: dc.float32[dim_m]):
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@dc.program
def fdtd_2d(TSTEPS: dc.int32, ex: dc.float32[dim_m,dim_n], ey: dc.float32[dim_m,dim_n],
                hz: dc.float32[dim_m,dim_n], _fict_: dc.float32[dim_m]):
    for t in range(TSTEPS):
        # ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])

@dc.program
def ico_stencil(TSTEPS: dc.int32, A: dc.float32[dim_m,dim_m], B: dc.float32[dim_m,dim_m]):
    for t in range(TSTEPS):
        for i in range(dim_m):
            for j in range(dim_m):                
                # The following seven loads should be buffered. In each iteration, we load 
                # only 3 new vertices, but we need 7. We can reuse 4 vertices in the streaming buffer.
                B[i,j] = (A[i, j] +  A[i - 1, j] + A[i - 1, j + 1] + A[i, j - 1] + \
                    A[i, j + 1] + A[i + 1, j - 1] + A[i + 1, j]) / 7
        A = B
        


#@dc.program
def horizontal_diffusion(TSTEPS: dc.int32, A: dc.float32[dim_m,dim_m], B: dc.float32[dim_m,dim_m]):
    dim_m = A.shape[0]
    for t in range(TSTEPS):
        for i in range(1, dim_m -1):
            for j in range(1, dim_m -1):                
                B[i,j] = (A[i-1, j] + \
                        A[i+1, j] + \
                        A[i, j-1] + \
                        A[i, j+1] + \
                        A[i, j]) / 5
        tmp = B
        B = A
        A = tmp

#@dc.program
def horizontal_diffusion_tiled(TSTEPS: dc.int32, 
                               tile_size: dc.int32,
                               A: dc.float32[dim_m,dim_m], 
                               B: dc.float32[dim_m,dim_m]):
    dim_m = A.shape[0]
    for t in range(TSTEPS):
        for ii in range(1, dim_m -1, tile_size):
            for jj in range(1, dim_m -1, tile_size):
                for i in range(ii, min(ii+tile_size, dim_m-1)):
                    for j in range(jj, min(jj+tile_size, dim_m-1)):                
                        B[i,j] = (A[i-1, j] + \
                                A[i+1, j] + \
                                A[i, j-1] + \
                                A[i, j+1] + \
                                A[i, j]) / 5
        tmp = B
        B = A
        A = tmp


def c2d(i,j, offset):
    return i+j, (j + i % 2)//2 - (i+1)//2 + offset

def d2c(x,y, offset):
    i = int(offset + x//2 - y)
    j = int((x+1)//2 - offset + y + int(offset + (x+1)//2 -y) % 2  + 1/2) - (i+x)%2
    return i, j

def g2l(coords, tile_size_i, tile_size_j):
    i,j = coords
    return i//tile_size_i, j//tile_size_j, i%tile_size_i + 1, j%tile_size_j + 1

def l2g(ti, tj, coords, tile_size_i, tile_size_j):
    i,j = coords
    return ti*tile_size_i + i - 1, tj*tile_size_j + j - 1
    
def horizontal_diffusion_tiled_buffered(TSTEPS: dc.int32, 
                               tile_size: dc.int32,
                               A: dc.float32[dim_m,dim_m], 
                               B: dc.float32[dim_m,dim_m]):
    dim_m = A.shape[0]
    local_tile_A = np.zeros((dim_m//tile_size, dim_m//tile_size, tile_size+2, tile_size+2))
    local_tile_B = np.zeros((dim_m//tile_size, dim_m//tile_size, tile_size+2, tile_size+2))
    buff_left = np.zeros(tile_size)
    buff_right = np.zeros(tile_size)
    buff_top = np.zeros(tile_size)
    buff_bottom = np.zeros(tile_size)

    # initialize local tile data
    ti = -1
    for ii in range(1, dim_m -1, tile_size):
        ti += 1
        tj = -1
        for jj in range(1, dim_m -1, tile_size):
            tj += 1
            local_tile_A[ti, tj,1:tile_size+1, 1:tile_size+1] = A[ii:ii+tile_size, jj:jj+tile_size]

    for t in range(TSTEPS):
        ti = -1
        for ii in range(1, dim_m -1, tile_size):
            ti += 1
            tj = -1
            for jj in range(1, dim_m -1, tile_size):
                tj += 1
                # receive halo points from neighbors
                if tj > 0:
                    local_tile_A[ti, tj, 1:tile_size+1,0] = \
                        local_tile_A[ti, tj-1, 1:tile_size+1,tile_size]
                if tj < dim_m//tile_size -1:
                    local_tile_A[ti, tj, 1:tile_size+1,tile_size+1] = \
                        local_tile_A[ti, tj+1, 1:tile_size+1,1]
                if ti > 0:
                    local_tile_A[ti, tj, 0, 1:tile_size+1] = \
                        local_tile_A[ti-1, tj, tile_size, 1:tile_size+1]
                if ti < dim_m//tile_size -1:
                    local_tile_A[ti, tj, tile_size+1, 1:tile_size+1] = \
                        local_tile_A[ti+1, tj, 1, 1:tile_size+1]
                
                # purely local computation
                for i in range(1, tile_size+1):
                    for j in range(1,tile_size+1):                
                        local_tile_B[ti, tj, i,j] = \
                               (local_tile_A[ti, tj, i-1, j] + \
                                local_tile_A[ti, tj, i+1, j] + \
                                local_tile_A[ti, tj, i, j-1] + \
                                local_tile_A[ti, tj, i, j+1] + \
                                local_tile_A[ti, tj, i, j]) / 5
        tmp = local_tile_B
        local_tile_B = local_tile_A
        local_tile_A = tmp

    # writeback local tile data
    ti = -1
    for ii in range(1, dim_m -1, tile_size):
        ti += 1
        tj = -1
        for jj in range(1, dim_m -1, tile_size):
            tj += 1
            A[ii:ii+tile_size, jj:jj+tile_size] = local_tile_A[ti, tj,1:tile_size+1, 1:tile_size+1]
            B[ii:ii+tile_size, jj:jj+tile_size] = local_tile_B[ti, tj,1:tile_size+1, 1:tile_size+1]


def horizontal_diffusion_diamond(TSTEPS: dc.int32, 
                               A: dc.float32[dim_m,dim_m], 
                               B: dc.float32[dim_m,dim_m]):
    dim_m = A.shape[0]

    dim_i = int(2.5*dim_m)
    diamondA = np.zeros((dim_i, dim_m+2))
    diamondB = np.zeros((dim_i, dim_m+2))

    offset = dim_m//2
    # reshape A to diamond shape
    for i in range(dim_m):
        for j in range(dim_m):
            x, y = c2d(i,j, offset)
            diamondA[x, y] = A[i,j]

    a = 1
    

    for t in range(TSTEPS):    
        # split into two cases: even and odd
        for x in range(1, dim_i, 2):
            for y in range(1,dim_m+1):  
                i,j = d2c(x,y, offset)
                if i <= 0 or i >= dim_m-1 or j <= 0 or j >= dim_m-1:
                    continue
                diamondB[x,y] = \
                       (diamondA[x-1, y] + \
                        diamondA[x-1, y+1] + \
                        diamondA[x+1, y] + \
                        diamondA[x+1, y+1] + \
                        diamondA[x, y]) / 5
        
        for x in range(2, dim_i-1, 2):
            for y in range(1,dim_m+1):   
                i,j = d2c(x,y, offset)
                if i <= 0 or i >= dim_m-1 or j <= 0 or j >= dim_m-1:
                    continue             
                diamondB[x,y] = \
                        (diamondA[x-1, y] + \
                        diamondA[x-1, y-1] + \
                        diamondA[x+1, y] + \
                        diamondA[x+1, y-1] + \
                        diamondA[x, y]) / 5
        tmp = diamondB
        diamondB = diamondA
        diamondA = tmp

    # reshape back from diamond shape
    for i in range(dim_m):
        for j in range(dim_m):
            x, y = c2d(i,j, offset)
            A[i,j] = diamondA[x, y]
            B[i,j] = diamondB[x, y]




def horizontal_diffusion_tiled_diamond(TSTEPS: dc.int32, 
                               tile_size: dc.int32,
                               A: dc.float32[dim_m,dim_m], 
                               B: dc.float32[dim_m,dim_m]):
    

    def diamond_tiles_to_cart(A, B):
        for i in range(1,dim_m -1):
            for j in range(1, dim_m -1):
                A[i,j] = local_tile_A[g2l(c2d(i,j, offset), tile_size_i, tile_size_j)]
                B[i,j] = local_tile_B[g2l(c2d(i,j, offset), tile_size_i, tile_size_j)]


    A_cp = copy.deepcopy(A)
    B_cp = copy.deepcopy(B)
    A_tmp = copy.deepcopy(A)
    B_tmp = copy.deepcopy(B)

    dim_m = A.shape[0]

    tile_size_i = 2*tile_size+1
    tile_size_j = tile_size

    dim_i = int(2.5*dim_m)
    diamondA = np.zeros((dim_i, dim_m+2))
    diamondB = np.zeros((dim_i, dim_m+2))
    local_tile_A = np.zeros((dim_i//tile_size_i + 1, dim_m//tile_size + 1, tile_size_i+2, tile_size+2))
    local_tile_B = np.zeros((dim_i//tile_size_i + 1, dim_m//tile_size + 1, tile_size_i+2, tile_size+2))

    offset = dim_m//2
    # reshape A to diamond shape and initialize local tile data
    for i in range(1,dim_m -1):
        for j in range(1, dim_m -1):
            x, y = c2d(i,j, offset)
            diamondA[x, y] = A[i,j]
            local_tile_A[g2l(c2d(i,j, offset), tile_size_i, tile_size_j)] = A[i,j]
    

    for t in range(TSTEPS):
        
        # reference computation
        for i in range(1, dim_m -1):
            for j in range(1, dim_m -1):                
                B_cp[i,j] = (A_cp[i-1, j] + \
                        A_cp[i+1, j] + \
                        A_cp[i, j-1] + \
                        A_cp[i, j+1] + \
                        A_cp[i, j]) / 5
                
                # A_cp[i-1, j], A_cp[i+1, j],A_cp[i, j-1], A_cp[i, j+1], A_cp[i, j]
        A_cp, B_cp = B_cp, A_cp

        for ti in range(local_tile_A.shape[0]):
            for tj in range(local_tile_A.shape[1]):
                # receive halo points from neighbors
                if tj > 0:
                    local_tile_A[ti, tj, 1:tile_size_i+1,0] = \
                        local_tile_A[ti, tj-1, 1:tile_size_i+1,tile_size]
                if tj < local_tile_A.shape[1] -1:
                    local_tile_A[ti, tj, 1:tile_size_i+1,tile_size+1] = \
                        local_tile_A[ti, tj+1, 1:tile_size_i+1,1]
                if ti > 0:
                    local_tile_A[ti, tj, 0, 1:tile_size+1] = \
                        local_tile_A[ti-1, tj, tile_size_i, 1:tile_size+1]
                if ti < local_tile_A.shape[0] -1:
                    local_tile_A[ti, tj, tile_size_i+1, 1:tile_size+1] = \
                        local_tile_A[ti+1, tj, 1, 1:tile_size+1]
                    
                # single corner points
                if ti > 0:
                    if ti % 2 == 1:
                        if tj < local_tile_A.shape[1] -1:
                            # looking to the right
                            local_tile_A[ti, tj, 0,tile_size + 1] = \
                                local_tile_A[ti-1, tj+1, tile_size_i, 1]
                    else:
                        if tj > 0:
                            # looking to the left
                            local_tile_A[ti, tj, 0, 0] = \
                                local_tile_A[ti-1, tj-1, tile_size_i, tile_size]
                        
                if ti < local_tile_A.shape[0] - 1:
                    if ti % 2 == 1:
                        if tj < local_tile_A.shape[1] -1:
                            # looking to the right
                            local_tile_A[ti, tj, tile_size_i + 1,tile_size + 1] = \
                                local_tile_A[ti+1, tj+1, 1, 1]
                    else:
                        if tj > 0:
                            # looking to the left
                            local_tile_A[ti, tj, tile_size_i + 1, 0] = \
                                local_tile_A[ti+1, tj-1, 1, tile_size]
                
                # purely local computation
                # split into two cases: even and odd
                for i in range(2 - ti % 2, tile_size_i + 1, 2):
                    for j in range(1,tile_size+1):   
                        i_d, j_d = l2g(ti, tj, (i,j), tile_size_i, tile_size_j)
                        i_c, j_c = d2c(i_d, j_d, offset) 
                        if i_c <= 0 or i_c >= dim_i-1 or j_c <= 0 or j_c >= dim_m-1:
                            continue        

                        local_tile_B[ti, tj, i,j] = \
                               (local_tile_A[ti, tj, i-1, j] + \
                                local_tile_A[ti, tj, i-1, j+1] + \
                                local_tile_A[ti, tj, i+1, j] + \
                                local_tile_A[ti, tj, i+1, j+1] + \
                                local_tile_A[ti, tj, i, j]) / 5
                        # local_tile_A[ti, tj, i-1, j], local_tile_A[ti, tj, i-1, j+1], local_tile_A[ti, tj, i+1, j], local_tile_A[ti, tj, i+1, j+1], local_tile_A[ti, tj, i, j]
                
                for i in range(2 - (ti+1) % 2, tile_size_i + 1, 2):
                    for j in range(1,tile_size+1):  
                        i_d, j_d = l2g(ti, tj, (i,j), tile_size_i, tile_size_j)
                        i_c, j_c = d2c(i_d, j_d, offset) 
                        if i_c <= 0 or i_c >= dim_i-1 or j_c <= 0 or j_c >= dim_m-1:
                            continue                
                        local_tile_B[ti, tj, i,j] = \
                            (local_tile_A[ti, tj, i-1, j] + \
                            local_tile_A[ti, tj, i-1, j-1] + \
                            local_tile_A[ti, tj, i+1, j] + \
                            local_tile_A[ti, tj, i+1, j-1] + \
                            local_tile_A[ti, tj, i, j]) / 5
                        # local_tile_A[ti, tj, i-1, j], local_tile_A[ti, tj, i-1, j-1], local_tile_A[ti, tj, i+1, j], local_tile_A[ti, tj, i+1, j-1], local_tile_A[ti, tj, i, j]

        local_tile_A, local_tile_B = local_tile_B, local_tile_A

        # reset boundary conditions
        for i in range(dim_m):
            local_tile_A[g2l(c2d(i,0, offset), tile_size_i, tile_size_j)] = 0
            local_tile_A[g2l(c2d(i,dim_m - 1, offset), tile_size_i, tile_size_j)] = 0
            
            local_tile_A[g2l(c2d(0,i, offset), tile_size_i, tile_size_j)] = 0
            local_tile_A[g2l(c2d(dim_m - 1, i, offset), tile_size_i, tile_size_j)] = 0


        diamond_tiles_to_cart(A_tmp, B_tmp)
        a = 1

    # writeback local tile data

    diamond_tiles_to_cart(A, B)


@dc.program
def gnn(A: dc.float32[dim_m,dim_k], B: dc.float32[dim_k,dim_m], C: dc.float32[dim_m,dim_m]):
    #return A @ B @ C
    D = np.zeros((dim_m,dim_n))
    E = np.zeros((dim_m,dim_n))
    for i in range(dim_m):
        for j in range(dim_n):
            for k in range(dim_k):
                D[i,j] += A[i,k] * B[k,j]
            E[i,j] = D[i,j] + C[i,j]
    return E


@dc.program
def monet(H: dc.float32[N,FIN], M: dc.float32[K,D], sigma: dc.float32[K,D], 
            Wa: dc.float32[FIN,K,FOUT], Wb: dc.float32[P_init,D], b: dc.float32[D], 
            pseudo: dc.float32[N,N, P_init]):
    P = np.einsum("ijk,kl->ijl", pseudo, Wb) + b  # b implicit broadcast to NxN

    C = P.reshape((N, N, 1, D)) - M.reshape((1, 1, K, D))
    S = sigma.reshape((1, 1, K, D))

    tmp0 = C * S
    tmp1 = np.sum(tmp0, axis=3)
    tmp2 = tmp1.reshape((N,N,K,1))
    tmp3 = np.exp(tmp2)
    G = tmp3
    # G = np.exp(np.sum(C * S, axis=-1, keepdims=True))

    tmp4 = np.einsum("ij,jkl->ikl", H, Wa)
    tmp5 = tmp4.reshape((1, N, K, FOUT))
    tmp6 = tmp5 * G
    tmp7 = np.max(tmp6, axis=0)
    H_prim = np.sum(tmp7,axis=1,
    )

    return H_prim
            

def test_stencils():
    np.set_printoptions(edgeitems=30, linewidth=100000, 
        formatter=dict(float=lambda x: "%.3g" % x))
    dim_n = 16 + 2
    tile_size = 4
    steps = 4
    A_in = np.random.rand(dim_n, dim_n).astype(np.float32)
    for i in range(dim_n):
        for j in range(dim_n):
            A_in[i,j] = (i*100 + j) % 1000
    boundary = 0
    A_in[0,:] = boundary
    A_in[-1,:] = boundary
    A_in[:,0] = boundary
    A_in[:,-1] = boundary
    B_in = np.zeros((dim_n, dim_n)).astype(np.float32)

    A_ref = copy.deepcopy(A_in)
    B_ref = copy.deepcopy(B_in)
    horizontal_diffusion(TSTEPS=steps, A= A_ref, B=B_ref)

    A_t = copy.deepcopy(A_in)
    B_t = copy.deepcopy(B_in)
    horizontal_diffusion_tiled(TSTEPS=steps, tile_size=tile_size, A=A_t, B=B_t )

    A_tb = copy.deepcopy(A_in)
    B_tb = copy.deepcopy(B_in)
    horizontal_diffusion_tiled_buffered(TSTEPS=steps, tile_size=tile_size, A=A_tb, B=B_tb )

    A_d = copy.deepcopy(A_in)
    B_d = copy.deepcopy(B_in)
    horizontal_diffusion_diamond(TSTEPS=steps, A=A_d, B=B_d )

    A_td = copy.deepcopy(A_in)
    B_td = copy.deepcopy(B_in)
    horizontal_diffusion_tiled_diamond(TSTEPS=steps, tile_size=tile_size, A=A_td, B=B_td )
    a = 1

if __name__ == "__main__":
    # test_stencils()
    # exit(0)
    # N = 5
    # K = 4
    # D = 3
    # FIN = 2
    # FOUT = 6
    # P_init = 2
    # H = np.arange(FIN * N).reshape(N, FIN)

    # Wa = np.arange(FIN * K * FOUT).reshape(FIN, K, FOUT)
    # Wb = np.arange(P_init * D).reshape(P_init, D)
    # b = np.arange(D)
    # M = np.ones((K, D))
    # sigma = np.empty((K, D))
    # pseudo = np.ones((N, N, P_init))

    # H_prim = monet(H, M, sigma, Wa, Wb, b, pseudo)


    sdfg = ico_stencil.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.save("tmp.sdfg", hash=False)    
    decomp_params = [("p", 255), ("Ss", 102400)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))
    decomp_params.append(('TSTEPS', 20))
    decomp_params.append(('dim_m', 20000))
    decomp_params.append(('dim_n', 1000))
    soap_result = perform_soap_analysis(sdfg, decomp_params,
                    generate_schedule = False)
    #soap_result.subgraphs[0].get_data_decomposition(0)
    print(soap_result.subgraphs[0].p_grid)
    a = 1
