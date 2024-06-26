import copy
import numpy as np
import dace as dc
import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis_from_ir, perform_soap_analysis_from_sdfg, perform_soap_analysis_einsum
from dace.transformation.estimator.soap.utils import d2sp
import numpy as np
import sympy as sp
import torch
from einops import rearrange

dim, head_dim, batch, seq, heads = (dc.symbol(s, dtype=dc.int64) for s in ('dim', 'head_dim','batch', 'seq', 'heads'))

head_dim = dim // heads

@dc.program
def Transformers_forward(x: dc.float16[batch,seq, dim], 
            to_qkv: dc.float32[dim, 3*dim],
            W_0: dc.float32[dim, dim]):
    # Step 1
    qkv: dc.float32[batch, seq, 3*dim] #= x @ to_qkv
    qkv = torch.einsum('b t emb, emb emb3 -> b t emb3', x, to_qkv) # [batch, tokens, dim*3*heads ]

    # Step 2
    # decomposition to q,v,k and cast to tuple
    # the resulted shape before casting to tuple will be:
    # [3, batch, heads, tokens, dim_head]
    q: dc.float32[batch, heads, seq, head_dim]
    k: dc.float32[batch, heads, seq, head_dim]
    v: dc.float32[batch, heads, seq, head_dim]
    q, k, v = tuple(rearrange(qkv, 'b t (head_d k h) -> k b h t head_d ', k=3, h=heads))

    # Step 3
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod: dc.float32[batch, heads, seq, seq]
    scaled_dot_prod = torch.einsum('b h t1 head_d , b h t2 head_d -> b h t1 t2', q, k)

    attention = torch.softmax(scaled_dot_prod, dim=-1)

    # Step 4. Calc result per batch and per head h
    out: dc.float32[batch, heads, seq, head_dim]
    out = torch.einsum('b h t1 t2 , b h t2 head_d -> b h t1 head_d', attention, v)

    # Step 5. Re-compose: merge heads with dim_head d
    out2: dc.float32[batch, seq, dim]
    out2 = rearrange(out, "b h t head_d -> b t (h head_d)")

    # Step 6. Apply final linear transformation layer
    ret: dc.float32[batch, seq, dim]
    ret = torch.einsum('b t emd1, emb1 emb2 -> b t emb2', out2, W_0)
    return ret


@dc.program
def Transformers_np_forward(x: dc.float16[batch,seq, dim], 
            to_qkv: dc.float32[dim, 3*dim],
            W_0: dc.float32[dim, dim]
            ):
    # Step 1
    qkv: dc.float32[batch, seq, 3*dim] #= x @ to_qkv
    qkv = np.einsum('bte,ef->btf', x, to_qkv) # [batch, tokens, dim*3*heads ]

    # Step 2
    # decomposition to q,v,k and cast to tuple
    # the resulted shape before casting to tuple will be:
    # [3, batch, heads, tokens, dim_head]
    q: dc.float32[batch, heads, seq, head_dim]
    k: dc.float32[batch, heads, seq, head_dim]
    v: dc.float32[batch, heads, seq, head_dim]
    q, k, v = tuple(rearrange(qkv, 'b t (head_d k h) -> k b h t head_d ', k=3, h=heads))

    # Step 3
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod: dc.float32[batch, heads, seq, seq]
    scaled_dot_prod = np.einsum('bhtd,bhud->bhtu', q, k) 

    # attention = torch.softmax(scaled_dot_prod, dim=-1)
    attention = scaled_dot_prod

    # Step 4. Calc result per batch and per head h
    out: dc.float32[batch, heads, seq, head_dim]
    out = np.einsum('bhtu,bhud->bhtd', attention, v)

    # Step 5. Re-compose: merge heads with dim_head d
    out2: dc.float32[batch, seq, dim]
    out2 = rearrange(out, "b h t head_d -> b t (h head_d)")

    # Step 6. Apply final linear transformation layer
    ret: dc.float32[batch, seq, dim]
    ret = np.einsum('bte,ef->btf', out2, W_0)
    return ret

@dc.program
def Transformers_forward_explicit(x: dc.float16[batch,seq, dim],
            W_q: dc.float32[dim, dim],
            W_k: dc.float32[dim, dim],
            W_v: dc.float32[dim, dim],
            W_0: dc.float32[dim, dim],
            out: dc.float32[batch, seq, dim]
            ):

    Q: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    K: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    V_t: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    V: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    A: dc.float32[heads, seq, seq] = np.zeros((heads, seq, seq), dtype=np.float32)

    for b in range(batch):
        for t in range(seq):
            for d in range(dim):
                for e in range(dim):
                    Q[t,d] += x[b,t,e] * W_q[e,d]
                    K[t,d] += x[b,t,e] * W_k[e,d]
                    V[t,d] += x[b,t,e] * W_v[e,d]


        for t1 in range(seq):
            for t2 in range(seq):
                for h in range(heads):
                    for d in range(head_dim):
                        A[h,t1,t2] += Q[t1,h*head_dim+d] * K[t2,h*head_dim+d]

        for t1 in range(seq):
            for h in range(heads):
                for d in range(head_dim):                        
                    for t2 in range(seq):
                        V[t1,h*head_dim+d] += A[h,t1,t2] * V[t2,h*head_dim+d]
        
        for t in range(seq):
            for d in range(dim):
                for e in range(dim):
                    out[b,t,d] += V[t,e] * W_0[e,d]

    return out



@dc.program
def Transformers_single_head_forward_explicit(x: dc.float16[batch,seq, dim],
            W_q: dc.float32[dim, dim],
            W_k: dc.float32[dim, dim],
            W_v: dc.float32[dim, dim],
            W_0: dc.float32[dim, dim],
            out: dc.float32[batch, seq, dim]
            ):

    Q: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    K: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    V: dc.float32[seq, dim] = np.zeros((seq, dim), dtype=np.float32)
    A: dc.float32[seq, seq] = np.zeros((seq, seq), dtype=np.float32)

    for b in range(batch):
        for t in range(seq):
            for d in range(dim):
                for e in range(dim):
                    Q[t,d] += x[b,t,e] * W_q[e,d]
                    K[t,d] += x[b,t,e] * W_k[e,d]
                    V[t,d] += x[b,t,e] * W_v[e,d]


        for t1 in range(seq):
            for t2 in range(seq):
                    for d in range(dim):
                        A[t1,t2] += Q[t1,d] * K[t2,d]

        for t1 in range(seq):
                for d in range(dim):                        
                    for t2 in range(seq):
                        V[t1,d] += A[t1,t2] * V[t2,d]
        
        for t in range(seq):
            for d in range(dim):
                for e in range(dim):
                    out[b,t,d] += V[t,e] * W_0[e,d]

    return out


def test_3MM():
    NI = 10
    NJ = 15
    NK = 20
    NL = 25
    NM = 30

    A = np.random.rand(NI, NJ).astype(np.float32)
    B = np.random.rand(NJ, NK).astype(np.float32)
    C = np.random.rand(NK, NL).astype(np.float32)
    D = np.random.rand(NL, NM).astype(np.float32)
    
    out1 = np.zeros((NI, NM), dtype=np.float32)
    out2 = np.zeros((NI, NM), dtype=np.float32)

    out1 = A@B@C@D

    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                for l in range(NL):
                    for m in range(NM):
                        out2[i,m]+=A[i,j]*B[j,k]*C[k,l]*D[l,m]

    res = np.allclose(out1, out2)
    a = 1

def test_mttkrp():
    NI = 10
    NJ = 15
    NK = 20
    NL = 25
    NA = 30
    X = np.random.rand(NI, NJ, NK).astype(np.float32)
    A = np.random.rand(NJ, NA).astype(np.float32)
    B = np.random.rand(NK, NA).astype(np.float32)
    C = np.random.rand(NA, NL).astype(np.float32)
    out1 = np.zeros((NI, NL), dtype=np.float32)
    out2 = np.zeros((NI, NL), dtype=np.float32)
    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                for l in range(NL):
                    for a in range(NA):
                        out1[i,l]+=X[i,j,k]*A[j,a]*B[k,a]*C[a,l]

    t0 = np.zeros((NJ, NK, NA), dtype=np.float32)
    t1 = np.zeros((NI, NA), dtype=np.float32)
    for j in range(NJ):
        for k in range(NK):
            for a in range(NA):
                t0[j,k,a]+=A[j,a]*B[k,a]

    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                for a in range(NA):
                    t1[i,a]+=X[i,j,k]*t0[j,k,a]

    for i in range(NI):
        for l in range(NL):
            for a in range(NA):
                out2[i,l]+=t1[i,a]*C[a,l]

    res = np.allclose(out1, out2)
    a = 1


from dace.transformation.auto import auto_optimize
import dace
from dace.transformation.interstate import LoopToMap, RefineNestedAccess


if __name__ == "__main__":
    # test_3MM()
    a = 1

    sdfg = Transformers_single_head_forward_explicit.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()
    sdfg.apply_transformations_repeated([LoopToMap, RefineNestedAccess], validate=True)
    # auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU)
    sdfg.save("tmp.sdfg", hash=False)    
    decomp_params = [("p", 255), ("Ss", 102400)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))
    decomp_params.append(('TSTEPS', 20))
    decomp_params.append(('dim_m', 20000))
    decomp_params.append(('dim_n', 1000))
    soap_result = perform_soap_analysis_from_sdfg(sdfg, decomp_params,
                    generate_schedule = False)
    #soap_result.subgraphs[0].get_data_decomposition(0)
    print(soap_result.subgraphs[0].p_grid)
    a = 1
