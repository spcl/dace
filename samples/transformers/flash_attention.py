import dace
import math
import torch
import numpy as np

N, H, L, E = dace.symbol('N'), dace.symbol('H'), 512, 128

desc = dace.float32[N*H, L, E] @ dace.StorageType.GPU_Global

# Blocking parameters
Br, Bc = 16,16 #128, (64 if E <= 64 else 32)
Bh = E #max(next_power_of_2(E), 16) - effectively Bh = E is almost always used
Tr = int(math.ceil(L / Br))
Tc = int(math.ceil(L / Bc))
num_stages = 4 if E <= 64 else 3
num_warps = 4

grid_dims = Tr, N*H

@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def flash_attention(Q: desc, K: desc, V: desc, O: desc, LogSumExp: desc, scale: dace.float32, exp2_opt: bool = False):
    #assert tuple(Q.shape) == (N, H, L, E)
    q = Q.reshape((N*H, Tr, Br, E))
    k = K.reshape((N*H, Tc, Bc, E))
    v = V.reshape((N*H, Tc, Bc, E))
    out = O.reshape((N*H, Tr, Br, E))
    logsumexp = LogSumExp.reshape((N*H, Tr, Br, E))
    if exp2_opt:
        scale *= 1 / math.log(2)

    for off_hb, start_m in dace.map[0:N*H, 0:Tr] @ dace.ScheduleType.GPU_Device:
        # Set up temporaries
        m = np.full([Br], -np.inf, np.float32)
        l = np.ones([Br], np.float32)
        acc_o = np.zeros([Br, Bh], np.float32)

        # qi = dace.ndarray([Br, E], dtype=dace.float16, storage=dace.StorageType.GPU_Shared)
        qi = q[off_hb, start_m] # Read block

        for start_n in range(Tc):  # Starting with not causal (causal would be up to (thread + 1) * Bc)
            kj = k[off_hb, start_n]  # Read block

            #for wid, tid in dace.map[0:num_warps, 0:32] @ dace.ScheduleType.GPU_ThreadBlock:

            # Q*K^T
            qkt = np.einsum('ik,jk->ij', qi, kj) #qi @ kj.T

            m_ij = np.maximum(m, np.max(qkt, axis=1) * scale)
            qkt *= scale
            qkt[:] = qkt - m_ij[:, None]

            # Softmax updates
            if exp2_opt:
                p_ij = np.exp2(qkt)
            else:
                p_ij = np.exp(qkt)
            l_ij = np.sum(p_ij, axis=1)
            expmsub = np.exp(m - m_ij)
            l *= expmsub
            l += l_ij
            acc_o[:] = acc_o * expmsub[:, None]

            # P*V
            vj = v[off_hb, start_n]  # Read block
            acc_o += p_ij.astype(V.dtype) @ vj

            # Update logsumexp
            m[:] = m_ij

        # Epilogue
        m += np.log2(l)
        acc_o[:] = acc_o / l[:, None]
        logsumexp[off_hb, start_m] = m[:, None]
        out[off_hb, start_m] = acc_o.astype(O.dtype)

def softmax(x, dim):
    """Perform softmax on x along dimension dim."""
    exps = np.exp(x - x.max(axis=dim, keepdims=True))
    return exps / exps.sum(axis=dim, keepdims=True)

def attn_forward_numpy(q, k, v, scale, mask=None):
    """Multi-head attention on queries q, keys k, and values v.

    q, k, v have shape (batch, sequence length, embedding).
    wq, wk, wv have shape (heads, proj size, embedding).
    wo has shape (embedding, embedding).
    in_b is a bias for each linear projection for each head,
    shape (3, heads, proj size).
    out_b is a bias for wo, shape (embedding,).
    scale is a scalar.
    mask has shape (sequence length, sequence length).

    Returns the attention output and intermediates (for backprop).

    """
    # Note: Need to do either the batch or head dimension explicitly,
    # NumPy can't broadcast the outer dimensions together.
    scores = np.matmul(q, np.transpose(k, (0, 2, 1)))

    if mask is not None:
        scores += mask

    scaled_scores = softmax(scale * scores, dim=-1)
    return np.matmul(scaled_scores, v)


N = 2
H = 12

device = 'cuda:0'
np.random.seed(1234)
q = np.random.rand(N, H, L, E).astype(np.float32) * 10 #torch.randn(N, L, H*E, device=device)#.half()
k = np.random.rand(N, H, L, E).astype(np.float32) #torch.randn(N, L, H*E, device=device)#.half()
v = np.random.rand(N, H, L, E).astype(np.float32) #torch.randn(N, L, H*E, device=device)#.half()
out = np.random.rand(N, H, L, E).astype(np.float32) #torch.randn(N, L, H*E, device=device)#.half()

mha = torch.nn.MultiheadAttention(E*H, H, bias=False, batch_first=True, device=device)
with torch.no_grad():
    mha.in_proj_weight[0:H*E, :] = torch.eye(H*E, device=device)
    mha.in_proj_weight[H*E:2*H*E, :] = torch.eye(H*E, device=device)
    mha.in_proj_weight[2*H*E:, :] = torch.eye(H*E, device=device)
    mha.out_proj.weight[:] = torch.eye(H*E, device=device)

def tp(x):
    return torch.tensor(x.transpose(0, 2, 1, 3).reshape(N, L, H*E)).to(device=device)


torchout, *p = mha(tp(q), tp(k), tp(v))
torchout = torchout.detach().cpu().numpy()
torchout = torchout.reshape(N, L, H, E).transpose(0, 2, 1, 3).reshape(N*H, L, E)
# print(torchout)
print('torch done')

q = torch.tensor(q.reshape(N*H, L, E), device='cuda:0')
k = torch.tensor(k.reshape(N*H, L, E), device='cuda:0')
v = torch.tensor(v.reshape(N*H, L, E), device='cuda:0')
fout = torch.randn(N * H, L, E, device='cuda:0')
logsumexp = torch.randn(N * H, L, E, device='cuda:0')
flash_attention(q, k, v, fout, logsumexp, 1/math.sqrt(E), N=N, H=H)
fout = fout.detach().cpu().numpy()
# out = o_w @ out

print('flash done')
print('Max abs difference flash - torch:', np.max(np.abs(torchout - fout)))
