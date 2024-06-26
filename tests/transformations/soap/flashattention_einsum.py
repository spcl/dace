import math
import torch
from functools import partial
from torch import nn, einsum
from torch.autograd.function import Function

from einops import rearrange

import numpy as np

# constants

EPSILON = 1e-10

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# flash attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf

# if yes, we don't apply softmax, no normalization, and no causal masking
# we only do the tensor contractions: A = Q @ K, and then A @ V, without any further processing
# without softmax(A), the attention is not normalized, and the output is not scaled by the inverse of the square root of the dimensionality of the query key vectors
ONLY_TENSOR_CONTRACTIONS = True

class FlashAttentionFunction(Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 1 in the v2 paper """

        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b n -> b 1 1 n')

        if not exists(mask):
            col_masks = (None,) * num_col_tiles
            mask = (col_masks,) * num_row_tiles 
        else:
            mask = ((mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split(q_bucket_size, dim = -2)
            mask = tuple(((row_mask,) * num_col_tiles) if row_mask.shape[-1] == 1 else row_mask.split(k_bucket_size, dim = -1) for row_mask in mask)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            mask,
            all_row_sums.split(q_bucket_size, dim = -2),
            all_row_maxes.split(q_bucket_size, dim = -2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                row_mask
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if not ONLY_TENSOR_CONTRACTIONS:
                    if exists(col_mask):
                        attn_weights.masked_fill_(~col_mask, max_neg_value)

                    if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                        causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                        attn_weights.masked_fill_(causal_mask, max_neg_value)

                    block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                    new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                    exp_weights = torch.exp(attn_weights - new_row_maxes)

                    if exists(col_mask):
                        exp_weights.masked_fill_(~col_mask, 0.)

                    block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                    exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                    exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                    new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                    oc.mul_(exp_row_max_diff).add_(exp_values)

                    row_maxes.copy_(new_row_maxes)
                    row_sums.copy_(new_row_sums)
                
                else:
                    a = 1
                    # perform only vc = qc @ kc and qc @ vc
                    

            oc.div_(row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
                row_mask
            )

            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = torch.exp(attn_weights - lsec)

                if exists(col_mask):
                    p.masked_fill_(~col_mask, 0.)

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim = -1, keepdims = True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None

# main class

# just flash attention in plain pytorch
# it will be way slower than implementing it in CUDA
# for tinkering and educational purposes

class FlashAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        q_bucket_size = 256,
        k_bucket_size = 512
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal

        dim_head = dim // heads
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # memory efficient attention related parameters
        # can be overriden on forward
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        x,
        context = None,
        mask = None,
        q_bucket_size = None,
        k_bucket_size = None,
    ):
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = FlashAttentionFunction.apply(q, k, v, mask, self.causal, q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')
        W_q = self.to_q.weight.data
        W_kv = self.to_kv.weight.data
        dim = W_q.shape[-1]
        W_k = W_kv[:dim]
        W_v = W_kv[dim:]
        W_0 = self.to_out.weight.data
        out2 = Transformers_forward_explicit(x, W_q, W_k, W_v, W_0, h)
        a = 1


        return self.to_out(out)
    

def Transformers_forward_explicit(x,
            W_q,
            W_k,
            W_v,
            W_0,
            heads
            ):

    batch, seq, dim = x.shape
    head_dim = dim // heads
    Q = np.zeros((seq, dim), dtype=np.float32)
    K = np.zeros((seq, dim), dtype=np.float32)
    V = np.zeros((seq, dim), dtype=np.float32)
    A = np.zeros((heads, seq, seq), dtype=np.float32)
    out = np.zeros((batch, seq, dim), dtype=np.float32)

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
    

if __name__ == "__main__":
    # generate some data
    b, n, dim = 2, 2048, 1024 
    x = torch.randn(b, n, dim)

    # instantiate the module
    attn = FlashAttention(dim = dim)

    # forward pass
    out = attn(x)
    print(out.shape)
    # torch.Size([2, 1024, 128])
    # backward pass
    out.sum().backward()