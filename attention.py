import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch import Tensor
import numpy as np
import copy
import math

def clones(module, N):
    """
    "Produce N identical layers."
    Use deepcopy the weight are indenpendent.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

'''
k_{\textrm{SE}}(x, x') = \sigma^2\exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
'''
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

'''
k_{\textrm{RQ}}(x, x') = \sigma^2 \left( 1 + \frac{(x - x')^2}{2 \alpha \ell^2} \right)^{-\alpha}
'''
def attention_RQ(query, key, value, mask=None, dropout=None, args=[99]):
    assert len(args) == 1, "Please supply 1 parameter in the args parameter"

    alpha = args[0]
    d_k = query.size(-1)

    query = query / torch.norm(query, dim=-1).unsqueeze(-1)
    key = key / torch.norm(key, dim=-1).unsqueeze(-1)
    
    scores = torch.pow((1 - torch.matmul(query, key.transpose(-2, -1)) / (2 * alpha * math.sqrt(d_k))), -alpha)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    p_attn = F.normalize(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    return torch.matmul(p_attn, value), p_attn

'''
k_{\textrm{Lin}}(x, x') = \sigma_b^2 + \sigma_v^2(x - c)(x' - c)
'''
def attention_linear(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    p_attn = F.normalize(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    return torch.matmul(p_attn, value), p_attn

'''
k_{\textrm{LocalPer}}(x, x') = k_{\textrm{Per}}(x, x')k_{\textrm{SE}}(x, x') = \sigma^2\exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right) \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
'''
def attention_LP(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1) 

    # rbf
    scores = torch.matmul(query, key.transpose(-2, -1))

    # periodic
    query = query / torch.norm(query, dim=-1).unsqueeze(-1)
    key = key / torch.norm(key, dim=-1).unsqueeze(-1)
    scores_ = 2 - torch.matmul(query, key.transpose(-2, -1))
    scores_ = np.pi * torch.sqrt(torch.abs(scores_))

    scores_ = -2 * torch.square(torch.sin(scores_)) / math.sqrt(d_k)

    scores += scores_
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

'''
k_{\textrm{Per}}(x, x') = \sigma^2\exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right)
'''
def attention_periodic(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    query = query / torch.norm(query, dim=-1).unsqueeze(-1)
    key = key / torch.norm(key, dim=-1).unsqueeze(-1)
    scores = 2 - torch.matmul(query, key.transpose(-2, -1))
    scores = np.pi * torch.sqrt(torch.abs(scores))

    scores = -2 * torch.square(torch.sin(scores)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn