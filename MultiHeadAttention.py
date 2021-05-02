from attention import attention, attention_RQ, attention_linear, attention_LP, attention_periodic

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

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, attention_="attention", args=[]):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.attention = attention
        self.args = args

    def attention_(self, query, key, value, mask, dropout, attention_="attention", args=[]):
      ATT = {"attention": attention,
             "attention_linear": attention_linear,
             "attention_periodic": attention_periodic,
             "attention_LP": attention_LP,
             "attention_RQ": attention_RQ}

      if (attention_ in ATT.keys()):
        return ATT[attention_](query, key, value, mask=mask, dropout=self.dropout)
      else:
        return attention_(query, key, value, mask=mask, dropout=self.dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0) # get batch size
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # parttion into h sections，switch 2,3 axis for computation. 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention_(query, key, value, mask=mask, dropout=self.dropout, attention_=self.attention, args=self.args)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x) # final linear layer