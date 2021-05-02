# pykernsformer

![alt text](https://img.shields.io/pypi/v/pykernsformer)
![alt text](https://img.shields.io/pypi/dd/pykernsformer?color=green&logo=red&logoColor=red)
![alt text](https://img.shields.io/pypi/pyversions/pykernsformer)

The pykernsformer module extends the `torch.nn.TransformerEncoderLayer` class to include custom attention formulas.

# Installation

You can install the pykernsformer package using pip as

`pip install pykernsformer`

# Usage

pykernsformer comes with the following built in attention kernels.

pykernsformer | Attention          | Formula | Citation       |
|-------------|--------------------|---------|----------------|
| `attention` | Regular            | $softmax(\frac{QK^T}{\sqrt{d_k}})V$   | Vaswani et al. |
| `attention_linear` | Linear             | $\frac{QK^T}{\sum_k QK^T}V$  |                |
| `attention_periodic` | Periodic           | $softmax(-\frac{2\sin^2(\pi\frac{\sqrt{2 - 2q_ik_j^T}}{p})}{\sqrt{d_k}})V$ |                |
| `attention_LP` | Locally Periodic     | $softmax(-\frac{2\sin^2(\pi\frac{\sqrt{2 - 2\hat{q}_i\hat{k}_j^T}}{p})}{\sqrt{d_k}} + \frac{{q_i}{k_j^T}}{\sqrt{d_k}})V$ |                |
| `attention_RQ` | Rational Quadratic | $\frac{\left( 1 + \frac{1}{\alpha \sqrt{d_k}} - \frac{2QK^T}{2 \alpha \sqrt{d_k}} \right)^{-\alpha}}{\sum_k \left( 1 + \frac{1}{\alpha \sqrt{d_k}} - \frac{2QK^T}{2 \alpha \sqrt{d_k}} \right)^{-\alpha}}V$ |                |

You can also implement your own attention function with the following signature:

```python
def attention_custom(query, key, value, mask=None, dropout=None):
    
    [...]
    
    p_attn = [...] # the attention matrix
    
    [...]

    return torch.matmul(p_attn, value), p_attn
```
