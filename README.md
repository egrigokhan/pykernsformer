# pykernsformer

![alt text](https://img.shields.io/pypi/v/pykernsformer)
![alt text](https://img.shields.io/pypi/dd/pykernsformer?color=green&logo=red&logoColor=red)
![alt text](https://img.shields.io/pypi/pyversions/pykernsformer)

The pykernsformer module extends the `torch.nn.TransformerEncoderLayer` class to include custom attention formulas.

| Attention          | Formula | Citation       |
|--------------------|---------|----------------|
| Regular            | $softmax(\frac{QK^T}{\sqrt{d_k}})V$   | Vaswani et al. |
| Linear             | $\frac{QK^T}{\sum_k QK^T}V$  |                |
| Periodic           | $softmax(-\frac{2\sin^2(\pi\frac{\sqrt{2 - 2q_ik_j^T}}{p})}{\sqrt{d_k}})V$ |                |
| Local Periodic     | $softmax(-\frac{2\sin^2(\pi\frac{\sqrt{2 - 2\hat{q}_i\hat{k}_j^T}}{p})}{\sqrt{d_k}} + \frac{{q_i {k_j^T}}{\sqrt{d_k}})V$ |                |
| Rational Quadratic | $\frac{\left( 1 + \frac{1}{\alpha \sqrt{d_k}} - \frac{2QK^T}{2 \alpha \sqrt{d_k}} \right)^{-\alpha}}{\sum_k \left( 1 + \frac{1}{\alpha \sqrt{d_k}} - \frac{2QK^T}{2 \alpha \sqrt{d_k}} \right)^{-\alpha}}V$ |                |

# Installation

You can install the pykernsformer package using pip as

`pip install pykernsformer`

# Usage

