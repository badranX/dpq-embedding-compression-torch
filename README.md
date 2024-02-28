Differentiable Product Quantization for Embedding Compression
================================================================================

This is a PyTorch implementation of the differentiable product quantization (DPQ) component (https://arxiv.org/abs/1908.09756). 

You can find the author's [original implementation in Tensorflow here](https://github.com/chentingpc/dpq_embedding_compression) 


## Installation

Create a virtual environment with Python 3 and then run `pip install -e .`
 
## Example
 
```python
import torch
from dpq import DPQ

vocab_size = 100
dpq_component = DPQ(vocab_size=vocab_size,
               K=16,
               D=32,
               emb_size=64,
               kdq_type='smx',
               kdq_d_in=0,
               kdq_share_subspace=True,
               additive_quantization=False)
x = torch.randint(vocab_size, (3,8))

input_emb, codes, losses = dpq_component(x, training=False)
```
