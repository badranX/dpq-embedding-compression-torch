import torch
import torch.nn as nn
from kd_quantizer import KDQuantizer


class KDQhparam(nn.Module):
  # A default KDQ parameter setting (demo)
  def __init__(self,
               vocab_size=100,
               K=16,
               D=32,
               emb_size=64,
               kdq_type='smx',
               kdq_d_in=0,
               kdq_share_subspace=True,
               additive_quantization=False):
    """
    Args:
      kdq_type: 'vq' or 'smx'
      kdq_d_in: when kdq_type == 'smx', we could reduce d_in
      kdq_share_subspace: whether or not to share the subspace among D.
    """
    super().__init__()
    
    d = emb_size
    d_in = d//D if kdq_d_in <= 0 else kdq_d_in  # could use diff. d_in/d_out for smx
    d_in = d if additive_quantization else d_in
    d_out = d if additive_quantization else d//D
    out_size = [D, emb_size] if additive_quantization else [emb_size]

    self.query_wemb = torch.randn([vocab_size, D * d_in], dtype=torch.float32)

    if kdq_type == "vq":
        assert self.kdq_d_in <= 0, (
            "kdq d_in cannot be changed (to %d) for vq" % self.kdq_d_in)
        tie_in_n_out = True
        dist_metric = "euclidean"
        beta, tau, softmax_BN = 0.0, 1.0, True
        share_subspace = kdq_share_subspace
    else:
        assert kdq_type == "smx", [
            "unknown kdq_type %s" % self.kdq_type]
        tie_in_n_out = False
        dist_metric = "dot"
        beta, tau, softmax_BN = 0.0, 1.0, True
        share_subspace = kdq_share_subspace

    self.kdq = KDQuantizer(K, D, d_in, d_out, tie_in_n_out,
                      dist_metric, share_subspace,
                      beta, tau, softmax_BN)
    self.d_in = d_in
    self.d_out = d_out
    self.out_size = out_size
    self.K = K
    self.D = D
    self.kdq_type = kdq_type
    self.kdq_d_in = kdq_d_in
    self.kdq_share_subspace = kdq_share_subspace
    self.additive_quantization = additive_quantization
    
    
  def forward(self, input, training=True):
    idxs = torch.reshape(input, [-1])
    input_emb = self.query_wemb[idxs]

    if self.kdq_type == "vq":
        assert self.kdq_d_in <= 0, (
            "kdq d_in cannot be changed (to %d) for vq" % self.kdq_d_in)
        tie_in_n_out = True
        dist_metric = "euclidean"
        beta, tau, softmax_BN = 0.0, 1.0, True
        share_subspace = self.kdq_share_subspace
    else:
        assert self.kdq_type == "smx", [
            "unknown kdq_type %s" % self.kdq_type]
        tie_in_n_out = False
        dist_metric = "dot"
        beta, tau, softmax_BN = 0.0, 1.0, True
        share_subspace = self.kdq_share_subspace
    _, input_emb, losses = self.kdq(torch.reshape(input_emb, [-1, self.D, self.d_in]),
                               is_training=training)
    final_size = list(input.shape) + list(self.out_size)
    input_emb = torch.reshape(input_emb, final_size)
    if self.additive_quantization:
      input_emb = torch.mean(input_emb, -2)
    return input_emb, losses


class FullEmbed(nn.Module):
  def __init__(self, vocab_size, emb_size,
                      name="full_emb"):
    super().__init__()
    self.embedding = torch.randn([vocab_size, emb_size])

  def full_embed(self, input, training=True):
    """Full embedding baseline.

    Args:
      input: int multi-dim tensor, entity idxs.
      vocab_size: int, vocab size
      emb_size: int, output embedding size

    Returns:
      input_emb: float tensor, embedding for entity idxs.
    """
    
    input_emb = self.embedding[input]
    return input_emb
  

if __name__=="__main__":
     
  x = torch.randint(3,(4,4,4))
  m = KDQhparam()
  y = m(x)
  print(y)