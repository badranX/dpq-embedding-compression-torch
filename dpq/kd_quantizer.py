import torch
from torch.nn import functional as F
import torch.nn as nn


def safer_log(x, eps=1e-10):
  """Avoid nan when x is zero by adding small eps.

  Note that if x.dtype=tf.float16, \forall eps, eps < 3e-8, is equal to zero.
  """
  return torch.log(x + eps)


def sample_gumbel(shape):
  """Sample from Gumbel(0, 1)"""
  U = torch.rand(shape)
  return -safer_log(-safer_log(U))


class KDQuantizer(nn.Module):
  def __init__(self, K, D, d_in, d_out, tie_in_n_out,
               query_metric="dot", shared_centroids=False,
               beta=0., tau=1.0, softmax_BN=True):
    """
    Args:
      K, D: int, size of KD code.
      d_in: dim of continuous input for each of D axis.
      d_out: dim of continuous output for each of D axis.
      tie_in_n_out: boolean, whether or not to tie the input/output centroids.
        If True, it is vector quantization, else it is tempering softmax.
      query_metric: string, which metric to use for input centroid matching.
      shared_centroids: boolean, whether or not to share centroids for
        different bits in D.
      beta: float, KDQ regularization coefficient.
      tau: float or None, (tempering) softmax temperature.
        If None, set to learnable.
      softmax_BN: whether to use BN in (tempering) softmax.
    """
    super().__init__()
    self._K = K
    self._D = D
    self._d_in = d_in
    self._d_out = d_out
    self._tie_in_n_out = tie_in_n_out
    self._query_metric = query_metric
    self._shared_centroids = shared_centroids
    self._beta = beta
    if tau is None:
      self._tau = 1.0
    else:
      self._tau = tau
    self._softmax_BN = softmax_BN

    # Create centroids for keys and values.
    D_to_create = 1 if shared_centroids else D
    #TODO check initilizing parameters compatiblity with tf
    centroids_k = torch.randn([D_to_create, K, d_in])
    if tie_in_n_out:
      centroids_v = centroids_k
    else:
      centroids_v =  torch.randn([D_to_create, K, d_out])
    if shared_centroids:
      centroids_k = torch.tile(centroids_k, [D, 1, 1])
      if tie_in_n_out:
        centroids_v = centroids_k
      else:
        centroids_v = torch.tile(centroids_v, [D, 1, 1])
    self._centroids_k = nn.Parameter(centroids_k)
    self._centroids_v = nn.Parameter(centroids_v)
    self.batch_norm = nn.BatchNorm1d(self._D, affine=False)

  def forward(self,
              inputs,
              sampling=False,
              is_training=True):
    """Returns quantized embeddings from centroids.

    Args:
      inputs: embedding tensor of shape (batch_size, D, d_in)

    Returns:
      code: (batch_size, D)
      embs_quantized: (batch_size, D, d_out)
    """
  # House keeping.
    centroids_k = self._centroids_k
    centroids_v = self._centroids_v

    # Compute distance (in a metric) between inputs and centroids_k
    # the response is in the shape of (batch_size, D, K)
    if self._query_metric == "euclidean":
      norm_1 = torch.sum(inputs**2, -1, keepdim=True)  # (bs, D, 1)
      norm_2 = torch.unsqueeze(torch.sum(centroids_k**2, -1), 0)  # (1, D, K)
      dot = torch.matmul(torch.permute(inputs, [1, 0, 2]),
                      torch.permute(centroids_k, [0, 2, 1]))  # (D, bs, K)
      response = -norm_1 + 2 * torch.permute(dot, [1, 0, 2]) - norm_2
    elif self._query_metric == "cosine":
      inputs = F.normalize(inputs, p=2, dim=-1)
      centroids_k = F.normalize(centroids_k, p=2, dim=-1)
      response = torch.matmul(torch.permute(inputs, [1, 0, 2]),
                      torch.permute(centroids_k, [0, 2, 1]))  # (D, bs, K)
      response = torch.permute(response, [1, 0, 2])
    elif self._query_metric == "dot":
      response = torch.matmul(torch.permute(inputs, [1, 0, 2]),
                      torch.permute(centroids_k, [0, 2, 1]))  # (D, bs, K)
      response = torch.permute(response, [1, 0, 2])
    else:
      raise ValueError("Unknown metric {}".format(self._query_metric))
    response = torch.reshape(response, [-1, self._D, self._K])
    if self._softmax_BN:
      # response = tf.contrib.layers.instance_norm(
      #    response, scale=False, center=False,
      #    trainable=False, data_format="NCHW")

      response = self.batch_norm(response)
      #TODO check compatability 
      ###TODO response = tf.layers.batch_normalization(response, scale=False, center=False, training=is_training)

      # Layer norm as alternative to BN.
      # response = tf.contrib.layers.layer_norm(
      #    response, scale=False, center=False)
    response_prob = torch.softmax(response / self._tau, -1)

    # Compute the codes based on response.
    codes = torch.argmax(response, -1)  # (batch_size, D)
    if sampling:
      response = safer_log(response_prob)
      noises = sample_gumbel(response.shape)
      neighbor_idxs = torch.argmax(response + noises, -1)  # (batch_size, D)
    else:
      neighbor_idxs = codes

    # Compute the outputs, which has shape (batch_size, D, d_out)
    if self._tie_in_n_out:
      if not self._shared_centroids:
        D_base = torch.tensor(
            [self._K*d for d in range(self._D)], dtype=torch.int64)
        neighbor_idxs += torch.unsqueeze(D_base, 0)  # (batch_size, D)
      neighbor_idxs = torch.reshape(neighbor_idxs, [-1])  # (batch_size * D)
      centroids_v = torch.reshape(centroids_v, [-1, self._d_out])
      outputs = centroids_v[neighbor_idxs]
      outputs = torch.reshape(outputs, [-1, self._D, self._d_out])
      outputs_final = (outputs - inputs).detach() + inputs
    else:
      nb_idxs_onehot = F.one_hot(neighbor_idxs,
                                  self._K)  # (batch_size, D, K)
      nb_idxs_onehot = response_prob - (response_prob - nb_idxs_onehot).detach()
      # nb_idxs_onehot = response_prob  # use continuous output
      outputs = torch.matmul(
          torch.permute(nb_idxs_onehot, [1, 0, 2]),  # (D, bs, K)
          centroids_v)  # (D, bs, d)
      outputs_final = torch.permute(outputs, [1, 0, 2])

    # Add regularization for updating centroids / stabilization.
    #TODO is_training?
    if is_training:
      #print("[INFO] Adding KDQ regularization.")
      if self._tie_in_n_out:
        alpha = 1.
        beta = self._beta
        gamma = 0.0
        reg = alpha * torch.mean(
            (outputs - inputs.detach())**2)
        reg += beta * torch.mean(
            (outputs.detach() - inputs)**2)
        minaxis = [0, 1] if self._shared_centroids else [0]
        reg += gamma * torch.mean(  # could sg(inputs), but still not eff.
            torch.mean(-response, minaxis))
      else:
        beta = self._beta
        reg = - beta * torch.mean(
            torch.sum(nb_idxs_onehot * safer_log(response_prob), [2]))
        # entropy regularization
        # reg = - beta * tf.reduce_mean(
        #    tf.reduce_sum(response_prob * safer_log(response_prob), [2]))
        
      losses = {"regulization_loss": reg}
      #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)

    return codes, outputs_final, losses


if __name__ == "__main__":
  # VQ
  kdq_demo = KDQuantizer(100, 10, 5, 5, True, "euclidean")
  codes_vq, outputs_vq, losses = kdq_demo(torch.randn([64, 10, 5]))
  # tempering softmax
  kdq_demo = KDQuantizer(100, 10, 5, 10, False, "dot")
  codes_ts, outputs_ts, losses = kdq_demo(torch.randn([64, 10, 5]))
  print(codes_ts)
  print(outputs_ts)
