import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from omegaconf.dictconfig import DictConfig


class TransformerDecoder(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.embed = nnx.Embed(num_embeddings=cfg.V, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs)
    self.pos_embed = nnx.Embed(num_embeddings=cfg.L, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs)
    self.blocks = [TransformerBlock(cfg, rngs) for _ in range(cfg.N)]
    self.out_ln = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.readout = nnx.Linear(in_features=cfg.D, out_features=cfg.V, use_bias=False, kernel_init=fsdp_init('zero', cfg), dtype=cfg.dtype, rngs=rngs)
  
  @nnx.jit
  def get_features(self, x):
    # Token + positional embedding
    h = self.embed(x) + self.pos_embed(jnp.arange(x.shape[1])[None, ...])  # [B, S, D]
    for block in self.blocks:
      h = block(h)
    return h
  
  @nnx.jit
  def get_features_and_logits(self, x):
    h = self.get_features(x)
    return h, self.readout(self.out_ln(h))

  def __call__(self, x):  # [B, S]
    h = self.get_features(x)
    h = self.out_ln(h)
    return self.readout(h)  # [B, S, O] where O is either cfg.O or cfg.V


class Attention(nnx.Module):
  """Custom multi-headed attention implementation with D x D projection matrices."""
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.num_heads = cfg.D // cfg.dh
    self.head_dim = cfg.dh
    self.scale = (1 / self.head_dim) ** 0.5
    # D x D projection matrices
    self.query_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.key_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.value_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.output_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('zero', cfg), dtype=cfg.dtype, rngs=rngs)
    
    # Layer normalization for query-key normalization
    self.q_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.k_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    
  def __call__(self, x, mask=None): # [B, S, D]
    B, S, D = x.shape
    H = self.num_heads
    
    q = self.query_proj(x) # [B, S, D]
    k = self.key_proj(x) # [B, S, D]
    v = self.value_proj(x) # [B, S, D]
    
    q = q.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
    
    q = self.q_norm(q)
    k = self.k_norm(k)
    
    attn_scores = jnp.einsum('bhsd,bhtd->bhst', q, k) * self.scale
    if mask is not None: attn_scores = jnp.where(mask, attn_scores, jnp.finfo(attn_scores.dtype).min)
    
    attn_probs = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum('bhst,bhtd->bhsd', attn_probs, v)
    
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    
    return self.output_proj(out)


class TransformerBlock(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.ln1 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.attn = Attention(cfg, rngs)
    self.ln2 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.mlp = Mlp(cfg, rngs)
    self.branch_multiplier = 1 / (cfg.N / 3) if cfg.scale_by_depth else 1
    
  def __call__(self, x):  # [B, S, D]
    h = self.ln1(x)
    # Create causal mask [B, S, S]
    mask = nnx.make_causal_mask(jnp.ones((x.shape[0], x.shape[1])), dtype=x.dtype)
    x = x + self.attn(h, mask=mask) * self.branch_multiplier
    return x + self.mlp(self.ln2(x)) * self.branch_multiplier


class Mlp(nnx.Module):
  """Multilayer perceptron."""
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(in_features=cfg.D, out_features=cfg.mlp_expansion*cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_kernel', cfg), dtype=cfg.dtype, rngs=rngs)
    self.fc2 = nnx.Linear(in_features=cfg.mlp_expansion*cfg.D, out_features=cfg.D, use_bias=False, kernel_init=fsdp_init('zero', cfg), dtype=cfg.dtype, rngs=rngs)
    
  def __call__(self, x):  # [B, S, D]
    h = jax.nn.gelu(self.fc1(x))  # [B, S, F]
    return self.fc2(h)  # [B, S, D]


def fsdp_init(layer_type: str, cfg: DictConfig):
  """Initialize weights with optional FSDP partitioning."""
  partition_fn = nnx.with_partitioning if cfg.fsdp_enabled else lambda x, _: x
  kernel_init = jax.nn.initializers.normal(stddev=cfg.init_std_mult*jnp.sqrt(1.0/cfg.D))
  embed_init = jax.nn.initializers.normal(stddev=cfg.init_std_mult*cfg.embed_init_std)
  zero_init = jax.nn.initializers.zeros
  match layer_type:
    case "embedding":  # [V, D]
      return partition_fn(embed_init, (None, "data"))
    case "attn_proj":  # [D, D]
      return partition_fn(kernel_init, ("data", None))
    case "mlp_kernel":  # [D, F]
      return partition_fn(kernel_init, ("data", None))
    case "zero":  # [D, O]
      return partition_fn(zero_init, ("data", None))
    case _:
      raise ValueError(f"unrecognized layer type: {layer_type}")


def create_sharded_model(c: DictConfig, mesh: Mesh, seed: int):
  """
  initialize sharded model without putting it on a single device
  https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html
  """

  @nnx.jit
  def initialize_sharded_model():
    model = TransformerDecoder(c, rngs=nnx.Rngs(seed)) # unsharded at this moment
    state = nnx.state(model) # the model's state, a pure pytree
    pspecs = nnx.get_partition_spec(state) # get annotations from state
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state) # the model is sharded now
    return model

  with mesh:
    model = initialize_sharded_model()

  return model