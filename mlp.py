import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig
from jax.sharding import Mesh

def fsdp_init(kind: str, model_cfg: DictConfig):
    """
    Initializes weights for MLP layers, aware of FSDP partitioning if enabled.
    Args:
        kind (str): Type of layer/weights to initialize ("embedding", "mlp_kernel", "readout").
        model_cfg (DictConfig): The model's configuration (e.g., cfg.model from main script)
                                expected to contain 'fsdp_enabled' and 'embed_init_std'.
    """
    # Use with_partitioning if fsdp_enabled is True in the model_cfg, otherwise, it's a no-op.
    # FSDP is not tested!
    partition_fn = nnx.with_partitioning if model_cfg.fsdp_enabled else (lambda func, _: func)

    # Define standard initializers
    variance_scaling_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    embedding_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    zeros_init = jax.nn.initializers.zeros

    if kind == "embedding":
        return partition_fn(embedding_init, (None, "data"))
    elif kind == "mlp_kernel":
        return partition_fn(variance_scaling_init, ("data", None))
    elif kind == "readout":
        return partition_fn(zeros_init, ("data", None))
    else:
        raise ValueError(f"Unknown initialization kind: {kind}")

class MLPBlock(nnx.Module):
    def __init__(self, model_cfg: DictConfig, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(model_cfg.D, use_scale=False, dtype=model_cfg.dtype, rngs=rngs)
        self.fc1 = nnx.Linear(
            in_features=model_cfg.D,
            out_features=model_cfg.D,
            use_bias=False,
            kernel_init=fsdp_init("mlp_kernel", model_cfg),
            rngs=rngs
        )
        self.fc2 = nnx.Linear(
            in_features=model_cfg.D,
            out_features=model_cfg.D,
            use_bias=False,
            kernel_init=fsdp_init("mlp_kernel", model_cfg),
            rngs=rngs
        )
        # Zero init output proj
        nnx.update(self.fc2.kernel, jnp.zeros_like(self.fc2.kernel))

        # Depth scaling multiplier (fixed to 1 at depth 6)
        self.depth_multiplier = 1.0 / (model_cfg.N / 6) if model_cfg.scale_by_depth else 1.0

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the MLP block with a residual connection."""
        x0 = x
        x = self.fc1(self.norm(x))
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x0 + x * self.depth_multiplier

class MLP(nnx.Module):
    def __init__(self, model_cfg: DictConfig, rngs: nnx.Rngs):
        # Embedding layer (1st layer)
        self.embed = nnx.Linear(
            in_features=model_cfg.V, # Input dimension
            out_features=model_cfg.D,
            use_bias=False,
            kernel_init=fsdp_init("embedding", model_cfg),
            rngs=rngs
        )

        self.blocks = [MLPBlock(model_cfg, rngs) for _ in range(model_cfg.N)]
        
        self.out_norm = nnx.RMSNorm(model_cfg.D, use_scale=False, dtype=model_cfg.dtype, rngs=rngs)
        # Readout layer (maps hidden dimension to a single output for regression)
        self.readout = nnx.Linear(
            in_features=model_cfg.D,
            out_features=1,
            use_bias=False,
            kernel_init=fsdp_init("readout", model_cfg), # Zero-initialized
            rngs=rngs
        )

    def get_features(self, x: jax.Array) -> jax.Array:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return h

    @nnx.jit
    def get_features_and_logits(self, x):
        h = self.get_features(x)
        return h, self.readout(self.out_norm(h))

    def __call__(self, x: jax.Array) -> jax.Array:
        features = self.get_features(x)
        return self.readout(self.out_norm(features))


def create_sharded_model(model_cfg: DictConfig, mesh: Mesh, seed: int) -> MLP:
    @nnx.jit
    def _initialize_and_shard_model():
        model_rngs = nnx.Rngs(params=jax.random.key(seed))
        model = MLP(model_cfg, model_rngs)
        model_state = nnx.state(model)
        partition_specs = nnx.get_partition_spec(model_state)
        sharded_state = jax.lax.with_sharding_constraint(model_state, partition_specs)
        nnx.update(model, sharded_state)
        return model
    with mesh:
        sharded_model = _initialize_and_shard_model()
    return sharded_model