import jax
import jax.numpy as jnp
from flax import nnx
from collections.abc import Mapping
from flax import traverse_util  # tiny helper, works without Flax too
from functools import partial
from typing import Any, Dict, Tuple
import jax.tree_util as tu
from jax.tree_util import tree_map
from jax import lax


def flatten_dict(d, prefix=None):
  if isinstance(d, Mapping):
    out = {}
    for k, v in d.items():
      nested_prefix = k if prefix is None else f'{prefix}/{k}'
      out |= flatten_dict(v, nested_prefix)
    return out
  else:
    return {prefix: d}

def get_scheduler(schedule, decay_frac, warmup, total_steps):
    """
    Returns a piecewise function of iteration => scale factor in [0, 1].
    """
    schedules = {
        'const': lambda _: 1.0,
        'cosine': lambda t: 0.5 * (1 + jnp.cos(jnp.pi * t)),
        'linear': lambda t: 1.0 - t,
        'quadratic': lambda t: 1.0 - t**2,
        'power': lambda t: (1 - t)**2,
        'exp': lambda t: 10.0 ** (-t),
        'multicosine': lambda t: 0.5 * (1 + jnp.cos(3*jnp.pi * t))
    }
    if schedule == 'const' or decay_frac == 0:
        base_fn = lambda _: 1.0
    else:
        base_fn = lambda t: schedules[schedule](
            jnp.maximum(0.0, (t - (1 - decay_frac)) / decay_frac)
        ).clip(min=0)
    if warmup > 0:
        return lambda t: jnp.where(
            t < warmup,
            t / warmup,
            base_fn((t - warmup)/(total_steps - warmup))
        )
    else:
        return lambda t: base_fn(t / total_steps)


@jax.jit
def tree_product(xs, ys):
    products = tu.tree_map(lambda x, y: jnp.sum(x*y), xs, ys)
    return sum(tu.tree_leaves(products))

@jax.jit
def tree_add(xs, ys, alpha=1.0):
    return tree_map(lambda x, y: x + alpha * y, xs, ys)

@jax.jit
def tree_take(t, idx):
    """Return the idx-th vector from a (order, …) pytree."""
    return tree_map(lambda a: a[idx], t)

@jax.jit
def tree_set(t, idx, value):
    """Set the idx-th vector to *value* in a (order, …) pytree."""
    return tree_map(lambda a, v: a.at[idx].set(v), t, value)

@partial(jax.jit, static_argnames=("rand_fn"))
def tree_random(params, key, rand_fn):
    keys = jax.random.split(key, len(tu.tree_leaves(params)))
    keys = tu.tree_unflatten(tu.tree_structure(params), keys)
    return tree_map(lambda p, k: rand_fn(k, p.shape, dtype=p.dtype), params, keys)

@jax.jit
def tree_normalize(vs):
    norm = jnp.sqrt(tree_product(vs, vs))
    return tree_map(lambda v: v / (norm + 1e-6), vs)

@jax.jit
def tree_orthogonalize(v, basis_vectors):
    for u in basis_vectors:
        proj = tree_product(v, u)
        v = tree_add(v, u, alpha=-proj)
    return tree_normalize(v)

# evaluation utilities
def zeros_like(pytree):
    """Create a pytree of zeros with the same structure/shapes as *pytree*."""
    return tu.tree_map(jnp.zeros_like, pytree)

def welford_step(mean, M2, sample, count):
    """One‑pass Welford pgrad on pytrees.

    Args:
      mean:   running mean (pytree).
      M2:     running sum of squares of deviations (pytree).
      sample: new sample (same structure).
      count:  current 1‑based index (int).
    Returns:
      (new_mean, new_M2)
    """
    delta  = tree_add(sample, mean, alpha=-1)                 # g − μ
    mean   = tree_add(mean, delta, alpha=1 / count)           # μ ← μ + δ/i
    delta2 = tree_add(sample, mean, alpha=-1)                 # g − μ′
    M2     = tree_add(M2, tu.tree_map(lambda x, y: x * y, delta, delta2))
    return mean, M2


def eval_step_fn(
    params: Any,
    dataset: Any,
    opt_state: Any,
    sqrt_P_inv: Any,
    loss_and_grad_fn: Any,
) -> Dict[str, Any]:
    """ Log validation loss and gradient metrics """

    zero_tree = zeros_like(params)
    carry = dict(
        opt_state       = opt_state,      # unchanged during scan
        test_loss       = 0.0,
        sum_pgrad_l2   = 0.0,            # Σ‖u_b‖²
        sum_grad_l2     = 0.0,            # Σ‖g_b‖²
        grad_mean       = zero_tree,      # Welford μ
        grad_M2         = zero_tree,      # Welford Σ(δ⋅δ′)
        pgrad_mean     = zero_tree,      # Welford μ
        pgrad_M2       = zero_tree,      # Welford Σ(δ⋅δ′)
        i               = 1,              # batch counter (1‑based)
    )

    def _body(carry: Dict[str, Any], batch: Any) -> Tuple[Dict[str, Any], None]:
        i = carry['i']

        loss, grads = loss_and_grad_fn(params, batch)
        pgrad      = sqrt_P_inv(grads, carry['opt_state'])  # P^{-1/2} g

        # Online mean / variance (Welford)
        grad_mean, grad_M2 = welford_step(carry['grad_mean'], carry['grad_M2'], grads, i)
        pgrad_mean, pgrad_M2 = welford_step(carry['pgrad_mean'], carry['pgrad_M2'], pgrad, i)

        new_carry = dict(
            opt_state      = carry['opt_state'],
            i              = i + 1,
            grad_mean      = grad_mean,
            grad_M2        = grad_M2,
            pgrad_mean    = pgrad_mean,
            pgrad_M2      = pgrad_M2,
            test_loss      = carry['test_loss'] + loss,
            sum_pgrad_l2  = carry['sum_pgrad_l2'] + tree_product(pgrad, pgrad),
            sum_grad_l2    = carry['sum_grad_l2']  + tree_product(grads, grads),
        )
        return new_carry, None

    carry, _ = jax.lax.scan(_body, carry, dataset)

    N = len(dataset) # Num batches
    # Averages
    test_loss    = carry['test_loss'] / N
    g2_mean  = carry['sum_grad_l2']  / N
    u2_mean = carry['sum_pgrad_l2'] / N
    gmean   = carry['grad_mean']
    umean = carry['pgrad_mean']
    gmean2 = tree_product(gmean, gmean)
    umean2 = tree_product(umean, umean)

    # Batch sizes
    if isinstance(dataset, dict):
        B = dataset['patches'][0].shape[0]
    else:
        first_sample = dataset[0]
        if isinstance(first_sample, (tuple, list)):
          B = dataset[0][0].shape[0]
        else:
          B = dataset[0].shape[0]

    # Welford final moments
    Bgvar_flat = tree_map(lambda x: x.reshape(-1), carry['grad_M2'])
    Bgvar_flat = jnp.concatenate(tu.tree_leaves(Bgvar_flat)) / N
    Bgvar = jnp.sum(Bgvar_flat) # Tr(Cov(g_batch)) per batch variance
    gvar = Bgvar * B # Tr(Cov(g_example)) per example variance

    # for preconditioned grads
    Buvar_flat = tree_map(lambda x: x.reshape(-1), carry['pgrad_M2'])
    Buvar_flat = jnp.concatenate(tu.tree_leaves(Buvar_flat)) / N
    Buvar = jnp.sum(Buvar_flat) # Tr(Cov(u_batch)) per batch variance
    uvar = Buvar * B # Tr(Cov(u_example)) per example variance
    
    metrics = dict(
        test_loss          = test_loss,
        gvar               = gvar,
        uvar               = uvar,
        Bgvar              = Bgvar,
        Buvar              = Buvar, # This is what we use for predicting loss curves
        g2_mean            = g2_mean,
        u2_mean            = u2_mean,
        gmean2             = gmean2,
        umean2             = umean2,
    )

    return metrics
